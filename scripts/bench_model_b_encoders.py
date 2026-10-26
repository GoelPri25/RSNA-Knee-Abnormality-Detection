"""Model B encoder benchmark (timing + equivalence only; no training run, no scoring).

Same head, data, augmentation and optimizer as scripts/swin3d_d96n_runner.py; only the encoder changes. Every encoder
keeps the B1 split: lower part shared across the 4 series, top part one copy per plane, then one token per depth
position (spatial mean) into the per-finding attention head.

  swin_tv_320     Swin3D-T as profiled (torchvision attention), 320
  swin_sdpa_320   the same weights and maths with fused attention (F.scaled_dot_product_attention + cached shift masks)
  swin_sdpa_256   as above at 256 (the 320 crop resized)
  vmae_s_224      VideoMAE V2 ViT-S/16 distilled from ViT-g (K710; HF OpenGVLab/VideoMAE2, Apache-2.0), tubelet
                  2 x 16 x 16, fixed 1-D sin-cos positions; blocks 0-8 shared, 9-11 + fc_norm per plane; 320 crop -> 224
  x3d_m_320       X3D-M (pytorchvideo, Apache-2.0); stem + stages 1-3 shared, stage 4 per plane; BN statistics frozen

Data: the timed studies are preloaded into RAM (compute only); separately, cold read throughput of the real
volume loader. Checks: Swin fused attention vs torchvision on one study (fp32, eval), pretrained weights load strictly.

Env: D96_CACHE (d96n), SWIN_WEIGHTS, VMAE_WEIGHTS, X3D_WEIGHTS, D96_OUT.
"""
import copy
import functools
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import swin3d_d96n_runner as b

VMAE = Path(os.environ.get("VMAE_WEIGHTS", "/root/vit_s_k710_dl_from_giant.pth"))
VMAE_SHA = "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"     # HF LFS sha256
X3D = Path(os.environ.get("X3D_WEIGHTS", "/root/X3D_M.pyth"))


# ---------------------------------------------------------------- Swin3D fused attention (same maths)
@functools.lru_cache(maxsize=64)
def _shift_mask(size_dhw, window_size, shift_size, device):
    import torchvision.models.video.swin_transformer as tvs
    return tvs._compute_attention_mask_3d(torch.zeros(1, device=device), size_dhw, window_size, shift_size)


def swa3d_sdpa(input, qkv_weight, proj_weight, relative_position_bias, window_size, num_heads, shift_size,
               attention_dropout=0.0, dropout=0.0, qkv_bias=None, proj_bias=None, training=True):
    """torchvision shifted_window_attention_3d with the softmax(QK^T * s + bias + mask) V step as one fused kernel."""
    import torchvision.models.video.swin_transformer as tvs
    b_, t, h, w, c = input.shape
    pad = tvs._compute_pad_size_3d((t, h, w), (window_size[0], window_size[1], window_size[2]))
    x = F.pad(input, (0, 0, 0, pad[2], 0, pad[1], 0, pad[0]))
    _, tp, hp, wp, _ = x.shape
    shifted = sum(shift_size) > 0
    if shifted:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
    nw = (tp // window_size[0]) * (hp // window_size[1]) * (wp // window_size[2])
    n = window_size[0] * window_size[1] * window_size[2]
    x = x.view(b_, tp // window_size[0], window_size[0], hp // window_size[1], window_size[1], wp // window_size[2],
               window_size[2], c).permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(b_ * nw, n, c)
    qkv = F.linear(x, qkv_weight, qkv_bias).reshape(x.size(0), n, 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    bias = relative_position_bias.to(q.dtype)                                     # [1, heads, n, n]
    if shifted:
        m = _shift_mask((tp, hp, wp), tuple(window_size), tuple(shift_size), x.device)  # [nW, n, n]
        bias = (bias + m[:, None].to(q.dtype)).repeat(b_, 1, 1, 1)               # [b * nW, heads, n, n]
    else:
        bias = bias.expand(x.size(0), -1, -1, -1)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=attention_dropout if training else 0.0)
    x = F.dropout(F.linear(x.transpose(1, 2).reshape(x.size(0), n, c), proj_weight, proj_bias), p=dropout,
                  training=training)
    x = x.view(b_, tp // window_size[0], hp // window_size[1], wp // window_size[2], window_size[0], window_size[1],
               window_size[2], c).permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b_, tp, hp, wp, c)
    if shifted:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
    return x[:, :t, :h, :w, :].contiguous()


def set_swin_attention(fused):
    import torchvision.models.video.swin_transformer as tvs
    if not hasattr(tvs, "_orig_swa3d"):
        tvs._orig_swa3d = tvs.shifted_window_attention_3d
    tvs.shifted_window_attention_3d = swa3d_sdpa if fused else tvs._orig_swa3d


# ---------------------------------------------------------------- shared head (as SwinB)
class HeadMixin:
    def init_head(self, d):
        self.norm = nn.LayerNorm(d)
        self.series = nn.Embedding(4, d)
        nn.init.trunc_normal_(self.series.weight, std=0.02)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, d))
        self.clsb = nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def forward(self, series):
        return b.SwinB.forward(self, series)


# ---------------------------------------------------------------- VideoMAE V2 ViT-S/16
class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__(); self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


class VAttn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        bsz, n, c = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        q, k, v = F.linear(x, self.qkv.weight, bias).reshape(bsz, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bsz, n, c))


class VMlp(nn.Module):
    def __init__(self, d):
        super().__init__(); self.fc1, self.act, self.fc2 = nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VBlock(nn.Module):
    def __init__(self, d, heads, dp):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(d, eps=1e-6), VAttn(d, heads)
        self.norm2, self.mlp = nn.LayerNorm(d, eps=1e-6), VMlp(d)
        self.drop_path = DropPath(dp)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


@functools.lru_cache(maxsize=64)
def sincos(n, d, device):
    """VideoMAE's fixed 1-D sinusoid table over the flattened (t, h, w) token index."""
    pos = np.arange(n)[:, None] / np.power(10000, 2 * (np.arange(d) // 2) / d)[None]
    pos[:, 0::2], pos[:, 1::2] = np.sin(pos[:, 0::2]), np.cos(pos[:, 1::2])
    return torch.tensor(pos, dtype=torch.float32, device=device)[None]


class VMAEB(HeadMixin, nn.Module):
    SHARED = 9

    def __init__(self, pretrained=True, drop_path=0.1):
        super().__init__()
        d, depth = 384, 12
        dpr = np.linspace(0, drop_path, depth)
        self.patch_embed = nn.Conv3d(3, d, (2, 16, 16), (2, 16, 16))
        blocks = [VBlock(d, 6, float(p)) for p in dpr]
        fc_norm = nn.LayerNorm(d, eps=1e-6)
        if pretrained:
            assert hashlib.sha256(VMAE.read_bytes()).hexdigest() == VMAE_SHA
            ck = torch.load(VMAE, map_location="cpu", weights_only=True)
            sd = ck.get("model", ck.get("module", ck))
            tmp = nn.Module(); tmp.patch_embed = nn.Module(); tmp.patch_embed.proj = self.patch_embed
            tmp.blocks, tmp.fc_norm = nn.ModuleList(blocks), fc_norm
            sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
            miss, extra = tmp.load_state_dict(sd, strict=False)
            assert not miss and not extra, (miss, extra)
        self.shared = nn.ModuleList(blocks[:self.SHARED])
        top = nn.ModuleList(blocks[self.SHARED:] + [fc_norm])
        self.branch = nn.ModuleList([copy.deepcopy(top) for _ in range(3)])
        self.init_head(d)

    def encode(self, v, j):
        x = self.patch_embed(v)                                                 # [1, 384, T', H', W']
        t_, h_, w_ = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + sincos(x.shape[1], x.shape[2], x.device).to(x.dtype)
        for m in self.shared:
            x = m(x)
        br = self.branch[b.PLANE[j]]
        for m in br[:-1]:
            x = m(x)
        return br[-1](x.view(t_, h_ * w_, -1).mean(1))                          # fc_norm after the spatial mean


# ---------------------------------------------------------------- X3D-M
class X3DB(HeadMixin, nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from pytorchvideo.models.hub.x3d import x3d_m
        net = x3d_m(pretrained=False)
        if pretrained:
            ck = torch.load(X3D, map_location="cpu", weights_only=True)
            net.load_state_dict(ck["model_state"], strict=True)
        blk = net.blocks
        assert len(blk) == 6
        self.patch_embed = blk[0]
        self.shared = nn.ModuleList(blk[1:4])
        self.branch = nn.ModuleList([copy.deepcopy(blk[4]) for _ in range(3)])
        self.init_head(192)

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():                                                # BN statistics frozen
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def encode(self, v, j):
        x = self.patch_embed(v)
        for m in self.shared:
            x = m(x)
        x = self.branch[b.PLANE[j]](x)                                          # [1, 192, T, H/32, W/32]
        return x.mean((3, 4))[0].transpose(0, 1)


# ---------------------------------------------------------------- benchmark
class Ram(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def bench():
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    labels, ref, meta, train_ids, val_ids, gold_ids = b.setup()
    prep = b.Prep(*b.kinetics_norm(), dev)
    t0 = time.monotonic()
    tr_items = [b.Volumes(train_ids[:56], meta, labels)[i] for i in range(56)]
    va_items = [b.Volumes(val_ids[:40], meta, {})[i] for i in range(40)]
    cold1 = (time.monotonic() - t0) / 96
    out = {"cold_read_1proc_s_study": round(cold1, 3)}
    for workers in (8, 16):
        ids = train_ids[200 + 150 * (workers // 16):350 + 150 * (workers // 16)]
        dl = torch.utils.data.DataLoader(b.Volumes(ids, meta, labels), batch_size=1, num_workers=workers,
                                         collate_fn=lambda z: z[0], prefetch_factor=4)
        t0 = time.monotonic()
        mb = sum(v[0].numel() for v in dl) / 2**20
        dt = time.monotonic() - t0
        out[f"cold_loader_{workers}w_s_study"] = round(dt / len(ids), 3)
        out[f"cold_loader_{workers}w_MBps"] = round(mb / dt, 1)
    print("IO " + json.dumps(out), flush=True)

    # equivalence: fused Swin attention vs torchvision (fp32, eval, one validation study)
    m = b.SwinB(pretrained=True).to(dev).eval()
    ser = prep(va_items[0][0], va_items[0][1], False, 320)
    with torch.no_grad():
        set_swin_attention(False); ref_out = m(ser)
        set_swin_attention(True); new_out = m(ser)
    d_eq = (ref_out - new_out).abs().max().item()
    out["swin_fused_vs_tv_max_abs_logit_fp32"] = d_eq
    print(f"EQUIV fused vs torchvision Swin attention: max |d logit| {d_eq:.2e}", flush=True)
    assert d_eq < 1e-3, d_eq
    del m; torch.cuda.empty_cache()

    configs = [("swin_tv_320", lambda: b.SwinB(True), False, 320), ("swin_sdpa_320", lambda: b.SwinB(True), True, 320),
               ("swin_sdpa_256", lambda: b.SwinB(True), True, 256), ("vmae_s_224", lambda: VMAEB(True), True, 224),
               ("x3d_m_320", lambda: X3DB(True), True, 320)]
    for name, make, fused, res in configs:
        set_swin_attention(fused)
        try:
            model = make().to(dev); model.train()
            n_par = sum(p.numel() for p in model.parameters()) / 1e6
            opt = torch.optim.AdamW(b.param_groups(model))
            torch.cuda.reset_peak_memory_stats(); n, t0 = 0, None
            dl = torch.utils.data.DataLoader(Ram(tr_items), batch_size=1, shuffle=True, num_workers=4,
                                             collate_fn=lambda z: z[0])
            for i, (vol, slot, y, w) in enumerate(dl):
                if i == 8:
                    torch.cuda.synchronize(); t0 = time.monotonic()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(prep(vol, slot, True, res)).float()
                loss = (F.binary_cross_entropy_with_logits(logit[0], y.to(dev), reduction="none") * w.to(dev)).sum() \
                    / w.sum().clamp_min(1)
                (loss / 8).backward()
                if (i + 1) % 8 == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
                n += t0 is not None
            torch.cuda.synchronize(); tr = (time.monotonic() - t0) / n
            mem = torch.cuda.max_memory_allocated() / 2**30
            model.eval()
            with torch.no_grad():
                torch.cuda.synchronize(); t0 = time.monotonic()
                for vol, slot, _, _ in va_items:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        model(prep(vol, slot, False, res))
                torch.cuda.synchronize(); te = (time.monotonic() - t0) / len(va_items)
            run_h = (tr * 3524 + te * 825 * 9 / 25) * 25 / 3600 + 4 * te * 883 / 3600   # eval at 9 of 25 epochs
            out[name] = {"params_M": round(n_par, 1), "train_s_study": round(tr, 3), "eval_s_study": round(te, 3),
                         "peak_gib": round(mem, 1), "run_h_25ep_compute": round(run_h, 2)}
            del model, opt
        except Exception as e:                                                  # report and continue
            out[name] = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
        torch.cuda.empty_cache()
        print("BENCH " + json.dumps({name: out[name]}), flush=True)
    set_swin_attention(False)
    print("BENCH_DONE " + json.dumps(out), flush=True)


if __name__ == "__main__":
    bench()
