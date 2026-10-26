"""Model B (Phase 2, pre-registered in the ledger before launch): a 3D video transformer on whole native-density series.

Not a fork of the 1e window recipe. Input = the d96n cache (every native slice of up to 4 series per study, capped at
40 per series: sagittal fluid, sagittal non-fluid, coronal fluid, axial fluid-preferred; 384 px over 140 mm, uint8).
Each series is one 3D volume [T, 320, 320] (grey replicated to 3 channels, Kinetics normalisation), the same layout in
training and evaluation.

Encoder: torchvision Swin3D-T (Kinetics-400 weights, loaded weights_only from the file baked into the image).
  shared across all series: patch embedding + stages 1-3 (features[0:5]);
  one branch per plane (sagittal = both sagittal series, coronal, axial): patch merging 3 + stage 4 + final norm
  (features[5:7] + norm), each branch initialised from the same pretrained weights.
Per series, stage-4 tokens [T/2, 10, 10, 768] are averaged over space -> one token per depth position. Head (as 1e's
Raptor, over tokens instead of windows): LayerNorm + learned series embedding, per-finding attention over all tokens
of all series, per-finding linear read-out.

Training: per-target labels (sha d40803d0...), fold 0 held out; one study per forward, 8 studies per optimizer step;
AdamW (backbone lr, head lr, wd 0.05 except norms / biases / relative position tables), 1 warm-up epoch, cosine to a
1% floor over 25 epochs, SWA of epochs 21-25 (LayerNorm only: no BN refresh), bf16 autocast, seed 2026.
Augmentation per study (1e's values): anatomical mirror p 0.5 (sagittal series reverse slice order, others flip
width), one affine (rotation +-10 deg, scale +-10%, shift +-6%) sampled at native pixel scale into the 320 crop, gamma
exp(+-0.25), gain +-10%. Evaluation: centre 320 crop, no TTA (primary); mirror TTA secondary.

Model B2 (SWIN_PHASE=swin3d_b2, pre-registered in the ledger): B1 overfit after epoch 10, so B2 trains less of the
encoder and regularises harder. Labels = label-v2 (teammate, sha dd0ab9ab..., LB 0.937 as 1e). Patch embedding +
stages 1-2 frozen (no gradient, no drop path); ONE stage-4 branch shared by all planes; drop path 0.25 (linear over
the 12 blocks, torchvision's rule); layer-wise lr decay 0.9 per unit from the top (B1's 5e-5 at the final norm);
per-finding attention over every stage-4 token (T/2 x 10 x 10 per series) instead of the spatial mean; 15 epochs,
SWA 11-15; early stop if val macro (12 targets) < 0.880 at epoch 9. Training augmentation adds: series dropout p 0.1
each (one always kept), affine / gamma / gain drawn per series, every second slice p 0.25 (series of >= 16 rows),
smooth bias field p 0.3, blur or sharpen p 0.2 each, Gaussian noise p 0.2 (no depth crop: the series ends hold
patella, suprapatellar recess, popliteal fossa and the compartment edges). Evaluation unchanged.

Env: D96_CACHE (d96n), D96_OUT, SWIN_PHASE (swin3d_b | swin3d_b2), SWIN_MODE (train | profile), SWIN_WEIGHTS;
SWIN_AMP (bf16 default | fp16 + GradScaler, for T4 GPUs without bf16); SWIN_CFG (JSON merged into the B2 config,
aug2 merged key by key: Kaggle ablation arms, logged in CFG). Unset = the Modal B1 / B2 runs unchanged.
Outputs match the 1e runner (val_swa.json, gold_swa.json, val_swa_mirror.json, history.json, ...) so
scripts/audit_phase1.py scores them unchanged.
"""
import contextlib
import copy
import functools
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

PHASE = os.environ.get("SWIN_PHASE", "swin3d_b")
os.environ["D96_PHASE"] = "r50_native_pertarget"      # 1e module used for IO only: native manifest, labels, scoring

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import coatnet_d96_runner as r

assert r.NATIVE_LAYOUT and r.CFG["sha256"]["labels"].startswith("d40803d0")
B2 = PHASE == "swin3d_b2"
if B2:                                                 # label-v2 (teammate): 1e labels + lateral OA / MCL / fx / contusion
    r.LABELS_FILE = "soft_labels_final_per_target_latD2.json"
    r.CFG["sha256"]["labels"] = "dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d"
OUT = Path(os.environ.get("D96_OUT", f"/root/vol/experiments/coatnet_d96_fold0/run_{PHASE}"))
WEIGHTS = Path(os.environ.get("SWIN_WEIGHTS", "/root/swin3d_t_k400.pth"))
CFG = {
    "version": "d96n_swin3d_t_b_v1", "res": 320, "img": 384, "epochs": 25, "warmup_epochs": 1, "lr_floor": 0.01,
    "swa_epochs": [21, 22, 23, 24, 25], "lr": 5e-5, "head_lr": 1e-3, "wd": 0.05, "accum": 8, "grad_clip": 1.0,
    "ckpt": False, "seed": 2026, "workers": 6,
    # B1 after the encoder benchmark: fused attention (same maths as torchvision, max |d logit| 7e-5 fp32; 0.233 ->
    # 0.163 s/study) and validation scored only at these epochs (enough for the pre-registered tuning rules)
    "attn": "sdpa", "eval_epochs": [1, 3, 5, 10, 15, 20, 21, 22, 23, 24, 25],
    "aug": {"mirror_p": 0.5, "rot_deg": 10.0, "scale": 0.10, "shift": 0.06, "gamma": 0.25, "gain": 0.10},
    "weights_sha256": "",   # filled from the baked file at start and logged
}
if B2:
    CFG.update({
        "version": "d96n_swin3d_t_b2_v1", "labels": "label-v2 dd0ab9ab", "epochs": 15, "swa_epochs": [11, 12, 13, 14, 15],
        "lr": 5e-5, "llrd": 0.9, "drop_path": 0.25, "freeze": 4, "branches": 1, "pool": "tokens",
        "eval_epochs": [1, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15], "kill": {"epoch": 9, "min_val": 0.880},
        "aug2": {"series_drop_p": 0.1, "stride2_p": 0.25, "stride2_min": 16,
                 "bias_p": 0.3, "bias_amp": 0.3, "blur_p": 0.2, "sharpen_p": 0.2, "noise_p": 0.2, "noise_std": 0.03}})
if os.environ.get("SWIN_CFG"):                         # ablation arm on top of B2 (label-v2)
    assert B2, "SWIN_CFG applies to swin3d_b2 only"
    _o = json.loads(os.environ["SWIN_CFG"])
    CFG.update({**_o, "aug2": {**CFG["aug2"], **_o.get("aug2", {})}})
AMP = torch.float16 if os.environ.get("SWIN_AMP") == "fp16" else torch.bfloat16
if AMP == torch.float16:
    CFG["amp"] = "fp16"
PLANE = {0: 0, 1: 0, 2: 1, 3: 2}                      # d96n slot -> branch: sagittal, coronal, axial
SAGITTAL = (0, 1)
TARGETS = r.TARGETS


# ---------------------------------------------------------------- data
class Volumes(torch.utils.data.Dataset):
    """One study -> (uint8 [n, 384, 384] rows of the d96n cache, slot per row [n], labels y [12], weights w [12])."""

    def __init__(self, ids, meta, labels):
        self.ids, self.meta, self.labels, self._mm = list(ids), meta, labels, {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        m = self.meta[self.ids[i]]
        if m["file"] not in self._mm:
            self._mm[m["file"]] = np.load(m["file"], mmap_mode="r")
        vol = np.array(self._mm[m["file"]][m["row"]:m["row"] + m["n"]])   # copy: the mmap slice is read-only
        y, w = self.labels.get(self.ids[i], (np.zeros(12, np.float32), np.zeros(12, np.float32)))
        return torch.from_numpy(vol), torch.from_numpy(m["slot"].copy()), torch.from_numpy(y), torch.from_numpy(w)


def loader(ds, shuffle, seed=0):
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=CFG["workers"], pin_memory=True,
                                       collate_fn=lambda b: b[0], prefetch_factor=4,
                                       generator=torch.Generator().manual_seed(seed))


class Prep:
    """GPU preprocessing of one study into [(slot, [1, 3, T, res, res])] in slot order (empty slots skipped)."""

    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
        self.device = device

    def draw(self):
        """One affine (into the 320 crop at native pixel scale) + gamma + gain."""
        a, f = CFG["aug"], CFG["res"] / CFG["img"]
        ang = math.radians(random.uniform(-1, 1) * a["rot_deg"])
        sc = 1 + random.uniform(-1, 1) * a["scale"]
        tx, ty = (random.uniform(-1, 1) * a["shift"] * 2 for _ in range(2))
        c, s = math.cos(ang) / sc * f, math.sin(ang) / sc * f
        theta = torch.tensor([[c, -s, tx], [s, c, ty]], device=self.device)
        return theta, math.exp(random.uniform(-1, 1) * a["gamma"]), 1 + random.uniform(-1, 1) * a["gain"]

    def mri(self, v, a2):
        """B2 MR-specific intensity augmentation of one series v [T, R, R] in [0, 1] (same draw for all its slices)."""
        if random.random() < a2["bias_p"]:                # smooth multiplicative bias field
            g = F.interpolate(torch.randn(1, 1, 4, 4, device=v.device), size=v.shape[-2:], mode="bicubic",
                              align_corners=True)[0, 0]
            v = v * torch.exp(random.uniform(0, a2["bias_amp"]) * g / g.abs().amax().clamp_min(1e-6))
        u = random.random()
        if u < a2["blur_p"] + a2["sharpen_p"]:            # Gaussian blur, or unsharp mask, sigma 0.5-1.0 px
            k = torch.exp(-torch.arange(-2, 3, device=v.device, dtype=v.dtype) ** 2 / (2 * random.uniform(0.5, 1.0) ** 2))
            k = k / k.sum()
            b = F.conv2d(F.conv2d(F.pad(v[:, None], (2, 2, 2, 2), mode="reflect"), k.view(1, 1, 1, 5)), k.view(1, 1, 5, 1))[:, 0]
            v = b if u < a2["blur_p"] else v + random.uniform(0.3, 0.8) * (v - b)
        if random.random() < a2["noise_p"]:
            v = v + torch.randn_like(v) * random.uniform(0, a2["noise_std"])
        return v.clamp(0, 1)

    def __call__(self, vol, slot, train, res, mirror=False):
        x = vol.to(self.device, non_blocking=True).float().div_(255.0)          # [n, 384, 384]
        slot = slot.numpy()
        a2 = CFG.get("aug2") if train else None
        if train:
            mirror = random.random() < CFG["aug"]["mirror_p"]                     # per study (anatomy stays consistent)
            if not a2:
                theta, gamma, gain = self.draw()                                  # B1: one draw per study
        spans = [(j, np.where(slot == j)[0]) for j in range(4)]
        spans = [(j, int(rw[0]), int(rw[-1]) + 1) for j, rw in spans if len(rw)]  # rows of a slot are contiguous
        if a2 and len(spans) > 1:                                                 # series dropout, one always kept
            keep = [sp for sp in spans if random.random() >= a2["series_drop_p"]]
            spans = keep or [random.choice(spans)]
        out = []
        for j, lo_r, hi_r in spans:
            v = x[lo_r:hi_r]
            if a2:
                if len(v) >= a2["stride2_min"] and random.random() < a2["stride2_p"]:   # thicker-slice spacing
                    v = v[random.randint(0, 1)::2]
                theta, gamma, gain = self.draw()                                  # B2: one draw per series
            if mirror:
                v = v.flip(0) if j in SAGITTAL else v.flip(-1)
            if train:                                    # warp into the 320 crop at native pixel scale
                grid = F.affine_grid(theta.expand(len(v), 2, 3), (len(v), 1, CFG["res"], CFG["res"]),
                                     align_corners=False)
                v = F.grid_sample(v[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
                v = (v.clamp_min(1e-4).pow(gamma) * gain).clamp_(0, 1)
                if a2:
                    with torch.autocast(self.device.type, enabled=False):
                        v = self.mri(v.float(), a2)
            else:
                lo = (CFG["img"] - CFG["res"]) // 2
                v = v[:, lo:lo + CFG["res"], lo:lo + CFG["res"]]
            if res != CFG["res"]:                        # smaller input: the same 140 x 320/384 mm view, resized
                v = F.interpolate(v[:, None], size=(res, res), mode="bilinear", antialias=True,
                                  align_corners=False)[:, 0]
            v = (v[None, None].expand(1, 3, *v.shape) - self.mean) / self.std     # [1, 3, T, res, res]
            out.append((j, v))
        return out


# ---------------------------------------------------------------- model
@functools.lru_cache(maxsize=64)
def _shift_mask(size_dhw, window_size, shift_size, device):
    import torchvision.models.video.swin_transformer as tvs
    return tvs._compute_attention_mask_3d(torch.zeros(1, device=device), size_dhw, window_size, shift_size)


def swa3d_sdpa(input, qkv_weight, proj_weight, relative_position_bias, window_size, num_heads, shift_size,
               attention_dropout=0.0, dropout=0.0, qkv_bias=None, proj_bias=None, training=True):
    """torchvision shifted_window_attention_3d (v0.20) with softmax(QK^T s + bias + mask) V as one fused kernel and the
    shift masks cached per shape; benchmarked equal to torchvision (scripts/bench_model_b_encoders.py)."""
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
        bias = (bias + m[:, None].to(q.dtype)).repeat(b_, 1, 1, 1)
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


def use_fused_attention():
    import torchvision.models.video.swin_transformer as tvs
    tvs.shifted_window_attention_3d = swa3d_sdpa


class SwinB(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models.video import swin3d_t
        net = swin3d_t(weights=None)
        if pretrained:
            net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
        f = net.features
        assert len(f) == 7 and type(f[5]).__name__ == "PatchMerging" and len(f[6]) == 2
        blocks = [b for k in (0, 2, 4, 6) for b in f[k]]
        assert len(blocks) == 12
        for i, b in enumerate(blocks):                    # torchvision's linear rule; 0.1 = the B1 default
            b.stochastic_depth.p = CFG.get("drop_path", 0.1) * i / (len(blocks) - 1)
        if CFG.get("t_patch", 2) == 1:                    # ablation: one slice per token (Kinetics kernel summed over
            pe, w = net.patch_embed, net.patch_embed.proj    # its 2 frames: same response on a repeated slice)
            proj = nn.Conv3d(w.in_channels, w.out_channels, kernel_size=(1, 4, 4), stride=(1, 4, 4))
            with torch.no_grad():
                proj.weight.copy_(w.weight.sum(2, keepdim=True)); proj.bias.copy_(w.bias)
            pe.proj, pe.tuple_patch_size = proj, (1, 4, 4)
        self.patch_embed, self.pos_drop = net.patch_embed, net.pos_drop
        self.shared = f[:5]
        self.branch = nn.ModuleList([nn.Sequential(copy.deepcopy(f[5]), copy.deepcopy(f[6]), copy.deepcopy(net.norm))
                                     for _ in range(CFG.get("branches", 3))])
        self.n_frozen = CFG.get("freeze", 0)              # B2: patch embedding + shared[:n_frozen] get no gradient
        for m in self.frozen():
            m.requires_grad_(False)
        d = net.num_features
        self.norm = nn.LayerNorm(d)
        self.series = nn.Embedding(4, d)
        nn.init.trunc_normal_(self.series.weight, std=0.02)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, d))
        self.clsb = nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def frozen(self):
        return [self.patch_embed, *self.shared[:self.n_frozen]] if self.n_frozen else []

    def train(self, mode=True):
        super().train(mode)
        for m in self.frozen():                          # frozen stages: no drop path
            m.eval()
        return self

    def run(self, m, h):
        return checkpoint(m, h, use_reentrant=False) if CFG["ckpt"] and self.training else m(h)

    def encode(self, v, j):
        """[1, 3, T, H, W] -> stage-4 tokens: [ceil(T / 2), 768] averaged over space (B1), or all
        [ceil(T / 2) * 10 * 10, 768] (B2, pool = tokens)."""
        k = self.n_frozen
        with torch.no_grad() if k else contextlib.nullcontext():
            h = self.pos_drop(self.patch_embed(v))                              # [1, T', H', W', C]
            for m in self.shared[:k]:
                h = m(h)
        for m in self.shared[k:]:
            h = self.run(m, h)
        for m in self.branch[PLANE[j] if len(self.branch) == 3 else 0]:
            h = self.run(m, h)
        return h.flatten(0, 3) if CFG.get("pool") == "tokens" else h.mean((2, 3))[0]

    def forward(self, series):
        if not series:                                                          # study with no usable series
            return self.clsb[None]
        toks = [self.encode(v, j) for j, v in series]
        ids = torch.cat([torch.full((len(t),), j, device=t.device) for t, (j, _) in zip(toks, series)])
        h = self.norm(torch.cat(toks)) + self.series(ids)                       # [N, 768]
        a = torch.softmax(self.att(h).float(), dim=0).to(h.dtype)               # [N, 12]
        return ((torch.einsum("nk,nf->kf", a, h) * self.clsW).sum(-1) + self.clsb)[None]


def backbone_units(model):
    """Encoder in depth order, one unit per block / patch merging / norm (a unit spans all branches)."""
    units = [[model.patch_embed]]
    for m in model.shared:
        units += [[b] for b in m] if isinstance(m, nn.Sequential) else [[m]]
    br = model.branch
    return units + [[b[0] for b in br]] + [[b[1][i] for b in br] for i in range(len(br[0][1]))] + [[b[2] for b in br]]


def param_groups(model):
    """AdamW groups: backbone lr * llrd ** (units above), wd 0.05 except norms / biases / relative position tables;
    frozen parameters left out; head at head_lr. llrd 1.0 (B1) = one backbone lr."""
    units, d, groups, seen = backbone_units(model), CFG.get("llrd", 1.0), [], set()
    for i, ms in enumerate(units):
        lr, dec, nod = CFG["lr"] * d ** (len(units) - 1 - i), [], []
        for m in ms:
            for n, p in m.named_parameters():
                seen.add(id(p))
                if p.requires_grad:
                    (nod if p.ndim <= 1 or "relative_position_bias_table" in n else dec).append(p)
        groups += [{"params": dec, "lr": lr, "weight_decay": CFG["wd"]}, {"params": nod, "lr": lr, "weight_decay": 0.0}]
    assert seen == {id(p) for n, p in model.named_parameters() if n.startswith(("patch_embed", "shared", "branch"))}
    head = [p for p in model.parameters() if id(p) not in seen]
    return [g for g in groups if g["params"]] + [{"params": head, "lr": CFG["head_lr"], "weight_decay": CFG["wd"]}]


@torch.no_grad()
def predict(model, prep, ids, meta, tta=False, res=None):
    model.eval()
    out = []
    for vol, slot, _, _ in loader(Volumes(ids, meta, {}), False):
        with torch.autocast("cuda", dtype=AMP):
            p = model(prep(vol, slot, False, res or CFG["res"])).float().sigmoid()
            if tta:
                p = 0.5 * (p + model(prep(vol, slot, False, res or CFG["res"], mirror=True)).float().sigmoid())
        out.append(p[0].cpu().numpy())
    return np.stack(out)


def val_bce(pred, ids, ref):
    """Validation loss: per-study weighted soft BCE against the validation labels (as the training loss)."""
    y = np.stack([ref[s][0] for s in ids]); w = np.stack([ref[s][1] for s in ids])
    p = np.clip(pred, 1e-6, 1 - 1e-6)
    b = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(((b * w).sum(1) / np.maximum(w.sum(1), 1)).mean())


def setup():
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    assert r.sha(r.ASSETS / "folds.csv") == r.CFG["sha256"]["folds"]
    fold = {row["study_id"]: row["fold"] for row in __import__("csv").DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest()
    gold_ids = sorted(s for s, m in meta.items() if m["gold"])
    train_ids = sorted(s for s in labels if fold[s] != "0")
    val_ids = sorted(s for s in labels if fold[s] == "0")
    assert len(train_ids) == 3524 and len(val_ids) == 825 and len(gold_ids) == 58
    assert set(val_ids) == set(ref) and not (set(gold_ids) & set(labels))
    assert all(s in meta for s in train_ids + val_ids + gold_ids)
    return labels, ref, meta, train_ids, val_ids, gold_ids


def kinetics_norm():
    from torchvision.models.video import Swin3D_T_Weights
    t = Swin3D_T_Weights.KINETICS400_V1.transforms()
    return list(t.mean), list(t.std)


# ---------------------------------------------------------------- profile (preflight checks + timing; no training run)
def profile():
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    if CFG["attn"] == "sdpa":                             # time and check the attention the run uses
        use_fused_attention()
    labels, ref, meta, train_ids, val_ids, gold_ids = setup()
    print(f"train {len(train_ids)}, validation {len(val_ids)}, gold {len(gold_ids)}", flush=True)
    rows = np.array([[int((meta[s]["slot"] == j).sum()) for j in range(4)] for s in train_ids + val_ids + gold_ids])
    print("SERIES rows per slot: present " + json.dumps([float((rows[:, j] > 0).mean().round(3)) for j in range(4)])
          + " median " + json.dumps([float(np.median(rows[rows[:, j] > 0, j])) for j in range(4)])
          + f" | study total median {np.median(rows.sum(1)):.0f}, p90 {np.percentile(rows.sum(1), 90):.0f}, "
          f"max {rows.sum(1).max()}, no series {(rows.sum(1) == 0).sum()}", flush=True)
    for s in train_ids + val_ids + gold_ids:                  # each slot's rows contiguous, in slot order
        sl = meta[s]["slot"]; v = sl[sl >= 0]
        assert (np.diff(v) >= 0).all() and all(np.ptp(np.where(sl == j)[0]) + 1 == (sl == j).sum()
                                               for j in range(4) if (sl == j).any()), s
    mean, std = kinetics_norm()
    sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    from torchvision.models.video import Swin3D_T_Weights
    url = Swin3D_T_Weights.KINETICS400_V1.url
    assert sha.startswith(url.rsplit("-", 1)[1].split(".")[0]), (url, sha)    # torchvision's hash-in-name check
    print(f"WEIGHTS {WEIGHTS.name} sha256 {sha}; kinetics mean {mean} std {std}", flush=True)
    prep = Prep(mean, std, dev)

    # (a) pretrained weights loaded; the three branches start equal to the pretrained stage 4
    model = SwinB(pretrained=True).to(dev)
    fresh = SwinB(pretrained=False)
    assert not torch.equal(model.shared[0][0].attn.qkv.weight.cpu(), fresh.shared[0][0].attn.qkv.weight)
    for b in model.branch:
        for (n1, p1), (n2, p2) in zip(b.named_parameters(), model.branch[0].named_parameters()):
            assert torch.equal(p1, p2), n1
    print(f"PARAMS shared {sum(p.numel() for p in [*model.patch_embed.parameters(), *model.shared.parameters()]) / 1e6:.1f}M, "
          f"branch {sum(p.numel() for p in model.branch[0].parameters()) / 1e6:.1f}M x {len(model.branch)}, "
          f"total {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, "
          f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M", flush=True)
    # (a2) drop path rule, frozen stages (no grad, eval mode in train()), optimizer groups
    sd = [b.stochastic_depth.p for k in (0, 2, 4) for b in model.shared[k]] + [b.stochastic_depth.p for b in model.branch[0][1]]
    assert abs(sd[-1] - CFG.get("drop_path", 0.1)) < 1e-9 and sd[0] == 0 and all(np.diff(sd) > 0), sd
    model.train()
    assert all(not m.training and not any(q.requires_grad for q in m.parameters()) for m in model.frozen())
    assert all(m.training for m in model.shared[model.n_frozen:])
    groups = param_groups(model)
    in_opt = {id(q) for g in groups for q in g["params"]}
    assert in_opt == {id(q) for q in model.parameters() if q.requires_grad} and sum(len(g["params"]) for g in groups) == len(in_opt)
    lrs = [g["lr"] for g in groups]
    print(f"DROP_PATH {[round(x, 3) for x in sd]}; FROZEN {len(model.frozen())} modules; GROUPS {len(groups)}, "
          f"backbone lr {min(lrs[:-1]):.2e}..{max(lrs[:-1]):.2e}, head {lrs[-1]:.0e}", flush=True)
    model.eval()
    # (b) eval crop equals the training warp with the identity affine (bilinear at pixel centres)
    vol, slot, _, _ = Volumes(val_ids[:1], meta, {})[0]
    ev = prep(vol, slot, False, 320)
    xg = vol.to(dev).float() / 255
    f_ = 320 / 384
    grid = F.affine_grid(torch.tensor([[f_, 0, 0], [0, f_, 0.]], device=dev).expand(len(xg), 2, 3),
                         (len(xg), 1, 320, 320), align_corners=False)
    warped = F.grid_sample(xg[:, None], grid, align_corners=False)[:, 0]
    lo = (384 - 320) // 2
    d_crop = (warped - xg[:, lo:lo + 320, lo:lo + 320]).abs().max().item()
    assert d_crop < 1e-5, d_crop
    # (c) mirror: sagittal series reverse depth, others flip width
    mm = prep(vol, slot, False, 320, mirror=True)
    for (j, a), (_, b) in zip(ev, mm):
        assert torch.equal(b, a.flip(2) if j in SAGITTAL else a.flip(-1)), j
    # (d) forward shapes, empty study, finite output
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        toks = [model.encode(v, j) for j, v in ev]
        out = model(ev)
        assert torch.equal(model([]), model.clsb[None])
    assert out.shape == (1, 12) and torch.isfinite(out.float()).all()
    per_t = 100 if CFG.get("pool") == "tokens" else 1                          # 320 / 32 = 10 x 10 stage-4 positions
    tp = CFG.get("t_patch", 2)
    assert all(t.shape == (math.ceil(v.shape[2] / tp) * per_t, 768) for t, (_, v) in zip(toks, ev))
    # (e) B2 training augmentation: shapes, finite, slots a subset in order, never empty, every op exercised
    if CFG.get("aug2"):
        a2, n_ser, n_full, rows_in, rows_out = CFG["aug2"], [], 0, 0, 0
        for s_ in train_ids[:200]:
            vol, slot, _, _ = Volumes([s_], meta, {})[0]
            if (slot >= 0).sum() == 0:
                continue
            tr_ = prep(vol, slot, True, 320)
            js = [j for j, _ in tr_]
            assert js and js == sorted(js) and set(js) <= set(slot.tolist()), (s_, js)
            for j, v in tr_:
                assert v.shape[:2] == (1, 3) and v.shape[-2:] == (320, 320) and torch.isfinite(v).all()
                assert 1 <= v.shape[2] <= int((slot == j).sum())
                rows_out += v.shape[2]
            rows_in += int((slot >= 0).sum()); n_ser.append(len(js)); n_full += len(js) == len(set(slot[slot >= 0].tolist()))
        print(f"AUG2_OK: {len(n_ser)} studies, mean series kept {np.mean(n_ser):.2f}, all kept {n_full / len(n_ser):.2f}, "
              f"rows kept {rows_out / rows_in:.2f}", flush=True)
    print(f"PREFLIGHT_OK: hashes, {len(meta)} manifests, crop == identity warp ({d_crop:.1e}), mirror rule, "
          f"branches equal at init, tokens {[tuple(t.shape) for t in toks]}", flush=True)

    # timing on real training studies (optimizer steps included), then eval
    timing = {}
    for res, ck in ((320, CFG["ckpt"]),) if B2 else ((320, False), (320, True), (256, False)):
        CFG["ckpt"] = ck
        model = SwinB(pretrained=True).to(dev); model.train()
        opt = torch.optim.AdamW(param_groups(model))
        scaler = torch.amp.GradScaler("cuda", enabled=AMP == torch.float16)
        torch.cuda.reset_peak_memory_stats(); n, t0 = 0, None
        try:
            for i, (vol, slot, y, w) in enumerate(loader(Volumes(train_ids[:56], meta, labels), True)):
                if i == 8:
                    torch.cuda.synchronize(); t0, rows0 = time.monotonic(), 0
                with torch.autocast("cuda", dtype=AMP):
                    logit = model(prep(vol, slot, True, res)).float()
                loss = (F.binary_cross_entropy_with_logits(logit[0], y.to(dev), reduction="none") * w.to(dev)).sum() \
                    / w.sum().clamp_min(1)
                assert torch.isfinite(loss), loss
                scaler.scale(loss / CFG["accum"]).backward()
                if (i + 1) % CFG["accum"] == 0:
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if t0 is not None:
                    n += 1; rows0 += len(vol)
            torch.cuda.synchronize(); tr = (time.monotonic() - t0) / n
            mem = torch.cuda.max_memory_allocated() / 2**30
            p0 = time.monotonic(); predict(model, prep, val_ids[:40], meta, res=res); te = (time.monotonic() - p0) / 40
            epoch_min = (tr * len(train_ids) + te * len(val_ids)) / 60
            timing[f"{res}{'_ckpt' if ck else ''}"] = {"train_s_study": round(tr, 3), "eval_s_study": round(te, 3),
                                                      "peak_gib": round(mem, 1), "rows_per_study": round(rows0 / n, 1),
                                                      "epoch_min": round(epoch_min, 1),
                                                      "run_h": round((tr * len(train_ids) * CFG["epochs"] + te * (
                                                          len(val_ids) * len(CFG["eval_epochs"]) + 4 * 883)) / 3600, 2)}
        except torch.cuda.OutOfMemoryError:
            timing[f"{res}{'_ckpt' if ck else ''}"] = "OOM"
        print("PROFILE " + json.dumps(timing), flush=True)
        del model, opt; torch.cuda.empty_cache()
    CFG["ckpt"] = False
    print("PROFILE_DONE " + json.dumps(timing), flush=True)


# ---------------------------------------------------------------- train
def main():
    started = time.monotonic()
    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    if CFG["attn"] == "sdpa":
        use_fused_attention()
    OUT.mkdir(parents=True, exist_ok=True)
    CFG["weights_sha256"] = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    labels, ref, meta, train_ids, val_ids, gold_ids = setup()
    print(f"CFG {json.dumps(CFG)}\ntrain {len(train_ids)}, validation {len(val_ids)}, gold {len(gold_ids)}", flush=True)
    prep = Prep(*kinetics_norm(), dev)
    model = SwinB(pretrained=True).to(dev)
    opt = torch.optim.AdamW(param_groups(model))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP == torch.float16)   # no-op for bf16 (Modal runs)
    steps_per_epoch = math.ceil(len(train_ids) / CFG["accum"])
    total, warm, floor = steps_per_epoch * CFG["epochs"], steps_per_epoch * CFG["warmup_epochs"], CFG["lr_floor"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else floor + (1 - floor) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    start, history, rec = 1, [], OUT / "recovery.pt"
    if rec.is_file():
        st = torch.load(rec, map_location="cpu", weights_only=False)
        assert st["cfg"] == CFG, "recovery config differs"
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        if "scaler" in st:
            scaler.load_state_dict(st["scaler"])
        start, history = st["epoch"] + 1, st["history"]
        random.setstate(st["py_rng"])
        if "torch_rng" in st:                            # B2 draws bias field / noise / drop path from torch's RNG
            torch.set_rng_state(st["torch_rng"]); torch.cuda.set_rng_state_all(st["cuda_rng"])
        print(f"RESUMED after epoch {st['epoch']}", flush=True)
        del st
    train_ds = Volumes(train_ids, meta, labels)
    for epoch in range(start, CFG["epochs"] + 1):
        model.train(); t0 = time.monotonic(); running, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        dl = loader(train_ds, True, seed=CFG["seed"] * 100 + epoch)
        for i, (vol, slot, y, w) in enumerate(dl):
            with torch.autocast("cuda", dtype=AMP):
                logit = model(prep(vol, slot, True, CFG["res"])).float()
            y, w = y.to(dev), w.to(dev)
            loss = (F.binary_cross_entropy_with_logits(logit[0], y, reduction="none") * w).sum() / w.sum().clamp_min(1)
            scaler.scale(loss / CFG["accum"]).backward()
            if (i + 1) % CFG["accum"] == 0 or i + 1 == len(dl):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            running += loss.item(); n += 1
            if (i + 1) % 400 == 0:
                print(f"  epoch {epoch} step {i + 1}/{len(dl)} loss {running / n:.4f} "
                      f"{(time.monotonic() - t0) / n:.3f}s/study", flush=True)
        train_s = time.monotonic() - t0
        score, per, vloss = None, None, None
        if epoch in CFG["eval_epochs"]:
            pred = predict(model, prep, val_ids, meta)
            score, per = r.macro_auc(pred, val_ids, ref)
            vloss = val_bce(pred, val_ids, ref)
            r.save_predictions(OUT / f"val_epoch{epoch:02d}.json", val_ids, pred, f"epoch {epoch}, 320 crop, no TTA")
        record = {"epoch": epoch, "train_loss": running / n, "val_macro_auc": score, "val_bce": vloss, "per_target": per,
                  "train_s": round(train_s, 1), "total_s": round(time.monotonic() - t0, 1),
                  "lr": [opt.param_groups[0]["lr"], opt.param_groups[-2]["lr"], opt.param_groups[-1]["lr"]]}
        history.append(record)
        scored = [h["val_macro_auc"] for h in history if h["val_macro_auc"] is not None]
        if epoch in CFG["swa_epochs"] or (score is not None and score >= max(scored)):
            torch.save(model.state_dict(), OUT / f"weights_epoch{epoch:02d}.pt")
        torch.save({"cfg": CFG, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "history": history, "py_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "scaler": scaler.state_dict()}, OUT / "recovery.tmp")
        os.replace(OUT / "recovery.tmp", rec)
        (OUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        # printed after the saves: the launcher commits the volume when it sees this line
        print(f"EPOCH {epoch}: loss {record['train_loss']:.4f} val macro AUC "
              f"{'-' if score is None else f'{score:.4f}'} val BCE {'-' if vloss is None else f'{vloss:.4f}'} "
              f"({record['total_s']:.0f}s) {json.dumps(per)}", flush=True)
        kill = CFG.get("kill")
        if kill and epoch == kill["epoch"] and score < kill["min_val"]:                    # pre-registered early stop
            summary = {"cfg": CFG, "history": history, "killed": f"epoch {epoch} val {score:.4f} < {kill['min_val']}",
                       "seconds": round(time.monotonic() - started, 1)}
            (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print("RUN_SUMMARY " + json.dumps({k: summary[k] for k in ("killed", "seconds")}), flush=True)
            return

    # ---- primary: weight average of the SWA epochs (LayerNorm model: no refresh)
    sd = [torch.load(OUT / f"weights_epoch{e:02d}.pt", map_location="cpu", weights_only=True) for e in CFG["swa_epochs"]]
    model.load_state_dict({k: sum(s[k].float() for s in sd) / len(sd) if sd[-1][k].is_floating_point() else sd[-1][k]
                           for k in sd[-1]})
    tag = f"swa{CFG['swa_epochs'][0]}_{CFG['swa_epochs'][-1]}"
    torch.save(model.state_dict(), OUT / f"weights_{tag}.pt")
    results = {}
    for name, tta in (("swa", False), ("swa_mirror", True)):
        for split, ids in (("val", val_ids), ("gold", gold_ids)):
            p = predict(model, prep, ids, meta, tta=tta)
            r.save_predictions(OUT / f"{split}_{name}.json", ids, p, f"{name}, 320 crop")
            if split == "val":
                results[name] = r.macro_auc(p, val_ids, ref)
    best = max((h for h in history if h["val_macro_auc"] is not None), key=lambda h: h["val_macro_auc"])
    model.load_state_dict(torch.load(OUT / f"weights_epoch{best['epoch']:02d}.pt", map_location="cpu", weights_only=True))
    for split, ids in (("val", val_ids), ("gold", gold_ids)):
        r.save_predictions(OUT / f"{split}_best_epoch.json", ids, predict(model, prep, ids, meta),
                           f"best epoch {best['epoch']} (selected on validation), 320 crop")
    summary = {"cfg": CFG, "history": history, "primary_swa": results["swa"], "swa_mirror_tta": results["swa_mirror"],
               "best_epoch": {"epoch": best["epoch"], "val_macro_auc": best["val_macro_auc"]},
               "seconds": round(time.monotonic() - started, 1),
               "note": "Gold predicted only; score locally. Primary = swa (pre-registered)."}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps({k: summary[k] for k in ("primary_swa", "swa_mirror_tta", "best_epoch", "seconds")}),
          flush=True)


if __name__ == "__main__":
    profile() if os.environ.get("SWIN_MODE") == "profile" else main()
