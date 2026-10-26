"""Exp r50-GL: the label-v2 r50 recipe trained on two views of each study, the whole field (global windows, as label-v2
r50) and per-finding zoomed crops (local), with one shared ResNet-50 and a residual reader that can only add to the
global answer.

Base: Exp 1e on label-v2 exactly (D96_PHASE=r50_gl: resnet50.tv_in1k from ImageNet, label-v2 sha dd0ab9ab..., seed
2026, 25-epoch cosine with a 1% floor, 1 warm-up epoch, backbone LR 1e-4, head LR 1e-3, wd 0.05, SWA 21-25, 2 studies x
4 accumulation, 16 / 94 windows, eval 320 centre crop, BN refresh on 256 training studies, 1e augmentation).
Crop placement: MS1's crop table (Exp MS1-CP, run_sig_lv2/crop_probe: per finding the gradient x activation centroid on
each of MS1's 4 most-attended windows, with MS1's attention weight; maps mark anatomy, not label: LOC-0-MS1), reused
unchanged. Box per finding, set from label-free centroid capture (MS1-GL crop checks, 2026-10-18): 128 px (46.7 mm) for
medial meniscus and contusion; 192 px (70 mm) for ACL, MCL, lateral meniscus, medial / lateral / PF OA, Baker's,
fracture; effusion and synovitis get no crop (diffuse findings; capture at 192 px 0.71 / 0.73). Every crop is resampled
to 384 px (3x / 2x zoom) in training and evaluation.
Training: per cropped finding one of the 4 windows drawn by MS1's attention, centre jitter +-16 px; evaluation: all 4,
weighted by MS1's attention (primary); top-1 crop per finding and 6 x (16 windows + top-1 crops) passes are descriptive.
Augmentation (training): ONE draw per study shared by both views (as MS1-GL): mirror, one affine (A, t), gamma, gain.
Globals: output -> input A u + t (1e). Crops: theta = [s A | p], s = box / 384, p = the anatomy point (a zoom of the
augmented whole image there). Evaluation: globals = 1e's 320 centre crop; crops theta = [s I | p].
BatchNorm: dual statistics. Global and crop images pass through the backbone in separate calls; every BatchNorm keeps
two sets of running statistics (global, crop) and shares its affine weights. Both sets are refreshed after SWA from
their own inputs (globals as 1e: 16 eval-centre windows at 384; crops: the top-1 crop per finding).
Heads:  global g = 1e's head (LayerNorm -> per-finding window attention -> W) on the global windows; BCE weight 1.
        local  l = crop feature -> own LayerNorm -> V_n of the crop's finding; per finding the attention-weighted mean
                   over its crops; auxiliary BCE x 0.5 on the 10 cropped findings only.
        fused    = g.detach() + delta, delta = CW1 reader (every global window and every crop a token; Linear 2048 -> 384
                   + LN, view-type, series, slice-position embeddings; crops + finding and in-plane centre embeddings;
                   2-layer pre-norm transformer, 8 heads, FFN 768, dropout 0.1, final LN; per-finding attention over all
                   tokens -> classifier initialised at ZERO, so the untrained fused output equals the global head). The
                   fused BCE reaches the backbone only through the reader; it can never pull the global head down.
Loss = BCE(fused) + BCE(global) + 0.5 x BCE_cropped(local).
Logged at epochs 5/10/15/20/25 and SWA: fold-0 AUC (label-v2 confident cells and the Steven validation labels) and val
BCE for fused / global / local / fused without crop tokens; the reader's pooling mass on crop tokens per finding.

v2 (2026-10-18, after v1's reader stayed dormant to epoch 10: fused = global, uniform crop mass): one change, the reader
classifier starts small random (trunc normal std 0.02, CW1's init) instead of zero, so its attention gets gradient from
step 1 (Exp 2a lesson: zero-started paths on r50 do not open); g.detach() still keeps the fused loss off the global head.
Fold 0 scored every epoch (about 30 s each).

Env: D96_CACHE, D96_PHASE=r50_gl, GL_CROPS (crop_probe dir), GL_OUT, GL_SMOKE=1 (end-to-end on a few studies),
GL_VERSION=v1 | v2.
"""
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coatnet_d96_runner as r

assert os.environ["D96_PHASE"] == "r50_gl"
OUT = Path(os.environ["GL_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
CP = Path(os.environ["GL_CROPS"]); SMOKE = os.environ.get("GL_SMOKE") == "1"
VER = os.environ.get("GL_VERSION", "v1"); assert VER in ("v1", "v2")
CFG = r.CFG; IMG, MAXS, K = CFG["img"], CFG["maxs"], CFG["eval_windows"]
BOX = {"ACL": 192, "MCL": 192, "Medial Meniscus": 128, "Lateral Meniscus": 192, "Medial OA": 192, "Lateral OA": 192,
       "PF OA": 192, "Baker's": 192, "Contusion": 128, "Fracture": 192}
GL = {"version": f"r50_gl_{VER}", "crop_table": "run_sig_lv2/crop_probe", "boxes": BOX, "crop_res": 384, "top": 4,
      "jitter_px": 16, "global_w": 1.0, "aux_local": 0.5, "bn": "dual", "fusion": "g.detach() + reader (zero-init cls)",
      "multi_passes": 6, "smoke_steps": 32,
      "reader": {"dim": 384, "layers": 2, "heads": 8, "ffn": 768, "dropout": 0.1, "proj_dropout": 0.2, "freqs": 4}}
if VER == "v2":                                          # v1's dict stays exactly as launched (its recovery.pt check)
    GL.update({"reader_cls_init": "trunc_normal 0.02", "eval_every_epoch": True})
CF = np.array([k for k, t in enumerate(r.TARGETS) if t in BOX])                 # the 10 cropped findings
BOXN = np.array([BOX.get(t, 0) for t in r.TARGETS])
CMASK = np.isin(np.arange(12), CF)
HEADS = ("fused", "global", "local", "fused_nocrop")


def crop_table(meta):
    """study -> (slice [12, TOP], row centre [12, TOP], col centre [12, TOP], MS1 attention weight [12, TOP])"""
    tab = {}
    for p in sorted(CP.glob("cp_*.npz")):
        z = np.load(p)
        for s, cen, at in zip(z["ids"], z["centres"], z["a_top"]):
            s = str(s); c = np.clip(r.centres(meta[s]["mask"], K, False), 1, MAXS - 2)
            tab[s] = (c[cen[..., 0].astype(int)], cen[..., 1].astype(int), cen[..., 2].astype(int), at.astype(np.float32))
    return tab


class GLStudies(r.Studies):
    """r.Studies + crop inputs for the cropped findings: full slices of the crop windows [kc, 3, IMG, IMG], their
    sagittal flag, context (position, known, slot), centre (row, col) px, finding index, weight within the finding."""

    def __init__(self, ids, meta, labels, k, train, tab, offset=None, top1=False):
        super().__init__(ids, meta, labels, k, train, offset); self.tab, self.top1 = tab, top1

    def __getitem__(self, i):
        x, sag, y, w, ctx = super().__getitem__(i)
        s = self.ids[i]; m = self.meta[s]
        vol = self._mm[m["file"]][m["row"]:m["row"] + MAXS]
        sl, rr, cc, at = (a[CF] for a in self.tab[s])
        n, T = len(CF), GL["top"]; A = np.arange(n)
        if self.train:                                       # one of the TOP windows per finding, by MS1 attention
            j = np.array([np.random.choice(T, p=at[q] / at[q].sum()) for q in range(n)])
            jit = np.random.randint(-GL["jitter_px"], GL["jitter_px"] + 1, size=(2, n))
            S, R, Q = sl[A, j], rr[A, j] + jit[0], cc[A, j] + jit[1]
            fid, wt = CF, np.ones(n, np.float32)
        elif self.top1:                                      # descriptive passes / BN refresh: MS1's top window
            j = at.argmax(1); S, R, Q = sl[A, j], rr[A, j], cc[A, j]
            fid, wt = CF, np.ones(n, np.float32)
        else:                                                # all TOP, MS1-attention-weighted within each finding
            S, R, Q = sl.ravel(), rr.ravel(), cc.ravel()
            fid, wt = np.repeat(CF, T), (at / at.sum(1, keepdims=True)).ravel()
        h = BOXN[fid] // 2
        R, Q = np.clip(R, h, IMG - h), np.clip(Q, h, IMG - h)
        self._S = S                                          # V4-IMP MC-PAIR: crop slices (read by GLStudies6)
        xc = np.ascontiguousarray(vol[np.stack([S - 1, S, S + 1], 1)])                  # [kc, 3, IMG, IMG]
        sagc = np.isin(r.SLOT_OF[S], r.SAGITTAL_SLOTS)
        ctxc = np.stack([m["pos"][S], m["known"][S], r.SLOT_OF[S].astype(np.float32)], 1).astype(np.float32)
        cen = np.stack([R, Q], 1).astype(np.float32)
        return (x, sag, y, w, ctx, torch.from_numpy(xc), torch.from_numpy(sagc), torch.from_numpy(ctxc),
                torch.from_numpy(cen), torch.from_numpy(fid.astype(np.int64)), torch.from_numpy(wt.astype(np.float32)))


def prepare_gl(x, sag, xc, sagc, cen, fid, train, dev, gres=None):
    """-> global images [b*k, 3, res, res], crop images [b*kc, 3, 384, 384], crop centres (x, y) in [-1, 1] [b, kc, 2].
    Training: one augmentation draw per study, shared by both views. Evaluation: globals 1e's centre crop to gres
    (default eval_res 320; train_res 384 = whole image, the BN refresh input)."""
    b, k, kc = x.shape[0], x.shape[1], xc.shape[1]
    x = x.to(dev, non_blocking=True).flatten(0, 1); xc = xc.to(dev, non_blocking=True).flatten(0, 1)
    sag = sag.to(dev).flatten(); sagc = sagc.to(dev).flatten()
    cen = cen.to(dev).view(-1, 2)
    s = torch.as_tensor(BOXN, dtype=torch.float32, device=dev)[fid.to(dev).flatten()].view(-1, 1, 1) / IMG
    px = torch.stack([(cen[:, 1] + 0.5) / IMG * 2 - 1, (cen[:, 0] + 0.5) / IMG * 2 - 1], 1)    # (x = col, y = row)
    a, cres = CFG["aug"], GL["crop_res"]
    if train:
        flip = torch.rand(b, device=dev) < a["mirror_p"]
        fg, fc = flip.repeat_interleave(k), flip.repeat_interleave(kc)
        x = torch.where(fg.view(-1, 1, 1, 1), r.mirror(x, sag), x)
        xc = torch.where(fc.view(-1, 1, 1, 1), r.mirror(xc, sagc), xc)
        px = torch.stack([torch.where(fc & ~sagc, -px[:, 0], px[:, 0]), px[:, 1]], 1)   # width flip moves the centre
    x = x.float().div_(255.0); xc = xc.float().div_(255.0)
    if train:
        res = CFG["train_res"]
        ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(a["rot_deg"])
        sc = 1 + (torch.rand(b, device=dev) * 2 - 1) * a["scale"]
        tx, ty = [(torch.rand(b, device=dev) * 2 - 1) * a["shift"] * 2 for _ in range(2)]
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        A = torch.stack([torch.stack([cos, -sin], 1), torch.stack([sin, cos], 1)], 1)                  # [b, 2, 2]
        fov = res / IMG if CFG.get("train_crop") else 1.0                          # KD-VIEW: the eval centre crop
        th_g = torch.cat([A * fov, torch.stack([tx, ty], 1)[..., None]], 2).repeat_interleave(k, 0)
        th_c = torch.cat([s * A.repeat_interleave(kc, 0), px[..., None]], 2)
        x = F.grid_sample(x, F.affine_grid(th_g, (b * k, 3, res, res), align_corners=False), mode="bilinear",
                          padding_mode="zeros", align_corners=False)
    else:
        res = gres or CFG["eval_res"]
        if res != x.shape[-1]:
            lo = (x.shape[-1] - res) // 2                                                 # 1e's centre crop
            x = x[..., lo:lo + res, lo:lo + res]
        th_c = torch.cat([s * torch.eye(2, device=dev).expand(b * kc, 2, 2), px[..., None]], 2)
    xc = F.grid_sample(xc, F.affine_grid(th_c, (b * kc, 3, cres, cres), align_corners=False), mode="bilinear",
                       padding_mode="zeros", align_corners=False)
    if train:
        gamma = torch.exp((torch.rand(b, device=dev) * 2 - 1) * a["gamma"])
        gain = 1 + (torch.rand(b, device=dev) * 2 - 1) * a["gain"]
        x = (x.clamp_min(1e-4).pow(gamma.repeat_interleave(k).view(-1, 1, 1, 1))
             * gain.repeat_interleave(k).view(-1, 1, 1, 1)).clamp_(0, 1)
        xc = (xc.clamp_min(1e-4).pow(gamma.repeat_interleave(kc).view(-1, 1, 1, 1))
              * gain.repeat_interleave(kc).view(-1, 1, 1, 1)).clamp_(0, 1)
    mean, std = r.MEAN.to(dev), r.STD.to(dev)
    x = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
    xc = ((xc - mean) / std).contiguous(memory_format=torch.channels_last)
    return x, xc, px.view(b, kc, 2)


class DualBN(nn.BatchNorm2d):
    """BatchNorm2d with a second set of running statistics for crop images (self.crop = True); affine shared."""

    def __init__(self, bn):
        super().__init__(bn.num_features, bn.eps, bn.momentum, bn.affine, bn.track_running_stats)
        self.load_state_dict(bn.state_dict())
        self.register_buffer("crop_mean", bn.running_mean.detach().clone())
        self.register_buffer("crop_var", bn.running_var.detach().clone())
        self.register_buffer("crop_nbt", bn.num_batches_tracked.detach().clone())
        self.crop = False

    def reset_running_stats(self):
        super().reset_running_stats()
        if hasattr(self, "crop_mean"):                       # the base constructor calls this before they exist
            self.crop_mean.zero_(); self.crop_var.fill_(1); self.crop_nbt.zero_()

    def forward(self, x):
        if not self.crop:
            return super().forward(x)
        f = 0.0
        if self.training:
            self.crop_nbt.add_(1)
            f = 1.0 / float(self.crop_nbt) if self.momentum is None else self.momentum
        return F.batch_norm(x, self.crop_mean, self.crop_var, self.weight, self.bias, self.training, f, self.eps)


def dual_bn(module):
    out = []
    for name, ch in module.named_children():
        if isinstance(ch, nn.BatchNorm2d):
            d = DualBN(ch); setattr(module, name, d); out.append(d)
        else:
            out += dual_bn(ch)
    return out


class Reader(nn.Module):
    """CW1 reader over global + crop tokens -> per-finding logit delta, pooling mass on crop tokens."""

    def __init__(self, d):
        super().__init__()
        c = GL["reader"]; D = c["dim"]
        self.inp = nn.Sequential(nn.Dropout(c["proj_dropout"]), nn.Linear(d, D), nn.LayerNorm(D))
        self.kind, self.slot, self.fid = nn.Embedding(2, D), nn.Embedding(5, D), nn.Embedding(12, D)
        self.pos = nn.Linear(2 * c["freqs"] + 1, D); self.xy = nn.Linear(4 * c["freqs"], D)
        layer = nn.TransformerEncoderLayer(D, c["heads"], c["ffn"], c["dropout"], activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, c["layers"], norm=nn.LayerNorm(D), enable_nested_tensor=False)
        self.att = nn.Sequential(nn.Linear(D, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, D)); self.clsb = nn.Parameter(torch.zeros(12))   # v1 zero: delta = 0
        for p in (self.kind.weight, self.slot.weight, self.fid.weight) + ((self.clsW,) if VER == "v2" else ()):
            nn.init.trunc_normal_(p, std=0.02)
        self.f = math.pi * torch.arange(1, c["freqs"] + 1, dtype=torch.float32)

    def place(self, ctx):
        """CW1 / Exp 2a anatomical position (sin / cos x known flag, + the flag) + series embedding"""
        f = self.f.to(ctx.device); pos, known = ctx[..., 0:1], ctx[..., 1:2]
        enc = torch.cat([torch.sin(pos * f), torch.cos(pos * f)], -1) * known
        return self.pos(torch.cat([enc, known], -1)) + self.slot(ctx[..., 2].long())

    def forward(self, hg, ctxg, hc=None, ctxc=None, fidc=None, pxc=None):
        t = self.inp(hg) + self.kind.weight[0] + self.place(ctxg)
        ng = t.shape[1]
        if hc is not None:
            f = self.f.to(pxc.device)
            xy = torch.cat([torch.sin(pxc[..., :1] * f), torch.cos(pxc[..., :1] * f),
                            torch.sin(pxc[..., 1:] * f), torch.cos(pxc[..., 1:] * f)], -1)
            tc = self.inp(hc) + self.kind.weight[1] + self.place(ctxc) + self.fid(fidc) + self.xy(xy)
            t = torch.cat([t, tc.to(t.dtype)], 1)
        z = self.enc(t)
        a = torch.softmax(self.att(z).float(), dim=1)                                    # [b, n, 12]
        logit = (torch.einsum("bnk,bnf->bkf", a.to(z.dtype), z) * self.clsW).sum(-1).float() + self.clsb
        mass = a[:, ng:].sum(1) if hc is not None else torch.zeros_like(logit)
        return logit, mass


class RaptorGL(r.Raptor):
    def __init__(self, res, pretrained):
        super().__init__(res, pretrained)                    # backbone, then 1e's head (same init order as 1e)
        d = self.backbone.num_features
        self.bns = dual_bn(self.backbone)
        assert len(self.bns) == 53, len(self.bns)            # ResNet-50: stem + 16 blocks x 3 + 4 downsample
        self.norm_l = nn.LayerNorm(d)
        self.clsV = nn.Parameter(torch.zeros(12, d)); self.clsvb = nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.clsV, std=0.02)
        self.reader = Reader(d)

    def encode(self, x, crop):
        for m in self.bns:
            m.crop = crop
        f = self.backbone(x)
        for m in self.bns:
            m.crop = False
        return f

    def forward(self, x, b, ctx, xc, ctxc, fidc, pxc, wc, nocrop=False):
        kc = xc.shape[0] // b
        hg = self.norm(self.encode(x, False).view(b, x.shape[0] // b, -1))
        hc = self.norm_l(self.encode(xc, True).view(b, kc, -1))
        a = torch.softmax(self.att(hg).float(), dim=1).to(hg.dtype)                       # 1e's head
        g = ((torch.einsum("bkn,bkf->bnf", a, hg) * self.clsW).sum(-1) + self.clsb).float()
        lc = ((hc * self.clsV[fidc]).sum(-1) + self.clsvb[fidc]).float()                # [b, kc], own finding
        l = torch.zeros(b, 12, device=x.device).scatter_add(1, fidc, lc * wc)
        dz, mass = self.reader(hg, ctx, hc, ctxc, fidc, pxc)
        out = [g.detach() + dz, g, l]
        if nocrop:
            out.append(g.detach() + self.reader(hg, ctx)[0])
        return out, mass


def bce(z, y, w):
    return (F.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() / w.sum().clamp_min(1)


def to_dev(dev, *t):
    return [v.to(dev, non_blocking=True) for v in t]


@torch.no_grad()
def predict(model, ids, meta, tab, dev, k=K, offset=None, top1=False):
    """-> sigmoid [4, n, 12] (fused, global, local, fused without crop tokens), reader crop mass [n, 12].
    The local head has no crops for effusion / synovitis: its columns there hold the global prediction."""
    model.eval()
    dl = torch.utils.data.DataLoader(GLStudies(ids, meta, {}, k, False, tab, offset, top1), batch_size=1,
                                     num_workers=CFG["workers"], pin_memory=True)
    out = np.zeros((4, len(ids), 12), np.float32); mass = np.zeros((len(ids), 12), np.float32)
    for i, (x, sag, _, _, ctx, xc, sagc, ctxc, cen, fid, wc) in enumerate(dl):
        xg, xcr, pxc = prepare_gl(x, sag, xc, sagc, cen, fid, False, dev)
        ctx, ctxc, fid, wc = to_dev(dev, ctx, ctxc, fid, wc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, ms = model(xg, 1, ctx, xcr, ctxc, fid, pxc, wc, nocrop=True)
        out[:, i] = torch.stack(o).float().sigmoid()[:, 0].cpu().numpy(); mass[i] = ms[0].float().cpu().numpy()
    out[2][:, ~CMASK] = out[1][:, ~CMASK]
    return out, mass


def score(p, mass, ids, ref, labels):
    rec = {}
    for j, nm in enumerate(HEADS):
        rec[f"lv2_{nm}"], per = r.macro_auc(p[j], ids, labels)
        rec[f"val_{nm}"] = r.macro_auc(p[j], ids, ref)[0]
        rec[f"lv2_bce_{nm}"] = r.val_bce(p[j], ids, labels); rec[f"val_bce_{nm}"] = r.val_bce(p[j], ids, ref)
        if nm == "local":
            rec["lv2_local10"] = float(np.nanmean([per[r.TARGETS[k]] for k in CF]))
        if nm in ("fused", "global"):
            rec[f"lv2_per_{nm}"] = per
    rec["crop_mass"] = dict(zip(r.TARGETS, mass.mean(0).round(3).tolist()))
    return rec


@torch.no_grad()
def bn_refresh(model, ids, meta, labels, tab, dev):
    """1e's refresh for the global statistics (16 eval-centre windows at 384, batch 2); the crop statistics from the
    same studies' top-1 crop per finding."""
    for m in model.bns:
        m.reset_running_stats(); m.momentum = None
    model.train()
    ds = GLStudies(ids, meta, labels, CFG["train_windows"], False, tab, top1=True)
    for x, sag, _, _, _, xc, sagc, _, cen, fid, _ in torch.utils.data.DataLoader(ds, batch_size=2,
                                                                                 num_workers=CFG["workers"]):
        xg, xcr, _ = prepare_gl(x, sag, xc, sagc, cen, fid, False, dev, gres=CFG["train_res"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.encode(xg, False); model.encode(xcr, True)


def main():
    started = time.monotonic()
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    assert r.sha(r.ASSETS / "folds.csv") == CFG["sha256"]["folds"]
    fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest()
    train_ids = sorted(s for s in labels if fold[s] != "0"); val_ids = sorted(s for s in labels if fold[s] == "0")
    assert len(train_ids) == 3524 and len(val_ids) == 825 and set(val_ids) == set(ref)
    tab = crop_table(meta)
    assert set(tab) >= set(train_ids) | set(val_ids), "crop table must cover every labelled study"
    print(f"train {len(train_ids)}, validation {len(val_ids)}; crops for {len(CF)} findings "
          + json.dumps({r.TARGETS[k]: int(BOXN[k]) for k in CF}), flush=True)

    model = RaptorGL(CFG["train_res"], pretrained=True).to(dev).to(memory_format=torch.channels_last)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": CFG["lr"]},
                             {"params": head, "lr": CFG["head_lr"]}], weight_decay=CFG["wd"])
    spe = math.ceil(len(train_ids) / (CFG["studies_per_step"] * CFG["accum"]))
    total, warm, floor = spe * CFG["epochs"], spe * CFG["warmup_epochs"], CFG["lr_floor"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else floor + (1 - floor) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))

    vids = val_ids[:16] if SMOKE else val_ids
    if SMOKE:                                            # v1 zero-init reader: fused must equal global before training
        p0, _ = predict(model, vids[:2], meta, tab, dev)
        print(f"INIT_CHECK max|fused - global| {np.abs(p0[0] - p0[1]).max():.2e} "
              f"max|nocrop - global| {np.abs(p0[3] - p0[1]).max():.2e}", flush=True)
    start, history, rec = 1, [], OUT / "recovery.pt"
    if rec.is_file():
        st = torch.load(rec, map_location="cpu", weights_only=False)
        assert st["gl"] == GL and st["cfg"] == CFG, "recovery config differs"
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, history = st["epoch"] + 1, st["history"]
        print(f"RESUMED after epoch {st['epoch']}", flush=True)
    train_ds = GLStudies(train_ids, meta, labels, CFG["train_windows"], True, tab)
    cmask = torch.from_numpy(CMASK.astype(np.float32)).to(dev)
    last = 1 if SMOKE else CFG["epochs"]
    for epoch in range(start, last + 1):
        model.train(); t0 = time.monotonic()
        dl = torch.utils.data.DataLoader(train_ds, batch_size=CFG["studies_per_step"], shuffle=True,
                                         num_workers=CFG["workers"], pin_memory=True,
                                         worker_init_fn=lambda w: np.random.seed((CFG["seed"] * 1000 + epoch * 100 + w) % 2**32))
        run = np.zeros(4); n = 0
        opt.zero_grad(set_to_none=True)
        for i, (x, sag, y, w, ctx, xc, sagc, ctxc, cen, fid, wc) in enumerate(dl):
            b = x.shape[0]
            xg, xcr, pxc = prepare_gl(x, sag, xc, sagc, cen, fid, True, dev)
            ctx, ctxc, fid, wc, y, w = to_dev(dev, ctx, ctxc, fid, wc, y, w)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                (fz, gz, lz), _ = model(xg, b, ctx, xcr, ctxc, fid, pxc, wc)
            parts = torch.stack([bce(fz, y, w), bce(gz, y, w), bce(lz, y, w * cmask)])
            loss = parts[0] + GL["global_w"] * parts[1] + GL["aux_local"] * parts[2]
            (loss / CFG["accum"]).backward()
            if (i + 1) % CFG["accum"] == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += np.append(parts.detach().cpu().numpy(), loss.item()) * b; n += b
            if (i + 1) % 400 == 0 or (SMOKE and i + 1 in (8, GL["smoke_steps"])):
                print(f"  epoch {epoch} step {i + 1}/{len(dl)} loss fused/global/local {run[0] / n:.4f}/{run[1] / n:.4f}/"
                      f"{run[2] / n:.4f} {(time.monotonic() - t0) / n:.3f}s/study "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
            if SMOKE and i + 1 == GL["smoke_steps"]:
                break
        train_s = time.monotonic() - t0
        rec_ = {"epoch": epoch, "train_loss": dict(zip(("fused", "global", "local", "total"), (run / n).round(5).tolist())),
                "train_s": round(train_s, 1), "lr": [g["lr"] for g in opt.param_groups],
                "reader_clsW_norm": float(model.reader.clsW.norm())}
        if epoch in CFG["eval_epochs"] or SMOKE or GL.get("eval_every_epoch"):
            te = time.monotonic(); p, mass = predict(model, vids, meta, tab, dev)
            rec_["eval_s_per_study"] = round((time.monotonic() - te) / len(vids), 3)
            rec_["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            if not SMOKE:
                rec_.update(score(p, mass, val_ids, ref, labels))
                for j, nm in enumerate(HEADS):
                    r.save_predictions(OUT / f"val_epoch{epoch:02d}_{nm}.json", val_ids, p[j], f"r50-GL epoch {epoch}, {nm}")
        history.append(rec_)
        print(f"EPOCH {epoch} " + json.dumps({k: v for k, v in rec_.items() if not k.startswith("lv2_per")}), flush=True)
        if epoch in CFG["swa_epochs"] or SMOKE:
            torch.save(model.state_dict(), OUT / f"weights_epoch{epoch:02d}.pt")
        if not SMOKE:
            torch.save({"gl": GL, "cfg": CFG, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch, "history": history}, OUT / "recovery.tmp")
            os.replace(OUT / "recovery.tmp", rec)
        (OUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    # ---- primary: weight average of the SWA epochs, both BN statistic sets refreshed
    swa = [1] if SMOKE else CFG["swa_epochs"]
    sds = [torch.load(OUT / f"weights_epoch{e:02d}.pt", map_location="cpu") for e in swa]
    avg = {k: (sum(s[k].float() for s in sds) / len(sds)).to(sds[-1][k].dtype) if sds[-1][k].is_floating_point()
           else sds[-1][k] for k in sds[-1]}
    model.load_state_dict(avg, strict=True)
    tr = time.monotonic()
    refresh = random.Random(CFG["seed"]).sample(train_ids, 8 if SMOKE else CFG["bn_refresh_studies"])
    bn_refresh(model, refresh, meta, labels, tab, dev)
    print(f"BN_REFRESH {len(refresh)} studies {time.monotonic() - tr:.0f}s; layer1.0.bn1 global/crop mean "
          f"{model.bns[1].running_mean[:4].tolist()} / {model.bns[1].crop_mean[:4].tolist()}", flush=True)
    tag = f"swa{swa[0]}_{swa[-1]}"
    torch.save(model.state_dict(), OUT / f"weights_{tag}.pt")
    p, mass = predict(model, vids, meta, tab, dev)
    res = score(p, mass, vids, ref, labels)
    for j, nm in enumerate(HEADS):
        r.save_predictions(OUT / f"val_swa_{nm}.json", vids, p[j], f"r50-GL SWA {swa}, {nm}")
    pt, _ = predict(model, vids, meta, tab, dev, top1=True)                      # descriptive: top-1 crop per finding
    r.save_predictions(OUT / "val_swa_fused_top1.json", vids, pt[0], "r50-GL SWA, fused, top-1 crop per finding")
    res["lv2_fused_top1"] = r.macro_auc(pt[0], vids, labels)[0]
    m = 2 if SMOKE else GL["multi_passes"]
    pm = np.mean([predict(model, vids, meta, tab, dev, k=CFG["train_windows"], offset=(j + 0.5) / m, top1=True)[0][0]
                  for j in range(m)], 0)
    r.save_predictions(OUT / "val_swa_fused_multi6.json", vids, pm, f"r50-GL SWA, fused, {m} x (16 windows + top-1 crops)")
    res["lv2_fused_multi6"] = r.macro_auc(pm, vids, labels)[0]
    summary = {"gl": GL, "cfg": CFG, "history": history, "swa": res, "seconds": round(time.monotonic() - started, 1)}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps({k: v for k, v in res.items() if not k.startswith(("lv2_per", "crop_mass"))}), flush=True)
    if SMOKE:
        print(f"SMOKE_DONE {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
