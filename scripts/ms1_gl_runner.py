"""Exp MS1-GL: MS1 trained on both views, the whole field (global) and per-finding zoomed crops (local), with one shared
MedSigLIP encoder and the CW1 cross-window reader as the fusion, so whole-field and crop tokens exchange information
before each finding is scored.

Start: MS1 SWA weights (run_sig_lv2/weights_swa21_25.pt).
Inputs per study: global = MS1's windows (16 random in training, the 94 eval windows at evaluation); local = per
finding a 128 px (46.7 mm) box of the 384 px image, all 3 slice channels, centred where frozen MS1's gradient x
activation map peaks on one of its 4 most-attended windows for that finding (Exp MS1-CP table; maps mark anatomy, not
label: LOC-0-MS1). Training: one of the 4 drawn by MS1's attention, centre jitter +-JIT px; evaluation: all 4.
Augmentation (training): ONE draw per study shared by both views: mirror (as MS1: sagittal reverses the slice
channels, others flip width, crop centre flipped with it), rotation / scale / shift (one affine A, t), gamma, gain.
Global windows: output -> input A u + t (MS1). Crops: the same transform cut around the anatomy point p (input
coordinates): theta = [s A | p], s = 128 / 384, i.e. exactly a zoom of the augmented whole image at that anatomy,
sampled from the full slice (no empty corners). Evaluation: plain resize (MS1) and theta = [s I | p].
Encoder: shared; crop images get a learned scale embedding (zero init) on their tokens before the blocks.
Heads:  global = MS1's own head (LayerNorm -> per-finding window attention -> W) on the global images;
        local  = crop feature -> own LayerNorm (init MS1's) -> V_n of the crop's finding (init MS1's W); per finding,
                 the attention-weighted mean over its crops;
        fused  = CW1 reader: every image a token (Linear 1152 -> 384 + LN on the LayerNorm feature, + view-type,
                 series, slice-position embeddings; crops + finding and in-plane centre embeddings), 2-layer pre-norm
                 transformer (8 heads, FFN 768, dropout 0.1, final LN), per-finding attention over ALL tokens ->
                 classifier. No gate, no bypass (Exp 2a lesson).
Loss = soft BCE(fused) + AUX x (BCE(global) + BCE(local)). AdamW, MS1's layer decay 0.9 at half MS1's peak (Exp 1b
continuation recipe), 1 warm-up epoch, cosine to 1% over 8 epochs, SWA 5-8; 1 study x 8 accumulation.
Fold 0 at 2/4/6/8 and SWA: fused (primary), global, local, fused without crop tokens; reader pooling mass on crop
tokens per finding; SWA also 6 x (16 windows + 12 top-1 crops) passes averaged (training token count, secondary).

Env: D96_CACHE, D96_PHASE=sig_lv2, MS1_WEIGHTS, GL_CROPS (crop_probe dir), GL_OUT, GL_SMOKE=1 (timing / memory only).
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

assert os.environ["D96_PHASE"] == "sig_lv2"
OUT = Path(os.environ["GL_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
CP = Path(os.environ["GL_CROPS"]); SMOKE = os.environ.get("GL_SMOKE") == "1"
GL = {"version": "ms1_gl_cw1_v1", "init": "run_sig_lv2/weights_swa21_25.pt", "crop": 128, "top": 4, "jitter_px": 16,
      "epochs": 8, "lr": 2.5e-5, "layer_decay": 0.9, "head_lr_old": 5e-4, "head_lr_new": 1e-3, "scale_lr": 1e-4,
      "wd": 0.05, "warmup_epochs": 1, "lr_floor": 0.01, "swa_epochs": [5, 6, 7, 8], "eval_epochs": [2, 4, 6, 8],
      "aux": 0.5, "studies_per_step": 1, "accum": 8, "train_windows": 16, "grad_clip": 1.0, "seed": 2026,
      "multi_passes": 6, "shared_aug": True,
      "reader": {"dim": 384, "layers": 2, "heads": 8, "ffn": 768, "dropout": 0.1, "proj_dropout": 0.2, "freqs": 4}}
CFG = r.CFG; IMG, MAXS, K = CFG["img"], CFG["maxs"], CFG["eval_windows"]
NEW = ("scale_emb", "norm_l.", "clsV", "clsvb", "reader.")
A12 = np.arange(12)


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
    """r.Studies + crop inputs: full slices of the crop windows [kc, 3, IMG, IMG], their sagittal flag, context
    (position, known, slot), centre (row, col) px, finding index, weight within the finding."""

    def __init__(self, ids, meta, labels, k, train, tab, offset=None, top1=False):
        super().__init__(ids, meta, labels, k, train, offset); self.tab, self.top1 = tab, top1

    def __getitem__(self, i):
        x, sag, y, w, ctx = super().__getitem__(i)
        s = self.ids[i]; m = self.meta[s]
        vol = self._mm[m["file"]][m["row"]:m["row"] + MAXS]
        sl, rr, cc, at = self.tab[s]
        T = GL["top"]
        if self.train:                                       # one of the TOP windows per finding, by MS1 attention
            j = np.array([np.random.choice(T, p=at[n] / at[n].sum()) for n in range(12)])
            jit = np.random.randint(-GL["jitter_px"], GL["jitter_px"] + 1, size=(2, 12))
            S, R, Q = sl[A12, j], rr[A12, j] + jit[0], cc[A12, j] + jit[1]
            fid, wt = A12, np.ones(12, np.float32)
        elif self.top1:                                      # secondary passes: MS1's top window per finding
            j = at.argmax(1); S, R, Q = sl[A12, j], rr[A12, j], cc[A12, j]
            fid, wt = A12, np.ones(12, np.float32)
        else:                                                # all TOP, MS1-attention-weighted within each finding
            S, R, Q = sl.ravel(), rr.ravel(), cc.ravel()
            fid, wt = np.repeat(A12, T), (at / at.sum(1, keepdims=True)).ravel()
        h = GL["crop"] // 2
        R, Q = np.clip(R, h, IMG - h), np.clip(Q, h, IMG - h)
        xc = np.ascontiguousarray(vol[np.stack([S - 1, S, S + 1], 1)])                  # [kc, 3, IMG, IMG]
        sagc = np.isin(r.SLOT_OF[S], r.SAGITTAL_SLOTS)
        ctxc = np.stack([m["pos"][S], m["known"][S], r.SLOT_OF[S].astype(np.float32)], 1).astype(np.float32)
        cen = np.stack([R, Q], 1).astype(np.float32)
        return (x, sag, y, w, ctx, torch.from_numpy(xc), torch.from_numpy(sagc), torch.from_numpy(ctxc),
                torch.from_numpy(cen), torch.from_numpy(fid.astype(np.int64)), torch.from_numpy(wt.astype(np.float32)))


def prepare_gl(x, sag, xc, sagc, cen, train, dev):
    """-> global images [b*k, 3, res, res], crop images [b*kc, 3, res, res], crop centres (x, y) in [-1, 1] [b, kc, 2].
    Training: one augmentation draw per study, shared by both views (see module docstring)."""
    b, k, kc = x.shape[0], x.shape[1], xc.shape[1]
    x = x.to(dev, non_blocking=True).flatten(0, 1); xc = xc.to(dev, non_blocking=True).flatten(0, 1)
    sag = sag.to(dev).flatten(); sagc = sagc.to(dev).flatten()
    cen = cen.to(dev).view(-1, 2)
    px = torch.stack([(cen[:, 1] + 0.5) / IMG * 2 - 1, (cen[:, 0] + 0.5) / IMG * 2 - 1], 1)    # (x = col, y = row)
    a, s = CFG["aug"], GL["crop"] / IMG
    res = CFG["train_res"] if train else CFG["eval_res"]
    if train:
        flip = torch.rand(b, device=dev) < a["mirror_p"]
        fg, fc = flip.repeat_interleave(k), flip.repeat_interleave(kc)
        x = torch.where(fg.view(-1, 1, 1, 1), r.mirror(x, sag), x)
        xc = torch.where(fc.view(-1, 1, 1, 1), r.mirror(xc, sagc), xc)
        px = torch.stack([torch.where(fc & ~sagc, -px[:, 0], px[:, 0]), px[:, 1]], 1)   # width flip moves the centre
    x = x.float().div_(255.0); xc = xc.float().div_(255.0)
    if train:
        ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(a["rot_deg"])
        sc = 1 + (torch.rand(b, device=dev) * 2 - 1) * a["scale"]
        tx, ty = [(torch.rand(b, device=dev) * 2 - 1) * a["shift"] * 2 for _ in range(2)]
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        A = torch.stack([torch.stack([cos, -sin], 1), torch.stack([sin, cos], 1)], 1)                  # [b, 2, 2]
        th_g = torch.cat([A, torch.stack([tx, ty], 1)[..., None]], 2).repeat_interleave(k, 0)
        th_c = torch.cat([s * A.repeat_interleave(kc, 0), px[..., None]], 2)
        x = F.grid_sample(x, F.affine_grid(th_g, (b * k, 3, res, res), align_corners=False), mode="bilinear",
                          padding_mode="zeros", align_corners=False)
    else:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)                    # MS1's eval
        th_c = torch.cat([s * torch.eye(2, device=dev).expand(b * kc, 2, 2), px[..., None]], 2)
    xc = F.grid_sample(xc, F.affine_grid(th_c, (b * kc, 3, res, res), align_corners=False), mode="bilinear",
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


class Reader(nn.Module):
    """CW1 reader over global + crop tokens -> per-finding logits, pooling mass on crop tokens."""

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
        self.clsW = nn.Parameter(torch.zeros(12, D)); self.clsb = nn.Parameter(torch.zeros(12))
        for p in (self.clsW, self.kind.weight, self.slot.weight, self.fid.weight):
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
        super().__init__(res, pretrained)
        d = self.backbone.num_features
        assert hasattr(self.backbone, "norm_pre")
        self.scale_emb = nn.Parameter(torch.zeros(d))
        self.norm_l = nn.LayerNorm(d)
        self.clsV = nn.Parameter(torch.zeros(12, d)); self.clsvb = nn.Parameter(torch.zeros(12))
        self.reader = Reader(d)
        self.is_crop = None
        self.backbone.norm_pre.register_forward_hook(self._scale)

    def _scale(self, mod, inp, out):
        if self.is_crop is None:
            return out
        return out + self.is_crop.view(-1, 1, 1).to(out.dtype) * self.scale_emb.to(out.dtype)

    def init_local(self):
        with torch.no_grad():
            self.norm_l.load_state_dict(self.norm.state_dict()); self.clsV.copy_(self.clsW); self.clsvb.copy_(self.clsb)

    def forward(self, x, b, ctx, xc, ctxc, fidc, pxc, wc, nocrop=False):
        ng, nc = x.shape[0], xc.shape[0]; kc = nc // b
        self.is_crop = torch.cat([torch.zeros(ng, device=x.device), torch.ones(nc, device=x.device)])
        f = self.backbone(torch.cat([x, xc]))
        self.is_crop = None
        hg = self.norm(f[:ng].view(b, ng // b, -1)); hc = self.norm_l(f[ng:].view(b, kc, -1))
        a = torch.softmax(self.att(hg).float(), dim=1).to(hg.dtype)                       # MS1's head
        g = ((torch.einsum("bkn,bkf->bnf", a, hg) * self.clsW).sum(-1) + self.clsb).float()
        lc = ((hc * self.clsV[fidc]).sum(-1) + self.clsvb[fidc]).float()                # [b, kc], own finding
        l = torch.zeros(b, 12, device=x.device).scatter_add(1, fidc, lc * wc)
        fz, mass = self.reader(hg, ctx, hc, ctxc, fidc, pxc)
        out = [fz, g, l]
        if nocrop:
            out.append(self.reader(hg, ctx)[0])
        return out, mass


def bce(z, y, w):
    return (F.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() / w.sum().clamp_min(1)


def batch_to(dev, ctx, ctxc, fid, wc):
    return ctx.to(dev), ctxc.to(dev), fid.to(dev), wc.to(dev)


@torch.no_grad()
def predict(model, ids, meta, tab, dev, k=K, offset=None, top1=False):
    """-> sigmoid [4, n, 12] (fused, global, local, fused without crop tokens), reader crop mass [n, 12]"""
    model.eval()
    dl = torch.utils.data.DataLoader(GLStudies(ids, meta, {}, k, False, tab, offset, top1), batch_size=1,
                                     num_workers=CFG["workers"], pin_memory=True)
    out = np.zeros((4, len(ids), 12), np.float32); mass = np.zeros((len(ids), 12), np.float32)
    for i, (x, sag, _, _, ctx, xc, sagc, ctxc, cen, fid, wc) in enumerate(dl):
        xg, xcr, pxc = prepare_gl(x, sag, xc, sagc, cen, False, dev)
        ctx, ctxc, fid, wc = batch_to(dev, ctx, ctxc, fid, wc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, ms = model(xg, 1, ctx, xcr, ctxc, fid, pxc, wc, nocrop=True)
        out[:, i] = torch.stack(o).float().sigmoid()[:, 0].cpu().numpy(); mass[i] = ms[0].float().cpu().numpy()
    return out, mass


HEADS = ("fused", "global", "local", "fused_nocrop")


def score(p, mass, val_ids, ref, labels):
    rec = {}
    for j, nm in enumerate(HEADS):
        rec[f"val_{nm}"] = r.macro_auc(p[j], val_ids, ref)[0]
        rec[f"lv2_{nm}"] = r.macro_auc(p[j], val_ids, labels)[0]
    rec["val_bce_fused"] = r.val_bce(p[0], val_ids, ref)
    rec["crop_mass"] = dict(zip(r.TARGETS, mass.mean(0).round(3).tolist()))
    return rec


def main():
    started = time.monotonic()
    torch.manual_seed(GL["seed"]); np.random.seed(GL["seed"]); random.seed(GL["seed"])
    dev = torch.device("cuda")
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    assert r.sha(r.ASSETS / "folds.csv") == CFG["sha256"]["folds"]
    fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest()
    train_ids = sorted(s for s in labels if fold[s] != "0"); val_ids = sorted(s for s in labels if fold[s] == "0")
    assert len(train_ids) == 3524 and len(val_ids) == 825 and set(val_ids) == set(ref)
    tab = crop_table(meta)
    assert set(tab) == set(train_ids) | set(val_ids), "crop table must cover every labelled study"
    model = RaptorGL(CFG["train_res"], pretrained=False)
    sd = torch.load(os.environ["MS1_WEIGHTS"], map_location="cpu", weights_only=True)
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not unexp and all(k.startswith(NEW) for k in miss), (miss, unexp)
    model.init_local()
    model = model.to(dev).to(memory_format=torch.channels_last)
    eval_model = RaptorGL(CFG["eval_res"], pretrained=False).to(dev).to(memory_format=torch.channels_last)
    eval_model.load_state_dict(model.state_dict(), strict=True)

    groups, lid = {}, r.layer_id(model)
    for n, p in model.backbone.named_parameters():
        groups.setdefault(lid(n), []).append(p)
    top = max(groups)
    pg = [{"params": ps, "lr": GL["lr"] * GL["layer_decay"] ** (top - i)} for i, ps in sorted(groups.items())]
    old = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and not n.startswith(NEW)]
    new = [p for n, p in model.named_parameters() if n.startswith(NEW) and n != "scale_emb"]
    pg += [{"params": old, "lr": GL["head_lr_old"]}, {"params": new, "lr": GL["head_lr_new"]},
           {"params": [model.scale_emb], "lr": GL["scale_lr"], "weight_decay": 0.0}]
    assert sum(len(g["params"]) for g in pg) == len(list(model.parameters()))
    opt = torch.optim.AdamW(pg, weight_decay=GL["wd"])
    spe = math.ceil(len(train_ids) / (GL["studies_per_step"] * GL["accum"]))
    total, warm, floor = spe * GL["epochs"], spe * GL["warmup_epochs"], GL["lr_floor"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else floor + (1 - floor) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))

    start, history, rec = 1, [], OUT / "recovery.pt"
    if rec.is_file():
        st = torch.load(rec, map_location="cpu", weights_only=False)
        assert st["gl"] == GL, "recovery config differs"
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, history = st["epoch"] + 1, st["history"]
        print(f"RESUMED after epoch {st['epoch']}", flush=True)
    train_ds = GLStudies(train_ids, meta, labels, GL["train_windows"], True, tab)
    for epoch in range(start, GL["epochs"] + 1):
        model.train(); t0 = time.monotonic()
        dl = torch.utils.data.DataLoader(train_ds, batch_size=GL["studies_per_step"], shuffle=True,
                                         num_workers=CFG["workers"], pin_memory=True,
                                         worker_init_fn=lambda w: np.random.seed((GL["seed"] * 1000 + epoch * 100 + w) % 2**32))
        run = np.zeros(3); n = 0
        opt.zero_grad(set_to_none=True)
        for i, (x, sag, y, w, ctx, xc, sagc, ctxc, cen, fid, wc) in enumerate(dl):
            b = x.shape[0]
            xg, xcr, pxc = prepare_gl(x, sag, xc, sagc, cen, True, dev)
            ctx, ctxc, fid, wc = batch_to(dev, ctx, ctxc, fid, wc)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                (fz, gz, lz), _ = model(xg, b, ctx, xcr, ctxc, fid, pxc, wc)
            y, w = y.to(dev), w.to(dev)
            parts = torch.stack([bce(fz, y, w), bce(gz, y, w), bce(lz, y, w)])
            loss = parts[0] + GL["aux"] * (parts[1] + parts[2])
            (loss / GL["accum"]).backward()
            if (i + 1) % GL["accum"] == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GL["grad_clip"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += parts.detach().cpu().numpy() * b; n += b
            if (i + 1) % 400 == 0 or (SMOKE and i + 1 in (8, 32)):
                print(f"  epoch {epoch} step {i + 1}/{len(dl)} loss fused/global/local {run[0] / n:.4f}/{run[1] / n:.4f}/"
                      f"{run[2] / n:.4f} {(time.monotonic() - t0) / n:.3f}s/study "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
            if SMOKE and i + 1 == 32:
                break
        train_s = time.monotonic() - t0
        eval_model.load_state_dict(model.state_dict(), strict=True)
        rec_ = {"epoch": epoch, "train_loss": (run / n).round(5).tolist(), "train_s": round(train_s, 1),
                "scale_emb_norm": float(model.scale_emb.norm())}
        if epoch in GL["eval_epochs"] or SMOKE:
            ids = val_ids[:16] if SMOKE else val_ids
            te = time.monotonic(); p, mass = predict(eval_model, ids, meta, tab, dev)
            rec_["eval_s_per_study"] = round((time.monotonic() - te) / len(ids), 3)
            rec_["eval_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            if not SMOKE:
                rec_.update(score(p, mass, val_ids, ref, labels))
                for j, nm in enumerate(HEADS):
                    r.save_predictions(OUT / f"val_epoch{epoch:02d}_{nm}.json", val_ids, p[j], f"MS1-GL epoch {epoch}, {nm}")
        history.append(rec_)
        print(f"EPOCH {epoch} " + json.dumps(rec_), flush=True)
        if SMOKE:
            print("SMOKE_DONE", flush=True); return
        if epoch in GL["swa_epochs"]:
            torch.save(model.state_dict(), OUT / f"weights_epoch{epoch:02d}.pt")
        torch.save({"gl": GL, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "history": history}, OUT / "recovery.tmp")
        os.replace(OUT / "recovery.tmp", rec)
        (OUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    sds = [torch.load(OUT / f"weights_epoch{e:02d}.pt", map_location="cpu") for e in GL["swa_epochs"]]
    avg = {k: (sum(s[k].float() for s in sds) / len(sds)).to(sds[-1][k].dtype) for k in sds[-1]}
    torch.save(avg, OUT / f"weights_swa{GL['swa_epochs'][0]}_{GL['swa_epochs'][-1]}.pt")
    eval_model.load_state_dict(avg, strict=True)
    p, mass = predict(eval_model, val_ids, meta, tab, dev)
    res = score(p, mass, val_ids, ref, labels)
    for j, nm in enumerate(HEADS):
        r.save_predictions(OUT / f"val_swa_{nm}.json", val_ids, p[j], f"MS1-GL SWA {GL['swa_epochs']}, {nm}")
    m = GL["multi_passes"]
    pm = np.mean([predict(eval_model, val_ids, meta, tab, dev, k=GL["train_windows"], offset=(j + 0.5) / m, top1=True)[0][0]
                  for j in range(m)], 0)
    r.save_predictions(OUT / "val_swa_fused_multi6.json", val_ids, pm, f"MS1-GL SWA, fused, {m} x (16 windows + 12 top-1 crops)")
    res["val_fused_multi6"] = r.macro_auc(pm, val_ids, ref)[0]; res["lv2_fused_multi6"] = r.macro_auc(pm, val_ids, labels)[0]
    summary = {"gl": GL, "cfg_version": CFG["version"], "history": history, "swa": res,
               "seconds": round(time.monotonic() - started, 1)}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps({k: v for k, v in res.items() if k != "crop_mass"}), flush=True)


if __name__ == "__main__":
    main()
