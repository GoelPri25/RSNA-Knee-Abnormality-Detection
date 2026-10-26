"""Exp J2-X: ResNet-50 (label-v2 / 1e recipe) + Swin3D-T (B2 recipe) trained together, output = the two heads' logit
average + a residual Delta from anatomy-aligned cross-attention. Pre-registration: docs/experiment_ledger.md row J2-X.

Why (J1-XAI, X0): J1's fused head was a mixture of two pooled vectors (no token met a token of the other branch), its
ResNet branch was starved (MCL), and output-level stacking across findings adds nothing (X0). J2-X keeps each branch on
its own full loss (no starvation) and puts any gain over averaging in one place:

  final = 1/2 (z_R + z_S).detach() + Delta,   Delta zero-initialised (the model starts as the ensemble)
  Delta: each ResNet window token (LayerNorm feature, 2048 -> 512) cross-attends to the Swin stage-4 tokens of the SAME
         series at the window's slice depth +- 1 token (3 x 10 x 10 tokens, + a learned null token; depth-offset and grid
         cell embeddings) in 2 pre-norm blocks; per-finding attention pooling over windows -> linear (zero init).
         Alignment is exact from the two caches' manifests: d96 row -> (series UID, native slice) -> d96n row -> the Swin
         token depth that row was fed at (after series dropout, stride-2 thinning and mirroring). d96 coronal non-fluid
         windows have no Swin series (null token only). Mirroring is drawn once per study and applied to both branches.
  Loss  BCE(final) + BCE(z_R) + BCE(z_S); the branch heads get no gradient from BCE(final) (detach), the backbones do
        (through Delta), so joint features can form without one branch taking over.
  Schedule  ResNet groups and Delta: 1e (25 epochs, 1 warm-up, cosine to 1%); Swin groups: B2 (15-epoch cosine) then the
        1% floor; SWA of epochs 21-25 + BN refresh of the ResNet branch; 2 studies x 4 accumulation.

Env: JOINT_CACHE_R, JOINT_CACHE_S, JOINT_OUT (as the J1 runner), J2X_MODE = train | profile, J2X_CHECK_DIR (profile).
"""
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ["JOINT_MODE"] = "j2x"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import joint_r50_swin_runner as j

rr, s, dev = j.rr, j.s, j.dev
OUT = Path(os.environ["JOINT_OUT"])
MODE = os.environ.get("J2X_MODE", "train")
CFG = {
    "version": "j2x_cross_residual_v1", "labels": "label-v2 dd0ab9ab", "epochs": 25, "s_epochs": 15, "warmup_epochs": 1,
    "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25], "studies_per_step": 2, "accum": 4,
    "r_lr": 1e-4, "r_head_lr": 1e-3, "x_lr": 1e-3, "wd": 0.05, "grad_clip": 1.0,
    "dim": 512, "heads": 8, "layers": 2, "pick_tol": 2, "bn_refresh_studies": 256, "seed": 2026, "workers": 8,
    "eval_epochs": [1, 3, 5, 10, 15, 20, 21, 22, 23, 24, 25], "kill": {"epoch": 5, "min_val": 0.83},
    "r_cfg": j.CFG["r_cfg"], "s_cfg": j.CFG["s_cfg"],
}
assert rr.CFG["aug"]["mirror_p"] == s.CFG["aug"]["mirror_p"] and rr.CFG["maxs"] == 96


# ---------------------------------------------------------------- alignment d96 row -> d96n row
def _rows(rec):
    out = {}
    for q_, sl in enumerate(rec["slots"]):
        if sl.get("series") and sl.get("picks"):
            for q, p in enumerate(sl["picks"]):
                out[sl["row_start"] + q] = (sl["series"], int(p), q_)
    return out


def build_align(ids):
    """study -> int64 [96]: the d96n row holding the same native slice (+- pick_tol) of the same series, else -1."""
    recs = {}
    for cache, tag in ((os.environ["JOINT_CACHE_R"], "r"), (os.environ["JOINT_CACHE_S"], "s")):
        for sh in sorted(Path(cache).glob("shard_*")):
            for line in (sh / "manifest.jsonl").read_text().splitlines():
                r = json.loads(line)
                recs.setdefault(r["study_id"], {})[tag] = r
    amap = {}
    for sid in ids:
        a, b = recs[sid]["r"], recs[sid]["s"]
        by = {}
        for row, (ser, p, q_) in _rows(b).items():
            assert row < b["n"] and b["slot_of"][row] == q_, (sid, row)                  # d96n slot rows are where the manifest says
            by.setdefault(ser, []).append((p, row))
        m = np.full(96, -1, np.int64)
        for row, (ser, p, _) in _rows(a).items():
            if ser in by and row < 96:
                pp, rw = min(by[ser], key=lambda z: abs(z[0] - p))
                if abs(pp - p) <= CFG["pick_tol"]:
                    m[row] = rw
        amap[sid] = m
    return amap


# ---------------------------------------------------------------- data
class StudiesC(rr.Studies):
    """rr.Studies (1e path) that also returns the window centre rows."""

    def __getitem__(self, i):
        assert not any(k in rr.CFG for k in ("alloc", "shift", "slice_gap")) and not self.guide
        rec, orig = {}, rr.centres

        def wrap(*a, **k):
            rec["c"] = orig(*a, **k)
            return rec["c"]
        rr.centres = wrap
        try:
            item = super().__getitem__(i)
        finally:
            rr.centres = orig
        return (*item, torch.from_numpy(np.clip(rec["c"], 1, rr.CFG["maxs"] - 2)))


class BothX(torch.utils.data.Dataset):
    def __init__(self, ids, meta_r, meta_s, labels, k, train, amap):
        self.ids, self.amap = list(ids), amap
        self.r = StudiesC(ids, meta_r, labels, k, train)
        self.s = s.Volumes(ids, meta_s, {})

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x, sag, y, w, ctx, crow = self.r[i]
        vol, slot, _, _ = self.s[i]
        return x, sag, y, w, ctx, crow, vol, slot, torch.from_numpy(self.amap[self.ids[i]])


def collate(batch):
    x, sag, y, w, ctx, crow, vol, slot, am = zip(*batch)
    return (torch.stack(x), torch.stack(sag), torch.stack(y), torch.stack(w), torch.stack(ctx), torch.stack(crow),
            list(vol), list(slot), list(am))


def loader(ds, shuffle, bs, seed):
    return torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=CFG["workers"], pin_memory=True, collate_fn=collate,
        prefetch_factor=2, generator=torch.Generator().manual_seed(seed),
        worker_init_fn=lambda wk: np.random.seed((seed * 1000 + wk) % 2**32))


class PrepX(s.Prep):
    """s.Prep with the per-study mirror passed in and, per fed series, the d96n rows in the order they were fed."""

    def __call__(self, vol, slot, train, res, mirror=False):
        x = vol.to(self.device, non_blocking=True).float().div_(255.0)
        slot = slot.numpy()
        a2 = s.CFG.get("aug2") if train else None
        if train and not a2:
            theta, gamma, gain = self.draw()
        spans = [(q, np.where(slot == q)[0]) for q in range(4)]
        spans = [(q, int(rw[0]), int(rw[-1]) + 1) for q, rw in spans if len(rw)]
        if a2 and len(spans) > 1:
            keep = [sp for sp in spans if random.random() >= a2["series_drop_p"]]
            spans = keep or [random.choice(spans)]
        out, rows = [], []
        for q, lo_r, hi_r in spans:
            v, rws = x[lo_r:hi_r], np.arange(lo_r, hi_r)
            if a2:
                if len(v) >= a2["stride2_min"] and random.random() < a2["stride2_p"]:
                    st = random.randint(0, 1)
                    v, rws = v[st::2], rws[st::2]
                theta, gamma, gain = self.draw()
            if mirror:
                if q in s.SAGITTAL:
                    v, rws = v.flip(0), rws[::-1].copy()
                else:
                    v = v.flip(-1)
            if train:
                grid = F.affine_grid(theta.expand(len(v), 2, 3), (len(v), 1, s.CFG["res"], s.CFG["res"]),
                                     align_corners=False)
                v = F.grid_sample(v[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
                v = (v.clamp_min(1e-4).pow(gamma) * gain).clamp_(0, 1)
                if a2:
                    with torch.autocast(self.device.type, enabled=False):
                        v = self.mri(v.float(), a2)
            else:
                lo = (s.CFG["img"] - s.CFG["res"]) // 2
                v = v[:, lo:lo + s.CFG["res"], lo:lo + s.CFG["res"]]
            if res != s.CFG["res"]:
                v = F.interpolate(v[:, None], size=(res, res), mode="bilinear", antialias=True, align_corners=False)[:, 0]
            out.append((q, (v[None, None].expand(1, 3, *v.shape) - self.mean) / self.std))
            rows.append((rws, lo_r, hi_r))
        return out, rows


def prepare_r(x, sag, flip, train, res):
    """rr.prepare; in training with the per-study mirror passed in (the rest of 1e's augmentation unchanged)."""
    if not train:
        return rr.prepare(x, sag, False, res, dev)
    b, k = x.shape[:2]
    x = x.to(dev, non_blocking=True).flatten(0, 1)
    sg = sag.to(dev).flatten()
    a = rr.CFG["aug"]
    fl = torch.tensor(flip, device=dev).repeat_interleave(k)
    x = torch.where(fl.view(-1, 1, 1, 1), rr.mirror(x, sg), x)
    x = x.float().div_(255.0)
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(a["rot_deg"])
    sc = 1 + (torch.rand(b, device=dev) * 2 - 1) * a["scale"]
    tx, ty = [(torch.rand(b, device=dev) * 2 - 1) * a["shift"] * 2 for _ in range(2)]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta.repeat_interleave(k, 0), (b * k, 3, res, res), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gamma = torch.exp((torch.rand(b, device=dev) * 2 - 1) * a["gamma"]).repeat_interleave(k).view(-1, 1, 1, 1)
    gain = (1 + (torch.rand(b, device=dev) * 2 - 1) * a["gain"]).repeat_interleave(k).view(-1, 1, 1, 1)
    x = (x.clamp_min(1e-4).pow(gamma) * gain).clamp_(0, 1)
    x = (x - rr.MEAN.to(dev)) / rr.STD.to(dev)
    return x.contiguous(memory_format=torch.channels_last)


def window_keys(crow, am, fed_rows, n_tok):
    """Per window: (offset of its Swin series in the concatenated tokens, depth t, T') or None. fed_rows: per fed series
    (rows in fed order, first row, end row) of the d96n study; a window matches the series whose row span holds its row."""
    out, offs = [], np.cumsum([0] + list(n_tok[:-1]))
    for c in crow:
        r, hit = int(am[int(c)]), None
        if r >= 0:
            for (rws, lo, hi), off, nt in zip(fed_rows, offs, n_tok):
                if lo <= r < hi:
                    d = np.abs(rws - r)
                    if d.min() <= CFG["pick_tol"]:
                        hit = (int(off), int(np.argmin(d)) // 2, nt // 100)
                    break
        out.append(hit)
    return out


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H = H
        self.nq, self.nk = nn.LayerNorm(D), nn.LayerNorm(D)
        self.q, self.kv, self.o = nn.Linear(D, D), nn.Linear(D, 2 * D), nn.Linear(D, D)
        self.n2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, keys, valid):
        """x [K, D]; keys [K, M, D]; valid [K, M] bool."""
        K, M, D = keys.shape
        h = self.H
        q = self.q(self.nq(x)).view(K, h, 1, D // h)
        k, v = self.kv(self.nk(keys)).view(K, M, 2, h, D // h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        x = x + self.o(a.reshape(K, D))
        return x + self.mlp(self.n2(x))


class J2X(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.R = rr.build(rr.CFG["train_res"], pretrained)
        assert type(self.R).__name__ == "Raptor"
        self.S = s.SwinB(pretrained=pretrained)
        D, dr, ds = CFG["dim"], self.R.backbone.num_features, self.S.clsW.shape[1]
        self.q_in = nn.Sequential(nn.LayerNorm(dr), nn.Linear(dr, D))
        self.kv_in = nn.Sequential(nn.LayerNorm(ds), nn.Linear(ds, D))
        self.cell = nn.Parameter(torch.zeros(100, D)); self.dz = nn.Parameter(torch.zeros(3, D))
        self.null = nn.Parameter(torch.zeros(1, D))
        for p_ in (self.cell, self.dz, self.null):
            nn.init.trunc_normal_(p_, std=0.02)
        self.blocks = nn.ModuleList([Block(D, CFG["heads"]) for _ in range(CFG["layers"])])
        self.att = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.dW = nn.Parameter(torch.zeros(12, D)); self.db = nn.Parameter(torch.zeros(12))

    def forward(self, xr, b, ctx, crow, fed, amaps):
        """fed: per study (series list [(slot, [1, 3, T, H, W])], rows per series). Returns final, z_R, z_S, Delta [b, 12]."""
        R, S = self.R, self.S
        f = R.backbone(xr).view(b, xr.shape[0] // b, -1)
        hr = R.context(R.norm(f), ctx)
        ar = torch.softmax(R.att(hr).float(), dim=1).to(hr.dtype)
        lr_ = ((torch.einsum("bkn,bkf->bnf", ar, hr) * R.clsW).sum(-1) + R.clsb).float()
        ls, dl = [], []
        for i, (series, rows) in enumerate(fed):
            K = hr.shape[1]
            x = self.q_in(hr[i])
            null = self.null.expand(K, 1, -1).to(x.dtype)
            if series:
                toks = [S.encode(v, q) for q, v in series]
                ids = torch.cat([torch.full((len(t),), q, device=t.device) for t, (q, _) in zip(toks, series)])
                hs = S.norm(torch.cat(toks)) + S.series(ids)
                a_s = torch.softmax(S.att(hs).float(), dim=0).to(hs.dtype)
                ls.append(((torch.einsum("nk,nf->kf", a_s, hs) * S.clsW).sum(-1) + S.clsb).float())
                kv = self.kv_in(hs)
                n_tok = [len(t) for t in toks]
                assert all(n % 100 == 0 for n in n_tok)
                hits = window_keys(crow[i].numpy(), amaps[i].numpy(), rows, n_tok)
                idx = torch.zeros(K, 3, 100, dtype=torch.long)
                ok = torch.zeros(K, 3, 100, dtype=torch.bool)
                for kk, h in enumerate(hits):
                    if h is None:
                        continue
                    off, t, T_ = h
                    for dz in range(3):
                        tt = t + dz - 1
                        if 0 <= tt < T_:
                            idx[kk, dz] = off + tt * 100 + torch.arange(100); ok[kk, dz] = True
                idx, ok = idx.to(kv.device), ok.to(kv.device)
                keys = kv[idx.view(-1)].view(K, 3, 100, -1) + self.dz[None, :, None].to(kv.dtype) + self.cell[None, None].to(kv.dtype)
                keys = torch.cat([null, keys.view(K, 300, -1)], 1)
                valid = torch.cat([torch.ones(K, 1, dtype=torch.bool, device=kv.device), ok.view(K, 300)], 1)
            else:
                ls.append(S.clsb.float())
                keys, valid = null, torch.ones(K, 1, dtype=torch.bool, device=x.device)
            keys = keys.to(x.dtype)
            for blk in self.blocks:
                x = blk(x, keys, valid)
            a = torch.softmax(self.att(x).float(), dim=0).to(x.dtype)                 # [K, 12]
            dl.append(((torch.einsum("kn,kf->nf", a, x) * self.dW).sum(-1) + self.db).float())
        ls, dl = torch.stack(ls), torch.stack(dl)
        return 0.5 * (lr_ + ls).detach() + dl, lr_, ls, dl


def param_groups(model):
    r_bb = list(model.R.backbone.parameters())
    r_head = [p for n, p in model.R.named_parameters() if not n.startswith("backbone.")]
    xs = [p for n, p in model.named_parameters() if not n.startswith(("R.", "S."))]
    gr = [{"params": r_bb, "lr": CFG["r_lr"], "weight_decay": CFG["wd"]},
          {"params": r_head, "lr": CFG["r_head_lr"], "weight_decay": CFG["wd"]}]
    gs = s.param_groups(model.S)
    gx = [{"params": xs, "lr": CFG["x_lr"], "weight_decay": CFG["wd"]}]
    n_opt = sum(p.numel() for g in gr + gs + gx for p in g["params"])
    assert n_opt == sum(p.numel() for p in model.parameters() if p.requires_grad)
    return gr + gs + gx, (len(gr), len(gs), len(gx))


def schedule(opt, sizes, spe):
    warm, floor = spe * CFG["warmup_epochs"], CFG["lr_floor"]

    def cos(total):
        return lambda q: (q + 1) / warm if q < warm else floor + (1 - floor) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, (q - warm) / max(1, total - warm))))
    main_, swin = cos(spe * CFG["epochs"]), cos(spe * CFG["s_epochs"])
    return torch.optim.lr_scheduler.LambdaLR(opt, [main_] * sizes[0] + [swin] * sizes[1] + [main_] * sizes[2])


bce = j.bce


def save_predictions(path, ids, pred, head, note):
    """rr.save_predictions with this run's version + head as the model field (rr's would say 1e)."""
    path.write_text(json.dumps({"model": f"{CFG['version']}:{head}", "note": note, "targets_in_order": rr.TARGETS,
                                "predictions": [{"study_id": x, "probabilities": [float(v) for v in q]}
                                                for x, q in zip(ids, pred)]}) + "\n")


def train_step(model, prep, batch):
    x, sag, y, w, ctx, crow, vols, slots, ams = batch
    b = x.shape[0]
    flip = [random.random() < rr.CFG["aug"]["mirror_p"] for _ in range(b)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fed = [prep(v, sl, True, s.CFG["res"], f) for v, sl, f in zip(vols, slots, flip)]
        fin, lr_, ls, _ = model(prepare_r(x, sag, flip, True, rr.CFG["train_res"]), b, ctx.to(dev), crow, fed, ams)
    y, w = y.to(dev), w.to(dev)
    parts = (bce(fin, y, w), bce(lr_, y, w), bce(ls, y, w))
    return parts[0] + parts[1] + parts[2], parts


@torch.no_grad()
def predict(model, prep, ids, meta_r, meta_s, amap, limit=None):
    model.eval()
    ids = ids[:limit] if limit else ids
    out = {"final": [], "r50": [], "swin": [], "delta": []}
    for x, sag, _, _, ctx, crow, vols, slots, ams in loader(
            BothX(ids, meta_r, meta_s, {}, rr.CFG["eval_windows"], False, amap), False, 1, 0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fed = [prep(vols[0], slots[0], False, s.CFG["res"])]
            fin, lr_, ls, dl = model(prepare_r(x, sag, None, False, rr.CFG["eval_res"]), 1, ctx.to(dev), crow, fed, ams)
        for k, v in (("final", fin.sigmoid()), ("r50", lr_.sigmoid()), ("swin", ls.sigmoid()), ("delta", dl)):
            out[k].append(v.float()[0].cpu().numpy())
    return {k: np.stack(v) for k, v in out.items()}


def ens_prob(p):
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-0.5 * (lg(p["r50"]) + lg(p["swin"]))))


# ---------------------------------------------------------------- profile: checks + timing, no run
def profile():
    t_all = time.monotonic()
    torch.manual_seed(CFG["seed"]); random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if s.CFG["attn"] == "sdpa":
        s.use_fused_attention()
    labels, ref, meta_r, meta_s, train_ids, val_ids, gold_ids = j.setup()
    print(f"CFG {json.dumps(CFG)}", flush=True)
    amap = build_align(train_ids + val_ids + gold_ids)

    # 0. alignment coverage on the evaluation windows (94 per study), by d96 slot
    cnt = np.zeros((5, 2))
    for sid in val_ids + train_ids[:800]:
        m = meta_r[sid]
        c = np.clip(rr.centres(m["mask"], rr.CFG["eval_windows"], False), 1, 94)
        sl = rr.SLOT_OF[c]
        for q in range(5):
            cnt[q] += [(amap[sid][c[sl == q]] >= 0).sum(), (sl == q).sum()]
    print("ALIGN eval windows matched to a Swin row, by d96 slot (sagFS/sagT1/corFS/corT1/axial): "
          + " ".join(f"{a / max(b, 1):.3f}" for a, b in cnt) + f"; all {cnt[:, 0].sum() / cnt[:, 1].sum():.3f}", flush=True)

    prep = PrepX(*s.kinetics_norm(), dev)
    # 1. mapping self-check in training mode (series dropout, stride 2, mirror): each matched window's token depth
    #    holds a fed row within pick_tol of the window's d96n row
    ds = BothX(train_ids[:64], meta_r, meta_s, labels, rr.CFG["train_windows"], True, amap)
    n_hit = n_all = 0
    for i in range(64):
        x, sag, y, w, ctx, crow, vol, slot, am = ds[i]
        fed, rows = prep(vol, slot, True, s.CFG["res"], i % 2 == 0)
        n_tok = [((v.shape[2] + 1) // 2) * 100 for _, v in fed]
        hits = window_keys(crow.numpy(), am.numpy(), rows, n_tok)
        for c, h in zip(crow.numpy(), hits):
            n_all += 1
            if h is None:
                continue
            n_hit += 1
            off, t, T_ = h
            k_ = int(np.searchsorted(np.cumsum([0] + n_tok), off))
            rws = rows[k_][0]
            assert min(abs(int(r) - int(am[int(c)])) for r in rws[2 * t:2 * t + 2]) <= CFG["pick_tol"], (i, c)
    print(f"MAP_OK training windows matched {n_hit}/{n_all}", flush=True)

    # 2. parity: 1e + B2 weights, fresh Delta (zero) -> heads reproduce their runs; final = logit average exactly
    chk = Path(os.environ["J2X_CHECK_DIR"])
    m = J2X(pretrained=False)
    m.R.load_state_dict(torch.load(chk / "run_r50_pertarget/weights_swa21_25.pt", map_location="cpu", weights_only=True), strict=True)
    m.S.load_state_dict(torch.load(chk / "run_swin3d_b2/weights_swa11_15.pt", map_location="cpu", weights_only=True), strict=True)
    m = m.to(dev); m.R.to(memory_format=torch.channels_last)
    p = predict(m, prep, val_ids, meta_r, meta_s, amap, limit=40)
    for name, run in (("r50", "run_r50_pertarget"), ("swin", "run_swin3d_b2")):
        saved = {q["study_id"]: q["probabilities"] for q in json.loads((chk / run / "val_swa.json").read_text())["predictions"]}
        diff = float(np.abs(p[name] - np.array([saved[x] for x in val_ids[:40]])).max())
        print(f"PARITY {name} head vs {run}/val_swa.json on 40 studies: max |dp| {diff:.4f}", flush=True)
        assert diff <= 0.01, f"{name} parity failed"
    d0 = float(np.abs(p["final"] - ens_prob(p)).max())
    print(f"PARITY final vs logit average at init: max |dp| {d0:.2e}; |Delta| max {np.abs(p['delta']).max():.2e}", flush=True)
    assert d0 < 1e-3
    del m; torch.cuda.empty_cache()

    # 3. fresh pretrained model: optimiser groups, steps; then BCE(final) alone reaches the backbones and Delta, not heads
    model = J2X(pretrained=True).to(dev); model.R.to(memory_format=torch.channels_last)
    groups, sizes = param_groups(model)
    opt = torch.optim.AdamW(groups)
    spe = math.ceil(len(train_ids) / (CFG["studies_per_step"] * CFG["accum"]))
    sched = schedule(opt, sizes, spe)
    model.train()
    it = iter(loader(BothX(train_ids, meta_r, meta_s, labels, rr.CFG["train_windows"], True, amap), True,
                     CFG["studies_per_step"], 1))
    torch.cuda.reset_peak_memory_stats()
    n, t0 = 0, None
    for i in range(28):
        if i == 4:
            torch.cuda.synchronize(); t0 = time.monotonic(); n = 0
        loss, parts = train_step(model, prep, next(it))
        (loss / CFG["accum"]).backward()
        if (i + 1) % CFG["accum"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"]); opt.step(); sched.step()
            opt.zero_grad(set_to_none=True)
        n += CFG["studies_per_step"]
    torch.cuda.synchronize()
    tr = (time.monotonic() - t0) / n
    print(f"STEP_OK loss {float(loss):.4f} parts {[round(float(v), 4) for v in parts]}; params "
          f"{sum(p_.numel() for p_ in model.parameters()) / 1e6:.1f}M, trainable "
          f"{sum(p_.numel() for p_ in model.parameters() if p_.requires_grad) / 1e6:.1f}M, Delta module "
          f"{sum(p_.numel() for n_, p_ in model.named_parameters() if not n_.startswith(('R.', 'S.'))) / 1e6:.1f}M", flush=True)
    opt.zero_grad(set_to_none=True)
    x, sag, y, w, ctx, crow, vols, slots, ams = next(it)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fed = [prep(v, sl, True, s.CFG["res"], False) for v, sl in zip(vols, slots)]
        fin, _, _, dl = model(prepare_r(x, sag, [False] * len(vols), True, rr.CFG["train_res"]), x.shape[0], ctx.to(dev), crow, fed, ams)
    bce(fin, y.to(dev), w.to(dev)).backward()
    g = {n_: (p_.grad is not None and float(p_.grad.abs().sum()) > 0) for n_, p_ in model.named_parameters() if p_.requires_grad}
    for part in ("R.backbone.", "S.branch.", "q_in.", "kv_in.", "blocks.", "att.", "dW", "cell", "dz"):
        assert any(v for n_, v in g.items() if n_.startswith(part)), f"BCE(final) does not reach {part}"
    for part in ("R.clsW", "R.att.", "S.clsW", "S.att."):
        assert not any(v for n_, v in g.items() if n_.startswith(part)), f"BCE(final) reaches the branch head {part}"
    print(f"GRAD_OK BCE(final) reaches both backbones and Delta, not the branch heads; |Delta| after "
          f"{28 // CFG['accum']} steps {float(dl.abs().max()):.3e}", flush=True)
    opt.zero_grad(set_to_none=True)
    te0 = time.monotonic(); predict(model, prep, val_ids, meta_r, meta_s, amap, limit=60); te = (time.monotonic() - te0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30
    ep, ev = tr * len(train_ids) / 60, te * len(val_ids) / 60
    total = CFG["epochs"] * ep + len(CFG["eval_epochs"]) * ev + 2 * ev * (len(val_ids) + len(gold_ids)) / len(val_ids)
    print(f"TIMING train {tr:.3f} s/study ({ep:.1f} min/epoch), eval {te:.3f} s/study ({ev:.1f} min/pass), peak "
          f"{peak:.1f} GiB; estimated run {total / 60:.2f} h ({CFG['epochs']} epochs, {len(CFG['eval_epochs'])} scored)", flush=True)
    print(f"PROFILE_OK in {time.monotonic() - t_all:.0f}s", flush=True)


# ---------------------------------------------------------------- training
def main():
    started = time.monotonic()
    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if s.CFG["attn"] == "sdpa":
        s.use_fused_attention()
    OUT.mkdir(parents=True, exist_ok=True)
    labels, ref, meta_r, meta_s, train_ids, val_ids, gold_ids = j.setup()
    print(f"CFG {json.dumps(CFG)}\ntrain {len(train_ids)}, validation {len(val_ids)}, gold {len(gold_ids)}", flush=True)
    amap = build_align(train_ids + val_ids + gold_ids)
    prep = PrepX(*s.kinetics_norm(), dev)
    model = J2X(pretrained=True).to(dev); model.R.to(memory_format=torch.channels_last)
    groups, sizes = param_groups(model)
    opt = torch.optim.AdamW(groups)
    spe = math.ceil(len(train_ids) / (CFG["studies_per_step"] * CFG["accum"]))
    sched = schedule(opt, sizes, spe)
    start, history, rec = 1, [], OUT / "recovery.pt"
    if rec.is_file():
        st = torch.load(rec, map_location="cpu", weights_only=False)
        assert st["cfg"] == CFG, "recovery config differs"
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, history = st["epoch"] + 1, st["history"]
        random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"]); torch.cuda.set_rng_state_all(st["cuda_rng"])
        print(f"RESUMED after epoch {st['epoch']}", flush=True)
        del st
    train_ds = BothX(train_ids, meta_r, meta_s, labels, rr.CFG["train_windows"], True, amap)
    for epoch in range(start, CFG["epochs"] + 1):
        model.train(); t0 = time.monotonic()
        run = np.zeros(4); n = 0
        opt.zero_grad(set_to_none=True)
        dl = loader(train_ds, True, CFG["studies_per_step"], CFG["seed"] * 100 + epoch)
        for i, batch in enumerate(dl):
            loss, parts = train_step(model, prep, batch)
            (loss / CFG["accum"]).backward()
            if (i + 1) % CFG["accum"] == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += [float(loss), *map(float, parts)]; n += 1
            if (i + 1) % 400 == 0:
                print(f"  epoch {epoch} step {i + 1}/{len(dl)} loss {run[0] / n:.4f} "
                      f"{(time.monotonic() - t0) / (n * CFG['studies_per_step']):.3f}s/study", flush=True)
        record = {"epoch": epoch, "train_loss": run[0] / n, "train_final": run[1] / n, "train_r50": run[2] / n,
                  "train_swin": run[3] / n, "val_macro_auc": None, "val_bce": None,
                  "train_s": round(time.monotonic() - t0, 1), "lr": [g["lr"] for g in opt.param_groups]}
        if epoch in CFG["eval_epochs"]:
            p = predict(model, prep, val_ids, meta_r, meta_s, amap)
            p["ens"] = ens_prob(p)
            for head in ("final", "r50", "swin", "ens"):
                sc, per = rr.macro_auc(p[head], val_ids, ref)
                record[f"val_{head}"] = sc; record[f"per_target_{head}"] = per
            record["val_macro_auc"] = record["val_final"]
            record["val_bce"] = s.val_bce(p["final"], val_ids, ref); record["val_bce_ens"] = s.val_bce(p["ens"], val_ids, ref)
            record["delta_sd"] = dict(zip(rr.TARGETS, np.round(p["delta"].std(0), 3).tolist()))
            save_predictions(OUT / f"val_epoch{epoch:02d}.json", val_ids, p["final"], "final", f"epoch {epoch}, no TTA")
        record["total_s"] = round(time.monotonic() - t0, 1)
        history.append(record)
        if epoch in CFG["swa_epochs"]:
            torch.save(model.state_dict(), OUT / f"weights_epoch{epoch:02d}.pt")
        torch.save({"cfg": CFG, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "history": history, "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, OUT / "recovery.tmp")
        os.replace(OUT / "recovery.tmp", rec)
        (OUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        v = record["val_macro_auc"]
        scored = "val - " if v is None else (
            f"val final {v:.4f} BCE {record['val_bce']:.4f} | ens {record['val_ens']:.4f} BCE {record['val_bce_ens']:.4f} "
            f"| Delta gain {v - record['val_ens']:+.4f} | r50 {record['val_r50']:.4f} swin {record['val_swin']:.4f} "
            f"| Delta sd {json.dumps(record['delta_sd'])} ")
        print(f"EPOCH {epoch}: loss {record['train_loss']:.4f} (final {record['train_final']:.4f} r50 "
              f"{record['train_r50']:.4f} swin {record['train_swin']:.4f}) {scored}({record['total_s']:.0f}s)", flush=True)
        kill = CFG["kill"]
        if epoch == kill["epoch"] and v is not None and v < kill["min_val"]:
            (OUT / "run_summary.json").write_text(json.dumps({"cfg": CFG, "history": history, "killed":
                f"epoch {epoch} val {v:.4f} < {kill['min_val']}"}, indent=2) + "\n")
            print(f"RUN_SUMMARY killed at epoch {epoch}", flush=True)
            return

    sd = [torch.load(OUT / f"weights_epoch{e:02d}.pt", map_location="cpu", weights_only=True) for e in CFG["swa_epochs"]]
    model.load_state_dict({k: (sum(q[k].float() for q in sd) / len(sd)).to(sd[-1][k].dtype)
                           if sd[-1][k].is_floating_point() else sd[-1][k] for k in sd[-1]})
    del sd
    j.refresh_bn(model, train_ids, meta_r, labels)
    tag = f"swa{CFG['swa_epochs'][0]}_{CFG['swa_epochs'][-1]}"
    torch.save(model.state_dict(), OUT / f"weights_{tag}.pt")
    results = {}
    for split, ids in (("val", val_ids), ("gold", gold_ids)):
        p = predict(model, prep, ids, meta_r, meta_s, amap)
        p["ens"] = ens_prob(p)
        for head, suffix in (("final", ""), ("r50", "_r50"), ("swin", "_swin"), ("ens", "_ens")):
            save_predictions(OUT / f"{split}_swa{suffix}.json", ids, p[head], head, f"SWA {tag}, 320, no TTA")
        np.save(OUT / f"{split}_swa_delta.npy", p["delta"])
        if split == "val":
            for head in ("final", "r50", "swin", "ens"):
                results[head] = rr.macro_auc(p[head], ids, ref)
    summary = {"cfg": CFG, "history": history, "swa": results, "seconds": round(time.monotonic() - started, 1),
               "note": "Primary = final (val_swa.json) vs ens (val_swa_ens.json, the same model's logit average)."}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps({k: v[0] for k, v in results.items()} | {"seconds": summary["seconds"]}), flush=True)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        profile() if MODE == "profile" else main()
