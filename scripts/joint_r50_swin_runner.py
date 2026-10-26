"""Exp J1: one network with a 2D / 2.5D ResNet-50 branch (Exp 1e) and a 3D Swin3D-T branch (Model B2), joined by
per-finding combined attention over the union of both branches' tokens. Pre-registration: docs/experiment_ledger.md
row J1.

  ResNet branch  Exp 1e exactly: resnet50.tv_in1k on 3-slice windows of the d96 cache (16 stratified windows at 384
                 in training, 94 at 320 for evaluation, 1e augmentation); tokens = LayerNorm window features [K, 2048]
                 + 1e's own per-finding attention head (deep supervision).
  Swin branch    Model B2 exactly: Swin3D-T Kinetics-400 on the d96n native volumes (patch embedding + stages 1-2
                 frozen, layer-wise lr decay 0.9, drop path 0.25, B2 augmentation); tokens = stage-4 tokens + series
                 embedding [N, 768] + B2's own per-finding attention head (deep supervision).
  Fusion         both token sets projected to 512 (Linear + LayerNorm) + a branch-type embedding; per finding, one
                 attention distribution over the union of the K + N tokens (1e's tanh attention MLP; each branch's
                 scores offset by -log(its token count) + a learned per-finding branch gate, so the branches start at
                 equal mass and the ResNet share does not move between 16 training and 94 evaluation windows) ->
                 pooled feature -> per-finding linear logit. Branch dropout: in training, with p 0.15 per study one branch's
                 tokens are removed from the fused pool (never both; the ResNet tokens stay if a study has no series).
  Loss           fused BCE + 0.5 x (ResNet-head BCE + Swin-head BCE), soft labels and weights as 1e (label-v2).
  Optimiser      AdamW; ResNet backbone 1e-4, ResNet head 1e-3 (1e); Swin groups exactly B2 (5e-5, decay 0.9, head
                 1e-3); fusion 1e-3; wd 0.05; 2 studies per forward x 4 accumulation = 8 per step (1e); one cosine
                 (1 warm-up epoch, 1% floor) over 20 epochs; SWA of epochs 16-20 + BN refresh of the ResNet branch.
Outputs (OUT): val / gold predictions of the fused head (val_swa.json = primary) and of each branch head
(val_swa_r50.json, val_swa_swin.json), per-epoch history with all three heads and the ResNet share of the fused
attention per finding, run_summary.json. Each runner module reads its configuration from the environment at import,
so this script imports the ResNet runner twice: once in native mode (inside the Swin runner) and once in d96 mode.

Env: JOINT_CACHE_R (d96), JOINT_CACHE_S (d96n), JOINT_OUT, JOINT_MODE = train | profile, JOINT_CHECK_DIR (profile:
experiment dir holding run_r50_pertarget / run_swin3d_b2 for the inference parity check).
"""
import contextlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ["D96_CACHE"] = os.environ["JOINT_CACHE_S"]
os.environ["SWIN_PHASE"] = "swin3d_b2"
os.environ.pop("SWIN_CFG", None); os.environ.pop("SWIN_AMP", None)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import swin3d_d96n_runner as s                       # B2 config; imports coatnet_d96_runner in native mode (s.r)

os.environ["D96_CACHE"] = os.environ["JOINT_CACHE_R"]
os.environ["D96_PHASE"] = "r50_pertarget"
_spec = importlib.util.spec_from_file_location("r50_d96_runner", Path(__file__).with_name("coatnet_d96_runner.py"))
rr = importlib.util.module_from_spec(_spec); sys.modules["r50_d96_runner"] = rr; _spec.loader.exec_module(rr)
assert not rr.NATIVE_LAYOUT and s.r.NATIVE_LAYOUT and rr.CFG["arch"] == "resnet50.tv_in1k" and s.B2
LABEL_V2 = ("soft_labels_final_per_target_latD2.json", "dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d")
rr.LABELS_FILE = LABEL_V2[0]; rr.CFG["sha256"]["labels"] = LABEL_V2[1]
assert s.r.LABELS_FILE == LABEL_V2[0] and s.r.CFG["sha256"]["labels"] == LABEL_V2[1]

OUT = Path(os.environ["JOINT_OUT"])
MODE = os.environ.get("JOINT_MODE", "train")
CFG = {
    "version": "joint_r50_swin3d_j1_v1", "labels": "label-v2 dd0ab9ab", "epochs": 20, "warmup_epochs": 1,
    "lr_floor": 0.01, "swa_epochs": [16, 17, 18, 19, 20], "studies_per_step": 2, "accum": 4,
    "r_lr": 1e-4, "r_head_lr": 1e-3, "fuse_lr": 1e-3, "wd": 0.05, "grad_clip": 1.0,
    "fuse_dim": 512, "aux_w": 0.5, "branch_drop_p": 0.15, "bn_refresh_studies": 256, "seed": 2026, "workers": 8,
    "eval_epochs": [1, 3, 5, 10, 13, 15, 16, 17, 18, 19, 20], "kill": {"epoch": 5, "min_val": 0.83},
    "r_cfg": {k: rr.CFG[k] for k in ("version", "train_res", "eval_res", "train_windows", "eval_windows", "aug")},
    "s_cfg": {k: s.CFG[k] for k in ("version", "res", "lr", "llrd", "head_lr", "drop_path", "freeze", "pool", "aug2")},
}
dev = torch.device("cuda")


# ---------------------------------------------------------------- data
class Both(torch.utils.data.Dataset):
    """One study -> 1e's windows (d96) and B2's volume (d96n)."""

    def __init__(self, ids, meta_r, meta_s, labels, k, train):
        self.r = rr.Studies(ids, meta_r, labels, k, train)
        self.s = s.Volumes(ids, meta_s, {})

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        x, sag, y, w, ctx = self.r[i]
        vol, slot, _, _ = self.s[i]
        return x, sag, y, w, ctx, vol, slot


def collate(batch):
    x, sag, y, w, ctx, vol, slot = zip(*batch)
    return torch.stack(x), torch.stack(sag), torch.stack(y), torch.stack(w), torch.stack(ctx), list(vol), list(slot)


def loader(ds, shuffle, bs, seed):
    return torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=CFG["workers"], pin_memory=True, collate_fn=collate,
        prefetch_factor=2, generator=torch.Generator().manual_seed(seed),
        worker_init_fn=lambda wk: np.random.seed((seed * 1000 + wk) % 2**32))


# ---------------------------------------------------------------- model
class JointNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.R = rr.build(rr.CFG["train_res"], pretrained)            # CNN: any input size, so one model for both
        assert type(self.R).__name__ == "Raptor"
        self.S = s.SwinB(pretrained=pretrained)
        D = CFG["fuse_dim"]
        self.pr = nn.Sequential(nn.Linear(self.R.backbone.num_features, D), nn.LayerNorm(D))
        self.ps = nn.Sequential(nn.Linear(self.S.clsW.shape[1], D), nn.LayerNorm(D))
        self.kind = nn.Parameter(torch.zeros(2, D))
        self.att = nn.Sequential(nn.Linear(D, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.gate = nn.Parameter(torch.zeros(2, 12))           # learned per-finding branch prior (log scale)
        self.clsW = nn.Parameter(torch.zeros(12, D)); self.clsb = nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.kind, std=0.02); nn.init.trunc_normal_(self.clsW, std=0.02)

    def forward(self, xr, b, ctx, series, drop=None):
        """xr [b*K, 3, H, W]; series: per study [(slot, [1, 3, T, H, W])]; drop: per study 0 keep, 1 no ResNet tokens
        in the fused pool, 2 no Swin tokens. Returns fused, ResNet-head and Swin-head logits [b, 12] and the fused
        attention mass on ResNet tokens [b, 12]."""
        R, S = self.R, self.S
        f = R.backbone(xr).view(b, xr.shape[0] // b, -1)
        hr = R.context(R.norm(f), ctx)
        ar = torch.softmax(R.att(hr).float(), dim=1).to(hr.dtype)
        lr_ = (torch.einsum("bkn,bkf->bnf", ar, hr) * R.clsW).sum(-1) + R.clsb
        lf, ls, share = [], [], []
        for i, sr in enumerate(series):
            ur = self.pr(hr[i]).float() + self.kind[0]
            us = None
            if sr:
                toks = [S.encode(v, j) for j, v in sr]
                ids = torch.cat([torch.full((len(t),), j, device=t.device) for t, (j, _) in zip(toks, sr)])
                hs = S.norm(torch.cat(toks)) + S.series(ids)
                a_s = torch.softmax(S.att(hs).float(), dim=0).to(hs.dtype)
                ls.append(((torch.einsum("nk,nf->kf", a_s, hs) * S.clsW).sum(-1) + S.clsb).float())
                us = self.ps(hs).float() + self.kind[1]
            else:
                ls.append(S.clsb.float())
            d = 0 if drop is None else drop[i]
            keep_r = d != 1 or us is None
            parts = ([(0, ur)] if keep_r else []) + ([(1, us)] if us is not None and d != 2 else [])
            u = torch.cat([q for _, q in parts])
            # one softmax over the union; each branch's tokens get -log(its token count) + a learned gate, so the
            # branches start at equal mass whatever their counts (16 / 94 windows vs ~8,000 Swin tokens)
            prior = torch.cat([(self.gate[k] - math.log(len(q))).expand(len(q), 12) for k, q in parts])
            a = torch.softmax(self.att(u).float() + prior, dim=0)                   # [K + N, 12]
            lf.append(((torch.einsum("nk,nf->kf", a.to(u.dtype), u) * self.clsW).sum(-1) + self.clsb).float())
            share.append(a[:len(ur)].sum(0) if keep_r else torch.zeros(12, device=u.device))
        return torch.stack(lf), lr_.float(), torch.stack(ls), torch.stack(share)


def param_groups(model):
    r_bb = list(model.R.backbone.parameters())
    r_head = [p for n, p in model.R.named_parameters() if not n.startswith("backbone.")]
    fuse = [p for n, p in model.named_parameters() if not n.startswith(("R.", "S."))]
    groups = ([{"params": r_bb, "lr": CFG["r_lr"], "weight_decay": CFG["wd"]},
               {"params": r_head, "lr": CFG["r_head_lr"], "weight_decay": CFG["wd"]}]
              + s.param_groups(model.S)
              + [{"params": fuse, "lr": CFG["fuse_lr"], "weight_decay": CFG["wd"]}])
    n_opt = sum(p.numel() for g in groups for p in g["params"])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_opt == n_train, (n_opt, n_train)
    return groups


# ---------------------------------------------------------------- shared steps
def setup():
    labels = rr.load_labels(rr.ASSETS / rr.LABELS_FILE, "labels")
    ref = rr.load_labels(rr.ASSETS / "validation_labels.json", "validation")
    assert rr.sha(rr.ASSETS / "folds.csv") == rr.CFG["sha256"]["folds"]
    fold = {row["study_id"]: row["fold"] for row in __import__("csv").DictReader((rr.ASSETS / "folds.csv").open())}
    meta_r, meta_s = rr.read_manifest(), s.r.read_manifest()
    gold_ids = sorted(x for x, m in meta_r.items() if m["gold"])
    assert gold_ids == sorted(x for x, m in meta_s.items() if m["gold"])
    train_ids = sorted(x for x in labels if fold[x] != "0")
    val_ids = sorted(x for x in labels if fold[x] == "0")
    assert len(train_ids) == 3524 and len(val_ids) == 825 and len(gold_ids) == 58
    assert set(val_ids) == set(ref) and not (set(gold_ids) & set(labels))
    assert all(x in meta_r and x in meta_s for x in train_ids + val_ids + gold_ids)
    return labels, ref, meta_r, meta_s, train_ids, val_ids, gold_ids


def bce(logit, y, w):
    return (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * w).sum() / w.sum().clamp_min(1)


def train_step(model, prep, batch, drop_p):
    x, sag, y, w, ctx, vols, slots = batch
    b = x.shape[0]
    drop = [0 if random.random() >= drop_p else random.choice((1, 2)) for _ in range(b)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        series = [prep(v, sl, True, s.CFG["res"]) for v, sl in zip(vols, slots)]
        lf, lr_, ls, _ = model(rr.prepare(x, sag, True, rr.CFG["train_res"], dev), b, ctx.to(dev), series, drop)
    y, w = y.to(dev), w.to(dev)
    parts = (bce(lf, y, w), bce(lr_, y, w), bce(ls, y, w))
    return parts[0] + CFG["aux_w"] * (parts[1] + parts[2]), parts


@torch.no_grad()
def predict(model, prep, ids, meta_r, meta_s, limit=None):
    """[n, 12] probabilities of the fused, ResNet and Swin heads + fused ResNet attention share; 320, no TTA."""
    model.eval()
    ids = ids[:limit] if limit else ids
    out = {"fused": [], "r50": [], "swin": [], "share": []}
    for x, sag, _, _, ctx, vols, slots in loader(Both(ids, meta_r, meta_s, {}, rr.CFG["eval_windows"], False),
                                                 False, 1, 0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            series = [prep(vols[0], slots[0], False, s.CFG["res"])]
            lf, lr_, ls, sh = model(rr.prepare(x, sag, False, rr.CFG["eval_res"], dev), 1, ctx.to(dev), series)
        for k, v in (("fused", lf.float().sigmoid()), ("r50", lr_.float().sigmoid()), ("swin", ls.float().sigmoid()),
                     ("share", sh.float())):
            out[k].append(v[0].cpu().numpy())
    return {k: np.stack(v) for k, v in out.items()}


def refresh_bn(model, train_ids, meta_r, labels):
    """1e's BN refresh of the ResNet branch (the Swin branch is LayerNorm only)."""
    bns = [m for m in model.R.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    model.train()
    pick = random.Random(CFG["seed"]).sample(train_ids, CFG["bn_refresh_studies"])
    ds = rr.Studies(pick, meta_r, labels, rr.CFG["train_windows"], False)
    with torch.no_grad():
        for x, sag, _, _, _ in torch.utils.data.DataLoader(ds, batch_size=2, num_workers=CFG["workers"]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.R.backbone(rr.prepare(x, sag, False, rr.CFG["train_res"], dev))


# ---------------------------------------------------------------- profile: checks + timing, no run
def profile():
    t_all = time.monotonic()
    torch.manual_seed(CFG["seed"]); random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
    if s.CFG["attn"] == "sdpa":
        s.use_fused_attention()
    labels, ref, meta_r, meta_s, train_ids, val_ids, gold_ids = setup()
    print(f"CFG {json.dumps(CFG)}\ntrain {len(train_ids)}, validation {len(val_ids)}, gold {len(gold_ids)}", flush=True)
    prep = s.Prep(*s.kinetics_norm(), dev)

    # 1. inference parity: the trained 1e and B2 weights inside JointNet reproduce their saved fold-0 predictions
    chk = Path(os.environ["JOINT_CHECK_DIR"])
    m = JointNet(pretrained=False)
    m.R.load_state_dict(torch.load(chk / "run_r50_pertarget/weights_swa21_25.pt", map_location="cpu",
                                   weights_only=True), strict=True)
    m.S.load_state_dict(torch.load(chk / "run_swin3d_b2/weights_swa11_15.pt", map_location="cpu",
                                   weights_only=True), strict=True)
    m = m.to(dev); m.R.to(memory_format=torch.channels_last)                   # 2D branch only (Swin is 5D)
    p = predict(m, prep, val_ids, meta_r, meta_s, limit=40)
    for name, run in (("r50", "run_r50_pertarget"), ("swin", "run_swin3d_b2")):
        saved = {q["study_id"]: q["probabilities"] for q in json.loads((chk / run / "val_swa.json").read_text())["predictions"]}
        diff = float(np.abs(p[name] - np.array([saved[x] for x in val_ids[:40]])).max())
        print(f"PARITY {name} head vs {run}/val_swa.json on 40 studies: max |dp| {diff:.4f}", flush=True)
        assert diff <= 0.01, f"{name} parity failed"
    del m; torch.cuda.empty_cache()

    # 2. fresh pretrained model: optimiser groups, frozen Swin stages, one step with gradients in every part
    model = JointNet(pretrained=True).to(dev); model.R.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(model))
    model.train()
    ds = Both(train_ids, meta_r, meta_s, labels, rr.CFG["train_windows"], True)
    it = iter(loader(ds, True, CFG["studies_per_step"], 1))
    batch = next(it)
    loss, parts = train_step(model, prep, batch, 0.0)
    loss.backward()
    g = {n: (p_.grad is not None and float(p_.grad.abs().sum()) > 0) for n, p_ in model.named_parameters() if p_.requires_grad}
    for part in ("R.backbone.", "R.att.", "R.clsW", "S.branch.", "S.att.", "S.clsW", "pr.", "ps.", "att.", "clsW", "kind", "gate"):
        assert any(v for n, v in g.items() if n.startswith(part)), f"no gradient reaches {part}"
    assert all(not p_.requires_grad for mod in model.S.frozen() for p_ in mod.parameters())
    print(f"GRAD_OK loss {float(loss):.4f} parts {[round(float(v), 4) for v in parts]}; params "
          f"{sum(p_.numel() for p_ in model.parameters()) / 1e6:.1f}M, trainable "
          f"{sum(p_.numel() for p_ in model.parameters() if p_.requires_grad) / 1e6:.1f}M", flush=True)
    opt.zero_grad(set_to_none=True)
    for d in ((1, 2), (2, 1)):                                                     # branch-dropout paths
        x, sag, y, w, ctx, vols, slots = batch
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            series = [prep(v, sl, True, s.CFG["res"]) for v, sl in zip(vols, slots)]
            lf, _, _, sh = model(rr.prepare(x, sag, True, rr.CFG["train_res"], dev), 2, ctx.to(dev), series, list(d))
        assert torch.isfinite(lf).all()
    print("DROP_OK", flush=True)

    # 3. timing: training steps (with optimiser) and evaluation, peak memory
    torch.cuda.reset_peak_memory_stats()
    n, t0 = 0, None
    for i in range(24):
        if i == 4:
            torch.cuda.synchronize(); t0 = time.monotonic(); n = 0
        loss, _ = train_step(model, prep, next(it), CFG["branch_drop_p"])
        (loss / CFG["accum"]).backward()
        if (i + 1) % CFG["accum"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"]); opt.step(); opt.zero_grad(set_to_none=True)
        n += CFG["studies_per_step"]
    torch.cuda.synchronize()
    tr = (time.monotonic() - t0) / n
    te0 = time.monotonic(); predict(model, prep, val_ids, meta_r, meta_s, limit=60); te = (time.monotonic() - te0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30
    ep = tr * len(train_ids) / 60
    ev = te * len(val_ids) / 60
    total = CFG["epochs"] * ep + len(CFG["eval_epochs"]) * ev + 2 * ev * (len(val_ids) + len(gold_ids)) / len(val_ids)
    print(f"TIMING train {tr:.3f} s/study ({ep:.1f} min/epoch), eval {te:.3f} s/study ({ev:.1f} min/pass), "
          f"peak {peak:.1f} GiB; estimated run {total / 60:.2f} h ({CFG['epochs']} epochs, "
          f"{len(CFG['eval_epochs'])} scored)", flush=True)
    print(f"PROFILE_OK in {time.monotonic() - t_all:.0f}s", flush=True)


# ---------------------------------------------------------------- training
def main():
    started = time.monotonic()
    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if s.CFG["attn"] == "sdpa":
        s.use_fused_attention()
    OUT.mkdir(parents=True, exist_ok=True)
    labels, ref, meta_r, meta_s, train_ids, val_ids, gold_ids = setup()
    print(f"CFG {json.dumps(CFG)}\ntrain {len(train_ids)}, validation {len(val_ids)}, gold {len(gold_ids)}", flush=True)
    prep = s.Prep(*s.kinetics_norm(), dev)
    model = JointNet(pretrained=True).to(dev); model.R.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(model))
    spe = math.ceil(len(train_ids) / (CFG["studies_per_step"] * CFG["accum"]))
    total, warm, floor = spe * CFG["epochs"], spe * CFG["warmup_epochs"], CFG["lr_floor"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda q: (q + 1) / warm if q < warm else floor + (1 - floor) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, (q - warm) / max(1, total - warm)))))
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
    train_ds = Both(train_ids, meta_r, meta_s, labels, rr.CFG["train_windows"], True)
    for epoch in range(start, CFG["epochs"] + 1):
        model.train(); t0 = time.monotonic()
        run = np.zeros(4); n = 0
        opt.zero_grad(set_to_none=True)
        dl = loader(train_ds, True, CFG["studies_per_step"], CFG["seed"] * 100 + epoch)
        for i, batch in enumerate(dl):
            loss, parts = train_step(model, prep, batch, CFG["branch_drop_p"])
            (loss / CFG["accum"]).backward()
            if (i + 1) % CFG["accum"] == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += [float(loss), *map(float, parts)]; n += 1
            if (i + 1) % 400 == 0:
                print(f"  epoch {epoch} step {i + 1}/{len(dl)} loss {run[0] / n:.4f} "
                      f"{(time.monotonic() - t0) / (n * CFG['studies_per_step']):.3f}s/study", flush=True)
        train_s = time.monotonic() - t0
        record = {"epoch": epoch, "train_loss": run[0] / n, "train_fused": run[1] / n, "train_r50": run[2] / n,
                  "train_swin": run[3] / n, "val_macro_auc": None, "val_bce": None, "train_s": round(train_s, 1),
                  "lr": [g["lr"] for g in opt.param_groups][:1] + [opt.param_groups[-1]["lr"]]}
        if epoch in CFG["eval_epochs"]:
            p = predict(model, prep, val_ids, meta_r, meta_s)
            for head in ("fused", "r50", "swin"):
                sc, per = rr.macro_auc(p[head], val_ids, ref)
                record[f"val_{head}"] = sc; record[f"per_target_{head}"] = per
            record["val_macro_auc"] = record["val_fused"]; record["val_bce"] = s.val_bce(p["fused"], val_ids, ref)
            record["val_mean_of_branches"] = rr.macro_auc(0.5 * (p["r50"] + p["swin"]), val_ids, ref)[0]
            record["r50_share"] = dict(zip(rr.TARGETS, np.round(p["share"].mean(0), 3).tolist()))
            rr.save_predictions(OUT / f"val_epoch{epoch:02d}.json", val_ids, p["fused"], f"epoch {epoch}, fused, no TTA")
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
        scored = "val macro AUC - val BCE - " if v is None else (
            f"val macro AUC {v:.4f} val BCE {record['val_bce']:.4f} | r50 {record['val_r50']:.4f} swin "
            f"{record['val_swin']:.4f} mean-of-branches {record['val_mean_of_branches']:.4f} | r50 share "
            f"{json.dumps(record['r50_share'])} ")
        print(f"EPOCH {epoch}: loss {record['train_loss']:.4f} (fused {record['train_fused']:.4f} r50 "
              f"{record['train_r50']:.4f} swin {record['train_swin']:.4f}) {scored}({record['total_s']:.0f}s)", flush=True)
        kill = CFG["kill"]
        if epoch == kill["epoch"] and v is not None and v < kill["min_val"]:                # pre-registered stop
            (OUT / "run_summary.json").write_text(json.dumps({"cfg": CFG, "history": history, "killed":
                f"epoch {epoch} val {v:.4f} < {kill['min_val']}"}, indent=2) + "\n")
            print(f"RUN_SUMMARY killed at epoch {epoch}", flush=True)
            return

    # ---- primary: SWA of the last 5 epochs, BN refresh of the ResNet branch
    sd = [torch.load(OUT / f"weights_epoch{e:02d}.pt", map_location="cpu", weights_only=True) for e in CFG["swa_epochs"]]
    model.load_state_dict({k: (sum(q[k].float() for q in sd) / len(sd)).to(sd[-1][k].dtype)
                           if sd[-1][k].is_floating_point() else sd[-1][k] for k in sd[-1]})
    del sd
    refresh_bn(model, train_ids, meta_r, labels)
    tag = f"swa{CFG['swa_epochs'][0]}_{CFG['swa_epochs'][-1]}"
    torch.save(model.state_dict(), OUT / f"weights_{tag}.pt")
    results = {}
    for split, ids in (("val", val_ids), ("gold", gold_ids)):
        p = predict(model, prep, ids, meta_r, meta_s)
        for head, suffix in (("fused", ""), ("r50", "_r50"), ("swin", "_swin")):
            rr.save_predictions(OUT / f"{split}_swa{suffix}.json", ids, p[head], f"SWA {tag}, {head} head, 320, no TTA")
        if split == "val":
            for head in ("fused", "r50", "swin"):
                results[head] = rr.macro_auc(p[head], ids, ref)
            results["mean_of_branches"] = rr.macro_auc(0.5 * (p["r50"] + p["swin"]), ids, ref)
            results["r50_share"] = dict(zip(rr.TARGETS, np.round(p["share"].mean(0), 3).tolist()))
            np.save(OUT / "val_swa_share.npy", p["share"])
    summary = {"cfg": CFG, "history": history, "swa": results, "seconds": round(time.monotonic() - started, 1),
               "note": "Primary = fused head SWA (val_swa.json); score with scripts/audit_joint_j1.py."}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps({k: (v[0] if isinstance(v, tuple) else v) for k, v in results.items()}
                                      | {"seconds": summary["seconds"]}), flush=True)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        profile() if MODE == "profile" else main()
