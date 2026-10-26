"""Exp r50-GL v3: one ResNet-50 that reads the whole study (global windows) and per-finding zoomed regions (local), with
each finding's answer a learned per-study mix of its own two views. Built from the r50-GL v2 diagnosis (XAI, MP-0).

What v2 got wrong and the v3 change (ledger r50-GL, r50-GL-XAI, MP-0):
  - The reader did not route: own-crop attention 0.03 = chance, 44% of its input attribution on OTHER findings' crops.
    -> no reader. Finding f reads only the global windows and its own regions (hard-wired).
  - The reader captured +0.0009 of the +0.0034 in-sample global / local complementarity (the gain was learnable).
    -> fused_f = beta_f g_f + (1 - beta_f) l_f; beta_f = sigmoid(gate([z_g,f ; z_l,f])), a per-study, per-finding gate
       on the two pooled features (MLP 4096 -> 128 -> 12, dropout 0.2), last layer small random (trunc normal 0.02,
       bias 0; v2's fix: zero-started paths on r50 stay dormant, v1 / Exp 2a) so training starts at beta ~ 0.5, the
       point v2 showed complementary (mean(g, l) - g +0.0047 fold 0, +0.0034 training), with gradient into the whole
       gate from step 1. No detach: the fused loss trains both views and the gate.
  - The local head trained on 1 crop per finding and was evaluated on 4 (top-1 alone 0.908 vs 4-crop 0.929); it
    stopped improving at epoch ~11 (0.928) and over-fitted more than the global (train - fold 0 gap 0.043 vs 0.034).
    -> 4 regions per finding in training AND evaluation (MS1's 4 windows / points, jitter +-16 px in training),
       pooled by a learned per-finding attention over the 4 (1e's head shape).
  - Greedy learning (one view solves the loss, the other starves): each view keeps its own full-weight BCE.
    Loss = BCE(fused) + BCE(global) + BCE(local, region findings). With the shared gate these three readouts are
    the "both / global only / regions only" states, so no hiding rate has to be set.
  - Regions only where a region is local and placed right: ACL, MCL, medial / lateral meniscus, medial / lateral OA,
    Baker's, contusion. None for effusion, synovitis (diffuse), PF OA (boxes on the tibiofemoral joint), fracture
    (crop head ~0.02 with the fracture in the box; extra inputs hurt it in MP-0). Those 4 findings: fused = global.
  - Baker's regions in the sagittal series: MS1 put 87% of its Baker's points on axial; a zoom probe chosen by CV in
    the training folds reads Baker's at 0.889 sagittal vs 0.851 axial. Each of MS1's 4 Baker's points is mapped
    through DICOM geometry into the sagittal series (scripts/multiplane_points.py, points4; window kept inside the
    series). All other findings keep MS1's plane (best or tied in the same CV).
Kept from v2: label-v2 r50 recipe (D96_PHASE=r50_gl: resnet50.tv_in1k ImageNet start, seed 2026, 25 epochs cosine,
1% floor, warm-up 1, LR 1e-4 / 1e-3, wd 0.05, SWA 21-25, 2 studies x 4 accumulation, 16 / 94 windows, 320 centre crop
in evaluation, BN refresh on 256 training studies), r50-GL boxes (128 px medial meniscus / contusion, 192 px others ->
384), one shared augmentation draw per study for both views, dual BatchNorm statistics (global / crop, shared affine).
Logged every epoch: fold-0 AUC + val BCE (label-v2 confident cells, Steven validation labels) for fused / global /
local, mean gate beta per finding.

Env: D96_CACHE, D96_PHASE=r50_gl, GL_CROPS (crop_probe dir), GL_POINTS4 (multiplane_points.npz), GL_OUT, GL_SMOKE=1.
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

import r50_gl_runner as g

r, CFG = g.r, g.CFG
OUT, SMOKE = g.OUT, g.SMOKE
IMG, MAXS, K = g.IMG, g.MAXS, g.K
R8 = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "Baker's", "Contusion"]
RF = np.array([r.TARGETS.index(t) for t in R8]); NR = 4
RMASK = np.isin(np.arange(12), RF)
V3 = {"version": "r50_gl_v3", "regions": R8, "boxes": {t: int(g.BOX[t]) for t in R8}, "regions_per_finding": NR,
      "region_source": "MS1 crop table (4 windows / points); Baker's mapped to sagittal (points4)", "jitter_px": 16,
      "region_pool": "per-finding attention over its 4 regions (Linear 2048-256, tanh, dropout 0.2, Linear 256-12)",
      "gate": {"input": "[z_global_f ; z_local_f]", "hidden": 128, "dropout": 0.2,
               "init": "last layer trunc normal 0.02, bias 0 (beta ~ 0.5)"},
      "loss": "BCE(fused) + BCE(global) + BCE(local, 8 region findings)", "bn": "dual", "smoke_steps": 32}
HEADS = ("fused", "global", "local")


def region_table(meta):
    """study -> (slice [8, 4], row centre [8, 4], col centre [8, 4]) for the 8 region findings, finding-major"""
    tab = g.crop_table(meta)
    z = np.load(os.environ["GL_POINTS4"])
    kb = r.TARGETS.index("Baker's"); out = {}
    for s, p in zip(z["ids"], z["points4"]):
        s = str(s); sl, rr, cc, _ = tab[s]
        sl, rr, cc = sl.astype(np.int64), rr.astype(np.float32), cc.astype(np.float32)
        q = p[kb, :, 0]                                                   # each of the 4 Baker's points in sagittal
        ok = q[:, 0] >= 0
        sl[kb, ok] = np.clip(q[ok, 0], q[ok, 6] + 1, q[ok, 7] - 1).astype(np.int64)
        rr[kb, ok], cc[kb, ok] = q[ok, 1], q[ok, 2]
        out[s] = (sl[RF], np.round(rr[RF]).astype(np.int64), np.round(cc[RF]).astype(np.int64))
    return out


class V3Studies(r.Studies):
    """r.Studies + the 32 regions (8 findings x 4): full slices of the region windows [32, 3, IMG, IMG], their sagittal
    flag, centre (row, col) px, finding index"""

    def __init__(self, ids, meta, labels, k, train, tab, offset=None):
        super().__init__(ids, meta, labels, k, train, offset); self.tab = tab

    def __getitem__(self, i):
        x, sag, y, w, ctx = super().__getitem__(i)
        s = self.ids[i]; m = self.meta[s]
        vol = self._mm[m["file"]][m["row"]:m["row"] + MAXS]
        S, R, Q = (a.ravel() for a in self.tab[s])
        if self.train:
            jit = np.random.randint(-V3["jitter_px"], V3["jitter_px"] + 1, size=(2, len(S)))
            R, Q = R + jit[0], Q + jit[1]
        fid = np.repeat(RF, NR); h = g.BOXN[fid] // 2
        R, Q = np.clip(R, h, IMG - h), np.clip(Q, h, IMG - h)
        xc = np.ascontiguousarray(vol[np.stack([S - 1, S, S + 1], 1)])
        sagc = np.isin(r.SLOT_OF[S], r.SAGITTAL_SLOTS)
        cen = np.stack([R, Q], 1).astype(np.float32)
        return (x, sag, y, w, ctx, torch.from_numpy(xc), torch.from_numpy(sagc), torch.from_numpy(cen),
                torch.from_numpy(fid.astype(np.int64)))


class RaptorGL3(r.Raptor):
    def __init__(self, res, pretrained):
        super().__init__(res, pretrained)                    # backbone, then 1e's head (same init order as 1e)
        d = self.backbone.num_features
        self.bns = g.dual_bn(self.backbone)
        assert len(self.bns) == 53, len(self.bns)
        self.norm_l = nn.LayerNorm(d)
        self.att_l = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.clsV = nn.Parameter(torch.zeros(12, d)); self.clsvb = nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.clsV, std=0.02)
        gt = V3["gate"]
        self.gate = nn.Sequential(nn.Dropout(gt["dropout"]), nn.Linear(2 * d, gt["hidden"]), nn.GELU(),
                                  nn.Dropout(gt["dropout"]), nn.Linear(gt["hidden"], 12))
        nn.init.trunc_normal_(self.gate[-1].weight, std=0.02); nn.init.zeros_(self.gate[-1].bias)
        self.register_buffer("rf", torch.as_tensor(RF), persistent=False)

    def encode(self, x, crop):
        for m in self.bns:
            m.crop = crop
        f = self.backbone(x)
        for m in self.bns:
            m.crop = False
        return f

    def forward(self, x, b, xc):
        """-> fused, global, local logits [b, 12] (local = global where a finding has no regions), gate beta [b, 8]"""
        hg = self.norm(self.encode(x, False).view(b, x.shape[0] // b, -1))
        a = torch.softmax(self.att(hg).float(), dim=1).to(hg.dtype)                        # 1e's head
        zg = torch.einsum("bkn,bkf->bnf", a, hg)                                          # [b, 12, d]
        gz = ((zg * self.clsW).sum(-1) + self.clsb).float()
        hc = self.norm_l(self.encode(xc, True).view(b, len(RF), NR, -1))                  # [b, 8, 4, d]
        rf = self.rf
        ac = self.att_l(hc).float().gather(-1, rf.view(1, -1, 1, 1).expand(b, -1, NR, 1))[..., 0]   # own finding
        ac = torch.softmax(ac, dim=2).to(hc.dtype)                                         # over its 4 regions
        zl = (ac[..., None] * hc).sum(2)                                                   # [b, 8, d]
        lr = ((zl * self.clsV[rf]).sum(-1) + self.clsvb[rf]).float()                      # [b, 8]
        beta = torch.sigmoid(self.gate(torch.cat([zg[:, rf], zl], -1)).float()
                             .gather(-1, rf.view(1, -1, 1).expand(b, -1, 1))[..., 0])     # [b, 8]
        lz = gz.clone(); lz[:, rf] = lr
        fz = gz.clone(); fz[:, rf] = beta * gz[:, rf] + (1 - beta) * lr
        return fz, gz, lz, beta


@torch.no_grad()
def predict(model, ids, meta, tab, dev, k=K):
    """-> sigmoid [3, n, 12] (fused, global, local), gate beta [n, 8]"""
    model.eval()
    dl = torch.utils.data.DataLoader(V3Studies(ids, meta, {}, k, False, tab), batch_size=1,
                                     num_workers=CFG["workers"], pin_memory=True)
    out = np.zeros((3, len(ids), 12), np.float32); beta = np.zeros((len(ids), len(RF)), np.float32)
    for i, (x, sag, _, _, _, xc, sagc, cen, fid) in enumerate(dl):
        xg, xcr, _ = g.prepare_gl(x, sag, xc, sagc, cen, fid, False, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fz, gz, lz, bt = model(xg, 1, xcr)
        out[:, i] = torch.stack([fz, gz, lz]).float().sigmoid()[:, 0].cpu().numpy(); beta[i] = bt[0].float().cpu().numpy()
    return out, beta


def score(p, beta, ids, ref, labels):
    rec = {}
    for j, nm in enumerate(HEADS):
        rec[f"lv2_{nm}"], per = r.macro_auc(p[j], ids, labels)
        rec[f"val_{nm}"] = r.macro_auc(p[j], ids, ref)[0]
        rec[f"lv2_bce_{nm}"] = r.val_bce(p[j], ids, labels); rec[f"val_bce_{nm}"] = r.val_bce(p[j], ids, ref)
        rec[f"lv2_per_{nm}"] = per
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    pm = p[1].copy(); pm[:, RF] = 1 / (1 + np.exp(-(lg(p[1][:, RF]) + lg(p[2][:, RF])) / 2))
    rec["lv2_halfgate"] = r.macro_auc(pm, ids, labels)[0]                     # descriptive: beta fixed at 0.5
    rec["gate_beta"] = dict(zip(R8, beta.mean(0).round(3).tolist()))
    rec["gate_beta_sd"] = dict(zip(R8, beta.std(0).round(3).tolist()))
    return rec


@torch.no_grad()
def bn_refresh(model, ids, meta, labels, tab, dev):
    """1e's refresh for the global statistics (16 eval-centre windows at 384, batch 2); crop statistics from the same
    studies' 32 evaluation regions."""
    for m in model.bns:
        m.reset_running_stats(); m.momentum = None
    model.train()
    ds = V3Studies(ids, meta, labels, CFG["train_windows"], False, tab)
    for x, sag, _, _, _, xc, sagc, cen, fid in torch.utils.data.DataLoader(ds, batch_size=2, num_workers=CFG["workers"]):
        xg, xcr, _ = g.prepare_gl(x, sag, xc, sagc, cen, fid, False, dev, gres=CFG["train_res"])
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
    tab = region_table(meta)
    assert set(tab) >= set(train_ids) | set(val_ids), "region table must cover every labelled study"
    print(f"train {len(train_ids)}, validation {len(val_ids)}; " + json.dumps(V3["boxes"]), flush=True)

    model = RaptorGL3(CFG["train_res"], pretrained=True).to(dev).to(memory_format=torch.channels_last)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": CFG["lr"]},
                             {"params": head, "lr": CFG["head_lr"]}], weight_decay=CFG["wd"])
    spe = math.ceil(len(train_ids) / (CFG["studies_per_step"] * CFG["accum"]))
    total, warm, floor = spe * CFG["epochs"], spe * CFG["warmup_epochs"], CFG["lr_floor"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else floor + (1 - floor) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))

    vids = val_ids[:16] if SMOKE else val_ids
    if SMOKE:                                            # small-init gate: beta ~ 0.5 before training
        p0, b0 = predict(model, vids[:2], meta, tab, dev)
        lg = lambda q: np.log(q / (1 - q))
        print(f"INIT_CHECK beta {b0.min():.4f}-{b0.max():.4f}; max|logit fused - mean(global, local)| "
              f"{np.abs(lg(p0[0]) - (lg(p0[1]) + lg(p0[2])) / 2).max():.2e}; no-region findings fused = global "
              f"{np.abs(p0[0][:, ~RMASK] - p0[1][:, ~RMASK]).max():.2e}", flush=True)
    start, history, rec = 1, [], OUT / "recovery.pt"
    if rec.is_file():
        st = torch.load(rec, map_location="cpu", weights_only=False)
        assert st["v3"] == V3 and st["cfg"] == CFG, "recovery config differs"
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, history = st["epoch"] + 1, st["history"]
        print(f"RESUMED after epoch {st['epoch']}", flush=True)
    train_ds = V3Studies(train_ids, meta, labels, CFG["train_windows"], True, tab)
    rmask = torch.from_numpy(RMASK.astype(np.float32)).to(dev)
    last = 1 if SMOKE else CFG["epochs"]
    for epoch in range(start, last + 1):
        model.train(); t0 = time.monotonic()
        dl = torch.utils.data.DataLoader(train_ds, batch_size=CFG["studies_per_step"], shuffle=True,
                                         num_workers=CFG["workers"], pin_memory=True,
                                         worker_init_fn=lambda w: np.random.seed((CFG["seed"] * 1000 + epoch * 100 + w) % 2**32))
        run = np.zeros(4); n = 0; bsum = np.zeros(len(RF))
        opt.zero_grad(set_to_none=True)
        for i, (x, sag, y, w, ctx, xc, sagc, cen, fid) in enumerate(dl):
            b = x.shape[0]
            xg, xcr, _ = g.prepare_gl(x, sag, xc, sagc, cen, fid, True, dev)
            y, w = g.to_dev(dev, y, w)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fz, gz, lz, bt = model(xg, b, xcr)
            parts = torch.stack([g.bce(fz, y, w), g.bce(gz, y, w), g.bce(lz, y, w * rmask)])
            loss = parts.sum()
            (loss / CFG["accum"]).backward()
            if (i + 1) % CFG["accum"] == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += np.append(parts.detach().cpu().numpy(), loss.item()) * b; n += b
            bsum += bt.detach().float().sum(0).cpu().numpy()
            if (i + 1) % 400 == 0 or (SMOKE and i + 1 in (8, V3["smoke_steps"])):
                print(f"  epoch {epoch} step {i + 1}/{len(dl)} loss fused/global/local {run[0] / n:.4f}/{run[1] / n:.4f}/"
                      f"{run[2] / n:.4f} {(time.monotonic() - t0) / n:.3f}s/study "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
            if SMOKE and i + 1 == V3["smoke_steps"]:
                break
        train_s = time.monotonic() - t0
        rec_ = {"epoch": epoch, "train_loss": dict(zip(("fused", "global", "local", "total"), (run / n).round(5).tolist())),
                "train_s": round(train_s, 1), "train_s_per_study": round(train_s / n, 4), "lr": [q["lr"] for q in opt.param_groups],
                "train_gate_beta": dict(zip(R8, (bsum / n).round(3).tolist()))}
        te = time.monotonic(); p, beta = predict(model, vids, meta, tab, dev)
        rec_["eval_s_per_study"] = round((time.monotonic() - te) / len(vids), 3)
        rec_["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        if not SMOKE:
            rec_.update(score(p, beta, val_ids, ref, labels))
            for j, nm in enumerate(HEADS):
                r.save_predictions(OUT / f"val_epoch{epoch:02d}_{nm}.json", val_ids, p[j], f"r50-GL v3 epoch {epoch}, {nm}")
        history.append(rec_)
        print(f"EPOCH {epoch} " + json.dumps({k: v for k, v in rec_.items() if not k.startswith("lv2_per")}), flush=True)
        if epoch in CFG["swa_epochs"] or SMOKE:
            torch.save(model.state_dict(), OUT / f"weights_epoch{epoch:02d}.pt")
        if not SMOKE:
            torch.save({"v3": V3, "cfg": CFG, "model": model.state_dict(), "opt": opt.state_dict(),
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
    p, beta = predict(model, vids, meta, tab, dev)
    res = score(p, beta, vids, ref, labels)
    for j, nm in enumerate(HEADS):
        r.save_predictions(OUT / f"val_swa_{nm}.json", vids, p[j], f"r50-GL v3 SWA {swa}, {nm}")
    np.save(OUT / "val_swa_gate_beta.npy", beta)
    summary = {"v3": V3, "cfg": CFG, "history": history, "swa": res, "seconds": round(time.monotonic() - started, 1)}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps({k: v for k, v in res.items() if not k.startswith("lv2_per")}), flush=True)
    if SMOKE:
        print(f"SMOKE_DONE {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
