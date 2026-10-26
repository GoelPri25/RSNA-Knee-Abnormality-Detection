"""Exp r50-GL-XAI scorer (ledger, pre-registered 2026-10-18). Reads run_r50_gl_v2/xai/xai_{val,train}.npz from
scripts/xai_r50_gl_gpu.py. Judge: label-v2 confident cells (weight > 0, y <= 0.3 / >= 0.7), 12-target mean AUC (10
cropped findings where stated), paired study bootstrap (2,000).
Q1 in-sample: training mean(g, l) - g vs fold 0.   Q2 routing: fused_route - fused.   Q3 reader use: fused_shuffle -
fused, attribution / attention on own crops.   Q4 placement: local_shift32 / 64 - local per finding, local CAM outer-ring
share and offset.   Q5 global view: parent384 - parent, g384 - global.   Q6: CAM box shares, GL vs parent.
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_r50_gl_xai.py <project root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); C = R / "repository/data/coatnet_d96_fold0"; RUN = C / "run_r50_gl_v2"; X = RUN / "xai"
lines = []
say = lambda s="": (print(s, flush=True), lines.append(s))
lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text()); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
CR = [k for k, t in enumerate(T) if t not in ("Effusion", "Synovitis")]
rng = np.random.default_rng(2026); B = 2000


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


class Split:
    def __init__(self, name):
        z = np.load(X / f"xai_{name}.npz")
        self.z = {k: z[k] for k in z.files}; self.ids = [str(s) for s in self.z["ids"]]
        assert [str(t) for t in self.z["targets"]] == T
        Y = np.array([lab[s][0] for s in self.ids]); W = np.array([lab[s][1] for s in self.ids])
        self.M = (W > 0) & ((Y <= .3) | (Y >= .7)); self.Y = Y >= .7
        self.z["mean_gl"] = (self.z["global"] + self.z["local"]) / 2
        self.boots = [rng.integers(0, len(self.ids), len(self.ids)) for _ in range(B)]

    def per(self, k, idx=None):
        x = self.z[k]; idx = np.arange(len(self.ids)) if idx is None else idx
        ok = ~np.isnan(x[idx]).any(1)
        idx = idx[ok]
        return np.array([auc(x[idx, j][self.M[idx, j]], self.Y[idx, j][self.M[idx, j]]) for j in range(12)])

    def diff(self, a, b, cols=None):
        cols = list(range(12)) if cols is None else cols
        d0 = np.nanmean(self.per(a)[cols]) - np.nanmean(self.per(b)[cols])
        bt = np.array([np.nanmean(self.per(a, i)[cols]) - np.nanmean(self.per(b, i)[cols]) for i in self.boots])
        lo, hi = np.percentile(bt, [2.5, 97.5])
        return d0, lo, hi, float(np.mean(bt > 0))


fmt = lambda d: f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}], {d[3]:.0%} > 0"
V = Split("val"); TR = Split("train")

say("r50-GL-XAI (v2 SWA) - fold 0 (825) and 825 training studies; label-v2 confident cells")
say("\nSanity: reproduction vs the run's val_swa files (per-target |dAUC|)")
for k, f in (("fused", "fused"), ("global", "global"), ("local", "local"), ("nocrop", "fused_nocrop")):
    d = json.loads((RUN / f"val_swa_{f}.json").read_text()); P = {x["study_id"]: np.array(x["probabilities"]) for x in d["predictions"]}
    ref = np.array([P[s] for s in V.ids]); V.z[f"_ref_{k}"] = ref
    if k == "local":
        V.z[f"_ref_{k}"][:, [7, 8]] = V.z["global"][:, [7, 8]]
    dm = np.nanmax(np.abs(V.per(k) - V.per(f"_ref_{k}")))
    say(f"  {k:7s} max |dAUC| {dm:.4f} {'OK' if dm <= .002 else 'CHECK'}; max |dp| {np.nanmax(np.abs(V.z[k] - V.z[f'_ref_{k}'])):.4f}")
say(f"  parent (label-v2 r50) 12-target {np.nanmean(V.per('parent')):.4f} (val_swa 0.9425)")

say("\nAUC per finding (fold 0)")
keys = ["parent", "parent384", "global", "g384", "local", "local_uniform", "local_top1", "local_shift32", "local_shift64",
        "mean_gl", "fused", "nocrop", "fused_route", "fused_shuffle", "fused_shift32", "fused_shift64"]
tab = pd.DataFrame({k: V.per(k) for k in keys}, index=T)
say(tab.T.assign(mean12=tab.mean(), mean10=tab.iloc[CR].mean()).round(3).to_string())

say("\nQ1 in-sample: mean(global, local) - global")
for nm, S in (("fold 0", V), ("training", TR)):
    pg, pl, pm = S.per("global"), S.per("local"), S.per("mean_gl")
    say(f"  {nm:8s}: global {np.nanmean(pg):.4f} local {np.nanmean(pl):.4f} mean {np.nanmean(pm):.4f}; mean - global {fmt(S.diff('mean_gl', 'global'))}")
    say("            per finding (mean - global): " + ", ".join(f"{T[k]} {pm[k] - pg[k]:+.3f}" for k in CR))
    say("            per finding (local - global): " + ", ".join(f"{T[k]} {pl[k] - pg[k]:+.3f}" for k in CR))
    say(f"            fused - global {fmt(S.diff('fused', 'global'))}")

say("\nQ2 routing (untrained): fused_route - fused " + fmt(V.diff("fused_route", "fused")))
say("Q3 reader use: fused_shuffle - fused (study 0 dropped) " + fmt(V.diff("fused_shuffle", "fused")))
for nm, S in (("fold 0", V), ("training", TR)):
    a, m = np.nanmean(S.z["attr"][:, CR], 0), np.nanmean(S.z["att"][:, CR], 0)
    say(f"  {nm:8s} attribution global / own / other (10 cropped): {a.mean(0).round(3).tolist()}; attention {m.mean(0).round(3).tolist()}")
    if nm == "fold 0":
        say("    own-crop attribution per finding: " + ", ".join(f"{T[k]} {np.nanmean(S.z['attr'][:, k, 1]):.3f}" for k in CR))
        say("    own-crop attention per finding:   " + ", ".join(f"{T[k]} {np.nanmean(S.z['att'][:, k, 1]):.3f}" for k in CR))

say(f"\nQ4 placement: crops moved {V.z['shift_px'].mean(0).round(1).tolist()} px (after clipping)")
say("  local_shift - local, local CAM outer-ring share (area 0.556) and centroid offset (px of 384), per finding")
for k in CR:
    ring, off = np.nanmean(V.z["cam_local"][:, k, 0]), np.nanmean(V.z["cam_local"][:, k, 1])
    say(f"  {T[k]:17s} shift32 {tab.at[T[k], 'local_shift32'] - tab.at[T[k], 'local']:+.3f}  shift64 "
        f"{tab.at[T[k], 'local_shift64'] - tab.at[T[k], 'local']:+.3f}  ring {ring:.3f}  offset {off:5.1f}px  (local - global "
        f"{tab.at[T[k], 'local'] - tab.at[T[k], 'global']:+.3f})")
say("  10 cropped: local_shift32 - local " + fmt(V.diff("local_shift32", "local", CR)) + "; shift64 " + fmt(V.diff("local_shift64", "local", CR)))
say("  local_uniform - local " + fmt(V.diff("local_uniform", "local", CR)) + "; local_top1 - local " + fmt(V.diff("local_top1", "local", CR)))

say("\nQ5 global view, whole 384 px window vs 320 centre crop")
say("  parent384 - parent " + fmt(V.diff("parent384", "parent")))
say("  g384 - global      " + fmt(V.diff("g384", "global")))
say("  per finding parent384 - parent: " + ", ".join(f"{t} {tab.at[t, 'parent384'] - tab.at[t, 'parent']:+.3f}" for t in T))

say("\nQ6 Grad-CAM of the global heads (fold 0, per finding; positives / negatives on confident cells)")
say("  columns: share of CAM on crop windows (chance), share inside crop boxes, box share within the crop windows (box area), top window is a crop window")
for k in CR:
    for nm in ("cam_gl", "cam_parent"):
        c = V.z[nm][:, k]; pos = V.M[:, k] & V.Y[:, k]; neg = V.M[:, k] & ~V.Y[:, k]
        f = lambda sel: (f"win {np.nanmean(c[sel, 0]):.2f} ({np.nanmean(c[sel, 3]):.2f}) box {np.nanmean(c[sel, 1]):.3f} "
                         f"in-win {np.nanmean(c[sel, 2]):.2f} ({np.nanmean(c[sel, 4]):.2f}) top {np.nanmean(c[sel, 5]):.2f}")
        say(f"  {T[k]:17s} {'GL    ' if nm == 'cam_gl' else 'parent'} pos: {f(pos)} | neg: {f(neg)}")
(X / "audit_xai.txt").write_text("\n".join(lines) + "\n")
