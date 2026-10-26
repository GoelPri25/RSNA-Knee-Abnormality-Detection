"""Exp L4C-D scorer (ledger row L4C-D, pre-registered 2026-10-18). Fold 0, label-v2 confident cells, 2,000 paired study
bootstraps against the ms1 condition (MS1's table = the training-time crop placement).
  CHECK      ms1 fused reproduces the run's val_swa_fused.json (0.9475).
  DECISION 1 self_plane - ms1 (fused, 12-target) >= -0.002 -> resubmit l4copy with plane-restricted self-location.
  DECISION 2 placement sensitivity = ms1 - shift10 on local10 (the 10 cropped findings): >= 0.005 -> jitter-trained arm;
             < 0.002 -> drop it; in between -> inconclusive (shift20 and slice2 shown for the shape of the curve).
  DECISION 3 window choice (self - ms1win_selfpts) vs points (ms1win_selfpts - ms1), fused and local10.
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_l4copy_diag.py <project root> <kaggle output dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(sys.argv[1]) / "repository/data/coatnet_d96_fold0"; K = Path(sys.argv[2])
lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text()); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
CROPPED = [k for k, t in enumerate(T) if t not in ("Effusion", "Synovitis")]
CONDS = ["ms1", "self", "self_plane", "ms1win_selfpts", "shift5", "shift10", "shift20", "slice2"]


def load(p):
    d = json.loads(p.read_text()); assert d["targets_in_order"] == T
    return {x["study_id"]: np.array(x["probabilities"]) for x in d["predictions"]}


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


P = {"run": load(C / "run_r50_gl4_mlx/v4_l4copy_s2026/val_swa_fused.json"), "global": load(K / "val_global.json")}
for c in CONDS:
    for h in ("fused", "local"):
        P[f"{c}_{h}"] = load(K / f"val_{c}_{h}.json")
ids = sorted(P["run"]); assert all(set(p) == set(ids) for p in P.values())
X = {k: np.array([v[s] for s in ids]) for k, v in P.items()}
Y = np.array([lab[s][0] for s in ids]); W = np.array([lab[s][1] for s in ids])
M = (W > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7


def per(x, i):
    return np.array([auc(x[i, k][M[i, k]], TT[i, k][M[i, k]]) for k in range(12)])


lines = []
say = lambda s="": (print(s, flush=True), lines.append(s))
al = np.arange(len(ids))
A = {k: per(x, al) for k, x in X.items()}
log = json.loads((K / "diag_log.json").read_text())
say(f"L4C-D: v4 l4copy s2026, fold 0, label-v2 confident cells ({len(ids)} studies); {log['device']}, {log['seconds']} s; "
    f"checks {json.dumps(log['checks'])}")
say(f"CHECK ms1 fused {A['ms1_fused'].mean():.4f} vs run val_swa_fused {A['run'].mean():.4f}; "
    f"max |dp| {np.abs(X['ms1_fused'] - X['run']).max():.4f}")
tf = pd.DataFrame({c: A[f"{c}_fused"] for c in CONDS}, index=T); tf.insert(0, "global", A["global"])
tl = pd.DataFrame({c: A[f"{c}_local"] for c in CONDS}, index=T).iloc[CROPPED]
say("\nFused AUC per finding"); say(tf.round(4).to_string())
say("mean " + "  ".join(f"{c} {tf[c].mean():.4f}" for c in tf.columns))
say("\nLocal AUC per cropped finding"); say(tl.round(4).to_string())
say("local10 " + "  ".join(f"{c} {tl[c].mean():.4f}" for c in tl.columns))

rng = np.random.default_rng(2026); B = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
bt = {k: np.array([per(X[k], i) for i in B]) for k in [f"{c}_{h}" for c in CONDS for h in ("fused", "local")]}
res = {}
say("\nPaired vs ms1 (fused 12-target | local10)")
for c in CONDS[1:]:
    df = np.nanmean(bt[f"{c}_fused"], 1) - np.nanmean(bt["ms1_fused"], 1)
    dl = np.nanmean(bt[f"{c}_local"][:, CROPPED], 1) - np.nanmean(bt["ms1_local"][:, CROPPED], 1)
    res[c] = (tf[c].mean() - tf["ms1"].mean(), tl[c].mean() - tl["ms1"].mean())
    say(f"  {c:15s} fused {res[c][0]:+.4f} [{np.percentile(df, 2.5):+.4f}, {np.percentile(df, 97.5):+.4f}] "
        f"{np.mean(df >= -0.002):.0%} >= -0.002 | local10 {res[c][1]:+.4f} [{np.percentile(dl, 2.5):+.4f}, {np.percentile(dl, 97.5):+.4f}]")
say("\nDECISION 1 (self_plane - ms1 fused >= -0.002): "
    + ("PASS -> resubmit with plane-restricted self-location" if res["self_plane"][0] >= -0.002 else "FAIL")
    + f"  [self_plane {res['self_plane'][0]:+.4f}; self {res['self'][0]:+.4f}]")
drop = -res["shift10"][1]
say(f"DECISION 2 (placement sensitivity, local10 drop at 10 mm = {drop:+.4f}; 5 mm {-res['shift5'][1]:+.4f}, "
    f"20 mm {-res['shift20'][1]:+.4f}, slice +-2 {-res['slice2'][1]:+.4f}): "
    + ("SENSITIVE -> jitter-trained arm" if drop >= 0.005 else "INSENSITIVE -> drop the jitter arm" if drop < 0.002 else "INCONCLUSIVE"))
say(f"DECISION 3 (self - ms1, fused {res['self'][0]:+.4f} / local10 {res['self'][1]:+.4f}) = points (ms1win_selfpts - ms1) "
    f"{res['ms1win_selfpts'][0]:+.4f} / {res['ms1win_selfpts'][1]:+.4f} + windows (self - ms1win_selfpts) "
    f"{res['self'][0] - res['ms1win_selfpts'][0]:+.4f} / {res['self'][1] - res['ms1win_selfpts'][1]:+.4f}")
say("crop pass s/study: " + json.dumps(log["crop_pass_s_per_study"]))
(K / "audit_l4copy_diag.txt").write_text("\n".join(lines) + "\n")
