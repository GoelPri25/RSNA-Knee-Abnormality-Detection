"""Exp SL-0v4 scorer (ledger row SL-0v4, pre-registered 2026-10-18). Fold 0, label-v2 confident cells, 12-target mean AUC,
2,000 paired study bootstraps.
  CHECK:     MS1-crop predictions reproduce the run's own val_swa_fused.json (max |dp| small; same weights, same table).
  DECISION:  fused(self) - fused(MS1) >= -0.002 (point) -> MS1-free Kaggle inference with self-located crops;
             otherwise the Kaggle notebook must run MS1 for crop placement.
  Descriptive: global (must be identical), local (10 cropped findings), per finding, bootstrap CI, agreement with MS1.
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_selfloc_v4.py <project root> <run name> [<run name> ...]
  run name = a dir under data/coatnet_d96_fold0/run_r50_gl4_mlx holding val_swa_*.json and selfloc/ (Modal output).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); C = R / "repository/data/coatnet_d96_fold0"; BASE = C / "run_r50_gl4_mlx"
lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text()); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
CROPPED = [k for k, t in enumerate(T) if t not in ("Effusion", "Synovitis")]


def load(p):
    d = json.loads(p.read_text()); assert d["targets_in_order"] == T
    return {x["study_id"]: np.array(x["probabilities"]) for x in d["predictions"]}


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


for run in sys.argv[2:]:
    d = BASE / run; sl = d / "selfloc"; lines = []
    say = lambda s="": (print(s, flush=True), lines.append(s))
    P = {f"{nm}_{h}": load(sl / f"val_{nm}_{h}.json") for nm in ("ms1", "self") for h in ("fused", "global", "local")}
    P["run_fused"] = load(d / "val_swa_fused.json")
    ids = sorted(P["run_fused"]); assert all(set(p) == set(ids) for p in P.values())
    X = {k: np.array([v[s] for s in ids]) for k, v in P.items()}
    Y = np.array([lab[s][0] for s in ids]); W = np.array([lab[s][1] for s in ids])
    M = (W > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7
    per = lambda x, i=None: np.array([auc(x[i if i is not None else slice(None), k][M[i if i is not None else slice(None), k]],
                                          TT[i if i is not None else slice(None), k][M[i if i is not None else slice(None), k]])
                                      for k in range(12)])
    log = json.loads((sl / "selfloc_log.json").read_text())
    say(f"SL-0v4 {run}: fold 0, label-v2 confident cells ({len(ids)} studies)")
    say(f"CHECK MS1-crop fused vs run val_swa_fused: max |dp| {np.abs(X['ms1_fused'] - X['run_fused']).max():.4f}, "
        f"AUC {per(X['ms1_fused']).mean():.4f} vs {per(X['run_fused']).mean():.4f}")
    say(f"global self vs MS1 max |dp| {np.abs(X['self_global'] - X['ms1_global']).max():.2e} (must be ~0)")
    tab = pd.DataFrame({k: per(X[k]) for k in ("ms1_fused", "self_fused", "ms1_local", "self_local", "ms1_global")}, index=T)
    tab["fused self-ms1"] = tab["self_fused"] - tab["ms1_fused"]
    tab["local self-ms1"] = tab["self_local"] - tab["ms1_local"]
    say(tab.round(4).to_string())
    say(f"mean: fused MS1 {tab['ms1_fused'].mean():.4f}, self {tab['self_fused'].mean():.4f}, "
        f"self - MS1 {tab['fused self-ms1'].mean():+.4f}; local10 MS1 {tab['ms1_local'].iloc[CROPPED].mean():.4f}, "
        f"self {tab['self_local'].iloc[CROPPED].mean():.4f}; global {tab['ms1_global'].mean():.4f}")
    rng = np.random.default_rng(2026); bt = []
    for _ in range(2000):
        i = rng.integers(0, len(ids), len(ids))
        bt.append(np.nanmean(per(X["self_fused"], i)) - np.nanmean(per(X["ms1_fused"], i)))
    bt = np.array(bt)
    say(f"bootstrap self - MS1 (fused): mean {bt.mean():+.4f}, 95% CI [{np.percentile(bt, 2.5):+.4f}, "
        f"{np.percentile(bt, 97.5):+.4f}], {np.mean(bt >= -0.002):.0%} of resamples >= -0.002")
    ok = tab["fused self-ms1"].mean() >= -0.002
    say(f"DECISION (self - MS1 >= -0.002): {'PASS -> MS1-free inference' if ok else 'FAIL -> Kaggle notebook needs MS1'}")
    say(f"speed (H100): locate {log['locate_s_per_study']} s/study, predict self {log['predict_self_s_per_study']} s/study")
    say("agreement with MS1: " + json.dumps(log["agreement_vs_ms1"]))
    (sl / "audit_selfloc_v4.txt").write_text("\n".join(lines) + "\n")
    say()
