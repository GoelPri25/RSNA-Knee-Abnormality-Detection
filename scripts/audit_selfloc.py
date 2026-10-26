"""Exp SL-0 scorer: r50-GL v3 SWA on its own regions (self-located) vs MS1's regions, fold 0, label-v2 confident cells
(weight > 0, y <= 0.3 / >= 0.7), 12-target mean AUC, paired study bootstrap (2,000).
  Outputs per region source: fused / global / local / halfgate (beta fixed 0.5 in logit space).
  DECISION (pre-registered 2026-10-18): self - MS1 >= -0.002 on the submitted output (fused, and halfgate) -> MS1 is
  dropped from inference with no retraining; otherwise a locator head trained on MS1's training-fold points (v3.1).
  Check: the MS1-region outputs reproduce run_r50_gl3/val_swa_*.json.
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_selfloc.py <project root> [selfloc dir, default run_r50_gl3/selfloc]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); C = R / "repository/data/coatnet_d96_fold0"
RUN = C / (sys.argv[2] if len(sys.argv) > 2 else "run_r50_gl3/selfloc"); OUT = RUN / "audit_selfloc.txt"; lines = []
say = lambda s="": (print(s, flush=True), lines.append(s))
lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text()); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}


def load(p):
    d = json.loads(p.read_text()); assert d["targets_in_order"] == T
    return {x["study_id"]: np.array(x["probabilities"]) for x in d["predictions"]}


P = {f"{src} {h}": load(RUN / f"val_{src}_{h}.json") for src in ("ms1", "self") for h in ("fused", "global", "local")}
P["lv2"] = load(C / "kaggle_fold0_labelv2/val_swa.json")
ids = sorted(P["lv2"]); assert all(set(p) == set(ids) for p in P.values())
X = {k: np.array([v[s] for s in ids]) for k, v in P.items()}
lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / np.clip(1 - q, 1e-6, 1))
for src in ("ms1", "self"):
    X[f"{src} halfgate"] = 1 / (1 + np.exp(-(lg(X[f"{src} global"]) + lg(X[f"{src} local"])) / 2))
ref = C / "run_r50_gl3/val_swa_fused.json"
if ref.is_file():
    q = load(ref); say(f"check: MS1-region fused vs run val_swa_fused max |dp| {max(np.abs(q[s] - P['ms1 fused'][s]).max() for s in ids):.2e}")
Y = np.array([lab[s][0] for s in ids]); W = np.array([lab[s][1] for s in ids])
M = (W > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(x, idx=None):
    idx = np.arange(len(ids)) if idx is None else idx
    return np.array([auc(x[idx, k][M[idx, k]], TT[idx, k][M[idx, k]]) for k in range(12)])


cols = ["lv2"] + [f"{s} {h}" for h in ("fused", "global", "local", "halfgate") for s in ("ms1", "self")]
tab = pd.DataFrame({k: per(X[k]) for k in cols}, index=T)
say(f"SL-0: r50-GL v3 SWA, MS1 vs self-located regions; fold 0 AUC, label-v2 confident cells ({len(ids)} studies)")
say(tab.round(4).to_string())
say("mean (12): " + ", ".join(f"{c} {tab[c].mean():.4f}" for c in cols))
rng = np.random.default_rng(2026); B = 2000
pairs = {f"self-ms1 {h}": (f"self {h}", f"ms1 {h}") for h in ("fused", "local", "halfgate")}
pairs.update({"self halfgate-lv2": ("self halfgate", "lv2"), "self fused-lv2": ("self fused", "lv2")})
need = sorted({a for p in pairs.values() for a in p}); bt = {k: [] for k in pairs}
for _ in range(B):
    i = rng.integers(0, len(ids), len(ids)); a = {k: np.nanmean(per(X[k], i)) for k in need}
    for k, (u, v) in pairs.items():
        bt[k].append(a[u] - a[v])
say("")
for k, (u, v) in pairs.items():
    lo, hi = np.percentile(bt[k], [2.5, 97.5])
    say(f"{k:18s}: {tab[u].mean() - tab[v].mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}], {np.mean(np.array(bt[k]) > 0):.0%} > 0")
ok = all(tab[f"self {h}"].mean() - tab[f"ms1 {h}"].mean() >= -.002 for h in ("fused", "halfgate"))
say("")
say(f"DECISION (self - MS1 >= -0.002 on fused and halfgate): {'PASS -> drop MS1 from inference' if ok else 'FAIL -> locator head (v3.1)'}")
log = RUN / "selfloc_log.json"
if log.is_file():
    d = json.loads(log.read_text())
    say(f"\nlocate {d['locate_s_per_study']} s/study, predict {d.get('predict_self_s_per_study')} s/study (H100)")
    say("agreement with MS1 regions: " + json.dumps(d["agreement_vs_ms1"]))
    say("Baker's unrestricted top-1 slot share (0/1 sagittal, 2/3 coronal, 4 axial): " + json.dumps(d["bakers_unrestricted_top1_slot_share"]))
OUT.write_text("\n".join(lines) + "\n")
