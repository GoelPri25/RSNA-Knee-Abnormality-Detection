"""Exp MS1-GL scorer (ledger, pre-registered 2026-10-17). Fold 0, label-v2 confident cells (weight > 0, y <= 0.3 / >= 0.7).
  Primary:     fused SWA vs MS1 SWA (run_sig_lv2/val_swa.json): 12-target mean AUC gain >= +0.003, paired study
               bootstrap (2,000) 95% lower bound > 0 -> LB submission.
  Attribution: fused vs the same model's global branch alone: gain >= +0.002, lower bound > 0 -> the local view adds
               complementary signal. Also: local alone, global alone vs MS1 (continued training), the reader without crop tokens
               (fused_nocrop) and the 6 x 16-window pass average (fused_multi6), per finding (descriptive).
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_ms1_gl.py <project root> [run dir name, default run_ms1_gl] [tag, default swa]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
RUN = C / (sys.argv[2] if len(sys.argv) > 2 else "run_ms1_gl"); TAG = sys.argv[3] if len(sys.argv) > 3 else "swa"
OUT = RUN / f"audit_ms1_gl_{TAG}.txt"; lines = []
say = lambda s="": (print(s, flush=True), lines.append(s))
lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text()); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}


def load(p):
    d = json.loads(p.read_text()); assert d["targets_in_order"] == T
    return {x["study_id"]: np.array(x["probabilities"]) for x in d["predictions"]}


name = lambda nm: f"val_{TAG}_{nm}.json"
P = {"MS1": load(C / "run_sig_lv2/val_swa.json"), **{nm: load(RUN / name(nm)) for nm in ("fused", "global", "local")},
     **{nm: load(RUN / name(nm)) for nm in ("fused_nocrop", "fused_multi6") if (RUN / name(nm)).is_file()}}   # descriptive
ids = sorted(P["MS1"]); assert all(set(p) == set(ids) for p in P.values())
X = {k: np.array([v[s] for s in ids]) for k, v in P.items()}
Y = np.array([lab[s][0] for s in ids]); W = np.array([lab[s][1] for s in ids])
M = (W > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(x, idx=None):
    idx = np.arange(len(ids)) if idx is None else idx
    return np.array([auc(x[idx, k][M[idx, k]], TT[idx, k][M[idx, k]]) for k in range(12)])


tab = pd.DataFrame({k: per(v) for k, v in X.items()}, index=T)
tab["fused-MS1"] = tab["fused"] - tab["MS1"]; tab["fused-global"] = tab["fused"] - tab["global"]
say(f"MS1-GL {TAG}: fold 0 AUC per finding, label-v2 confident cells ({len(ids)} studies)")
say(tab.round(4).to_string()); say("mean: " + ", ".join(f"{c} {tab[c].mean():+.4f}" if "-" in c else f"{c} {tab[c].mean():.4f}" for c in tab))
rng = np.random.default_rng(2026); B = 2000
bt = {"fused-MS1": [], "fused-global": [], "global-MS1": []}
for _ in range(B):
    i = rng.integers(0, len(ids), len(ids)); a = {k: np.nanmean(per(v, i)) for k, v in X.items()}
    bt["fused-MS1"].append(a["fused"] - a["MS1"]); bt["fused-global"].append(a["fused"] - a["global"]); bt["global-MS1"].append(a["global"] - a["MS1"])
ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in bt.items()}
say("")
for k, v in ci.items():
    say(f"{k}: mean {np.mean(bt[k]):+.4f}, 95% CI [{v[0]:+.4f}, {v[1]:+.4f}]")
g1 = tab["fused-MS1"].mean() >= .003 and ci["fused-MS1"][0] > 0
g2 = tab["fused-global"].mean() >= .002 and ci["fused-global"][0] > 0
say(f"PRIMARY (fused vs MS1 >= +0.003, CI > 0): {'PASS -> LB submission' if g1 else 'FAIL'}")
say(f"ATTRIBUTION (fused vs own global >= +0.002, CI > 0): {'PASS: local view adds complementary signal' if g2 else 'FAIL: no reliable local contribution'}")
OUT.write_text("\n".join(lines) + "\n")
