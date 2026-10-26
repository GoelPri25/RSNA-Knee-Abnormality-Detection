"""Exp 6a secondary: diversity of SWA members (cyclical tail vs 1e's cosine tail), from saved per-epoch fold-0 predictions.

For each run and member set: per-member 10-target AUC (fold-0 Steven-v4 ruler, confident cells), mean pairwise
Spearman rank correlation between members (averaged over the 10 targets; lower = more diverse), and the AUC of the
prediction-level average of members (an ensemble upper reference for the weight average, which the runner scores as
val_swa). Cross-seed reference: average of 1e and 1e-s2 SWA predictions.
Usage (container): python -I scripts/audit_d96_swa_diversity.py <project root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); C = R / "repository/data/coatnet_d96_fold0"
ref = json.loads((R / "repository/data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]


def load(p):
    d = json.loads(Path(p).read_text()); m = {r["study_id"]: r["probabilities"] for r in d["predictions"]}
    tg = d["targets_in_order"]; a = np.array([m[s] for s in ids], float)
    return a[:, [tg.index(t) for t in T]]


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ten(S):
    return np.mean([auc(S[mask[:, k], k], truth[mask[:, k], k]) for k in TEN])


def rankcorr(A, B):
    return np.mean([pd.Series(A[:, k]).corr(pd.Series(B[:, k]), method="spearman") for k in TEN])


for run, eps in (("run_r50_pertarget", range(21, 26)), ("run_r50_pertarget", range(18, 26)),
                 ("run_r50_pertarget_seed", range(21, 26)), ("run_r50_swa_cyclic", range(18, 26)),
                 ("run_r50_swa_cyclic", range(21, 26))):
    P = [load(C / run / f"val_epoch{e:02d}.json") for e in eps]
    rc = np.mean([rankcorr(P[i], P[j]) for i in range(len(P)) for j in range(i + 1, len(P))])
    swa = C / run / "val_swa.json"
    print(f"{run:24s} epochs {eps.start}-{eps.stop - 1}: members {' '.join(f'{ten(p):.4f}' for p in P)} | "
          f"mean member {np.mean([ten(p) for p in P]):.4f} | pairwise rank corr {rc:.4f} | prediction-average {ten(np.mean(P, 0)):.4f}"
          + (f" | weight-average (val_swa) {ten(load(swa)):.4f}" if swa.is_file() else ""))
A, B = load(C / "run_r50_pertarget/val_swa.json"), load(C / "run_r50_pertarget_seed/val_swa.json")
print(f"cross-seed reference: 1e SWA {ten(A):.4f}, 1e-s2 SWA {ten(B):.4f}, rank corr {rankcorr(A, B):.4f}, average {ten((A + B) / 2):.4f}")
