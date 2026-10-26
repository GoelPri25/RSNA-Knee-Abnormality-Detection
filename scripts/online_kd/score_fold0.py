"""Fold-0 comparison of a student against E4: macro AUC on confident cells (<= 0.3 / >= 0.7) of label-v2 and of the
Steven-v4 validation labels, with a paired study bootstrap (2,000 resamples).
Usage: python score_fold0.py ASSETS NEW_val_swa_fused.json E4_val_swa_fused.json [more.json ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]


def labels(p):
    d = json.loads(Path(p).read_text())
    return {r["study_id"]: np.array([np.nan if v is None else v for v in r["probabilities"]], float) for r in d["labels"]}


def preds(p):
    d = json.loads(Path(p).read_text())
    rows = d["predictions"] if isinstance(d, dict) and "predictions" in d else d
    return {r["study_id"]: np.array(r["probabilities"], float) for r in rows}


def auc(s, y):
    if y.all() or not y.any():
        return np.nan
    r = np.argsort(np.argsort(s)) + 1.0
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(S, Y, idx):
    out = []
    for k in range(12):
        y = Y[idx, k]; c = (y <= .3) | (y >= .7)
        out.append(auc(S[idx, k][c], y[c] >= .7))
    return float(np.nanmean(out))


A = Path(sys.argv[1]); files = sys.argv[2:]
rulers = {"label-v2": labels(A / "soft_labels_final_per_target_latD2.json"), "steven-v4": labels(A / "validation_labels.json")}
P = [preds(f) for f in files]
ids = sorted(set(rulers["steven-v4"]) & set.intersection(*[set(p) for p in P]))
print(f"{len(ids)} fold-0 studies")
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
for name, lab in rulers.items():
    Y = np.array([lab[s] for s in ids]); full = np.arange(len(ids))
    S = [np.array([p[s] for s in ids]) for p in P]
    base = [macro(s, Y, full) for s in S]
    line = " | ".join(f"{Path(f).parent.name}/{Path(f).name}: {b:.4f}" for f, b in zip(files, base))
    print(f"[{name}] {line}")
    for j in range(1, len(S)):
        d = np.array([macro(S[0], Y, i) - macro(S[j], Y, i) for i in boot])
        print(f"   first - {Path(files[j]).parent.name}: {base[0] - base[j]:+.4f}  95% [{np.percentile(d, 2.5):+.4f}, "
              f"{np.percentile(d, 97.5):+.4f}]  P(>0) {np.mean(d > 0):.2f}")
