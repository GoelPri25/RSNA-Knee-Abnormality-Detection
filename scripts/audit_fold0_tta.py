"""Score the pre-registered fold-0 TTA sets (kernel tumis001/rsna-r50-tta-fold0) against Steven-v4.

Per model and set: macro AUC, change vs 'none' with 2,000 paired study bootstraps, per-target changes.
Adopt a set only if it improves both models with >= 90% of bootstraps better. Also reports the quad with the two
r50 224 members replaced by their TTA versions (other members unchanged), for information.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_fold0_tta.py /proj repository/data/r50_tta/kaggle_output
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, TTA = Path(sys.argv[1]), sys.argv[2]
D = "repository/data/"
REF = D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json"
OTHER = [D + "highres336/results/oof_predictions.json", D + "efficientnet_v31_d2/results/oof_predictions.json"]
MODELS, SETS = ["d2_ep15", "hpo_v1_ep17"], ["none", "tta5", "tta9", "int3"]
ref = json.loads((R / REF).read_text()); T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7


def load(p):
    d = json.loads((R / p).read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
    tg = d.get("targets_in_order", d.get("targets", T)); a = np.array([m[s] for s in ids], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, yy):
    r = pd.Series(s).rank().to_numpy(); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per(S, idx):
    return np.array([auc(S[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(len(T))])


P = {(m, s): load(f"{TTA}/{m}__{s}.json") for m in MODELS for s in SETS}
O = [load(p) for p in OTHER]
n = len(ids); full = np.arange(n); rng = np.random.default_rng(0); boots = [rng.integers(0, n, n) for _ in range(2000)]
rows, pt = [], {}
for m in MODELS:
    base_b = np.array([per(P[m, "none"], b).mean() for b in boots])
    for s in SETS:
        d = np.array([per(P[m, s], b).mean() for b in boots]) - base_b
        rows.append({"model": m, "set": s, "macro": per(P[m, s], full).mean(),
                     "delta": per(P[m, s], full).mean() - per(P[m, "none"], full).mean(),
                     "p2.5": np.percentile(d, 2.5), "p97.5": np.percentile(d, 97.5), "share_better": (d > 0).mean()})
        pt[f"{m}:{s}"] = per(P[m, s], full) - per(P[m, "none"], full)
quad = {s: np.mean([P["d2_ep15", s], P["hpo_v1_ep17", s]] + O, 0) for s in SETS}
qb = np.array([per(quad["none"], b).mean() for b in boots])
for s in SETS:
    d = np.array([per(quad[s], b).mean() for b in boots]) - qb
    rows.append({"model": "quad (2 members TTA)", "set": s, "macro": per(quad[s], full).mean(),
                 "delta": per(quad[s], full).mean() - per(quad["none"], full).mean(),
                 "p2.5": np.percentile(d, 2.5), "p97.5": np.percentile(d, 97.5), "share_better": (d > 0).mean()})
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(4).to_string(index=False)); print()
print(pd.DataFrame(pt, index=T).round(4).to_string())
