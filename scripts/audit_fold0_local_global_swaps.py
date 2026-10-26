"""Quad ensemble vs quad + local_global and vs swapping one quad member for local_global (fold 0, Steven-v4).

Equal probability mean throughout (no fitting). 2,000 paired study bootstraps of each option minus the quad.
Two candidates: local_global (selected on Steven-v4) and local_global_hpo (selected on HPO-v3.1-D2 labels).
Same reference, mask, loader and AUC as audit_fold0_checkpoint_zoo.py.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_fold0_local_global_swaps.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = "repository/data/"
REF = D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json"
QUAD = {
    "d2_224": D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json",
    "d2_336": D + "highres336/results/oof_predictions.json",
    "hpo_v1": D + "hpo_v1_resnet50_auc_margin_fold0/results/baseline_validation_predictions.json",
    "effnet_d2_224": D + "efficientnet_v31_d2/results/oof_predictions.json",
}
LG = {"local_global": D + "local_global/results/oof_predictions.json",
      "local_global_hpo": D + "local_global_hpo_validation/results/oof_predictions.json"}
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


def macro(S, idx):
    return np.mean([auc(S[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(len(T))])


P = {k: load(v) for k, v in {**QUAD, **LG}.items()}
opts = {"quad": list(QUAD)}
for g in LG:
    opts[f"quad+{g}"] = list(QUAD) + [g]
    opts[f"quad+both_lg"] = list(QUAD) + list(LG)
    for m in QUAD:
        opts[f"swap {m}->{g}"] = [x for x in QUAD if x != m] + [g]
S = {k: np.mean([P[m] for m in v], 0) for k, v in opts.items()}
n = len(ids); full = np.arange(n); rng = np.random.default_rng(0)
boots = [rng.integers(0, n, n) for _ in range(2000)]
base_b = np.array([macro(S["quad"], b) for b in boots])
rows = []
for k in opts:
    d = np.array([macro(S[k], b) for b in boots]) - base_b
    rows.append({"option": k, "fold0_macro": macro(S[k], full), "delta": macro(S[k], full) - macro(S["quad"], full),
                 "p2.5": np.percentile(d, 2.5), "p97.5": np.percentile(d, 97.5), "share_boot_better": (d > 0).mean()})
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).set_index("option").sort_values("fold0_macro", ascending=False).round(4).to_string())
