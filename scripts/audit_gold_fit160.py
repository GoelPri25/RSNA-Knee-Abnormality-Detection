"""Gold-58 per-target AUC of the Fit160 r34 (HPO-v1 +281, ep15, LB 0.903) vs 140 mm checkpoints (development evidence).

Closest 140 mm counterpart: r34_hpo_v1_auc_control_fold0 (same HPO-v1 +281 labels and training set, 140 mm cache,
epoch 22, LB 0.910). Also shown: r34 plateau60, r50 D2 (0.925) and the d96 SWA models.

Container:
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_gold_fit160.py /proj
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"
rows = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8"))]
fit = json.loads((D / "fit160_gold_audit/gold_fit160_predictions.json").read_text())
T = fit["targets_in_order"]
gold = {r["StudyInstanceUID"]: [int(r[t]) for t in T] for r in rows if all(r[t] in ("0", "1") for t in T)}
ids = fit["gold_study_ids"]; assert set(ids) == set(gold)
y = np.array([gold[s] for s in ids]).astype(bool)

zoo = json.loads((D / "share/gold_auc_checkpoint_bundle/gold_image_model_predictions.json").read_text())
assert zoo["gold_study_ids"] == ids and zoo["targets_in_order"] == T
M = {"fit160_r34_ep15 (LB .903)": np.array(fit["gold_probabilities"])}
for k, lab in (("r34_hpo_v1_auc_control_fold0", "140mm_r34_auc_ctrl (LB .910)"), ("r34_plateau60_fold0", "140mm_r34_plateau60"),
               ("r50_hpo_v31_d2_fold0", "140mm_r50_D2 (LB .925)")):
    M[lab] = np.array(zoo["models"][k]["gold_probabilities"])
for run, lab in (("run_continue", "d96_coatnet_1b (LB .933)"), ("run_r50", "d96_r50_1c (LB .933)")):
    d = json.loads((D / f"coatnet_d96_fold0/{run}/gold_swa.json").read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]}
    M[lab] = np.array([m[s] for s in ids])


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(S, idx):
    return np.array([auc(S[idx, k], y[idx, k]) for k in range(len(T))])


full = np.arange(len(ids))
tab = pd.DataFrame({k: per(S, full) for k, S in M.items()}, index=T)
tab.loc["MACRO"] = tab.mean()
print("Gold-58 AUC (development evidence only); gold positives per target:", dict(zip(T, y.sum(0).tolist())))
print(tab.round(3).to_string())

rng = np.random.default_rng(0)
boots = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
A, B = M["fit160_r34_ep15 (LB .903)"], M["140mm_r34_auc_ctrl (LB .910)"]
d = np.array([[x - z for x, z in zip(per(A, b), per(B, b))] for b in boots])
print("\nFit160 minus 140 mm r34 (auc_ctrl), paired bootstrap per target:")
for k, t in enumerate(T):
    v = d[:, k][np.isfinite(d[:, k])]
    print(f"  {t:17s} {tab.iloc[k, 0] - tab.iloc[k, 1]:+.3f}  [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]  "
          f"{(v > 0).mean():.0%} better")
mac = np.nanmean(d, 1)
print(f"  MACRO             {tab.iloc[-1, 0] - tab.iloc[-1, 1]:+.3f}  [{np.percentile(mac, 2.5):+.3f}, {np.percentile(mac, 97.5):+.3f}]  "
      f"{(mac > 0).mean():.0%} better")
