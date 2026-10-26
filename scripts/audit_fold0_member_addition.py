"""Does a candidate model add to the current submitted ensemble? (fold 0, Steven-v4 reference)

Reusable member check. The base is the current best public ensemble (equal per-study probability mean). For the
candidate it reports: AUC alone, base + candidate, the candidate swapped in for each base member, rank correlation with
the base mean (diversity), and paired study-bootstrap deltas vs the base. Same reference, mask, loader and AUC as the
other fold-0 audits. Saved validation predictions only; no training, no GPU, no gold.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_fold0_member_addition.py /proj /out NAME path/to/oof_predictions.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT, NAME, CAND = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
OUT.mkdir(parents=True, exist_ok=True)
REF = "repository/data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json"
D = "repository/data/"
BASE_MEMBERS = {  # public 0.937 (2026-10-09)
    "d2_224": D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json",
    "d2_336": D + "highres336/results/oof_predictions.json",
    "hpo_v1": D + "hpo_v1_resnet50_auc_margin_fold0/results/baseline_validation_predictions.json",
    "effnet_d2_224": D + "efficientnet_v31_d2/results/oof_predictions.json",
}
DRAWS = 1000
ref = json.loads((R / REF).read_text())
T = ref["targets_in_order"]
ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7


def load(p):
    d = json.loads((R / p).read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
    tg = d.get("targets_in_order", d.get("targets", T))
    assert set(ids) <= m.keys(), p
    a = np.array([m[s] for s in ids], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, yy):
    r = pd.Series(s).rank().to_numpy(); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_target(P, idx):
    return np.array([auc(P[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(len(T))])


P = {k: load(v) for k, v in BASE_MEMBERS.items()}
P[NAME] = load(CAND)
base = list(BASE_MEMBERS)
S = {NAME: P[NAME], "base": np.mean([P[m] for m in base], 0), f"base+{NAME}": np.mean([P[m] for m in base + [NAME]], 0)}
for m in base:
    S[f"{NAME}_for_{m}"] = np.mean([P[k] for k in base if k != m] + [P[NAME]], 0)
n = len(ids); full = np.arange(n)
tab = pd.DataFrame({k: per_target(v, full) for k, v in S.items()}, index=T)
tab.loc["MACRO"] = tab.mean()
tab.round(5).to_csv(OUT / f"{NAME}_member_auc.csv", index_label="target")

rng = np.random.default_rng(0)
dl = {k: [] for k in S if k != "base"}
for _ in range(DRAWS):
    idx = rng.integers(0, n, n)
    b = per_target(S["base"], idx)
    for k in dl:
        dl[k].append((per_target(S[k], idx) - b).mean())
bt = pd.DataFrame({k: {"macro": tab.at["MACRO", k], "delta_vs_base": tab.at["MACRO", k] - tab.at["MACRO", "base"],
                       "ci_low": np.percentile(v, 2.5), "ci_high": np.percentile(v, 97.5)} for k, v in dl.items()}).T
bt.round(5).to_csv(OUT / f"{NAME}_member_bootstrap.csv", index_label="candidate")
corr = pd.Series({k: np.mean([pd.Series(P[k][:, j]).rank().corr(pd.Series(S["base"][:, j]).rank()) for j in range(len(T))])
                  for k in base + [NAME]}, name="mean_rank_corr_vs_base")
corr.round(4).to_csv(OUT / f"{NAME}_member_corr.csv", index_label="member")
pd.set_option("display.width", 200)
print(tab.round(3).to_string()); print(); print(bt.round(4).to_string()); print(); print(corr.round(3).to_string())
