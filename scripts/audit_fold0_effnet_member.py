"""Does the existing EfficientNetV2-S (HPO v3.1-D2, fold 0) add to the 0.935 ResNet50 trio? (fold 0, Steven-v4 reference)

Plain per-study probability means only (the submitted rule). Same reference, mask, loader and AUC as
audit_fold0_submission_candidates.py. Saved validation predictions only; no training, no GPU, no gold.
Checkpoints were selected on these studies, so read deltas, not absolute values.

Container:
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_fold0_effnet_member.py /proj /out
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
REF = "repository/data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json"
ref = json.loads((R / REF).read_text())
T = ref["targets_in_order"]
ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
D = "repository/data/"
M = {
    "d2_224": D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json",
    "d2_336": D + "highres336/results/oof_predictions.json",
    "hpo_v1": D + "hpo_v1_resnet50_auc_margin_fold0/results/baseline_validation_predictions.json",
    "effnet_d2_224": D + "efficientnet_v31_d2/results/oof_predictions.json",
}
ENS = {
    "trio": ["d2_224", "d2_336", "hpo_v1"],
    "trio+effnet": ["d2_224", "d2_336", "hpo_v1", "effnet_d2_224"],
    "effnet_for_d2_224": ["effnet_d2_224", "d2_336", "hpo_v1"],
    "effnet_for_hpo_v1": ["d2_224", "d2_336", "effnet_d2_224"],
}
BASE, DRAWS = "trio", 1000


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


P = {k: load(v) for k, v in M.items()}
S = {**P, **{k: np.mean([P[m] for m in v], 0) for k, v in ENS.items()}}
n = len(ids); full = np.arange(n)
tab = pd.DataFrame({k: per_target(v, full) for k, v in S.items()}, index=T)
tab.loc["MACRO"] = tab.mean()
tab.round(5).to_csv(OUT / "fold0_effnet_member_auc.csv", index_label="target")

rng = np.random.default_rng(0)
dl = {k: [] for k in S if k != BASE}
for _ in range(DRAWS):
    idx = rng.integers(0, n, n)
    b = per_target(S[BASE], idx)
    for k in dl:
        dl[k].append((per_target(S[k], idx) - b).mean())
bt = pd.DataFrame({k: {"macro": tab.at["MACRO", k], "delta_vs_trio": tab.at["MACRO", k] - tab.at["MACRO", BASE],
                       "ci_low": np.percentile(v, 2.5), "ci_high": np.percentile(v, 97.5)} for k, v in dl.items()}).T
bt.round(5).to_csv(OUT / "fold0_effnet_member_bootstrap.csv", index_label="candidate")

# diversity: per-target Spearman of each member with d2_224 and with the trio mean
corr = pd.DataFrame({k: [pd.Series(P[k][:, j]).rank().corr(pd.Series(S["trio"][:, j]).rank()) for j in range(len(T))]
                     for k in M}, index=T)
corr.loc["MEAN"] = corr.mean()
corr.round(4).to_csv(OUT / "fold0_member_rank_corr_vs_trio.csv", index_label="target")
pd.set_option("display.width", 200)
print(tab.round(3).to_string()); print(); print(bt.round(4).to_string()); print(); print(corr.loc[["MEAN"]].round(3).to_string())
