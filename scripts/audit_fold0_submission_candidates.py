"""Candidate submission ensembles on the fixed 825 fold-0 studies (Steven-v4 reference), including the 336 px run.

Same reference, confidence mask and AUC as audit_fold0_ensemble.py; saved validation predictions only (no training,
no GPU, no gold). Every checkpoint was selected on these studies, so absolute values are optimistic; deltas between
ensembles that share members are the useful part.

Container:
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_fold0_submission_candidates.py /proj /out
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
CANDS = {
    "d2_224": "repository/data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json",
    "d2_224_rerun": "repository/data/hpo_v31_trainval_resnet50/results/hpo_v31_trainval_resnet50_fresh/run/oof_predictions.json",
    "d2_336": "repository/data/highres336/results/oof_predictions.json",
    "hpo_v1": "repository/data/hpo_v1_resnet50_auc_margin_fold0/results/baseline_validation_predictions.json",
    "hpo_v2": "repository/data/hpo_v2_provenance_resnet50_fold0/kaggle_output/hpo_v2_provenance_resnet50_fold0/run/oof_predictions.json",
}
ENS = {
    "d2_224+hpo_v1": ["d2_224", "hpo_v1"],
    "d2_336+hpo_v1": ["d2_336", "hpo_v1"],
    "d2_336+d2_224": ["d2_336", "d2_224"],
    "d2_336+d2_224+hpo_v1": ["d2_336", "d2_224", "hpo_v1"],
    "d2_336+hpo_v1+hpo_v2": ["d2_336", "hpo_v1", "hpo_v2"],
    "d2_336+d2_224+hpo_v1+hpo_v2": ["d2_336", "d2_224", "hpo_v1", "hpo_v2"],
    "all5": list(CANDS),
}
BASE, DRAWS = "d2_224", 1000


def load(p):
    d = json.loads((R / p).read_text())
    if "scores" in d:
        m = dict(zip(d["study_ids"], d["scores"])); tg = d.get("targets_in_order", T)
    else:
        m = {r["study_id"]: r["probabilities"] for r in d["predictions"]}; tg = d.get("targets_in_order", d.get("targets", T))
    assert set(ids) <= m.keys(), p
    a = np.array([m[s] for s in ids], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, yy):
    r = pd.Series(s).rank().to_numpy(); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per_target(Pm, idx):
    return np.array([auc(Pm[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(len(T))])


def rankavg(Ps):
    return np.mean([np.apply_along_axis(lambda c: pd.Series(c).rank(pct=True).to_numpy(), 0, x) for x in Ps], 0)


def logitavg(Ps):
    c = [np.clip(x, 1e-6, 1 - 1e-6) for x in Ps]
    return np.mean([np.log(x / (1 - x)) for x in c], 0)


def probavg(Ps):
    return np.mean(Ps, 0)


P = {k: load(p) for k, p in CANDS.items()}
S = {**P, **{k: rankavg([P[m] for m in v]) for k, v in ENS.items()}}
# Per-study combinations (no dependence on how many test studies are scored together)
TRIO = ENS["d2_336+d2_224+hpo_v1"]
S["trio_logit_mean"] = logitavg([P[m] for m in TRIO])
S["trio_prob_mean"] = probavg([P[m] for m in TRIO])
# Order-statistic combinations: max, min, median, and the single most confident member (furthest from 0.5)
_X = np.stack([P[m] for m in TRIO], -1)
S["trio_max"], S["trio_min"], S["trio_median"] = _X.max(-1), _X.min(-1), np.median(_X, -1)
S["trio_most_confident"] = np.take_along_axis(_X, np.abs(_X - .5).argmax(-1)[..., None], -1)[..., 0]
# Spread of each member's logits on fold 0 (if scales differ a lot, a plain mean is dominated by one model)
spread = pd.DataFrame({m: np.log(np.clip(P[m], 1e-6, 1 - 1e-6) / (1 - np.clip(P[m], 1e-6, 1 - 1e-6))).std(0) for m in TRIO}, index=T)
spread.round(3).to_csv(OUT / "fold0_trio_logit_std.csv", index_label="target")
print(spread.round(2).to_string()); print()
full = np.arange(len(ids))
tab = pd.DataFrame({k: per_target(v, full) for k, v in S.items()}, index=T)
tab.loc["MACRO"] = tab.mean()
tab.round(5).to_csv(OUT / "fold0_submission_candidates_auc.csv", index_label="target")

rng = np.random.default_rng(0)
dl = {k: [] for k in S if k != BASE}
for _ in range(DRAWS):
    idx = rng.integers(0, len(ids), len(ids))
    b = np.nanmean(per_target(S[BASE], idx))
    for k in dl:
        dl[k].append(np.nanmean(per_target(S[k], idx)) - b)
bt = pd.DataFrame({k: {"macro": tab.at["MACRO", k], "delta_vs_" + BASE: tab.at["MACRO", k] - tab.at["MACRO", BASE],
                       "ci_low": np.percentile(v, 2.5), "ci_high": np.percentile(v, 97.5)} for k, v in dl.items()}).T
bt.round(5).to_csv(OUT / "fold0_submission_candidates_bootstrap.csv", index_label="candidate")
pd.set_option("display.width", 250)
print(tab.round(3).to_string()); print(); print(bt.round(4).to_string())
