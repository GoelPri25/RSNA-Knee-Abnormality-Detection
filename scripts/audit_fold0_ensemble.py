"""Single models vs prediction ensembles on the fixed 825 fold-0 validation studies (Steven-v4 reference).

Uses saved validation predictions only (no training, no GPU, no gold). Same confidence mask as the trainer
(reference <= 0.3 or >= 0.7, positive weight). Ensembles are averages of per-target percentile ranks or of logits.
Caveat: every model's checkpoint was selected on these same studies, so absolute values are optimistic.

Container:
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_fold0_ensemble.py /proj /out
"""
import hashlib
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
    "resnet_hpo_v1": "repository/data/hpo_v1_resnet50_auc_margin_fold0/results/baseline_validation_predictions.json",
    "resnet_hpo_v31_d2": "repository/data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json",
    "resnet_hpo_v2": "repository/data/hpo_v2_provenance_resnet50_fold0/kaggle_output/hpo_v2_provenance_resnet50_fold0/run/oof_predictions.json",
    "plane_split_v1": "repository/data/teammate_plane_split_balanced_wideband/output/hpo_v1_plane_split_balanced_wideband_fold0/run/oof_predictions.json",
    "topk_plateau60": "repository/data/cnn_topk_plateau60/cnn_topk_support_v1_plateau60/oof_predictions.json",
    "topk_pairwise": "data/kaggle_cli/cnn_pairwise_auc_fold0/cnn_topk_pairwise_auc_fold0/oof_predictions.json",
    "topk_v1": "cnn_topk_experiment/outputs/cnn_topk_support_v1/oof_predictions.json",
}


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
    return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_target(P, idx=None):
    idx = np.arange(len(ids)) if idx is None else idx
    return np.array([auc(P[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(len(T))])


def rankavg(Ps):
    return np.mean([np.apply_along_axis(lambda c: pd.Series(c).rank(pct=True).to_numpy(), 0, P) for P in Ps], 0)


def logitavg(Ps):
    L = np.mean([np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6))) for P in Ps], 0)
    return 1 / (1 + np.exp(-L))


P = {k: load(p) for k, p in CANDS.items()}
rows = {k: per_target(a) for k, a in P.items()}
best3 = sorted(rows, key=lambda k: -rows[k].mean())[:3]
E = {"ens_v1_v31d2_rank": rankavg([P["resnet_hpo_v1"], P["resnet_hpo_v31_d2"]]),
     "ens_v1_v31d2_logit": logitavg([P["resnet_hpo_v1"], P["resnet_hpo_v31_d2"]]),
     "ens_all7_rank": rankavg(list(P.values())),
     "ens_top3_rank": rankavg([P[k] for k in best3])}
tab = pd.DataFrame({**rows, **{k: per_target(v) for k, v in E.items()}}, index=T)
tab.loc["MACRO"] = tab.mean()
tab.insert(0, "reference_positives", list((truth & mask).sum(0)) + [np.nan])
tab.round(5).to_csv(OUT / "fold0_auc_single_and_ensemble.csv", index_label="target")
corr = pd.DataFrame({k: [np.corrcoef(pd.Series(P[k][:, j]).rank(), pd.Series(P["resnet_hpo_v31_d2"][:, j]).rank())[0, 1]
                         for j in range(len(T))] for k in P}, index=T)
corr.round(4).to_csv(OUT / "fold0_prediction_rank_corr_vs_v31d2.csv", index_label="target")
base, ens = P["resnet_hpo_v31_d2"], E["ens_v1_v31d2_rank"]
rng = np.random.default_rng(0); d = []
for _ in range(500):
    i = rng.integers(0, len(ids), len(ids))
    d.append(per_target(ens, i).mean() - per_target(base, i).mean())
res = {"top3": best3, "ens_v1_v31d2_rank_minus_v31d2": round(float(tab.at["MACRO", "ens_v1_v31d2_rank"] - tab.at["MACRO", "resnet_hpo_v31_d2"]), 5),
       "bootstrap_95": [round(float(x), 5) for x in np.percentile(d, [2.5, 97.5])], "bootstrap_draws": 500, "seed": 0,
       "inputs_sha256": {k: hashlib.sha256((R / p).read_bytes()).hexdigest() for k, p in {**CANDS, "reference": REF}.items()}}
(OUT / "fold0_ensemble_summary.json").write_text(json.dumps(res, indent=1) + "\n")
print(tab.round(4).to_string()); print(json.dumps({k: v for k, v in res.items() if k != "inputs_sha256"}))
