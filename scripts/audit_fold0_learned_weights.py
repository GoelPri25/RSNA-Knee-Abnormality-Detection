"""Do learned ensemble weights beat a plain probability mean out of sample? (fold 0, Steven-v4 reference)

Members: r50 v3.1-D2 224px, r50 v3.1-D2 336px, r50 HPO-v1 224px (the 0.935 public trio). Weighted probability means
with non-negative weights summing to 1 (grid step 0.05) are fitted to maximise masked AUC on one half of the 825
fold-0 studies and scored on the other half, then the halves swap. Repeated over 50 random splits (seed 0).
Two variants: one weight vector per target (24 free weights) and one shared across targets (2 free weights).
Same reference, confidence mask and loader as audit_fold0_submission_candidates.py. No training, no GPU, no gold.

Container:
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_fold0_learned_weights.py /proj /out
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
TRIO = {
    "d2_224": "repository/data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json",
    "d2_336": "repository/data/highres336/results/oof_predictions.json",
    "hpo_v1": "repository/data/hpo_v1_resnet50_auc_margin_fold0/results/baseline_validation_predictions.json",
}
SPLITS, STEP = 50, 0.05


def load(p):
    d = json.loads((R / p).read_text())
    if "scores" in d:
        m = dict(zip(d["study_ids"], d["scores"])); tg = d.get("targets_in_order", T)
    else:
        m = {r["study_id"]: r["probabilities"] for r in d["predictions"]}; tg = d.get("targets_in_order", d.get("targets", T))
    assert set(ids) <= m.keys(), p
    a = np.array([m[s] for s in ids], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc_many(S, yy):
    """AUC of each column of S (n x c) against boolean yy (n)."""
    r = S.argsort(0).argsort(0) + 1.0
    n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


X = np.stack([load(p) for p in TRIO.values()], -1)  # n x targets x 3
g = np.round(np.arange(0, 1 + 1e-9, STEP), 2)
W = np.array([(a, b, round(1 - a - b, 2)) for a in g for b in g if a + b <= 1 + 1e-9])  # grid x 3
EQ = np.argmin(np.abs(W - 1 / 3).sum(1))  # closest grid point to equal weights, used only for reference
n = len(ids)


def grid_auc(idx, k):
    m = idx[mask[idx, k]]
    return auc_many(X[m, k, :] @ W.T, truth[m, k])  # AUC for every grid weight


def eq_auc(idx, k):
    m = idx[mask[idx, k]]
    return auc_many(X[m, k, :].mean(1, keepdims=True), truth[m, k])[0]


rng = np.random.default_rng(0)
rows = []
chosen = {t: [] for t in T}
for s in range(SPLITS):
    perm = rng.permutation(n); halves = (perm[: n // 2], perm[n // 2:])
    for fit, ev in (halves, halves[::-1]):
        G_fit = np.array([grid_auc(fit, k) for k in range(len(T))])  # targets x grid
        G_ev = np.array([grid_auc(ev, k) for k in range(len(T))])
        per_t = G_ev[np.arange(len(T)), G_fit.argmax(1)]
        shared = G_ev[:, G_fit.mean(0).argmax()]
        eq = np.array([eq_auc(ev, k) for k in range(len(T))])
        for k, t in enumerate(T):
            chosen[t].append(W[G_fit[k].argmax()])
        rows.append({"split": s, "equal_mean": eq.mean(), "per_target_weights": per_t.mean(), "shared_weights": shared.mean(),
                     "in_sample_per_target_gain": G_fit.max(1).mean() - np.array([eq_auc(fit, k) for k in range(len(T))]).mean()})
res = pd.DataFrame(rows)
res.to_csv(OUT / "fold0_learned_weights_splits.csv", index=False)
summ = pd.DataFrame({c: {"mean_delta_vs_equal": (res[c] - res.equal_mean).mean(),
                         "p2.5": np.percentile(res[c] - res.equal_mean, 2.5), "p97.5": np.percentile(res[c] - res.equal_mean, 97.5),
                         "share_of_evals_better": ((res[c] - res.equal_mean) > 0).mean()}
                     for c in ("per_target_weights", "shared_weights")}).T
summ.loc["in_sample_per_target (optimism)", "mean_delta_vs_equal"] = res.in_sample_per_target_gain.mean()
summ.round(5).to_csv(OUT / "fold0_learned_weights_summary.csv", index_label="variant")
cw = pd.DataFrame({t: np.mean(v, 0) for t, v in chosen.items()}, index=list(TRIO)).T
cw["sd_of_d2_336_weight"] = [np.std([c[1] for c in chosen[t]]) for t in T]
cw.round(3).to_csv(OUT / "fold0_learned_weights_mean_chosen.csv", index_label="target")
pd.set_option("display.width", 200)
print(summ.round(4).to_string()); print(); print(cw.round(2).to_string())
