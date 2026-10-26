"""Better ways to combine the 0.937 quad than per-target AUC weights: honest fold-0 estimate by cross-fitting.

Rules compared against the equal probability mean (submitted 0.937 rule):
  logit_mean     - mean of member logits (no fitting)
  platt_logit    - per member and target, Platt scaling (a*logit+b, fitted by L2-regularised log-likelihood on the
                   selection half), then mean of calibrated logits
  cross_target   - per target, L2 logistic regression on the equal-mean logits of all 12 targets (lets correlated
                   findings inform each other), fitted on the selection half
  cross_shrunk   - cross_target score averaged with the equal-mean logit of the target (rank-normalised halves)
Fitted rules are chosen on a random half of fold-0 and scored on the other half (repeated, both directions).
Same reference, mask, loader and AUC as audit_fold0_member_addition.py. Saved validation predictions only.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_fold0_combination_rules.py /proj /out
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
REF = "repository/data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json"
D = "repository/data/"
MEMBERS = {  # public 0.937 (2026-10-09)
    "d2_224": D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json",
    "d2_336": D + "highres336/results/oof_predictions.json",
    "hpo_v1": D + "hpo_v1_resnet50_auc_margin_fold0/results/baseline_validation_predictions.json",
    "effnet_d2_224": D + "efficientnet_v31_d2/results/oof_predictions.json",
}
SPLITS = 100
L2_PLATT = 1.0
L2_STACK = 10.0
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


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def fit_logistic(X, t, l2, iters=50):
    """Newton's method for L2-regularised logistic regression; the intercept (last column) is not penalised."""
    X = np.column_stack([X, np.ones(len(X))]); beta = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2); pen[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        g = X.T @ (p - t) + pen * beta
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen) + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(H, g); beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def apply(beta, X):
    return np.column_stack([X, np.ones(len(X))]) @ beta


P = np.stack([load(v) for v in MEMBERS.values()])  # members x studies x targets
L = logit(P)
EQ_P = P.mean(0); EQ_L = logit(EQ_P); LOGIT_MEAN = L.mean(0)
n, K = len(ids), len(T)


def scores_for(sel, ev):
    out = {"equal": EQ_P[ev], "logit_mean": LOGIT_MEAN[ev]}
    platt = np.zeros((len(ev), K)); cross = np.zeros((len(ev), K))
    Xs, Xe = EQ_L[sel], EQ_L[ev]
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    for k in range(K):
        keep = mask[sel, k]; t = truth[sel, k][keep].astype(float)
        cal = [apply(fit_logistic(L[m, sel, k][keep, None], t, L2_PLATT), L[m, ev, k][:, None]) for m in range(len(P))]
        platt[:, k] = np.mean(cal, 0)
        beta = fit_logistic(((Xs - mu) / sd)[keep], t, L2_STACK)
        cross[:, k] = apply(beta, (Xe - mu) / sd)
    out["platt_logit"] = platt; out["cross_target"] = cross
    rank = lambda a: pd.DataFrame(a).rank(pct=True).to_numpy()
    out["cross_shrunk"] = (rank(cross) + rank(EQ_P[ev])) / 2
    return out


def macro(S, ev):
    return np.mean([auc(S[:, k][mask[ev, k]], truth[ev, k][mask[ev, k]]) for k in range(K)])


rng = np.random.default_rng(0); rules = None; rows = []
for _ in range(SPLITS):
    perm = rng.permutation(n); halves = (perm[: n // 2], perm[n // 2:])
    for sel, ev in (halves, halves[::-1]):
        S = scores_for(sel, ev); rules = list(S)
        rows.append({r: macro(S[r], ev) for r in rules})
S = pd.DataFrame(rows)
Dl = S[rules[1:]].sub(S["equal"], axis=0)
summary = pd.DataFrame({"mean_heldout_macro": S.mean(), "delta_vs_equal": [0.0] + list(Dl.mean()),
                        "delta_p2.5": [0.0] + list(Dl.quantile(.025)), "delta_p97.5": [0.0] + list(Dl.quantile(.975)),
                        "share_splits_better": [np.nan] + list((Dl > 0).mean())}, index=rules)
summary.round(5).to_csv(OUT / "combination_rules_crossfit.csv", index_label="rule")
full = np.arange(n)
print(f"full fold-0 (no fitting): equal {macro(EQ_P, full):.4f}  logit_mean {macro(LOGIT_MEAN, full):.4f}")
print(f"members: {list(MEMBERS)}; {SPLITS} random half-splits x 2 directions; L2 platt {L2_PLATT}, stack {L2_STACK}")
pd.set_option("display.width", 200)
print(summary.round(4).to_string())
