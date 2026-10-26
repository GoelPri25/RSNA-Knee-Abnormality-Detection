"""Per-target ensemble weights for the 0.937 quad: honest fold-0 estimate by cross-fitting.

For each target, weights over the 4 members are chosen on a random half of the fold-0 studies and scored on the other
half (repeated splits, both directions), so the estimate never scores weights on the studies that chose them.
Rules compared against the equal mean (the submitted 0.937 rule):
  best_member   - per target, the single member with the highest AUC on the selection half
  grid_weights  - per target, the best weight vector on a simplex grid (step 0.25) on the selection half
  shrunk_grid   - grid weights averaged 50/50 with equal weights
Same reference, mask, loader and AUC as audit_fold0_member_addition.py. Saved validation predictions only.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_fold0_per_target_weights.py /proj /out
"""
import itertools
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
SPLITS = 200
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


names = list(MEMBERS)
P = np.stack([load(MEMBERS[k]) for k in names])  # members x studies x targets
M = len(names)
GRID = [np.array(g) / 4 for g in itertools.product(range(5), repeat=M) if sum(g) == 4]
EQUAL = np.full(M, 1 / M)


def t_auc(weights, k, idx):
    s = np.tensordot(weights, P[:, idx, k], axes=1)
    keep = mask[idx, k]
    return auc(s[keep], truth[idx, k][keep])


def choose(k, idx):
    member_aucs = [t_auc(np.eye(M)[j], k, idx) for j in range(M)]
    grid = max(GRID, key=lambda g: t_auc(g, k, idx))
    return {"equal": EQUAL, "best_member": np.eye(M)[int(np.argmax(member_aucs))],
            "grid_weights": grid, "shrunk_grid": (grid + EQUAL) / 2}


n = len(ids); rng = np.random.default_rng(0)
rules = ["equal", "best_member", "grid_weights", "shrunk_grid"]
scores = {r: [] for r in rules}
for _ in range(SPLITS):
    perm = rng.permutation(n); halves = (perm[: n // 2], perm[n // 2:])
    for sel, ev in (halves, halves[::-1]):
        per_rule = {r: [] for r in rules}
        for k in range(len(T)):
            chosen = choose(k, sel)
            for r in rules:
                per_rule[r].append(t_auc(chosen[r], k, ev))
        for r in rules:
            scores[r].append(np.mean(per_rule[r]))
S = pd.DataFrame(scores)
D_ = S[rules[1:]].sub(S["equal"], axis=0)
summary = pd.DataFrame({"mean_heldout_macro": S.mean(), "delta_vs_equal": [0.0] + list(D_.mean()),
                        "delta_p2.5": [0.0] + list(D_.quantile(.025)), "delta_p97.5": [0.0] + list(D_.quantile(.975)),
                        "share_splits_better": [np.nan] + list((D_ > 0).mean())}, index=rules)
summary.round(5).to_csv(OUT / "per_target_weights_crossfit.csv", index_label="rule")

full = np.arange(n)
member_tab = pd.DataFrame({names[j]: [t_auc(np.eye(M)[j], k, full) for k in range(len(T))] for j in range(M)}, index=T)
member_tab["equal_mean"] = [t_auc(EQUAL, k, full) for k in range(len(T))]
fullw = {T[k]: choose(k, full)["grid_weights"].round(2).tolist() for k in range(len(T))}
member_tab["grid_weights_full_fold0"] = [str(fullw[t]) for t in T]
member_tab.round(4).to_csv(OUT / "per_target_member_auc.csv", index_label="target")
pd.set_option("display.width", 220)
print(member_tab.round(3).to_string()); print()
print(f"members order: {names}; {SPLITS} random half-splits x 2 directions")
print(summary.round(4).to_string())
