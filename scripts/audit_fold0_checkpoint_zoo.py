"""Inventory every saved fold-0 prediction file and test what the checkpoint collection can add (fold 0, Steven-v4).

1. For each candidate with predictions on all 825 fold-0 studies: AUC alone, mean rank correlation with the 0.937 quad,
   and the fold-0 change when added to the quad as an equal member.
2. Greedy forward selection with replacement (Caruana) over all candidates, cross-fitted: the selection is made on a
   random half of fold-0 and scored on the other half (repeated, both directions), against the quad on the same half.
Same reference, mask, loader and AUC as audit_fold0_member_addition.py. Saved validation predictions only.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_fold0_checkpoint_zoo.py /proj /out
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
REF = "repository/data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json"
D = "repository/data/"
QUAD = {
    "d2_224": D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json",
    "d2_336": D + "highres336/results/oof_predictions.json",
    "hpo_v1": D + "hpo_v1_resnet50_auc_margin_fold0/results/baseline_validation_predictions.json",
    "effnet_d2_224": D + "efficientnet_v31_d2/results/oof_predictions.json",
}
# fold-0 held out in all of these; trainval (fold 0 trained on), fold-1 and smoke outputs are excluded
CANDIDATES = {
    "r34_plateau60": D + "cnn_topk_plateau60/cnn_topk_support_v1_plateau60/oof_predictions.json",
    "convnext_tiny": D + "convnext_tiny_v31_d2/results/oof_predictions.json",
    "r50_hpo_v2": D + "hpo_v2_provenance_resnet50_fold0/kaggle_output/hpo_v2_provenance_resnet50_fold0/run/oof_predictions.json",
    "r34_local_global_hpo": D + "local_global_hpo_validation/results/oof_predictions.json",
    "r34_local_global": D + "local_global/results/oof_predictions.json",
    "r50_cosine_best": D + "r50_cosine_ema_seed2026/results/oof_predictions.json",
    "r50_cosine_ema": D + "r50_cosine_ema_seed2026/results/oof_predictions_ema.json",
    "r50_synovitis_w": D + "synovitis_weight_candidate_seed2026/published/oof_predictions.json",
    "r34_plane_split": D + "teammate_plane_split_balanced_wideband/output/hpo_v1_plane_split_balanced_wideband_fold0/run/oof_predictions.json",
}
for extra in sys.argv[3:]:  # optional NAME=repo-relative/path/oof_predictions.json
    name, path = extra.split("=", 1)
    CANDIDATES[name] = path
SPLITS, STEPS = 50, 12
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
    if not set(ids) <= m.keys():
        return None
    a = np.array([m[s] for s in ids], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, yy):
    r = pd.Series(s).rank().to_numpy(); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(S, idx):
    return np.mean([auc(S[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(len(T))])


P = {k: load(v) for k, v in QUAD.items()}
skipped = []
for k, v in CANDIDATES.items():
    if not (R / v).is_file():
        skipped.append((k, "missing file")); continue
    a = load(v)
    if a is None:
        skipped.append((k, "does not cover all 825 fold-0 studies")); continue
    P[k] = a
quad = list(QUAD); base = np.mean([P[m] for m in quad], 0)
n = len(ids); full = np.arange(n)
rows = []
for k, a in P.items():
    corr = np.mean([pd.Series(a[:, j]).rank().corr(pd.Series(base[:, j]).rank()) for j in range(len(T))])
    add = np.mean([P[m] for m in quad] + [a], 0) if k not in quad else base
    rows.append({"model": k, "in_quad": k in quad, "auc_alone": macro(a, full), "rank_corr_vs_quad": corr,
                 "quad_plus_this_minus_quad": macro(add, full) - macro(base, full) if k not in quad else 0.0})
inv = pd.DataFrame(rows).set_index("model").sort_values("auc_alone", ascending=False)
inv.round(4).to_csv(OUT / "zoo_inventory.csv")

names = list(P)


def greedy(idx):
    """Caruana selection with replacement, initialised with the quad; returns member counts."""
    counts = Counter(quad); S = sum(P[m] for m in quad); cur = macro(S / len(quad), idx)
    for _ in range(STEPS):
        best = max(names, key=lambda m: macro((S + P[m]) / (sum(counts.values()) + 1), idx))
        val = macro((S + P[best]) / (sum(counts.values()) + 1), idx)
        if val <= cur:
            break
        counts[best] += 1; S = S + P[best]; cur = val
    return counts


rng = np.random.default_rng(0); deltas = []; picked = Counter()
for _ in range(SPLITS):
    perm = rng.permutation(n); halves = (perm[: n // 2], perm[n // 2:])
    for sel, ev in (halves, halves[::-1]):
        c = greedy(sel)
        S = sum(P[m] * c[m] for m in c) / sum(c.values())
        deltas.append(macro(S, ev) - macro(base, ev))
        picked.update({m: 1 for m in c if m not in quad or c[m] > 1})
deltas = np.array(deltas)
full_counts = greedy(full)
pd.set_option("display.width", 200)
print(inv.round(4).to_string()); print()
for k, why in skipped:
    print(f"skipped {k}: {why}")
print(f"\ngreedy (cross-fitted, {SPLITS}x2 half-splits): delta vs quad {deltas.mean():+.4f} "
      f"[{np.percentile(deltas, 2.5):+.4f}, {np.percentile(deltas, 97.5):+.4f}], better in {(deltas > 0).mean():.0%}")
print("models picked beyond the quad (share of splits):", {k: round(v / len(deltas), 2) for k, v in picked.most_common()})
print("greedy on full fold-0 (in-sample, optimistic):", dict(full_counts))
json.dump({"crossfit_delta_mean": deltas.mean(), "p2.5": np.percentile(deltas, 2.5), "p97.5": np.percentile(deltas, 97.5),
           "share_better": (deltas > 0).mean(), "full_counts": dict(full_counts), "skipped": skipped},
          (OUT / "zoo_greedy.json").open("w"), indent=2, default=float)
