"""Which equal-mean ensemble subsets of our fold-0 models are worth their members? (fold 0, Steven-v4 confident cells)

1. Exhaustive search over all subsets of the candidate members: best subset per size, and where hepta / octa sit.
2. Honest estimate of what subset search buys: 200 random half splits of the 825 studies; pick the best subset on
   half A, score it and hepta on half B. Selection on all 825 studies is optimistic by construction.
Exp 1e is trained on Steven-v4 for Lateral OA and Baker's (inflated on fold 0), so every number is also given on the
10 other targets.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_ensemble_subsets.py /proj
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]


def load(p):
    d = json.loads(Path(p).read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
    tg = d.get("targets_in_order", d.get("targets", T)); a = np.array([m[s] for s in ids], float)
    return a[:, [list(tg).index(t) for t in T]]


M = {"d2_224": D / "r50_tta/kaggle_output/d2_ep15__none.json", "hpo1_224": D / "r50_tta/kaggle_output/hpo_v1_ep17__none.json",
     "d2_336": D / "highres336/results/oof_predictions.json", "effnet": D / "efficientnet_v31_d2/results/oof_predictions.json",
     "1b_coat": C / "run_continue/val_swa.json", "1c": C / "run_r50/val_swa.json", "1e": C / "run_r50_pertarget/val_swa.json",
     "2a": C / "run_r50_ctx/val_swa.json", "2c": C / "run_r50_plane/val_swa.json", "3a": C / "run_r50_border/val_swa.json",
     "3b": C / "run_r50_state/val_swa.json"}
names = list(M); P = np.stack([load(M[k]) for k in names])          # [m, n, 12]
HEPTA = {"d2_224", "hpo1_224", "d2_336", "effnet", "1b_coat", "1c", "1e"}
n = len(ids)


def auc_vec(s, idx, k):
    """AUC per subset row for target k on studies idx. s: [S, n]."""
    sel = idx[mask[idx, k]]; t = truth[sel, k]; n1 = t.sum(); n0 = len(t) - n1
    r = s[:, sel].argsort(1).argsort(1) + 1.0                         # ties negligible for averaged probabilities
    return (r[:, t].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


subsets = [c for r in range(1, len(names) + 1) for c in itertools.combinations(range(len(names)), r)]
S = np.zeros((len(subsets), len(names)), bool)
for i, c in enumerate(subsets):
    S[i, list(c)] = True
E = np.einsum("sm,mnk->snk", S.astype(np.float32), P.astype(np.float32)) / S.sum(1)[:, None, None]   # [subsets, n, 12]


def scores(idx):
    per = np.stack([auc_vec(E[..., k], idx, k) for k in range(12)], 1)  # [subsets, 12]
    return per.mean(1), per[:, TEN].mean(1)


full = np.arange(n)
s12, s10 = scores(full)
key = lambda members: subsets.index(tuple(sorted(names.index(m) for m in members)))
ih, io = key(HEPTA), key(HEPTA | {"2c"})
lab = lambda i: "+".join(names[j] for j in subsets[i])
print(f"{len(names)} candidates, {len(subsets)} subsets. In-sample fold-0 macro (12-target / 10-target):")
print(f"  hepta            {s12[ih]:.4f} / {s10[ih]:.4f}   [{lab(ih)}]")
print(f"  octa (hepta+2c)  {s12[io]:.4f} / {s10[io]:.4f}")
print("\nBest subset per size (ranked on 10-target, in-sample, optimistic):")
size = S.sum(1)
for r in range(1, len(names) + 1):
    i = np.where(size == r)[0][np.argmax(s10[size == r])]
    print(f"  {r:2d} members: {s12[i]:.4f} / {s10[i]:.4f}  {lab(i)}")
top = np.argsort(-s10)[:8]
print("\nTop 8 subsets overall (10-target):")
for i in top:
    print(f"  {s12[i]:.4f} / {s10[i]:.4f}  ({size[i]}) {lab(i)}")

rng = np.random.default_rng(0); gains, picks, gains_o = [], [], []
for _ in range(200):
    perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
    a12, a10 = scores(A); b12, b10 = scores(B)
    i = int(np.argmax(a10)); picks.append(lab(i))
    gains.append(b10[i] - b10[ih]); gains_o.append(b10[io] - b10[ih])
g = np.array(gains); go = np.array(gains_o)
print(f"\nHonest subset search (200 half splits, select on A, score on B, 10-target):")
print(f"  best-on-A subset vs hepta on B: mean {g.mean():+.4f}, 2.5-97.5% [{np.percentile(g, 2.5):+.4f}, {np.percentile(g, 97.5):+.4f}], better in {(g > 0).mean():.0%}")
print(f"  fixed octa vs hepta on B:        mean {go.mean():+.4f}, 2.5-97.5% [{np.percentile(go, 2.5):+.4f}, {np.percentile(go, 97.5):+.4f}], better in {(go > 0).mean():.0%}")
from collections import Counter
print("  most frequent picks on A:", Counter(picks).most_common(4))
print("  member frequency in picks:", {m: round(sum(m in p.split("+") for p in picks) / len(picks), 2) for m in names})
