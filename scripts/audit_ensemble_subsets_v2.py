"""Lean ensemble search (<= 5 equal-mean members) over every fold-0 model we have, including the 2026-10-12/13 runs.

Same scoring as audit_ensemble_subsets.py (Steven-v4 confident cells, 10-target macro = without Lateral OA / Baker's).
1. In-sample best subset per size and top subsets overall (optimistic: selected and scored on the same 825 studies).
2. Named candidates vs L4 (d2_224 + 1b + 2c + 1e) with 2,000 paired bootstraps.
3. Honest estimate: 200 half splits; pick the best subset on half A, score it vs L4 on half B.
Development evidence only; fold 0 has shown ~no LB signal for singles (see fold0-lb-offset in the ledger).

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_ensemble_subsets_v2.py /proj
"""
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]
MAXK = 5


def load(p):
    d = json.loads(Path(p).read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
    tg = d.get("targets_in_order", d.get("targets", T)); a = np.array([m[s] for s in ids], float)
    return a[:, [list(tg).index(t) for t in T]]


M = {"d2_224": D / "r50_tta/kaggle_output/d2_ep15__none.json", "1b": C / "run_continue/val_swa.json",
     "2c": C / "run_r50_plane/val_swa.json", "1e": C / "run_r50_pertarget/val_swa.json",
     "1e_s27": C / "run_r50_pertarget_seed/val_swa.json", "1c": C / "run_r50/val_swa.json",
     "2a": C / "run_r50_ctx/val_swa.json", "3a": C / "run_r50_border/val_swa.json", "3b": C / "run_r50_state/val_swa.json",
     "2c_pt": C / "run_r50_plane_pertarget/val_swa.json", "native_pt": C / "run_r50_native_pertarget/val_swa.json",
     "5a_ssl": C / "run_r50_ssl_pertarget/val_swa.json", "6a_cyc": C / "run_r50_swa_cyclic/val_swa.json",
     "6b_alloc": C / "run_r50_series_alloc/val_swa.json", "v3": C / "run_r50_spatial/val_swa.json",
     "C0": C / "run_r50_os16_gap/val_swa.json", "v3_os32": C / "run_r50_spatial_os32/val_swa.json"}
names = list(M); P = np.stack([load(M[k]) for k in names]).astype(np.float32)      # [m, n, 12]
n = len(ids); L4 = ("d2_224", "1b", "2c", "1e")

subsets = [c for r in range(1, MAXK + 1) for c in itertools.combinations(range(len(names)), r)]
S = np.zeros((len(subsets), len(names)), np.float32)
for i, c in enumerate(subsets):
    S[i, list(c)] = 1
E = np.einsum("sm,mnk->snk", S, P) / S.sum(1)[:, None, None]


def auc_rows(s, idx, k):
    sel = idx[mask[idx, k]]; t = truth[sel, k]; n1 = t.sum(); n0 = len(t) - n1
    r = s[:, sel].argsort(1).argsort(1) + 1.0
    return (r[:, t].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def scores(idx, X=E):
    per = np.stack([auc_rows(X[..., k], idx, k) for k in range(12)], 1)
    return per.mean(1), per[:, TEN].mean(1), per


full = np.arange(n)
s12, s10, per = scores(full)
key = lambda ms: subsets.index(tuple(sorted(names.index(m) for m in ms)))
lab = lambda i: "+".join(names[j] for j in subsets[i])
size = S.sum(1).astype(int)
iL4 = key(L4)
print(f"{len(names)} candidates, {len(subsets)} subsets of <= {MAXK}. In-sample fold-0 macro (12-target / 10-target)")
print(f"  L4 {s12[iL4]:.4f} / {s10[iL4]:.4f}   [{lab(iL4)}]")
print("\nSingles:")
for m in names:
    i = key((m,)); print(f"  {m:10s} {s12[i]:.4f} / {s10[i]:.4f}")
print("\nBest subset per size (10-target, in-sample, optimistic):")
for r in range(1, MAXK + 1):
    i = np.where(size == r)[0][np.argmax(s10[size == r])]
    print(f"  {r} members: {s12[i]:.4f} / {s10[i]:.4f}  {lab(i)}")
print("\nTop 12 subsets overall (10-target):")
for i in np.argsort(-s10)[:12]:
    print(f"  {s12[i]:.4f} / {s10[i]:.4f}  ({size[i]}) {lab(i)}")

# ---- named candidates vs L4, paired bootstrap
rng = np.random.default_rng(0); boots = [rng.integers(0, n, n) for _ in range(2000)]
NAMED = {"L4: 1e -> v3_os32": ("d2_224", "1b", "2c", "v3_os32"),
         "L4 + v3_os32": L4 + ("v3_os32",),
         "L4 + 1e_s27": L4 + ("1e_s27",),
         "L4 + v3": L4 + ("v3",),
         "L4: 1e -> v3": ("d2_224", "1b", "2c", "v3"),
         "L4 + C0": L4 + ("C0",)}
best_by_size = {r: np.where(size == r)[0][np.argmax(s10[size == r])] for r in (3, 4, 5)}
idx_named = {k: key(v) for k, v in NAMED.items()} | {f"best-{r} in-sample": i for r, i in best_by_size.items()}
X = E[[iL4] + list(idx_named.values())]
bs = np.array([scores(b, X)[1] for b in boots])                                # [2000, 1 + named]
print("\nNamed candidates vs L4 (10-target, 2,000 paired bootstraps; best-k are selected in-sample, so optimistic):")
for j, (k, i) in enumerate(idx_named.items(), 1):
    d = bs[:, j] - bs[:, 0]
    print(f"  {k:22s} {s10[i]:.4f}  d {s10[i] - s10[iL4]:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] "
          f"{(d > 0).mean():.0%} better   {lab(i)}")

# ---- honest: select on half A, score on half B
g, picks, gn = [], [], {k: [] for k in NAMED}
for _ in range(200):
    perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
    a10 = scores(A)[1]; b10 = scores(B)[1]
    i = int(np.argmax(a10)); picks.append(lab(i)); g.append(b10[i] - b10[iL4])
    for k, v in NAMED.items():
        gn[k].append(b10[key(v)] - b10[iL4])
g = np.array(g)
print("\nHonest search (200 half splits, select best <=5 subset on A, score vs L4 on B, 10-target):")
print(f"  best-on-A vs L4 on B: mean {g.mean():+.4f} [{np.percentile(g, 2.5):+.4f}, {np.percentile(g, 97.5):+.4f}], better in {(g > 0).mean():.0%}")
for k, v in gn.items():
    v = np.array(v); print(f"  fixed {k:20s}: mean {v.mean():+.4f} [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}], better in {(v > 0).mean():.0%}")
print("  most frequent picks on A:", Counter(picks).most_common(5))
print("  member frequency in picks:", {m: round(sum(m in p.split("+") for p in picks) / len(picks), 2) for m in names})

# ---- diversity: rank correlation with 1e (mean over targets)
print("\nMean per-target Spearman vs 1e (lower = more diverse):")
rk = lambda a: a.argsort(0).argsort(0).astype(float)
R1 = rk(P[names.index("1e")])
for m in names:
    Rm = rk(P[names.index(m)])
    print(f"  {m:10s} {np.mean([np.corrcoef(Rm[:, k], R1[:, k])[0, 1] for k in range(12)]):.3f}")
