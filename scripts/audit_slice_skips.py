"""SK-0: are v4's misses concentrated where the d96 cache dropped acquired slices? (radiologist: follow a structure
through consecutive slices; host meniscus rubric = tear on >= 2 slices.) The cache keeps at most 26 / 22 / 18 / 12 / 18
slices of SagF / SagNF / CorF / CorNF / Ax (evenly spaced picks), so thin-slice series lose slices.
Per study and slot: skip = 1 - distinct picks / n_files (0 = every acquired slice kept).
Per target, fold 0, label-v2 confident cells, v4 l4copy SWA fused (s2026 and s2027 averaged in rank):
  AUC in studies whose relevant slot has skip = 0 vs skip > 0 (1,000 study bootstraps of the difference)
  Spearman of each positive's rank among negatives (its AUC contribution) with skip.
Relevant slot: ACL / menisci / OA / Baker's / contusion / fracture / effusion / synovitis -> SagF; MCL -> CorF; also all.
Pre-registered decision (2026-10-19, before running): native gap-free windows become a lane if, for menisci or ACL
(SagF) or MCL (CorF), AUC(skip > 0) - AUC(skip = 0) <= -0.010 [amended after the first run found no skip = 0 study: highest vs lowest skip third] with the 95% interval below 0 on the averaged model,
and the same sign for both seeds. Otherwise slice skipping is not where v4 loses.
Usage (container): python -I audit_slice_skips.py <project root> <out dir> <v4 run root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, X, V = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); X.mkdir(parents=True, exist_ok=True)
rows = {}
for i in range(4):
    for l in (R / f"repository/data/coatnet_d96_fold0/manifests/manifest_{i}.jsonl").read_text().splitlines():
        m = json.loads(l); rows[m["study_id"]] = m
lv = json.load(open(R / "repository/data/coatnet_d96_fold0/assets/soft_labels_final_per_target_latD2.json")); T = lv["targets_in_order"]
lab = {r["study_id"]: np.array([.5 if p is None else p for p in r["probabilities"]]) for r in lv["labels"]}


def preds(seed):
    d = json.load(open(V / f"v4_l4copy_{seed}" / "val_swa_fused.json"))
    return {r["study_id"]: np.array(r["probabilities"]) for r in d["predictions"]}


P1, P2 = preds("s2026"), preds("s2027")
ids = sorted(set(P1) & set(rows)); n = len(ids)
Y = np.array([lab[s] for s in ids]); M = (Y <= .3) | (Y >= .7); TT = Y >= .7
S1 = np.array([P1[s] for s in ids]); S2 = np.array([P2[s] for s in ids])
rk = lambda S: pd.DataFrame(S).rank(pct=True).to_numpy()
SA = (rk(S1) + rk(S2)) / 2
NAMES = ["SagF", "SagNF", "CorF", "CorNF", "Ax"]
skip = np.full((n, 5), np.nan); nf = np.full((n, 5), np.nan)
for i, s in enumerate(ids):
    for j, sl in enumerate(rows[s]["slots"]):
        if sl.get("series") and sl.get("n_files"):
            nf[i, j] = sl["n_files"]; skip[i, j] = 1 - len(set(sl["picks"])) / sl["n_files"]
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


pr(f"{n} fold-0 studies")
for j, nm in enumerate(NAMES):
    v = skip[:, j]; ok = np.isfinite(v)
    pr(f"{nm:5s} present {ok.mean():.0%} | n_files p50 {np.nanmedian(nf[:, j]):.0f} | skip = 0 in {np.mean(v[ok] == 0):.0%}, "
       f"skip p50 {np.nanmedian(v):.2f} p90 {np.nanpercentile(v, 90):.2f}")
REL = {t: 0 for t in T}; REL["MCL"] = 2
rng = np.random.default_rng(2026); BS = [rng.integers(0, n, n) for _ in range(1000)]
res = {}
pr("\nper target (relevant slot): AUC lowest vs highest skip third (averaged model; s2026 / s2027 deltas), positives' rank vs skip Spearman")
for k, t in enumerate(T):
    for j in sorted({REL[t], 2 if REL[t] == 0 else 0}):            # relevant slot first in the decision; the other as context
        v = skip[:, j]; ok = np.isfinite(v) & M[:, k]
        q1, q2 = np.nanpercentile(v[ok], [100 / 3, 200 / 3])           # amended 2026-10-19: no study has skip = 0
        lo, hi = ok & (v <= q1), ok & (v > q2)                         # lowest vs highest third (same thresholds)
        if TT[lo, k].sum() < 10 or TT[hi, k].sum() < 10:
            pr(f"  {t:17s} {NAMES[j]:5s} too few positives in a stratum ({TT[lo, k].sum()} / {TT[hi, k].sum()})"); continue
        d = lambda S, idx=None: (auc(S[hi if idx is None else idx[hi[idx]], k], TT[hi if idx is None else idx[hi[idx]], k])
                                 - auc(S[lo if idx is None else idx[lo[idx]], k], TT[lo if idx is None else idx[lo[idx]], k]))
        bd = np.array([d(SA, b) for b in BS])
        pos = np.where(ok & TT[:, k])[0]; neg = SA[ok & ~TT[:, k], k]
        contrib = np.array([(neg < SA[i, k]).mean() for i in pos])
        sp = pd.Series(contrib).corr(pd.Series(v[pos]), method="spearman")
        res[(t, NAMES[j])] = (d(SA), np.percentile(bd, 97.5), d(S1), d(S2))
        pr(f"  {t:17s} {NAMES[j]:5s} skip=0 {auc(SA[lo, k], TT[lo, k]):.4f} (n+ {TT[lo, k].sum()}, skip <= {q1:.2f}) high {auc(SA[hi, k], TT[hi, k]):.4f} "
           f"(n+ {TT[hi, k].sum()}) d {d(SA):+.4f} [{np.percentile(bd, 2.5):+.4f}, {np.percentile(bd, 97.5):+.4f}] "
           f"s2026 {d(S1):+.4f} s2027 {d(S2):+.4f} | rho {sp:+.2f}")
pr("\n## pre-registered decision")
hit = []
for t, sl in (("Medial Meniscus", "SagF"), ("Lateral Meniscus", "SagF"), ("ACL", "SagF"), ("MCL", "CorF")):
    if (t, sl) in res:
        dv, up, a, b = res[(t, sl)]
        flag = dv <= -0.010 and up < 0 and a < 0 and b < 0
        pr(f"{t} {sl}: d {dv:+.4f} (upper {up:+.4f}) seeds {a:+.4f} / {b:+.4f} -> {'SKIP EFFECT' if flag else 'no'}")
        hit += [t] if flag else []
pr("DECISION: " + ("native gap-free windows lane for " + ", ".join(hit) if hit else "no lane: slice skipping is not where v4 loses"))
pr("Caveat: skip > 0 means thinner slices (more files) = different protocols / sites; a skip effect is confounded with protocol.")
(X / "audit_slice_skips.txt").write_text("\n".join(L) + "\n")
