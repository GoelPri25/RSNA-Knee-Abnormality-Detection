"""Is the d96 cache an information bottleneck for the KD student? (CPU, read-only; development evidence only.)
The student (PUB-KD-R ra / rc) mimics a Raptor teacher that read the native series; the d96 cache keeps at most
26 / 22 / 18 / 12 / 18 evenly spaced slices per slot (SK-0: skip p50 SagF 13%, CorF 40%, Ax 44%).
If cache loss limits mimicry, student-teacher rank disagreement should rise with the slot's skip, more than the
teacher's own internal disagreement (Raptor 384 vs 224, same native input) does.
Per fold-0 study: D(x, y) = mean over findings of |pct rank x - pct rank y|; Spearman with skip per slot, and the
disagreement in the highest vs lowest skip third (1,000 study bootstraps of the difference of differences).
Output data/pipeline_study/study_input_fidelity.txt.  Usage: python3 scripts/study_input_fidelity.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_predictor as LP  # noqa: E402

T, IDS, C = LP.T, LP.IDS, LP.C
OUT = LP.D / "pipeline_study"; OUT.mkdir(exist_ok=True)
rows = {}
for i in range(4):
    for l in (C / f"manifests/manifest_{i}.jsonl").read_text().splitlines():
        m = json.loads(l); rows[m["study_id"]] = m
ids = [s for s in IDS if s in rows]; ix = [LP.POS[s] for s in ids]; n = len(ids)
NAMES = ["SagF", "SagNF", "CorF", "CorNF", "Ax"]
skip = np.full((n, 5), np.nan)
for i, s in enumerate(ids):
    for j, sl in enumerate(rows[s]["slots"]):
        if sl.get("series") and sl.get("n_files"):
            skip[i, j] = 1 - len(set(sl["picks"])) / sl["n_files"]
S = C / "pubkd/students_r"
pred = {"ra": LP.load_pred(S / "pubkd_kd_ra/val_swa_global.json")[ix], "rc": LP.load_pred(S / "pubkd_kd_rc/val_swa_global.json")[ix],
        "v4": LP.load_pred(C / "run_r50_gl4_mlx/v4_l4copy_s2026/val_swa_global.json")[ix],
        "teacher": LP.PUB["rap"][ix], "r384": LP.PUB["r384"][ix], "r224": LP.PUB["r224"][ix]}
pct = {k: pd.DataFrame(v).rank(pct=True).to_numpy() for k, v in pred.items()}
D = lambda a, b, fs=range(12): np.abs(pct[a][:, fs] - pct[b][:, fs]).mean(1)
pairs = {"ra~teacher": D("ra", "teacher"), "rc~teacher": D("rc", "teacher"), "v4~teacher": D("v4", "teacher"),
         "r384~r224": D("r384", "r224"), "ra~rc": D("ra", "rc")}
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


sp = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())
pr(f"{n} fold-0 studies; D = mean |pct rank difference| over 12 findings (global-branch files: no crop placement effect)")
pr("mean D: " + "  ".join(f"{k} {v.mean():.4f}" for k, v in pairs.items()))
pr("\nSpearman(D, slot skip) per slot")
pr(f"{'pair':12s} " + " ".join(f"{x:>7s}" for x in NAMES))
for k, v in pairs.items():
    pr(f"{k:12s} " + " ".join(f"{sp(v[np.isfinite(skip[:, j])], skip[np.isfinite(skip[:, j]), j]):+7.3f}" for j in range(5)))
rng = np.random.default_rng(0); BS = [rng.integers(0, n, n) for _ in range(1000)]
pr("\nhighest - lowest skip third: D(ra~teacher) change minus D(r384~r224) change (student-specific input effect)")
for j, nm in enumerate(NAMES):
    v = skip[:, j]; ok = np.isfinite(v); q1, q2 = np.nanpercentile(v[ok], [100 / 3, 200 / 3])
    lo, hi = ok & (v <= q1), ok & (v > q2)
    f = lambda idx: ((pairs["ra~teacher"][idx][hi[idx]].mean() - pairs["ra~teacher"][idx][lo[idx]].mean())
                     - (pairs["r384~r224"][idx][hi[idx]].mean() - pairs["r384~r224"][idx][lo[idx]].mean()))
    b = np.array([f(i) for i in BS])
    pr(f"  {nm:5s} skip thirds <= {q1:.2f} / > {q2:.2f}: ra~teacher lo {pairs['ra~teacher'][lo].mean():.4f} hi {pairs['ra~teacher'][hi].mean():.4f} | "
       f"r384~r224 lo {pairs['r384~r224'][lo].mean():.4f} hi {pairs['r384~r224'][hi].mean():.4f} | diff-in-diff {f(np.arange(n)):+.4f} "
       f"[{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]")
REL = {"SagF": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "CorF": [1, 10, 11], "Ax": [7, 8, 9]}
pr("\nper finding group: Spearman(D_findings(ra~teacher), skip) vs same for r384~r224")
for nm, fs in REL.items():
    j = NAMES.index(nm); ok = np.isfinite(skip[:, j])
    a = sp(D("ra", "teacher", fs)[ok], skip[ok, j]); b = sp(D("r384", "r224", fs)[ok], skip[ok, j])
    pr(f"  {nm:5s} findings {[T[k][:6] for k in fs]}: ra~teacher {a:+.3f}  r384~r224 {b:+.3f}")
(OUT / "study_input_fidelity.txt").write_text("\n".join(L) + "\n")
