"""Pre-specified lean ensembles vs octa (fold 0, Steven-v4 confident cells, 2,000 paired bootstraps).

Candidates are fixed by principle, not by fold-0 search (the split-half check showed search buys ~nothing):
  L3   d2_224 + 1b_coat + 2c          one member per family: 2D quad CNN, CoAtNet d96, per-plane r50 d96
  L4   L3 + 1e                         + the best single on LB (0.936)
  D4   1b_coat + 1c + 1e + 2c          d96-only: one shared preprocessing pass, no quad code in the notebook
  D3   1b_coat + 1e + 2c               d96-only without 1c (2c is 1c's head variant)
Lean passes if 10-target delta vs octa >= -0.001 (equivalence margin), reported with its interval.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_lean_ensemble.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(S, idx):
    return np.array([auc(S[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(12)])


M = {"d2_224": D / "r50_tta/kaggle_output/d2_ep15__none.json", "hpo1_224": D / "r50_tta/kaggle_output/hpo_v1_ep17__none.json",
     "d2_336": D / "highres336/results/oof_predictions.json", "effnet": D / "efficientnet_v31_d2/results/oof_predictions.json",
     "1b_coat": C / "run_continue/val_swa.json", "1c": C / "run_r50/val_swa.json", "1e": C / "run_r50_pertarget/val_swa.json",
     "2c": C / "run_r50_plane/val_swa.json"}
P = {k: load(v) for k, v in M.items()}
ens = lambda ms: np.mean([P[m] for m in ms], 0)
OCTA = list(M); HEPTA = [m for m in OCTA if m != "2c"]
CAND = {"L3": ["d2_224", "1b_coat", "2c"], "L4": ["d2_224", "1b_coat", "2c", "1e"],
        "D4": ["1b_coat", "1c", "1e", "2c"], "D3": ["1b_coat", "1e", "2c"], "hepta": HEPTA}
rng = np.random.default_rng(0); n = len(ids); full = np.arange(n)
boots = [rng.integers(0, n, n) for _ in range(2000)]
O = ens(OCTA); po = per(O, full); bo = [per(O, b) for b in boots]
print(f"octa: 12-target {po.mean():.4f} / 10-target {po[TEN].mean():.4f}")
rows = {"octa": po}
for name, ms in CAND.items():
    E = ens(ms); pe = per(E, full); rows[name] = pe
    d = np.array([[(pb := per(E, b)).mean() - o.mean(), pb[TEN].mean() - o[TEN].mean()] for b, o in zip(boots, bo)])
    d10 = pe[TEN].mean() - po[TEN].mean()
    print(f"{name:6s} ({len(ms)}) 12t {pe.mean():.4f} d {pe.mean() - po.mean():+.4f} [{np.percentile(d[:, 0], 2.5):+.4f}, {np.percentile(d[:, 0], 97.5):+.4f}] | "
          f"10t {pe[TEN].mean():.4f} d {d10:+.4f} [{np.percentile(d[:, 1], 2.5):+.4f}, {np.percentile(d[:, 1], 97.5):+.4f}] "
          f"{(d[:, 1] > 0).mean():.0%} better | equiv (d10 >= -0.001): {'yes' if d10 >= -0.001 else 'no'}   {'+'.join(ms)}")
t = pd.DataFrame(rows, index=T); t.loc["MACRO10"] = t.iloc[TEN].mean()
print("\nPer target:"); print(t.round(4).to_string())
