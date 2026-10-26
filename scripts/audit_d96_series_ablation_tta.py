"""Score the Kaggle series-ablation / shift-TTA output (kaggle/submission_kernels/d96_r50_series_ablation_tta) for
Exp 1c and Exp 1e on fold 0 against Steven-v4 (confident cells).

1. Series ablation: per target, AUC with one series / plane removed from attention minus the full model's AUC.
   A large drop = that series carries information the model uses; ~0 = redundant or unused.
2. Shift TTA: mean probability over the centre view + 4 axis shifts (tta5) or + all 8 shifts (tta9), vs the centre
   view alone; paired bootstraps. Also: does this kernel's centre view reproduce the saved val_swa scores?

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_series_ablation_tta.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
z = np.load(D / "series_ablation_tta/kaggle_output/series_ablation_tta.npz")
ids = list(z["ids"]); V = list(z["variants"])
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; m = {r["study_id"]: r for r in ref["labels"]}
y = np.array([[.5 if p is None else p for p in m[s]["probabilities"]] for s in ids])
w = np.array([m[s]["weights"] for s in ids])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(S, idx):
    return np.array([auc(S[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(12)])


rng = np.random.default_rng(0); n = len(ids); full = np.arange(n)
boots = [rng.integers(0, n, n) for _ in range(2000)]


def compare(A, B):
    d = np.array([per(A, b).mean() - per(B, b).mean() for b in boots])
    return per(A, full).mean() - per(B, full).mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


sig = lambda a: 1 / (1 + np.exp(-a))
for key, run in (("1c", "run_r50"), ("1e", "run_r50_pertarget")):
    L = z[f"logits_{key}"]; P = sig(L)
    print(f"================ Exp {key}  (NaN cells: {int(np.isnan(L).sum())})")
    saved = json.loads((C / run / "val_swa.json").read_text())
    sm = {r["study_id"]: r["probabilities"] for r in saved["predictions"]}
    S0 = np.array([sm[s] for s in ids])
    print(f"centre view reproduces saved val_swa: macro {per(P[:, 0], full).mean():.4f} vs {per(S0, full).mean():.4f}; "
          f"max |dp| {np.nanmax(np.abs(P[:, 0] - S0)):.4f}")
    base = per(P[:, 0], full)
    tab = {"full": base}
    for j, v in enumerate(V):
        if v.startswith(("drop_", "only_")):
            tab[v] = per(P[:, j], full) - base
    t = pd.DataFrame(tab, index=T); t.loc["MACRO"] = t.mean()
    print("\nSeries ablation: AUC change when the series/plane is removed from attention (full = absolute AUC):")
    print(t.round(4).to_string())
    sh = [j for j, v in enumerate(V) if v.startswith("shift_")]
    tta5 = P[:, [0] + sh[:4]].mean(1); tta9 = P[:, [0] + sh].mean(1)
    for name, M in (("tta5 (centre + 4 axis shifts)", tta5), ("tta9 (centre + 8 shifts)", tta9)):
        d, lo, hi, s = compare(M, P[:, 0])
        print(f"\nshift {name}: {per(M, full).mean():.4f} vs centre {base.mean():.4f}, delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {s:.1%} better")
    pt = pd.DataFrame({"centre": base, "tta9": per(tta9, full)}, index=T); pt["delta"] = pt["tta9"] - pt["centre"]
    print(pt.round(4).to_string())
    print()
