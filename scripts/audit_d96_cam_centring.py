"""Analyse the Kaggle CAM output of Exp 1c on fold 0 (kaggle/submission_kernels/d96_r50_cam_centring_audit).

For each target, on Steven-v4 confident-positive studies only: per slot, the positive part of the 12x12 CAM (full
384 input = 140 mm, 11.7 mm per cell) summed over that slot's windows. Reported:
  - slot share: share of the finding's attention mass per slot (which planes the model reads)
  - centroid offset (mm) of the CAM mass from the crop centre; share of studies > 20 mm
  - mass outside the eval crop (the outer ring of cells; eval = centre 320 px = 117 mm)
  - r90: half-width (mm) of the smallest centred square holding 90% of the mass
A joint-centred crop would help if evidence were often far off-centre or spread to the edges.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_cam_centring.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"
z = np.load(D / "cam_centring_audit/kaggle_output/cam_fold0.npz")
summ = json.loads((D / "cam_centring_audit/kaggle_output/cam_summary.json").read_text())
print("kernel summary:", json.dumps(summ))
ids = list(z["ids"]); cam = z["cam"].astype(np.float32); att = z["att"]; ok = z["ok"]
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; m = {r["study_id"]: r for r in ref["labels"]}
y = np.array([[.5 if p is None else p for p in m[s]["probabilities"]] for s in ids])
w = np.array([m[s]["weights"] for s in ids])
pos = (w > 0) & (y >= .7) & ok[:, None]
SLOTS = ["Sag fluid", "Sag non-fluid", "Cor fluid", "Cor non-fluid", "Axial"]
cell = summ["field_mm"] / 12; ctr = (np.arange(12) - 5.5) * cell
YY, XX = np.meshgrid(ctr, ctr, indexing="ij"); RING = np.maximum(np.abs(YY), np.abs(XX))
EVAL_HALF = summ["field_mm"] * 320 / 384 / 2
outer = RING > EVAL_HALF
print(f"{ok.sum()} studies with CAMs; cell {cell:.1f} mm; eval half-width {EVAL_HALF:.1f} mm "
      f"(outer ring = {outer.sum()} of 144 cells)\n")


def stats(P):
    """P: [12,12] non-negative mass. centroid offset (mm), outer-ring share, r90 half-width (mm)."""
    s = P.sum()
    if s <= 0:
        return np.nan, np.nan, np.nan
    cy, cx = (P * YY).sum() / s, (P * XX).sum() / s
    rs = np.sort(np.unique(RING)); cum = np.array([P[RING <= r].sum() / s for r in rs])
    return float(np.hypot(cy, cx)), float(P[outer].sum() / s), float(rs[np.argmax(cum >= .9)] + cell / 2)


rows, share_rows = [], []
for k, t in enumerate(T):
    idx = np.where(pos[:, k])[0]
    share = att[idx, k].sum(0) / max(att[idx, k].sum(), 1e-9)
    share_rows.append({"target": t, "n_pos": len(idx), **{s: round(float(v), 2) for s, v in zip(SLOTS, share)}})
    for j, s in enumerate(SLOTS):
        st = np.array([stats(np.clip(cam[i, k, j].reshape(12, 12), 0, None)) for i in idx])
        wt = att[idx, k, j]                                        # weight studies by how much this slot is used
        good = np.isfinite(st[:, 0]) & (wt > 0)
        if good.sum() < 5:
            continue
        st, wt = st[good], wt[good]
        rows.append({"target": t, "slot": s, "slot_share": share[j], "centroid_mm_med": np.median(st[:, 0]),
                     ">20mm": float((wt * (st[:, 0] > 20)).sum() / wt.sum()),
                     "mass_outside_eval": float((wt * st[:, 1]).sum() / wt.sum()),
                     "r90_mm_med": np.median(st[:, 2])})
print("Slot share of each finding's attention (confident-positive studies):")
print(pd.DataFrame(share_rows).set_index("target").to_string())
df = pd.DataFrame(rows)
top = df.sort_values("slot_share", ascending=False).groupby("target").head(2).sort_values(["target", "slot_share"], ascending=[True, False])
print("\nTwo most-used slots per target: CAM centring (positives; studies weighted by the slot's attention):")
print(top.round(3).to_string(index=False))
print("\nAll target x slot (attention-weighted means over targets):")
agg = df.assign(wt=df.slot_share).groupby("slot").apply(
    lambda g: pd.Series({c: np.average(g[c], weights=g.wt) for c in ("centroid_mm_med", ">20mm", "mass_outside_eval", "r90_mm_med")}))
print(agg.round(3).to_string())
