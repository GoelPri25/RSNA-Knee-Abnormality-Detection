"""Idea 2 (joint-centred input), step 1: is the knee off-centre in the d96 crops? Geometry only, no images, no GPU.

The cache crops a centred 140 mm square (eval: centre 320/384 px = 117 mm) from every series. Each series also places
the knee in 3D: its stack of slices plus its in-plane field. For each study, every slot's 3D box centre is computed
from DICOM geometry (ipp, iop, pixel spacing; rows/cols from series_fov.csv), and the consensus knee centre is the
per-axis median over the study's slots. Reported per slot: the in-plane offset (mm) of the crop centre from that
consensus, i.e. how far the crop is shifted relative to where the other planes put the knee. Also the direct
sagittal-vs-coronal superior-inferior disagreement (both planes contain S-I in-plane). This measures centring
consistency between planes; it does not locate the joint line from image content.

Also: effective input resolution vs CoPAS (Qiu et al. 2024: 448 px ROI at 0.28 mm resampled to 224 = 0.56 mm/px).

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_centring.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
fov = pd.read_csv(D / "external/rsna-knee-crop-fov-check/series_fov.csv").drop_duplicates("series").set_index("series")
NAMES = ["Sag fluid", "Sag non-fluid", "Cor fluid", "Cor non-fluid", "Axial"]
CROP, EVAL = 140.0, 140.0 * 320 / 384


def box_centre(slot):
    """Mean 3D image-centre over the picked slices, plus the in-plane unit axes (row dir, col dir) of the series."""
    f = fov.loc[slot["series"]]; rows, cols = int(f["rows"]), int(f["cols"])
    cs = []
    for g in slot["geometry"]:
        ipp, iop, ps = np.array(g["ipp"]), np.array(g["iop"]), g["ps"]
        cs.append(ipp + iop[:3] * ps[1] * (cols - 1) / 2 + iop[3:] * ps[0] * (rows - 1) / 2)
    iop = np.array(slot["geometry"][len(slot["geometry"]) // 2]["iop"])
    return np.mean(cs, 0), iop[:3], iop[3:], float(np.median([g["ps"][0] for g in slot["geometry"]])), rows * slot["geometry"][0]["ps"][0]


recs, skipped = [], 0
for mf in sorted((C / "manifests").glob("manifest_*.jsonl")):
    for line in mf.read_text().splitlines():
        m = json.loads(line)
        if m.get("gold"):
            continue
        boxes = {}
        for name, s in zip(NAMES, m["slots"]):
            if s.get("series") and s.get("geometry") and s["series"] in fov.index:
                boxes[name] = box_centre(s)
            elif s.get("series"):
                skipped += 1
        if len(boxes) < 3:
            continue
        cons = np.median(np.stack([b[0] for b in boxes.values()]), 0)
        sag = boxes.get("Sag fluid"); cor = boxes.get("Cor fluid")
        rec = {"study": m["study_id"], "si_sag_minus_cor": (sag[0][2] - cor[0][2]) if sag and cor else np.nan}
        for name, (c, u, v, ps, fv) in boxes.items():
            d = cons - c
            rec[name] = float(np.hypot(d @ u, d @ v)); rec[name + " fov"] = fv
        recs.append(rec)
df = pd.DataFrame(recs)
print(f"{len(df)} non-gold studies with >= 3 located slots; slots without geometry/fov skipped: {skipped}")
print(f"crop half-widths: train {CROP / 2:.1f} mm, eval {EVAL / 2:.1f} mm\n")
print("In-plane offset (mm) of each slot's crop centre from the consensus knee centre (median over the study's slots):")
rows = []
for n in NAMES:
    x = df[n].dropna()
    rows.append({"slot": n, "n": len(x), "median": x.median(), "p75": x.quantile(.75), "p90": x.quantile(.9),
                 "p99": x.quantile(.99), ">10mm": (x > 10).mean(), ">20mm": (x > 20).mean(), ">30mm": (x > 30).mean()})
t = pd.DataFrame(rows).set_index("slot")
for c in (">10mm", ">20mm", ">30mm"):
    t[c] = (t[c] * 100).round(1).astype(str) + "%"
print(t.round(1).to_string())
si = df["si_sag_minus_cor"].dropna()
print(f"\nSagittal fluid vs coronal fluid box centre, superior-inferior (mm): median {si.median():+.1f}, "
      f"|d| p50 {si.abs().median():.1f}, p90 {si.abs().quantile(.9):.1f}, >20 mm {(si.abs() > 20).mean():.1%}")
worst = df.assign(mx=df[NAMES].max(axis=1)).sort_values("mx", ascending=False)
print(f"studies with any slot > 30 mm off consensus: {(worst['mx'] > 30).mean():.1%}; top 5:")
print(worst[["study"] + NAMES].head(5).round(1).to_string(index=False))

print("\nEffective resolution (mm per input pixel):")
print(f"  ours: train {CROP / 384:.3f} (140 mm / 384 px), eval same scale on the 320 px centre ({EVAL:.0f} mm)")
print(f"  CoPAS: 0.28 mm resample, 448 px ROI -> 224 px = {0.28 * 448 / 224:.3f} mm/px (ROI {0.28 * 448:.0f} mm); "
      f"their no-ROI arm: 576 px ({0.28 * 576:.0f} mm) -> 224 = {0.28 * 576 / 224:.3f} mm/px")
