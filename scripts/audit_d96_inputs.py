"""What the d96 recipe actually feeds the model, per slot: sequence type picked vs available, slice coverage,
spacing, and physical field of view vs the native series FOV.

Inputs: d96 manifests (data/coatnet_d96_fold0/manifests/manifest_{0..3}.jsonl), the competition's
train_series.csv (data/competition_metadata/), native series FOVs (data/external/rsna-knee-crop-fov-check/series_fov.csv).

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_inputs.py /proj/repository
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "data"
rows = [json.loads(l) for i in range(4) for l in (D / f"coatnet_d96_fold0/manifests/manifest_{i}.jsonl").read_text().splitlines()]
series = pd.read_csv(D / "competition_metadata/train_series.csv")
fov = pd.read_csv(D / "external/rsna-knee-crop-fov-check/series_fov.csv").set_index("series")
avail = series.groupby(["StudyInstanceUID", "Anatomical_Plane"]).agg(
    n=("SeriesInstanceUID", "size"), fs=("Fluid_Sensitive", "max")).reset_index()
avail = {(r.StudyInstanceUID, r.Anatomical_Plane): (r.n, r.fs) for r in avail.itertuples()}
NAMES = ["Sag fluid", "Sag non-fluid", "Cor fluid", "Cor non-fluid", "Axial any"]
EVAL_MM = 140.0 * 320 / 384

print(f"{len(rows)} studies; native series per study: "
      f"{series.groupby('StudyInstanceUID').size().describe()[['mean', '50%', 'max']].round(1).to_dict()}")
print("Series inventory (plane x fluid x fat-sat):")
print(series.groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]).size().to_string())

for j, name in enumerate(NAMES):
    slots = [r["slots"][j] for r in rows]
    filled = [s for s in slots if s["series"]]
    fluid = Counter(str(s.get("fluid")) for s in filled)
    nfiles = np.array([s["n_files"] for s in filled])
    k = slots[0]["k"]
    # through-plane spacing between consecutive picks, and of the native series
    gaps, native = [], []
    for s in filled:
        g = [x for x in s.get("geometry", []) if x and x["ipp"] and x["iop"]]
        if len(g) >= 2:
            r_, c_ = np.array(g[0]["iop"][:3]), np.array(g[0]["iop"][3:]); nrm = np.cross(r_, c_)
            pos = np.array([np.dot(x["ipp"], nrm) for x in g])
            d = np.abs(np.diff(pos)); d = d[d > 1e-3]
            if len(d):
                gaps.append(np.median(d))
                step = np.median(np.diff(s["picks"])) or 1
                native.append(np.median(d) / step)
    fv = [fov.loc[s["series"], "fov_row_mm"] for s in filled if s["series"] in fov.index]
    print(f"\n[{name}] filled {len(filled)}/{len(slots)}; picked series Fluid_Sensitive: {dict(fluid)}")
    print(f"  native slices n: median {np.median(nfiles):.0f} (p10 {np.percentile(nfiles, 10):.0f}, p90 {np.percentile(nfiles, 90):.0f}); "
          f"k={k}; share of studies where k >= n (all slices kept): {(nfiles <= k).mean():.0%}")
    if gaps:
        print(f"  native slice spacing median {np.median(native):.1f} mm; gap between picked slices median "
              f"{np.median(gaps):.1f} mm (p90 {np.percentile(gaps, 90):.1f})")
    if fv:
        fv = np.array(fv)
        print(f"  native FOV median {np.median(fv):.0f} mm (p10 {np.percentile(fv, 10):.0f}, p90 {np.percentile(fv, 90):.0f}); "
              f"train crop 140 mm keeps {np.median(np.minimum(140, fv) / fv):.0%}; eval crop {EVAL_MM:.0f} mm keeps "
              f"{np.median(np.minimum(EVAL_MM, fv) / fv):.0%} of the side")

ax_pick_nonfs = ax_fs_avail = 0
for r in rows:
    s = r["slots"][4]
    n, fs = avail.get((r["study_id"], "Axial"), (0, 0))
    if s["series"] and str(s.get("fluid")) == "0" and fs == 1:
        ax_pick_nonfs += 1
    ax_fs_avail += fs == 1
print(f"\nAxial: fluid-sensitive axial available in {ax_fs_avail} studies; slot picked a NON-fluid axial although a "
      f"fluid one existed in {ax_pick_nonfs} ({ax_pick_nonfs / max(ax_fs_avail, 1):.0%})")
for j, plane in ((0, "Sagittal"), (2, "Coronal")):
    miss = sum(1 for r in rows if r["slots"][j]["series"] and str(r["slots"][j].get("fluid")) == "0"
               and avail.get((r["study_id"], plane), (0, 0))[1] == 1)
    print(f"{plane} fluid slot got a non-fluid series although fluid existed: {miss}")
