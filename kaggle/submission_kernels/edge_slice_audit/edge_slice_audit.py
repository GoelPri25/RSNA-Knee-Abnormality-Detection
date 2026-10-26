"""What lies in the slices the 20-80% central band drops? (CPU only, training studies, no model, no gold)

Uses the cache's own series selection, physical ordering, series-shared clip bounds and 140 mm / 224 px crop, but
renders every raw slice of each selected series instead of the 12 cached ones.

1. Tissue profile: for a fixed random sample of training studies, the share of supported crop pixels above a low
   intensity floor for every raw slice, binned by relative stack position per plane. Says whether the edges hold
   tissue or mostly background; it is not a diagnostic measure.
2. Montages: for one training study per target (first sorted study whose HPO v3.1-D2 training label is >= 0.9) plus
   two all-negative studies, every raw slice per series with its role marked: green = cached, grey = inside the band
   but skipped, red = outside the band.
Outputs in /kaggle/working/edge_audit/.
"""
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw

SOURCE = Path("/kaggle/input/datasets/tumis001/rsna-cnn-topk-support-source-v1/skanee")
DATA = next(p for p in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                        Path("/kaggle/input/rsna-knee-abnormality-detection")) if (p / "train_series.csv").is_file())
LABELS = next(Path("/kaggle/input").rglob("soft_labels_hpo_v3_1_d2_structured.json"))
OUT = Path("/kaggle/working/edge_audit"); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOURCE))
from src.features.dicom import decode_pixels, order_headers, series_bounds  # noqa: E402
from src.images.prepare import physical_crop_with_support, pixel_spacing, select_series, select_slice_indices, to_uint8  # noqa: E402

FOV, SIZE, SLOTS, PER_SLOT, BAND, CLIP = 140.0, 224, 4, 12, (0.2, 0.8), (1.0, 99.0)
PROFILE_STUDIES, THUMB, FLOOR, BINS = 150, 96, 25, 10

series_by_study = {}
with (DATA / "train_series.csv").open(newline="") as f:
    for r in csv.DictReader(f):
        series_by_study.setdefault(r["StudyInstanceUID"], []).append(r)
lab = json.loads(LABELS.read_text())
T = lab["targets_in_order"]
labels = {r["study_id"]: r["probabilities"] for r in lab["labels"] if r["study_id"] in series_by_study}
print(f"train studies with series: {len(series_by_study)}, labelled: {len(labels)}", flush=True)
pool = ThreadPoolExecutor(8)


def render_series(study, row):
    d = DATA / "train_series" / study / row["SeriesInstanceUID"]
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() == ".dcm")
    items = [(p, pydicom.dcmread(p, stop_before_pixels=True)) for p in paths]
    ordered = order_headers(items, 0)
    if ordered.order_source == "unordered" or not ordered.files:
        return None
    n = len(ordered.files)
    cached = set(select_slice_indices(n, PER_SLOT, *BAND))
    lo, hi = int(np.floor(n * BAND[0])), int(np.ceil(n * BAND[1]))

    def dec(p):
        ds = pydicom.dcmread(p)
        return decode_pixels(ds), getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1", pixel_spacing(ds)

    dl = list(pool.map(dec, ordered.files))
    low, high = series_bounds([x for x, _, _ in dl], *CLIP)
    sp0 = next((s for _, _, s in dl if s is not None), None)
    imgs, tissue = [], []
    for px, inv, sp in dl:
        sup = np.isfinite(px)
        im = to_uint8(px, low, high, inv); im[~sup] = 0
        c, cs = physical_crop_with_support(im.astype(np.float32), sup, sp if sp is not None else sp0, FOV, SIZE)
        imgs.append(c.round().astype(np.uint8))
        s = cs > 127
        tissue.append(float((c[s] > FLOOR).mean()) if s.any() else 0.0)
    roles = ["cached" if i in cached else ("band_skipped" if lo <= i < hi else "outside") for i in range(n)]
    return {"plane": row.get("Anatomical_Plane"), "fluid": row.get("Fluid_Sensitive"), "n": n,
            "imgs": imgs, "tissue": tissue, "roles": roles, "fallback": not (lo <= min(cached) and max(cached) < hi)}


# 1. tissue profile
rng = random.Random(2026)
sample = rng.sample(sorted(labels), PROFILE_STUDIES)
prof = []
for k, s in enumerate(sample):
    for row in select_series(series_by_study[s], SLOTS):
        try:
            r = render_series(s, row)
        except Exception as e:  # keep going; count failures
            print("fail", s[-8:], type(e).__name__, e, flush=True); continue
        if r is None:
            continue
        for i, (t, role) in enumerate(zip(r["tissue"], r["roles"])):
            prof.append({"study": s, "plane": r["plane"], "n": r["n"], "i": i, "rel": i / max(r["n"] - 1, 1),
                         "role": role, "tissue": t})
    if k % 25 == 0:
        print(f"profile {k}/{PROFILE_STUDIES}", flush=True)
with (OUT / "tissue_profile.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(prof[0])); w.writeheader(); w.writerows(prof)
summary = {}
for plane in sorted({p["plane"] for p in prof}):
    rows = [p for p in prof if p["plane"] == plane]
    bins = [[p["tissue"] for p in rows if min(int(p["rel"] * BINS), BINS - 1) == b] for b in range(BINS)]
    summary[plane] = {"median_tissue_by_decile": [round(float(np.median(b)), 3) if b else None for b in bins],
                      "median_tissue_by_role": {r: round(float(np.median([p["tissue"] for p in rows if p["role"] == r])), 3)
                                                for r in ("outside", "band_skipped", "cached")},
                      "share_outside_slices_with_tissue_ge_0.2": round(float(np.mean(
                          [p["tissue"] >= .2 for p in rows if p["role"] == "outside"])), 3)}
(OUT / "tissue_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1), flush=True)

# 2. montages
picks = []
for j, t in enumerate(T):
    s = next((s for s in sorted(labels) if labels[s][j] is not None and labels[s][j] >= .9 and s not in picks), None)
    if s:
        picks.append((t, s))
neg = [s for s in sorted(labels) if all(p is not None and p <= .1 for p in labels[s])][:2]
picks += [("all_negative", s) for s in neg]
COLOR = {"cached": (0, 200, 0), "band_skipped": (110, 110, 110), "outside": (230, 0, 0)}
index = []
for t, s in picks:
    rows = [r for r in (render_series(s, row) for row in select_series(series_by_study[s], SLOTS)) if r]
    width = max(r["n"] for r in rows)
    canvas = Image.new("RGB", (width * (THUMB + 4) + 4, len(rows) * (THUMB + 18) + 4), (0, 0, 0))
    dr = ImageDraw.Draw(canvas)
    for y, r in enumerate(rows):
        oy = 4 + y * (THUMB + 18)
        dr.text((4, oy), f"{r['plane']} fluid={r['fluid']} n={r['n']}", fill=(255, 255, 0))
        for x, (im, role) in enumerate(zip(r["imgs"], r["roles"])):
            ox = 4 + x * (THUMB + 4)
            dr.rectangle([ox - 2, oy + 12, ox + THUMB + 1, oy + 14 + THUMB + 1], outline=COLOR[role], width=2)
            canvas.paste(Image.fromarray(im).resize((THUMB, THUMB)).convert("RGB"), (ox, oy + 14))
    name = f"{len(index):02d}_{t.replace(' ', '_').replace(chr(39), '')}.png"
    canvas.save(OUT / name)
    index.append({"file": name, "target": t, "study_id": s,
                  "label_row": dict(zip(T, labels[s])), "series": [(r["plane"], r["n"]) for r in rows]})
(OUT / "montage_index.json").write_text(json.dumps(index, indent=2))
print("montages:", [i["file"] for i in index], flush=True)
