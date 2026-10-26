"""Native-capped d96 cache ("d96n"): the d96 corpus at the series' own slice density (CPU only).

Differences from the d96 cache (kaggle/experiment_kernels/coatnet_d96_cache_shard*/run.py), everything else equal:
  * series slots: Sagittal fluid | Sagittal non-fluid | Coronal fluid | Axial fluid-preferred (coronal non-fluid dropped:
    the series ablation showed it carries no signal; axial now prefers the fluid-sensitive series, as the other slots do);
  * slices: every native slice in the 2%..98% range of the sorted series; a series with more than CAP slices
    (thin-slice 3D acquisitions) is resampled evenly to CAP;
  * variable length: a study's slices are stored back to back (no padding, empty slots take no rows), minimum 3 rows.
    manifest: n (rows), slot_of (slot index per row), mask (row has any non-zero pixel), per-slot meta as before.
Same per-series 2/98 intensity window over the picked slices, MONOCHROME1 inversion, 140 mm centred crop, 384 px.

Output (/kaggle/working/d96n/): vol_NNN.npy uint8 [sum of n over the part's studies, 384, 384], manifest.jsonl.
Shard = sha256(study) % SHARDS.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SHARD, SHARDS = 2, 6
LIMIT = None                # small int for a smoke run, None for the full shard
IMG, CROP_MM, CAP, MIN_ROWS = 384, 140.0, 40, 3
SLOTS = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Axial", 1)]
STUDIES_PER_PART = 100
LABELS_SHA = "38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def pick(paths):
    found = [p for p in paths if p.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one input among {paths}; found {found}")
    return found[0]


DATA = pick([Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
             Path("/kaggle/input/rsna-knee-abnormality-detection")])
LABELS = pick([Path("/kaggle/input/datasets/tumis001/rsna-knee-hpo-v31-d2-report-labels/soft_labels_hpo_v3_1_d2_structured.json"),
               Path("/kaggle/input/rsna-knee-hpo-v31-d2-report-labels/soft_labels_hpo_v3_1_d2_structured.json")])
OUT = Path("/kaggle/working/d96n")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def header(path):
    import pydicom
    try:
        h = pydicom.dcmread(path, stop_before_pixels=True)
        iop = getattr(h, "ImageOrientationPatient", None)
        ipp = getattr(h, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r, c = np.array(iop[:3], float), np.array(iop[3:], float)
            pos = float(np.dot(np.array(ipp, float), np.cross(r, c)))
        else:
            pos = float(getattr(h, "InstanceNumber", 0) or 0)
        ps = getattr(h, "PixelSpacing", None)
        geo = {"ipp": [float(v) for v in ipp] if ipp is not None else None,
               "iop": [float(v) for v in iop] if iop is not None else None,
               "ps": [float(v) for v in ps] if ps is not None else None}
        lat = {k: str(getattr(h, k, "") or "") for k in ("Laterality", "ImageLaterality", "BodyPartExamined")}
        return pos, path, (float(ps[0]) if ps is not None else 0.5), True, geo, lat
    except Exception:
        return 0.0, path, 0.5, False, None, None


def pixels(path):
    import pydicom
    from pydicom.pixels import apply_modality_lut
    d = pydicom.dcmread(path)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def crop_resize(a, ps):
    import cv2
    h, w = a.shape
    cpx = min(int(round(CROP_MM / max(ps, 1e-3))), h, w)
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (IMG, IMG), interpolation=cv2.INTER_AREA)


def native_picks(n):
    """Every slice in 2%..98% (same bounds as d96); evenly resampled to CAP if more."""
    if n <= 1:
        return [0] * n
    lo, hi = int(n * 0.02), max(int(n * 0.98) - 1, int(n * 0.02))
    if hi - lo + 1 <= CAP:
        return list(range(lo, hi + 1))
    return [int(p) for p in np.linspace(lo, hi, CAP).round().astype(int)]


def build_study(job):
    study_id, rows = job
    slices, slot_of, used, slots_meta = [], [], set(), []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for j, (plane, fluid) in enumerate(SLOTS):
            cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
            pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
            row = (pref or cands or [None])[0]
            meta = {"plane": plane, "fluid_wanted": fluid, "row_start": len(slices), "k": 0, "series": None}
            slots_meta.append(meta)
            if row is None:
                continue
            used.add(row["SeriesInstanceUID"])
            files = sorted((DATA / "train_series" / study_id / row["SeriesInstanceUID"]).glob("*.dcm"))
            heads = list(pool.map(header, files))
            ok_ps = [h[2] for h in heads if h[3]]
            med_ps = float(np.median(ok_ps)) if ok_ps else 0.5
            heads.sort(key=lambda h: h[0])
            n = len(heads)
            meta.update(series=row["SeriesInstanceUID"], fluid=row.get("Fluid_Sensitive"),
                        fat=row.get("Fat_Suppression"), n_files=n,
                        laterality=next((h[5] for h in heads if h[5]), None))
            picks = native_picks(n)
            if not picks:
                continue
            meta.update(picks=picks, k=len(picks), geometry=[heads[p][4] for p in picks])

            def load(p):
                try:
                    return pixels(heads[p][1]), heads[p][2]
                except Exception:
                    return None, med_ps
            got = list(pool.map(load, picks))
            valid = [a for a, _ in got if a is not None]
            meta["decode_errors"] = len(got) - len(valid)
            loq, hiq = (np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0]) if valid else (0.0, 1.0))
            for a, ps in got:
                if a is None:
                    slices.append(np.zeros((IMG, IMG), np.uint8))
                else:
                    aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                    slices.append((crop_resize(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8))
                slot_of.append(j)
    while len(slices) < MIN_ROWS:                     # keep the window code valid for (near-)empty studies
        slices.append(np.zeros((IMG, IMG), np.uint8)); slot_of.append(-1)
    vol = np.stack(slices)
    mask = vol.reshape(len(vol), -1).max(1) > 0
    return study_id, vol, {"study_id": study_id, "n": len(vol), "slot_of": slot_of,
                           "mask": mask.astype(int).tolist(), "slots": slots_meta}


def main():
    started = time.monotonic()
    assert hashlib.sha256(LABELS.read_bytes()).hexdigest() == LABELS_SHA
    weak = {r["study_id"] for r in json.loads(LABELS.read_text())["labels"]}
    gold = {r["StudyInstanceUID"] for r in read_csv(DATA / "train.csv")
            if all(r.get(t, "") in {"0", "1"} for t in TARGETS)}
    assert len(weak) == 4349 and len(gold) == 58 and not weak & gold, (len(weak), len(gold))
    studies = sorted(s for s in weak | gold if int(hashlib.sha256(s.encode()).hexdigest(), 16) % SHARDS == SHARD)
    studies = studies[:LIMIT]
    by_study = {}
    for row in read_csv(DATA / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    print(f"shard {SHARD}/{SHARDS}: {len(studies)} studies ({len(gold & set(studies))} gold)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = (OUT / "manifest.jsonl").open("w", encoding="utf-8")
    part, buf, rows_in_buf, done = 0, [], 0, 0
    stats = {"studies": 0, "rows": 0, "empty_slots": 0, "decode_errors": 0, "masked_rows": 0, "capped_series": 0}

    def flush():
        nonlocal part, buf, rows_in_buf
        if buf:
            np.save(OUT / f"vol_{part:03d}.npy", np.concatenate(buf))
            part, buf, rows_in_buf = part + 1, [], 0

    with Pool(4) as procs:
        for study_id, vol, rec in procs.imap(build_study, [(s, by_study.get(s, [])) for s in studies], chunksize=1):
            rec.update(part=f"vol_{part:03d}.npy", row_start=rows_in_buf, gold=study_id in gold)
            manifest.write(json.dumps(rec) + "\n")
            buf.append(vol); rows_in_buf += len(vol)
            stats["studies"] += 1; stats["rows"] += len(vol)
            stats["empty_slots"] += sum(m["series"] is None for m in rec["slots"])
            stats["decode_errors"] += sum(m.get("decode_errors", 0) for m in rec["slots"])
            stats["masked_rows"] += len(vol) - sum(rec["mask"])
            stats["capped_series"] += sum(m.get("n_files", 0) > CAP + 2 and m["k"] == CAP for m in rec["slots"])
            if len(buf) == STUDIES_PER_PART:
                flush()
            done += 1
            if done % 50 == 0 or done == len(studies):
                print(f"{done}/{len(studies)} studies; {stats['rows']} rows; {time.monotonic() - started:.0f}s", flush=True)
    flush()
    manifest.close()
    summary = {"shard": [SHARD, SHARDS], "img": IMG, "crop_mm": CROP_MM, "cap": CAP, "slots": SLOTS, **stats,
               "rows_per_study_mean": round(stats["rows"] / max(stats["studies"], 1), 1),
               "parts": part, "seconds": round(time.monotonic() - started, 1),
               "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
