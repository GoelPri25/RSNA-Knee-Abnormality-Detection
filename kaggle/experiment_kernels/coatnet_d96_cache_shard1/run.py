"""Experiment 1 cache: the public 0.949 CoAtNet corpus layout ("d96"), rebuilt on our studies (CPU only).

Per study: 96 uint8 slices at 384 px from 5 slots, as in the public recipe:
  Sagittal fluid-sensitive 26 | Sagittal non-fluid 22 | Coronal fluid 18 | Coronal non-fluid 12 | Axial any 18.
Series per slot: first unused series of that plane in CSV order, preferring the requested Fluid_Sensitive value.
Slices: sorted by position along the slice normal, k picks evenly over 2%..98% of the series.
Intensity: one 2/98 percentile window per series over its picked slices (after the modality LUT,
MONOCHROME1 inverted). Crop: centred square of 140 mm (row spacing), INTER_AREA resize to 384.
An empty slot leaves zeros; mask = slice has any non-zero pixel.

Also records, per picked slice, its DICOM geometry (position, orientation, spacing) and series laterality tags,
so experiment 2 can add 3D coordinates without rebuilding the cache.

Output (/kaggle/working/d96/): vol_NNN.npy uint8 [studies_in_part * 96, 384, 384] and manifest.jsonl
(one line per study: part, row_start, mask, slot series, picks, geometry). Shard = sha256(study) % SHARDS.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SHARD, SHARDS = 1, 4
LIMIT = None                     # set to a small int for a smoke run
IMG, CROP_MM = 384, 140.0
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(k for _, _, k in SLOTS)                       # 96
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
OUT = Path("/kaggle/working/d96")


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


def build_study(job):
    study_id, rows = job
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx, used, slots_meta = 0, set(), []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for plane, fluid, k in SLOTS:
            cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
            pref = [r for r in cands if fluid in (0, 1) and int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
            row = (pref or cands or [None])[0]
            meta = {"plane": plane, "fluid_wanted": fluid, "k": k, "row_start": idx, "series": None}
            slots_meta.append(meta)
            if row is None:
                idx += k
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
            if n == 0:
                idx += k
                continue
            lo, hi = int(n * 0.02), int(n * 0.98) - 1
            hi = max(hi, lo)
            picks = [int(min(p, n - 1)) for p in (np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k)]
            meta["picks"] = picks
            meta["geometry"] = [heads[p][4] for p in picks]

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
                if a is not None:
                    aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                    vol[idx] = (crop_resize(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8)
                idx += 1
    mask = (vol.reshape(MAXS, -1).max(1) > 0)
    return study_id, vol, {"study_id": study_id, "mask": mask.astype(int).tolist(), "slots": slots_meta}


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
    part, buf, done = 0, [], 0
    stats = {"studies": 0, "empty_slots": 0, "decode_errors": 0, "masked_slices": 0}

    def flush():
        nonlocal part, buf
        if buf:
            np.save(OUT / f"vol_{part:03d}.npy", np.concatenate(buf))
            part, buf = part + 1, []

    with Pool(4) as procs:
        for study_id, vol, rec in procs.imap(build_study, [(s, by_study.get(s, [])) for s in studies], chunksize=1):
            rec.update(part=f"vol_{part:03d}.npy", row_start=len(buf) * MAXS, gold=study_id in gold)
            manifest.write(json.dumps(rec) + "\n")
            buf.append(vol)
            stats["studies"] += 1
            stats["empty_slots"] += sum(m["series"] is None for m in rec["slots"])
            stats["decode_errors"] += sum(m.get("decode_errors", 0) for m in rec["slots"])
            stats["masked_slices"] += MAXS - sum(rec["mask"])
            if len(buf) == STUDIES_PER_PART:
                flush()
            done += 1
            if done % 50 == 0 or done == len(studies):
                print(f"{done}/{len(studies)} studies; {time.monotonic() - started:.0f}s", flush=True)
    flush()
    manifest.close()
    summary = {"shard": [SHARD, SHARDS], "img": IMG, "crop_mm": CROP_MM, "slots": SLOTS, **stats,
               "parts": part, "seconds": round(time.monotonic() - started, 1),
               "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
