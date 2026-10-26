"""Build deduplicated native-neighbor slices; shared preprocessing for inference.

Requires the pinned baseline source on PYTHONPATH. No GPU or label inference.
Each committed part contains complete studies; manifest replacement is atomic.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from src.features.dicom import decode_pixels, normalized_positions, read_series, series_bounds, slice_normal
from src.images.prepare import physical_crop_with_support, pixel_spacing, select_series, select_slice_indices, to_uint8
from src.reports.schema import TARGETS
from local_global_data import SCHEMA


def triplets(centres, count):
    centres = np.asarray(centres, dtype=np.int64)
    return np.clip(centres[:, None] + np.array([-1, 0, 1]), 0, count - 1)


def acquisition_metadata(dicom, job):
    normal = slice_normal(dicom)
    plane = None
    if normal is not None and np.max(np.abs(normal)) >= 0.8:
        plane = ("Sagittal", "Coronal", "Axial")[np.argmax(np.abs(normal))]
    fat = job.get("fat_suppression")
    if str(fat) not in ("0", "1"):
        # Missing FS tags are unknown, not proof that fat was not suppressed.
        options = getattr(dicom, "ScanOptions", [])
        options = [options] if isinstance(options, str) else list(options)
        suppression = str(getattr(dicom, "SpectrallySelectedSuppression", "")).upper()
        fat = "1" if "FS" in options or suppression == "FAT" else ("0" if suppression == "NONE" else None)
    return {"plane": plane, "reported_plane": job.get("plane"), "fat_suppression": fat}


def prepare_slot(job, cache):
    import pydicom
    ordered = read_series(Path(job["folder"]))
    size = cache["image_size"]
    empty = np.zeros((0, size, size), dtype=np.uint8)
    record = {**{k: job.get(k) for k in ("study_id", "series_id", "slot", "plane", "fluid_sensitive", "fat_suppression")},
              "schema": SCHEMA, "order_source": ordered.order_source, "exceptions": ordered.exceptions,
              "decode_errors": 0, "count": 0, "stored_count": 0, "positions": [], "triplet_offsets": [],
              "center_indices": [], "native_indices": [], "files": [], "center_files": [],
              "neighbor_gaps_mm": [], "spacing_mm": None}
    if ordered.order_source in cache["exclude_order_sources"] or not ordered.files:
        return record, empty, empty
    centres = select_slice_indices(len(ordered.files), cache["slices_per_slot"], *cache["central_band"])
    indices = np.unique(triplets(centres, len(ordered.files)))
    decoded = {}
    spacing, first = None, None
    for index in indices:
        try:
            d = pydicom.dcmread(ordered.files[index])
            pixels = decode_pixels(d)
            if pixels.ndim != 2:
                raise ValueError("Expected 2D DICOM slice")
            decoded[int(index)] = (pixels, getattr(d, "PhotometricInterpretation", "") == "MONOCHROME1")
            # Match baseline spacing reference: first successful selected centre.
            if index in centres and first is None:
                first, spacing = d, pixel_spacing(d)
        except Exception:
            record["decode_errors"] += 1
    centres = [i for i in centres if i in decoded]
    if not centres:
        return record, empty, empty
    mapping = triplets(centres, len(ordered.files))
    indices = np.unique(mapping)
    row_of_index = {int(v): i for i, v in enumerate(indices)}
    # Preserve the original percentile reference, NOT all newly decoded neighbors.
    low, high = series_bounds([decoded[i][0] for i in centres], *cache["clip_percentiles"])
    images, masks = [], []
    for index in indices:
        if int(index) not in decoded:
            images.append(np.zeros((size, size), dtype=np.uint8))
            masks.append(np.zeros((size, size), dtype=np.uint8))
            continue
        pixels, inverted = decoded[int(index)]
        support = np.isfinite(pixels)
        image = to_uint8(pixels, low, high, inverted)
        image[~support] = 0
        image, support = physical_crop_with_support(image.astype(np.float32), support, spacing,
                                                    cache["field_of_view_mm"], size)
        images.append(image.round().astype(np.uint8)); masks.append(support)
    positions = normalized_positions(ordered.coordinates)
    coords = np.asarray(ordered.coordinates)
    gaps = np.abs(coords[mapping[:, [0, 2]]] - coords[np.asarray(centres), None])
    if ordered.order_source != "physical":
        gaps[:] = 0  # InstanceNumber differences are not millimetres.
    record.update(acquisition_metadata(first, job))
    record.update(count=len(centres), stored_count=len(indices), positions=[positions[i] for i in centres],
                  triplet_offsets=[[row_of_index[int(v)] for v in row] for row in mapping],
                  center_indices=centres, native_indices=indices.tolist(),
                  files=[ordered.files[i].name for i in indices], center_files=[ordered.files[i].name for i in centres],
                  neighbor_gaps_mm=gaps.tolist(), spacing_mm=list(spacing) if spacing else None,
                  clip_bounds=[low, high], missing_native_indices=[int(i) for i in indices if int(i) not in decoded])
    return record, np.stack(images), np.stack(masks)


class SlotDataset:
    def __init__(self, jobs, cache): self.jobs, self.cache = jobs, cache
    def __len__(self): return len(self.jobs)
    def __getitem__(self, index): return prepare_slot(self.jobs[index], self.cache)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--studies-per-part", type=int, default=50)
    p.add_argument("--time-budget-hours", type=float, default=10.5)
    a = p.parse_args()
    if not 0 <= a.shard_index < a.shard_count: raise ValueError("Invalid shard")
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    config = json.loads(a.config.read_text()); cache = config["cache"]
    out = a.output_dir; out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    records = [json.loads(s) for s in manifest_path.read_text().splitlines() if s] if manifest_path.is_file() else []
    identity = {"schema": SCHEMA, "config_sha256": hashlib.sha256(a.config.read_bytes()).hexdigest(),
                "shard": [a.shard_index, a.shard_count], "split": a.split}
    identity_path = out / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity: raise ValueError("Existing cache identity mismatch")
    atomic_json(identity_path, identity)
    studies = read_csv(a.data_root / f"{a.split}.csv")
    if a.split == "train": studies = [r for r in studies if not all(r.get(t, "") in {"0", "1"} for t in TARGETS)]
    expected = sorted(r["StudyInstanceUID"] for r in studies if int(hashlib.sha256(r["StudyInstanceUID"].encode()).hexdigest(), 16) % a.shard_count == a.shard_index)
    done = {r["study_id"] for r in records}
    if not done.issubset(expected): raise ValueError("Manifest has unexpected studies")
    by_study = {}
    for r in read_csv(a.data_root / f"{a.split}_series.csv"): by_study.setdefault(r["StudyInstanceUID"], []).append(r)
    jobs = []
    for sid in expected:
        if sid in done: continue
        selected = select_series(by_study.get(sid, []), cache["series_slots"])
        if not selected: raise ValueError(f"No selected series: {sid}")
        for slot, r in enumerate(selected):
            jobs.append({"study_id": sid, "series_id": r["SeriesInstanceUID"], "slot": slot,
                         "folder": str(a.data_root / f"{a.split}_series" / sid / r["SeriesInstanceUID"]),
                         "plane": r.get("Anatomical_Plane"), "fluid_sensitive": r.get("Fluid_Sensitive"),
                         "fat_suppression": r.get("Fat_Suppression")})
    print(f"NATIVE_CACHE shard={a.shard_index}/{a.shard_count} expected={len(expected)} jobs={len(jobs)}", flush=True)
    # conversion to numpy in the parent avoids shared float tensor buffers.
    loader = torch.utils.data.DataLoader(SlotDataset(jobs, cache), batch_size=None, num_workers=a.workers,
                                        collate_fn=lambda x: x, prefetch_factor=1 if a.workers else None)
    part_index = 1 + max([int(r["part"].split("_")[1].split(".")[0]) for r in records], default=-1)
    pending, images, masks, stored, current = [], [], [], 0, None

    def flush():
        nonlocal pending, images, masks, stored, part_index, records
        if not pending: return
        names = [f"{kind}_{part_index:03d}.npy" for kind in ("slices", "support")]
        for name, buffers in zip(names, (images, masks)):
            # Disk-backed concatenation bounds RAM to the buffered input arrays.
            temp = out / (name + ".tmp")
            array = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8, shape=(stored, cache["image_size"], cache["image_size"]))
            row = 0
            for block in buffers: array[row:row + len(block)] = block; row += len(block)
            array.flush(); del array
            temp.replace(out / name)
        new = [{**r, "part": names[0], "support_part": names[1]} for r in pending]
        temp = manifest_path.with_suffix(".jsonl.tmp")
        with temp.open("w") as f:
            for r in records + new: f.write(json.dumps(r, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        temp.replace(manifest_path)
        records += new
        print(f"COMMITTED part={part_index} studies={len({r['study_id'] for r in records})}/{len(expected)} stored={stored}", flush=True)
        pending, images, masks, stored = [], [], [], 0; part_index += 1

    for r, x, support in loader:
        if r["study_id"] != current:
            if time.monotonic() - started > a.time_budget_hours * 3600: break
            if len({item["study_id"] for item in pending}) >= a.studies_per_part: flush()
            current = r["study_id"]
        r["row_start"] = stored
        pending.append(r); images.append(x); masks.append(support); stored += len(x)
    flush()
    complete = {r["study_id"] for r in records} == set(expected)
    summary = {**identity, "cache": cache, "complete": complete, "studies": len({r["study_id"] for r in records}),
               "windows": sum(r["count"] for r in records), "stored_slices": sum(r["stored_count"] for r in records),
               "decode_errors": sum(r["decode_errors"] for r in records), "seconds": time.monotonic() - started}
    atomic_json(out / "cache_summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not complete: raise RuntimeError("Cache incomplete; resume from committed full studies")


if __name__ == "__main__": main()
