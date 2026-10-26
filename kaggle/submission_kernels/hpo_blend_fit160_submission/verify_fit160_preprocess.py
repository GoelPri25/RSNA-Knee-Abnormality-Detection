"""CPU-only parity check against the teammate's cached gold images.

This reads training DICOM pixels only to verify preprocessing parity. It never
loads train.csv labels or sends predictions to the competition.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def find_data_root() -> Path:
    candidates = (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    )
    for root in candidates:
        if (root / "train_series.csv").is_file():
            return root
    raise FileNotFoundError("Competition train_series.csv is unavailable")


def find_cache_root() -> Path:
    # Kaggle's attached-kernel mount has used these two layouts. Check only
    # fixed paths; never walk the competition's DICOM tree.
    roots = (
        Path("/kaggle/input/kernels/priyanjaliml/rsna-knee-fit160-cache-gold"),
        Path("/kaggle/input/notebooks/priyanjaliml/rsna-knee-fit160-cache-gold"),
    )
    for root in roots:
        for candidate in (root, root / "slice_cache", root / "output" / "slice_cache",
                          root / "gold_download" / "slice_cache"):
            if (candidate / "cache_summary.json").is_file():
                return candidate
    raise FileNotFoundError(
        "Fit160 gold cache not found at fixed attached-kernel paths: "
        + ", ".join(str(p) for p in roots)
    )


def physical_fit_with_support(image, support, spacing, minimum_fov_mm, size):
    from PIL import Image

    image = np.asarray(image, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    if image.ndim != 2 or image.shape != support.shape:
        raise ValueError("image and support must be same-size 2D arrays")
    rows, columns = image.shape
    if spacing is None:
        canvas_rows = canvas_columns = max(rows, columns)
    else:
        row_mm, column_mm = map(float, spacing)
        if not (np.isfinite(row_mm) and np.isfinite(column_mm)
                and row_mm > 0 and column_mm > 0):
            raise ValueError(f"Invalid pixel spacing: {spacing}")
        fov = max(float(minimum_fov_mm), rows * row_mm, columns * column_mm)
        canvas_rows = max(rows, int(round(fov / row_mm)))
        canvas_columns = max(columns, int(round(fov / column_mm)))
    top, left = (canvas_rows - rows) // 2, (canvas_columns - columns) // 2
    canvas = np.zeros((canvas_rows, canvas_columns), dtype=np.float32)
    mask = np.zeros((canvas_rows, canvas_columns), dtype=np.uint8)
    canvas[top:top + rows, left:left + columns] = image
    mask[top:top + rows, left:left + columns] = support
    resized = np.asarray(
        Image.fromarray(canvas).resize((size, size), Image.BILINEAR), dtype=np.float32
    )
    resized_support = np.asarray(
        Image.fromarray(mask * 255).resize((size, size), Image.NEAREST), dtype=np.uint8
    )
    return resized, resized_support


def read_series_parallel(folder, plane, workers, pydicom, order_headers, ordered_series_type):
    paths = [path for path in sorted(folder.iterdir())
             if path.is_file() and path.suffix.lower() == ".dcm"]
    if not paths:
        return ordered_series_type([], [], "unordered", 0, ["no_readable_headers"])

    def header(path):
        try:
            return path, pydicom.dcmread(path, stop_before_pixels=True), None
        except Exception as error:
            return path, None, error

    results = list(workers.map(header, paths))
    items = [(path, dataset) for path, dataset, error in results if error is None]
    errors = len(results) - len(items)
    if not items:
        return ordered_series_type([], [], "unordered", errors, ["no_readable_headers"])
    return order_headers(items, errors, plane=plane, canonicalize=True)


def main() -> None:
    data_root = find_data_root()
    source = Path("/kaggle/input/datasets/tumis001/rsna-cnn-topk-support-source-v1/skanee")
    if not (source / "src/features/dicom.py").is_file():
        raise FileNotFoundError(f"Expected source at fixed Kaggle path: {source}")
    cache_root = find_cache_root()
    cache_meta = json.loads((cache_root / "cache_summary.json").read_text())
    cache = cache_meta["cache"]
    if (cache.get("field_of_view_mm"), cache.get("fov_mode"),
            cache.get("canonicalize_slice_direction"), cache.get("image_size")) != (
            160.0, "fit", True, 224):
        raise ValueError(f"Unexpected reference cache config: {cache}")

    sys.path.insert(0, str(source))
    import pydicom
    from src.features.dicom import (
        OrderedSeries, decode_pixels, order_headers, series_bounds,
    )
    from src.images.prepare import pixel_spacing, select_slice_indices, to_uint8

    manifest_path = cache_root / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    image_cache = np.load(cache_root / "slices_000.npy", mmap_mode="r")
    support_cache = np.load(cache_root / "support_000.npy", mmap_mode="r")
    compared = 0
    image_mismatches = support_mismatches = ordering_mismatches = 0
    max_abs = 0
    workers = ThreadPoolExecutor(max_workers=4)
    for index, record in enumerate(records, start=1):
        if not record["count"]:
            continue
        series_dir = data_root / "train_series" / record["study_id"] / record["series_id"]
        ordered = read_series_parallel(
            series_dir, record["plane"], workers, pydicom, order_headers, OrderedSeries
        )
        if ordered.order_source != "physical" or not ordered.files:
            ordering_mismatches += 1
            continue
        chosen = select_slice_indices(
            len(ordered.files), cache["slices_per_slot"], *cache["central_band"]
        )
        decoded = []
        for position in chosen:
            dicom = pydicom.dcmread(ordered.files[position])
            pixels = decode_pixels(dicom)
            decoded.append((
                position, pixels,
                getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1",
            ))
        if [ordered.files[pos].name for pos, _, _ in decoded] != record["files"]:
            ordering_mismatches += 1
            continue
        low, high = series_bounds(
            [pixels for _, pixels, _ in decoded], *cache["clip_percentiles"]
        )
        spacing = record.get("spacing_mm")
        spacing = tuple(spacing) if spacing else None
        for local_index, (_, pixels, inverted) in enumerate(decoded):
            start = record["row_start"] + local_index
            expected = image_cache[start]
            expected_support = support_cache[start]
            valid = np.isfinite(pixels)
            image = to_uint8(pixels, low, high, inverted)
            image[~valid] = 0
            generated, generated_support = physical_fit_with_support(
                image.astype(np.float32), valid, spacing,
                cache["field_of_view_mm"], cache["image_size"],
            )
            generated = generated.round().astype(np.uint8)
            compared += 1
            if not np.array_equal(generated, expected):
                image_mismatches += 1
                max_abs = max(max_abs, int(np.abs(
                    generated.astype(np.int16) - expected.astype(np.int16)
                ).max()))
            if not np.array_equal(generated_support, expected_support):
                support_mismatches += 1
        if index % 20 == 0 or index == len(records):
            print(f"checked {index}/{len(records)} cache slots; slices={compared}", flush=True)
    workers.shutdown(wait=True)

    result = {
        "cache_studies": cache_meta["studies_in_manifest"],
        "cache_slots": len(records),
        "slices_compared": compared,
        "image_slice_mismatches": image_mismatches,
        "support_slice_mismatches": support_mismatches,
        "ordering_mismatches": ordering_mismatches,
        "max_abs_intensity_difference": max_abs,
        "labels_read": False,
        "parity_pass": compared == int(cache_meta["run_stats"]["slices"])
            and image_mismatches == 0 and support_mismatches == 0
            and ordering_mismatches == 0,
    }
    print(json.dumps(result, indent=2), flush=True)
    Path("/kaggle/working/fit160_preprocess_parity.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if not result["parity_pass"]:
        raise RuntimeError("Fit160 preprocessing did not match the reference cache")


if __name__ == "__main__":
    main()
