"""Submit the pinned epoch-15 HPO-blend +281 Fit160 checkpoint."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


EXPECTED_CHECKPOINT_SHA256 = (
    "427dd72baa09185055b5b013e3b6a8e54c6f148e6a7dbbb630f2a9ed2a11a512"
)
EXPECTED_CHECKPOINT_EPOCH = 15
EXPECTED_REPORT_VALIDATION_AUC = 0.8818230427951721
EXPECTED_REPORT_VALIDATION_BCE = 0.5142054332386363
FIT160_CACHE = {
    "field_of_view_mm": 160.0,
    "fov_mode": "fit",
    "canonicalize_slice_direction": True,
    "image_size": 224,
    "series_slots": 4,
    "slices_per_slot": 12,
    "central_band": (0.2, 0.8),
    "clip_percentiles": (1.0, 99.0),
    "exclude_order_sources": ("unordered",),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_series_parallel(folder, plane, workers, pydicom, order_headers, ordered_series_type):
    """Read only this series' DICOM headers concurrently, preserving stable order."""
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


def physical_fit_with_support(image, support, spacing, minimum_fov_mm, size):
    """Fit the full acquired field into a >=160mm square canvas, preserving support."""
    from PIL import Image

    image = np.asarray(image, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    if image.ndim != 2 or image.shape != support.shape:
        raise ValueError("image and acquired-support mask must be matching 2D arrays")
    rows, columns = image.shape
    if spacing is None:
        canvas_rows = canvas_columns = max(rows, columns)
    else:
        row_mm, column_mm = map(float, spacing)
        if not (np.isfinite(row_mm) and np.isfinite(column_mm)
                and row_mm > 0 and column_mm > 0):
            raise ValueError(f"Invalid DICOM pixel spacing: {spacing}")
        canvas_fov_mm = max(minimum_fov_mm, rows * row_mm, columns * column_mm)
        canvas_rows = max(rows, int(round(canvas_fov_mm / row_mm)))
        canvas_columns = max(columns, int(round(canvas_fov_mm / column_mm)))
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


def main() -> None:
    started = time.monotonic()
    source = Path("/kaggle/input/datasets/tumis001/rsna-cnn-topk-support-source-v1/skanee")
    artifact_dir = Path(
        "/kaggle/input/datasets/tumis001/rsna-hpo-blend-fit160-best-epoch15"
    )
    checkpoint_path = artifact_dir / "checkpoint_best.pt"
    run_summary_path = artifact_dir / "run_summary.json"
    if not (source / "scripts/train_cnn_baseline.py").is_file():
        raise FileNotFoundError(f"Model source package is missing: {source}")
    if not checkpoint_path.is_file() or not run_summary_path.is_file():
        raise FileNotFoundError("Pinned Fit160 checkpoint or run summary is missing")
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    best = run_summary["best"]
    if best.get("epoch") != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError("Run summary does not identify the pinned epoch-15 checkpoint")
    if abs(best.get("macro_auc", -1) - EXPECTED_REPORT_VALIDATION_AUC) > 1e-9:
        raise ValueError("Pinned report-fold AUC does not match run summary")
    if abs(best.get("validation_bce", -1) - EXPECTED_REPORT_VALIDATION_BCE) > 1e-9:
        raise ValueError("Pinned validation BCE does not match run summary")
    if (run_summary.get("train_studies"), run_summary.get("validation_studies")) != (3524, 825):
        raise ValueError("Unexpected fixed-fold train/validation counts")

    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected checkpoint SHA-256: {checkpoint_hash}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError("Checkpoint epoch differs from the pinned epoch")
    if checkpoint.get("config", {}).get("model", {}).get("backbone") != "resnet34":
        raise ValueError("Checkpoint is not the expected ResNet-34")

    data_root = next((p for p in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ) if (p / "sample_submission.csv").is_file()), None)
    if data_root is None:
        raise FileNotFoundError("Competition sample_submission.csv was not attached")

    sys.path.insert(0, str(source))
    from src.features.dicom import (
        OrderedSeries, decode_pixels, order_headers, series_bounds,
    )
    from src.images.dataset import stack_channels
    from src.images.prepare import pixel_spacing, select_series, select_slice_indices, to_uint8
    from src.reports.schema import TARGETS
    from scripts.train_cnn_baseline import IMAGENET_MEAN, IMAGENET_STD, SliceModel
    import pydicom

    sample_rows = read_csv(data_root / "sample_submission.csv")
    expected_columns = ["StudyInstanceUID", *TARGETS]
    if not sample_rows or list(sample_rows[0]) != expected_columns:
        raise ValueError("Sample submission columns differ from the model target order")
    study_ids = [row["StudyInstanceUID"] for row in sample_rows]
    if len(set(study_ids)) != len(study_ids):
        raise ValueError("Duplicate study IDs in sample submission")
    fallback = {row["StudyInstanceUID"]: [float(row[target]) for target in TARGETS]
                for row in sample_rows}
    output_path = Path("/kaggle/working/submission.csv")
    # Preserve a valid scoreable file if an unexpected test-study issue occurs.
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(expected_columns)
        for study_id in study_ids:
            writer.writerow([study_id, *[f"{value:.8f}" for value in fallback[study_id]]])
    series_by_study: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(data_root / "test_series.csv"):
        series_by_study.setdefault(row["StudyInstanceUID"], []).append(row)

    model_config = checkpoint["config"]["model"]
    model = SliceModel(model_config["backbone"], False, model_config["dropout"],
                       len(TARGETS), model_config.get("top_k_fraction", 0.1))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # T4 convolution inference benefits from NHWC/channels-last tensors.
        # Keep autotuning off because study window counts vary. AMP remains
        # enabled; no graph compilation or engine build is needed.
        torch.backends.cudnn.benchmark = False
        model.to(device=device, memory_format=torch.channels_last).eval()
    else:
        torch.set_num_threads(2)
        model.to(device).eval()
    use_amp = device.type == "cuda"
    print(f"Pinned checkpoint SHA256={checkpoint_hash}; epoch={checkpoint['epoch']}; "
          f"device={device}; preprocessing=Fit160 fit-to-canvas", flush=True)

    predictions: dict[str, list[float]] = {}
    stats = {"studies": len(study_ids), "windows": 0,
             "decode_failures": 0, "empty_studies": 0}
    stage_seconds = {"header_io": 0.0, "pixel_decode": 0.0,
                     "fit_resize": 0.0, "model": 0.0}
    with ThreadPoolExecutor(max_workers=4) as io_workers, torch.inference_mode():
        for study_number, study_id in enumerate(study_ids, start=1):
            images, supports, slots = [], [], []
            selected = select_series(series_by_study.get(study_id, []), FIT160_CACHE["series_slots"])
            for slot, row in enumerate(selected):
                series_dir = data_root / "test_series" / study_id / row["SeriesInstanceUID"]
                timer = time.monotonic()
                ordered = read_series_parallel(
                    series_dir, row.get("Anatomical_Plane"), io_workers,
                    pydicom, order_headers, OrderedSeries,
                )
                stage_seconds["header_io"] += time.monotonic() - timer
                if ordered.order_source in FIT160_CACHE["exclude_order_sources"] or not ordered.files:
                    continue
                decoded = []
                selected_positions = select_slice_indices(
                    len(ordered.files), FIT160_CACHE["slices_per_slot"],
                    *FIT160_CACHE["central_band"]
                )

                def decode_one(position):
                    try:
                        dicom = pydicom.dcmread(ordered.files[position])
                        pixels = decode_pixels(dicom)
                        if pixels.ndim != 2:
                            raise ValueError(f"Expected 2D slice, got {pixels.shape}")
                        return (position, pixels,
                                getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1",
                                pixel_spacing(dicom), None)
                    except Exception as error:
                        return position, None, False, None, error

                timer = time.monotonic()
                decoded_results = list(io_workers.map(decode_one, selected_positions))
                stage_seconds["pixel_decode"] += time.monotonic() - timer
                for position, pixels, monochrome1, spacing_value, error in decoded_results:
                    if error is not None:
                        stats["decode_failures"] += 1
                        print(f"Decode failure {study_id}/{row['SeriesInstanceUID']} "
                              f"slice={position}: {type(error).__name__}: {error}", flush=True)
                    else:
                        decoded.append((pixels, monochrome1, spacing_value))
                if not decoded:
                    continue
                low, high = series_bounds([item[0] for item in decoded],
                                          *FIT160_CACHE["clip_percentiles"])
                spacing = next((item[2] for item in decoded if item[2] is not None), None)
                timer = time.monotonic()
                for pixels, monochrome1, slice_spacing in decoded:
                    support = np.isfinite(pixels)
                    image = to_uint8(pixels, low, high, monochrome1)
                    image[~support] = 0
                    fitted, fitted_support = physical_fit_with_support(
                        image, support,
                        slice_spacing if slice_spacing is not None else spacing,
                        FIT160_CACHE["field_of_view_mm"], FIT160_CACHE["image_size"]
                    )
                    images.append(fitted.round().astype(np.uint8))
                    supports.append(fitted_support)
                    slots.append(slot)
                stage_seconds["fit_resize"] += time.monotonic() - timer

            if not images:
                stats["empty_studies"] += 1
                predictions[study_id] = fallback[study_id]
                continue
            image_channels = stack_channels(np.stack(images), np.asarray(slots))
            support_channels = stack_channels(np.stack(supports), np.asarray(slots))
            model_input = np.concatenate((image_channels, support_channels), axis=1)
            batch = torch.from_numpy(np.ascontiguousarray(model_input)).to(
                device=device, dtype=torch.float32
            )
            batch[:, :3] = (batch[:, :3] / 255.0 - float(IMAGENET_MEAN)) / float(IMAGENET_STD)
            batch[:, 3:] /= 255.0
            if device.type == "cuda":
                batch = batch.contiguous(memory_format=torch.channels_last)
            owner = torch.zeros(len(batch), dtype=torch.long, device=device)
            timer = time.monotonic()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                # SliceModel already pools these windows into one study row.
                logits = model(batch, owner, 1).float().squeeze(0)
            stage_seconds["model"] += time.monotonic() - timer
            probability = torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist()
            if len(probability) != len(TARGETS) or not np.isfinite(probability).all():
                raise ValueError(f"Invalid predictions for {study_id}")
            if min(probability) < 0 or max(probability) > 1:
                raise ValueError(f"Out-of-range probabilities for {study_id}")
            predictions[study_id] = probability
            stats["windows"] += len(batch)
            if study_number % 100 == 0 or study_number == len(study_ids):
                print(f"{study_number}/{len(study_ids)} studies; "
                      f"{time.monotonic() - started:.1f}s elapsed", flush=True)

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(expected_columns)
        for study_id in study_ids:
            writer.writerow([study_id, *[f"{value:.8f}" for value in predictions[study_id]]])
    written = read_csv(output_path)
    if len(written) != len(sample_rows) or list(written[0]) != expected_columns:
        raise ValueError("Generated submission shape or target order is invalid")
    if [row["StudyInstanceUID"] for row in written] != study_ids:
        raise ValueError("Generated submission row order differs from sample submission")
    values = np.asarray([[float(row[target]) for target in TARGETS] for row in written])
    if values.shape != (len(study_ids), len(TARGETS)) or not np.isfinite(values).all():
        raise ValueError("Generated submission contains non-finite values")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("Generated probabilities must be in [0,1]")
    summary = {
        **stats, "device": str(device), "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
        "checkpoint_sha256": checkpoint_hash,
        "training_label_source": "HPO v1 blended report labels",
        "report_validation_macro_auc": EXPECTED_REPORT_VALIDATION_AUC,
        "report_validation_bce": EXPECTED_REPORT_VALIDATION_BCE,
        "training_studies": 3524, "validation_studies": 825,
        "preprocessing": "Fit160 fit-to-canvas, support-aware, 224x224",
        "stage_seconds": {key: round(value, 3) for key, value in stage_seconds.items()},
        "seconds": round(time.monotonic() - started, 1),
        "submission": str(output_path),
    }
    Path("/kaggle/working/inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
