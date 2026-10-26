"""Most-confident member of three pinned ResNet50 checkpoints (per study and target, the probability furthest from 0.5)."""

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


MODEL_SPECS = [
    {"name": "v31_d2_224", "dataset": "rsna-hpo-v31-d2-resnet50-best",
     "sha256": "4091948e1ce9000e8f4ab7dbd7d859de9370a31d3fe09bd09bf99eaf45d8e48c",
     "epoch": 15, "version": "hpo_v31_d2_resnet50_fold0", "size": 224},
    {"name": "v31_d2_336", "dataset": "rsna-resnet50-v31-trainval-336px-fold0",
     "sha256": "51aa5e5b907bfce0f0ce770dd70f546b0189ff3d0b8687358459d63958fbe9d7",
     "epoch": 17, "version": "hpo_v31_trainval_resnet50_336px", "size": 336},
    {"name": "hpo_v1_224", "dataset": "rsna-hpo-blend-add281-resnet50-fold0",
     "sha256": "7c76e8760b6989fb5bb3afa2fe14cf016ac766981ad612c94fc058acb703120a",
     "epoch": 17, "version": "cnn_topk_support_v1_resnet50", "size": 224},
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    started = time.monotonic()
    source = Path("/kaggle/input/datasets/tumis001/rsna-cnn-topk-support-source-v1/skanee")
    if not (source / "scripts/train_cnn_baseline.py").is_file():
        raise FileNotFoundError(f"Missing architecture source: {source}")
    data_root = next((path for path in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ) if (path / "sample_submission.csv").is_file()), None)
    if data_root is None:
        raise FileNotFoundError("Competition sample_submission.csv was not attached")

    sys.path.insert(0, str(source))
    from src.features.dicom import (
        OrderedSeries, decode_pixels, order_headers, series_bounds,
    )
    from src.images.dataset import stack_channels
    from src.images.prepare import (
        physical_crop_with_support, pixel_spacing, select_series,
        select_slice_indices, to_uint8,
    )
    from src.reports.schema import TARGETS
    from scripts.train_cnn_baseline import IMAGENET_MEAN, IMAGENET_STD, SliceModel
    import pydicom

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = False
    models = []
    cache_config = None
    for spec in MODEL_SPECS:
        checkpoint_path = Path("/kaggle/input/datasets/tumis001") / spec["dataset"] / "checkpoint_best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing pinned checkpoint: {checkpoint_path}")
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {spec['name']}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if checkpoint["epoch"] != spec["epoch"] or config["version"] != spec["version"]:
            raise ValueError(f"Checkpoint provenance mismatch: {spec['name']}")
        cfg = config["cache"]
        expected_cache = {"field_of_view_mm": 140.0, "image_size": spec["size"],
                          "series_slots": 4, "slices_per_slot": 12,
                          "central_band": [0.2, 0.8], "clip_percentiles": [1.0, 99.0],
                          "exclude_order_sources": ["unordered"]}
        if cfg != expected_cache:
            raise ValueError(f"Preprocessing mismatch: {spec['name']}: {cfg}")
        model_config = config["model"]
        if model_config["backbone"] != "resnet50" or model_config["top_k_fraction"] != 0.1:
            raise ValueError(f"Architecture mismatch: {spec['name']}")
        model = SliceModel("resnet50", False, model_config["dropout"], len(TARGETS), 0.1)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        if use_amp:
            model.to(memory_format=torch.channels_last)
        models.append((spec, model))
        cache_config = cfg
        del checkpoint
        print(f"Loaded {spec['name']}: epoch {spec['epoch']}, {spec['size']}px", flush=True)
    sizes = sorted({spec["size"] for spec in MODEL_SPECS})
    sample_rows = read_csv(data_root / "sample_submission.csv")
    if not sample_rows or list(sample_rows[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("sample submission columns differ from the model target order")
    study_ids = [row["StudyInstanceUID"] for row in sample_rows]
    if len(set(study_ids)) != len(study_ids):
        raise ValueError("sample_submission.csv contains duplicate study IDs")
    fallback = {row["StudyInstanceUID"]: [float(row[target]) for target in TARGETS]
                for row in sample_rows}
    output_path = Path("/kaggle/working/submission.csv")
    # Leave a valid schema-correct file in place even if an unexpected DICOM
    # issue interrupts inference. Successful predictions replace it below.
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["StudyInstanceUID", *TARGETS])
        for study_id in study_ids:
            writer.writerow([study_id, *[f"{value:.8f}" for value in fallback[study_id]]])
    series_by_study: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(data_root / "test_series.csv"):
        series_by_study.setdefault(row["StudyInstanceUID"], []).append(row)

    print(f"Studies: {len(study_ids)}; device: {device}; ensemble: most-confident member per target", flush=True)
    individual_predictions = {spec["name"]: {} for spec in MODEL_SPECS}
    predictions: dict[str, list[float]] = {}
    stats = {"studies": len(study_ids), "windows": 0,
             "decode_failures": 0, "empty_studies": 0}
    stage_seconds = {"header_io": 0.0, "pixel_decode": 0.0, "model": 0.0}
    mean = float(IMAGENET_MEAN)
    std = float(IMAGENET_STD)
    # The test set contains only a few studies. DICOM filesystem reads dominate
    # runtime, so overlap header and selected-pixel I/O while keeping every
    # study's original sorted-file order and preprocessing unchanged.
    with ThreadPoolExecutor(max_workers=4) as io_workers, torch.inference_mode():
        for study_number, study_id in enumerate(study_ids, start=1):
            images = {size: [] for size in sizes}
            supports = {size: [] for size in sizes}
            slots: list[int] = []
            for slot, row in enumerate(select_series(
                    series_by_study.get(study_id, []), cache_config["series_slots"])):
                series_dir = data_root / "test_series" / study_id / row["SeriesInstanceUID"]
                paths = [path for path in sorted(series_dir.iterdir())
                         if path.is_file() and path.suffix.lower() == ".dcm"]

                def read_header(path):
                    try:
                        return path, pydicom.dcmread(path, stop_before_pixels=True), None
                    except Exception as error:
                        return path, None, error

                timer = time.monotonic()
                header_results = list(io_workers.map(read_header, paths))
                stage_seconds["header_io"] += time.monotonic() - timer
                header_items = [(path, header) for path, header, error in header_results
                                if error is None]
                header_errors = len(header_results) - len(header_items)
                if header_items:
                    # Match read_series() defaults exactly: no anatomical-plane
                    # filtering or direction canonicalization.
                    ordered = order_headers(header_items, header_errors)
                else:
                    ordered = OrderedSeries([], [], "unordered", header_errors,
                                            ["no_readable_headers"])
                if ordered.order_source in cache_config["exclude_order_sources"] or not ordered.files:
                    continue

                selected_positions = select_slice_indices(
                    len(ordered.files), cache_config["slices_per_slot"],
                    *cache_config["central_band"])

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
                decoded: list[tuple[np.ndarray, bool, tuple[float, float] | None]] = []
                for position, pixels, inverted, spacing_value, error in decoded_results:
                    if error is not None:
                        stats["decode_failures"] += 1
                        print(f"Decode failure in {study_id}, series {row['SeriesInstanceUID']} "
                              f"slice={position}: {type(error).__name__}: {error}", flush=True)
                    else:
                        decoded.append((pixels, inverted, spacing_value))
                if not decoded:
                    continue
                low, high = series_bounds([pixels for pixels, _, _ in decoded],
                                          *cache_config["clip_percentiles"])
                spacing = next((value for _, _, value in decoded if value is not None), None)
                for pixels, inverted, slice_spacing in decoded:
                    support = np.isfinite(pixels)
                    image = to_uint8(pixels, low, high, inverted)
                    image[~support] = 0
                    for size in sizes:
                        cropped, cropped_support = physical_crop_with_support(
                            image.astype(np.float32), support,
                            slice_spacing if slice_spacing is not None else spacing,
                            cache_config["field_of_view_mm"], size)
                        images[size].append(cropped.round().astype(np.uint8))
                        supports[size].append(cropped_support)
                    slots.append(slot)

            if not images[sizes[0]]:
                stats["empty_studies"] += 1
                predictions[study_id] = fallback[study_id]
                for spec in MODEL_SPECS:
                    individual_predictions[spec["name"]][study_id] = fallback[study_id]
                continue

            inputs = {}
            for size in sizes:
                image_channels = stack_channels(np.stack(images[size]), np.asarray(slots))
                support_channels = stack_channels(np.stack(supports[size]), np.asarray(slots))
                array = np.concatenate((image_channels, support_channels), axis=1)
                batch = torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=torch.float32)
                batch[:, :3] = (batch[:, :3] / 255.0 - mean) / std
                batch[:, 3:] /= 255.0
                if use_amp:
                    batch = batch.contiguous(memory_format=torch.channels_last)
                inputs[size] = batch
            # One model call per study preserves global top-k pooling. This reader
            # selects at most 48 windows, so no chunkwise top-k approximation.
            assert len(slots) <= 48
            values = []
            timer = time.monotonic()
            for spec, model in models:
                batch = inputs[spec["size"]]
                owner = torch.zeros(len(batch), dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(batch, owner, 1)
                prob = torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()
                if prob.shape != (len(TARGETS),) or not np.isfinite(prob).all():
                    raise ValueError(f"Invalid model output: {study_id}, {spec['name']}")
                individual_predictions[spec["name"]][study_id] = prob.tolist()
                values.append(prob)
            stage_seconds["model"] += time.monotonic() - timer
            stacked = np.stack(values).astype(np.float64)
            pick = np.abs(stacked - 0.5).argmax(axis=0)
            predictions[study_id] = stacked[pick, np.arange(stacked.shape[1])].tolist()
            stats["windows"] += len(slots)
            if study_number % 100 == 0 or study_number == len(study_ids):
                print(f"{study_number}/{len(study_ids)} studies; "
                      f"{time.monotonic() - started:.1f}s elapsed", flush=True)

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["StudyInstanceUID", *TARGETS])
        for study_id in study_ids:
            writer.writerow([study_id, *[f"{value:.8f}" for value in predictions[study_id]]])

    written = read_csv(output_path)
    if len(written) != len(sample_rows):
        raise ValueError(f"Wrote {len(written)} rows, expected {len(sample_rows)}")
    if not written or list(written[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("Saved submission columns do not match sample_submission.csv")
    if [row["StudyInstanceUID"] for row in written] != study_ids:
        raise ValueError("Submission study order does not match sample_submission.csv")
    written_values = np.asarray([[float(row[target]) for target in TARGETS] for row in written])
    if written_values.shape != (len(study_ids), len(TARGETS)) \
            or not np.isfinite(written_values).all() \
            or np.any(written_values < 0) or np.any(written_values > 1):
        raise ValueError("Saved submission contains invalid probabilities")
    Path("/kaggle/working/individual_predictions.json").write_text(
        json.dumps(individual_predictions, indent=2) + "\n", encoding="utf-8")
    summary = {**stats, "device": str(device), "combination": "most-confident member per study and target (max |p - 0.5|)",
               "models": MODEL_SPECS, "shared_dicom_reads": True,
               "stage_seconds": {key: round(value, 3) for key, value in stage_seconds.items()},
               "seconds": round(time.monotonic() - started, 1), "submission": str(output_path)}
    Path("/kaggle/working/inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
