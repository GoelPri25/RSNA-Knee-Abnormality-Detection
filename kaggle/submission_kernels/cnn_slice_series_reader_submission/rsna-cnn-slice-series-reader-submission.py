"""Inference for the selected fold-0 slice-series CNN reader."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


INPUT = Path("/kaggle/input")
DATA_CANDIDATES = (
    INPUT / "competitions/rsna-knee-abnormality-detection",
    INPUT / "rsna-knee-abnormality-detection",
)
BASE_SOURCE_CANDIDATES = (
    INPUT / "datasets/tumis001/rsna-cnn-topk-support-source-v1/skanee",
    INPUT / "datasets/tumis001/rsna-cnn-topk-support-source-v1",
)
READER_SOURCE_CANDIDATES = (
    INPUT / "datasets/tumis001/rsna-cnn-slice-series-reader-source-v1/skanee",
    INPUT / "datasets/tumis001/rsna-cnn-slice-series-reader-source-v1/skanee/skanee",
    INPUT / "datasets/tumis001/rsna-cnn-slice-series-reader-source-v1",
)
CHECKPOINT = INPUT / (
    "notebooks/tumis001/rsna-cnn-slice-series-reader-fold-0/"
    "cnn_slice_series_reader_fold0/run/checkpoint_best.pt"
)


def locate(candidates: tuple[Path, ...], required: str, description: str) -> Path:
    for path in candidates:
        if (path / required).is_file():
            return path
    raise FileNotFoundError(
        f"Could not find {description}; checked fixed paths: "
        + ", ".join(str(path) for path in candidates)
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def plane_id(value: str) -> int:
    value = str(value or "").strip().lower()
    return 1 if "sag" in value else 2 if "cor" in value else 3 if "axi" in value or "trans" in value else 0


def fluid_id(value: str) -> int:
    value = str(value or "").strip().lower()
    if value in {"1", "true", "yes", "y"} or ("fluid" in value and "non" not in value):
        return 1
    if value in {"0", "false", "no", "n"} or "non-fluid" in value or "not fluid" in value:
        return 2
    return 0


def main() -> None:
    started = time.monotonic()
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Selected training checkpoint is missing: {CHECKPOINT}")
    data_root = locate(DATA_CANDIDATES, "sample_submission.csv", "competition data")
    base_source = locate(BASE_SOURCE_CANDIDATES, "src/features/dicom.py", "DICOM preprocessing source")
    reader_source = locate(READER_SOURCE_CANDIDATES, "scripts/train_cnn_slice_series.py", "reader source")

    # Put the experiment overlay first, so its exact model implementation and
    # target schema take precedence; DICOM/crop helpers come from the proven CNN source.
    sys.path.insert(0, str(base_source))
    sys.path.insert(0, str(reader_source))
    import src
    # The experiment source is an overlay with its own ``src`` package; make
    # the proven DICOM helpers visible under that same package namespace.
    src.__path__.append(str(base_source / "src"))
    import src.images
    src.images.__path__.append(str(base_source / "src" / "images"))
    from src.features.dicom import decode_pixels, normalized_positions, read_series, series_bounds
    from src.images.prepare import (
        physical_crop_with_support,
        pixel_spacing,
        select_series,
        select_slice_indices,
        to_uint8,
    )
    from src.reports.schema import TARGETS
    from scripts.train_cnn_slice_series import IMAGENET_MEAN, IMAGENET_STD, OrderedSeriesReader

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if not {"state_dict", "config", "epoch"}.issubset(checkpoint):
        raise ValueError("Checkpoint is missing state_dict, config, or epoch")
    config = checkpoint["config"]
    expected_epoch = 16
    if int(checkpoint["epoch"]) != expected_epoch:
        raise ValueError(f"Expected selected epoch {expected_epoch}, found {checkpoint['epoch']}")
    model_config = config["model"]
    cache_config = config["cache"]

    sample_rows = read_csv(data_root / "sample_submission.csv")
    test_rows = read_csv(data_root / "test.csv")
    test_series_rows = read_csv(data_root / "test_series.csv")
    if not sample_rows or list(sample_rows[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("sample-submission columns do not match the trained target order")
    study_ids = [row["StudyInstanceUID"] for row in sample_rows]
    test_ids = {row["StudyInstanceUID"] for row in test_rows}
    if len(study_ids) != len(set(study_ids)) or set(study_ids) != test_ids:
        raise ValueError("sample-submission and official test study IDs do not match exactly")
    series_by_study: dict[str, list[dict[str, str]]] = {}
    for row in test_series_rows:
        series_by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    fallback = {row["StudyInstanceUID"]: [float(row[target]) for target in TARGETS]
                for row in sample_rows}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    # All trained ResNet weights are included in state_dict. Avoid torchvision's
    # default ImageNet download because this kernel is intentionally offline.
    model_config = dict(model_config)
    model_config["pretrained"] = False
    inference_config = dict(config)
    inference_config["model"] = model_config
    model = OrderedSeriesReader(inference_config, len(TARGETS)).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    use_amp = device.type == "cuda"
    print(f"Device: {device}; model: ResNet34 slice-series reader; selected epoch: {expected_epoch}", flush=True)
    print(f"Data: {data_root}; test studies: {len(study_ids)}; test series: {len(test_series_rows)}", flush=True)

    predictions: dict[str, list[float]] = {}
    stats = {"studies": len(study_ids), "slices": 0, "decode_failures": 0,
             "skipped_unordered_series": 0, "empty_studies": 0}
    size = int(cache_config["image_size"])
    wanted_slices = int(cache_config["slices_per_slot"])
    low_pct, high_pct = cache_config["clip_percentiles"]
    fov_mm = float(cache_config["field_of_view_mm"])
    excluded_order = set(cache_config.get("exclude_order_sources", []))

    with torch.inference_mode():
        for study_number, study_id in enumerate(study_ids, start=1):
            image_blocks: list[np.ndarray] = []
            support_blocks: list[np.ndarray] = []
            slots: list[int] = []
            token_indices: list[int] = []
            positions: list[float] = []
            planes = np.zeros((1, int(config["data"]["series_slots"])), dtype=np.int64)
            fluids = np.zeros_like(planes)

            selected_series = select_series(
                series_by_study.get(study_id, []), int(config["data"]["series_slots"]))
            for slot, row in enumerate(selected_series):
                planes[0, slot] = plane_id(row.get("Anatomical_Plane", ""))
                fluids[0, slot] = fluid_id(row.get("Fluid_Sensitive", ""))
                ordered = read_series(data_root / "test_series" / study_id / row["SeriesInstanceUID"])
                if ordered.order_source in excluded_order or not ordered.files:
                    stats["skipped_unordered_series"] += 1
                    continue
                indices = select_slice_indices(
                    len(ordered.files), wanted_slices, *cache_config["central_band"])
                decoded: list[tuple[np.ndarray, bool, tuple[float, float] | None, float]] = []
                normalized = normalized_positions(ordered.coordinates)
                for index in indices:
                    try:
                        dicom = __import__("pydicom").dcmread(ordered.files[index])
                        pixels = decode_pixels(dicom)
                        if pixels.ndim != 2:
                            raise ValueError(f"Expected a 2D DICOM slice, got {pixels.shape}")
                        decoded.append((pixels,
                                        getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1",
                                        pixel_spacing(dicom),
                                        float(normalized[index])))
                    except Exception as error:
                        stats["decode_failures"] += 1
                        print(f"Decode failure: {study_id}/{row['SeriesInstanceUID']} "
                              f"{type(error).__name__}: {error}", flush=True)
                if not decoded:
                    continue
                low, high = series_bounds([item[0] for item in decoded], low_pct, high_pct)
                default_spacing = next((item[2] for item in decoded if item[2] is not None), None)
                for token_index, (pixels, inverted, spacing, position) in enumerate(decoded):
                    support = np.isfinite(pixels)
                    image = to_uint8(pixels, low, high, inverted)
                    image[~support] = 0
                    cropped, cropped_support = physical_crop_with_support(
                        image.astype(np.float32), support,
                        spacing if spacing is not None else default_spacing,
                        fov_mm, size)
                    image_blocks.append(cropped[None])
                    support_blocks.append(cropped_support[None])
                    slots.append(slot)
                    token_indices.append(token_index)
                    positions.append(position)

            if not image_blocks:
                stats["empty_studies"] += 1
                predictions[study_id] = fallback[study_id]
                continue

            images = np.concatenate(image_blocks, axis=0).astype(np.float32)[:, None] / 255.0
            support = np.concatenate(support_blocks, axis=0).astype(np.float32)[:, None] / 255.0
            images = (images - IMAGENET_MEAN) / IMAGENET_STD
            inputs = torch.from_numpy(np.concatenate((images, support), axis=1)).to(device)
            study_index = torch.zeros(len(inputs), dtype=torch.long, device=device)
            slot_tensor = torch.tensor(slots, dtype=torch.long, device=device)
            token_tensor = torch.tensor(token_indices, dtype=torch.long, device=device)
            position_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
            plane_tensor = torch.from_numpy(planes).to(device)
            fluid_tensor = torch.from_numpy(fluids).to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(inputs, study_index, slot_tensor, token_tensor,
                               position_tensor, plane_tensor, fluid_tensor, 1)
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()[0]
            if probabilities.shape != (len(TARGETS),) or not np.isfinite(probabilities).all():
                raise ValueError(f"Invalid model output for study {study_id}")
            if np.any(probabilities < 0) or np.any(probabilities > 1):
                raise ValueError(f"Out-of-range probability for study {study_id}")
            predictions[study_id] = probabilities.tolist()
            stats["slices"] += len(inputs)
            if study_number % 100 == 0 or study_number == len(study_ids):
                print(f"{study_number}/{len(study_ids)} studies; "
                      f"{time.monotonic() - started:.1f}s elapsed", flush=True)

    if set(predictions) != set(study_ids):
        raise ValueError("Missing or extra test study predictions")
    output_path = Path("/kaggle/working/submission.csv")
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["StudyInstanceUID", *TARGETS])
        for study_id in study_ids:
            writer.writerow([study_id, *[f"{value:.9g}" for value in predictions[study_id]]])

    check = read_csv(output_path)
    values = np.asarray([[float(row[target]) for target in TARGETS] for row in check])
    if (len(check) != len(sample_rows)
            or [row["StudyInstanceUID"] for row in check] != study_ids
            or list(check[0]) != ["StudyInstanceUID", *TARGETS]
            or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Round-trip submission validation failed")

    summary = {
        **stats,
        "device": str(device),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "training_report_validation_macro_auc": 0.8771392918127184,
        "training_report_validation_bce": 0.5240884225055424,
        "seconds": round(time.monotonic() - started, 1),
        "submission_shape": list(values.shape),
        "submission_schema_valid": True,
        "raw_dicom_stays_in_kaggle": True,
    }
    Path("/kaggle/working/inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Submission CSV passed schema and probability checks:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
