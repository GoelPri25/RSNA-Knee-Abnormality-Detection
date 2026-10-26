"""Submit the best checkpoint from the HPO blend +281 AUC-control run."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


EXPECTED_CHECKPOINT_SHA256 = "c1e822bd797ff0a89cd1905276c0fd4c9ad3b5bdfd51cebf95507a597bfd3e4e"
EXPECTED_CHECKPOINT_EPOCH = 22
EXPECTED_REPORT_VALIDATION_AUC = 0.8858800134599393


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    started = time.monotonic()
    source = Path("/kaggle/input/datasets/tumis001/rsna-cnn-topk-support-source-v1/skanee")
    artifact_dir = Path(
        "/kaggle/input/datasets/tumis001/rsna-hpo-blend-add281-auc-control-fold0"
    )
    checkpoint_path = artifact_dir / "checkpoint_best.pt"
    history_path = artifact_dir / "training_history.json"
    progress_path = artifact_dir / "run_progress.json"
    manifest_path = artifact_dir / "run_manifest.json"
    for required in (source / "scripts/train_cnn_baseline.py", checkpoint_path,
                     history_path, progress_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required submission input is missing: {required}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best_row = max(history, key=lambda row: row["validation_macro_auc"])
    if best_row.get("epoch") != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError("Training history does not select the expected best-checkpoint epoch")
    if abs(best_row.get("validation_macro_auc", -1) - EXPECTED_REPORT_VALIDATION_AUC) > 1e-9:
        raise ValueError("Training-history AUC differs from the pinned submission checkpoint")
    if (progress.get("best_epoch"), progress.get("best_macro_auc")) != (
            EXPECTED_CHECKPOINT_EPOCH, EXPECTED_REPORT_VALIDATION_AUC):
        raise ValueError("Progress metadata differs from the pinned best checkpoint")
    if (manifest.get("train_studies_expected"), manifest.get("validation_studies_expected")) != (3524, 825):
        raise ValueError("Manifest does not describe the intended +281/fixed-validation split")
    if manifest.get("fixed_validation_ids_unchanged") is not True or manifest.get("gold_labels_loaded") is not False:
        raise ValueError("Manifest does not confirm the fixed report-only validation and gold lock")
    data_root = next((path for path in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ) if (path / "sample_submission.csv").is_file()), None)
    if data_root is None:
        raise FileNotFoundError("Competition sample_submission.csv was not attached")

    sys.path.insert(0, str(source))
    from src.features.dicom import decode_pixels, read_series, series_bounds
    from src.images.dataset import stack_channels
    from src.images.prepare import (
        physical_crop_with_support, pixel_spacing, select_series,
        select_slice_indices, to_uint8,
    )
    from src.reports.schema import TARGETS
    from scripts.train_cnn_baseline import IMAGENET_MEAN, IMAGENET_STD, SliceModel

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain config and state_dict")
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected best-checkpoint SHA-256: {checkpoint_hash}")
    if checkpoint.get("epoch") != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError(f"Expected selected checkpoint epoch {EXPECTED_CHECKPOINT_EPOCH}; "
                         f"got {checkpoint.get('epoch')}")
    config = checkpoint["config"]
    if config.get("version") != "cnn_hpo_v1_blend_add281_auc_control_tpu":
        raise ValueError(f"Unexpected checkpoint config: {config.get('version')}")
    if checkpoint["epoch"] != best_row["epoch"]:
        raise ValueError("Checkpoint epoch differs from the history-selected best epoch")
    cache_config = config["cache"]
    model_config = config["model"]
    if model_config.get("input_channels") not in (None, 6):
        raise ValueError(f"Expected 6-channel checkpoint, got {model_config.get('input_channels')}")

    sample_rows = read_csv(data_root / "sample_submission.csv")
    if not sample_rows or list(sample_rows[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("sample submission columns differ from the model target order")
    study_ids = [row["StudyInstanceUID"] for row in sample_rows]
    if len(set(study_ids)) != len(study_ids):
        raise ValueError("sample_submission.csv contains duplicate study IDs")
    fallback = {row["StudyInstanceUID"]: [float(row[target]) for target in TARGETS]
                for row in sample_rows}
    series_by_study: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(data_root / "test_series.csv"):
        series_by_study.setdefault(row["StudyInstanceUID"], []).append(row)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SliceModel(model_config["backbone"], False, model_config["dropout"], len(TARGETS),
                       model_config.get("top_k_fraction", 0.1))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    use_amp = device.type == "cuda"
    print(f"Source: {source}\nCheckpoint: {checkpoint_path}\nData: {data_root}", flush=True)
    print(f"Studies: {len(study_ids)} | checkpoint epoch: {checkpoint.get('epoch')} | "
          f"device: {device} | input: 6 x {cache_config['image_size']} x "
          f"{cache_config['image_size']}", flush=True)

    predictions: dict[str, list[float]] = {}
    stats = {"studies": len(study_ids), "windows": 0,
             "decode_failures": 0, "empty_studies": 0}
    mean = float(IMAGENET_MEAN)
    std = float(IMAGENET_STD)
    with torch.inference_mode():
        for study_number, study_id in enumerate(study_ids, start=1):
            images: list[np.ndarray] = []
            supports: list[np.ndarray] = []
            slots: list[int] = []
            for slot, row in enumerate(select_series(
                    series_by_study.get(study_id, []), cache_config["series_slots"])):
                ordered = read_series(data_root / "test_series" / study_id / row["SeriesInstanceUID"])
                if ordered.order_source in cache_config["exclude_order_sources"] or not ordered.files:
                    continue
                decoded: list[tuple[np.ndarray, bool, tuple[float, float] | None]] = []
                for position in select_slice_indices(
                        len(ordered.files), cache_config["slices_per_slot"],
                        *cache_config["central_band"]):
                    try:
                        dicom = __import__("pydicom").dcmread(ordered.files[position])
                        pixels = decode_pixels(dicom)
                        if pixels.ndim != 2:
                            raise ValueError(f"Expected 2D slice, got {pixels.shape}")
                        decoded.append((pixels,
                                        getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1",
                                        pixel_spacing(dicom)))
                    except Exception as error:
                        stats["decode_failures"] += 1
                        print(f"Decode failure in {study_id}, series {row['SeriesInstanceUID']}: "
                              f"{type(error).__name__}: {error}", flush=True)
                if not decoded:
                    continue
                low, high = series_bounds([pixels for pixels, _, _ in decoded],
                                          *cache_config["clip_percentiles"])
                spacing = next((value for _, _, value in decoded if value is not None), None)
                for pixels, inverted, slice_spacing in decoded:
                    support = np.isfinite(pixels)
                    image = to_uint8(pixels, low, high, inverted)
                    image[~support] = 0
                    cropped, cropped_support = physical_crop_with_support(
                        image.astype(np.float32), support,
                        slice_spacing if slice_spacing is not None else spacing,
                        cache_config["field_of_view_mm"], cache_config["image_size"])
                    images.append(cropped.round().astype(np.uint8))
                    supports.append(cropped_support)
                    slots.append(slot)

            if not images:
                stats["empty_studies"] += 1
                predictions[study_id] = fallback[study_id]
                continue

            image_channels = stack_channels(np.stack(images), np.asarray(slots))
            support_channels = stack_channels(np.stack(supports), np.asarray(slots))
            model_input = np.concatenate((image_channels, support_channels), axis=1)
            batch = torch.from_numpy(np.ascontiguousarray(model_input)).to(device=device, dtype=torch.float32)
            batch[:, :3] = (batch[:, :3] / 255.0 - mean) / std
            batch[:, 3:] /= 255.0
            chunks = []
            for begin in range(0, len(batch), 64):
                chunk = batch[begin:begin + 64]
                owner = torch.zeros(len(chunk), dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    chunks.append(model(chunk, owner, 1).float() * len(chunk))
            logits = torch.cat(chunks).sum(dim=0) / len(batch)
            probability = torch.sigmoid(logits).cpu().tolist()
            if len(probability) != len(TARGETS) or not np.isfinite(probability).all():
                raise ValueError(f"Invalid predictions for study {study_id}")
            if min(probability) < 0 or max(probability) > 1:
                raise ValueError(f"Predictions outside [0,1] for study {study_id}")
            predictions[study_id] = probability
            stats["windows"] += len(batch)
            if study_number % 100 == 0 or study_number == len(study_ids):
                print(f"{study_number}/{len(study_ids)} studies; "
                      f"{time.monotonic() - started:.1f}s elapsed", flush=True)

    output_path = Path("/kaggle/working/submission.csv")
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
    summary = {**stats, "device": str(device), "checkpoint_epoch": checkpoint.get("epoch"),
               "checkpoint_sha256": checkpoint_hash,
               "training_label_source": "HPO v1 blended report labels (train); Steven v4 (fixed report validation)",
               "report_validation_macro_auc": EXPECTED_REPORT_VALIDATION_AUC,
               "training_studies": manifest["train_studies_expected"],
               "validation_studies": manifest["validation_studies_expected"],
               "config_version": config.get("version"),
               "pooling": model_config.get("pooling"),
               "top_k_fraction": model_config.get("top_k_fraction", 0.1),
               "seconds": round(time.monotonic() - started, 1),
               "submission": str(output_path)}
    Path("/kaggle/working/inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
