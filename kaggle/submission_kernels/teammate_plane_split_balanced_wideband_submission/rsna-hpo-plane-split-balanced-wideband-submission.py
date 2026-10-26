#!/usr/bin/env python3
"""Submit the report-fold-best checkpoint from the wideband plane-split run.

Inputs are pinned to explicit Kaggle mount paths. No input-tree traversal is used.
The 58 gold cases are not loaded by this inference notebook.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ASSET_ROOT_CANDIDATES = (
    Path("/kaggle/input/datasets/tumis001/tumis-rsna-plane-split-wb-1005"),
    Path("/kaggle/input/tumis-rsna-plane-split-wb-1005"),
)
SOURCE_ROOT_CANDIDATES = (
    Path("/kaggle/input/datasets/tumis001/tumis-rsna-plane-split-wb-1005/skanee"),
    Path("/kaggle/input/tumis-rsna-plane-split-wb-1005/skanee"),
)
COMPETITION_ROOT_CANDIDATES = (
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
)

CHECKPOINT_RELATIVE = Path("checkpoint/checkpoint_best.pt")
SUMMARY_RELATIVE = Path("checkpoint/run_summary.json")
EXPECTED_SHA256 = "0caab4cf3325e0051d4d1ca8844dfec2c22a30fbc352f0db7f79085d23189406"
EXPECTED_EPOCH = 15
EXPECTED_AUC = 0.8877762467244824
EXPECTED_VERSION = "hpo_v1_blend_plane_split_balanced_wideband"
PLANES = ("Sagittal", "Coronal", "Axial")


def exact_dir(candidates: tuple[Path, ...], marker: str, description: str) -> Path:
    found = [path for path in candidates if (path / marker).is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {description}; found {[str(x) for x in found]}")
    return found[0]


def exact_file(candidates: tuple[Path, ...], description: str) -> Path:
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {description}; found {[str(x) for x in found]}")
    return found[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def chunked_study_logits(model, batch: torch.Tensor, planes: torch.Tensor,
                         trainer, chunk_size: int = 24) -> torch.Tensor:
    """Run windows in bounded batches, then apply the model's exact global plane pooling."""
    window_logits = []
    backbone = model.backbone
    for start in range(0, len(batch), chunk_size):
        stop = min(start + chunk_size, len(batch))
        image_chunk = batch[start:stop]
        plane_chunk = planes[start:stop].clamp_min(0)
        shared = backbone.layer2(backbone.layer1(
            backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(image_chunk))))
        ))
        logits = shared.new_zeros((len(image_chunk), model.plane_trust.shape[1]))
        for plane_index in range(len(PLANES)):
            rows = (plane_chunk == plane_index).nonzero().squeeze(1)
            if len(rows):
                features = model.plane_layer4[plane_index](
                    model.plane_layer3[plane_index](shared[rows])
                )
                features = torch.flatten(backbone.avgpool(features), 1)
                logits = logits.index_put(
                    (rows,), model.plane_head[plane_index](features).to(logits.dtype)
                )
        window_logits.append(logits)

    all_logits = torch.cat(window_logits, dim=0)
    owner = torch.zeros(len(all_logits), dtype=torch.long, device=all_logits.device)
    routed_planes = planes.clamp_min(0)
    per_plane, present = trainer.plane_topk_mean(
        all_logits, owner, routed_planes, 1, model.top_k_fraction, model.plane_min_k
    )
    trust = F.softplus(model.plane_trust).unsqueeze(0) * present.unsqueeze(-1)
    return (trust * per_plane).sum(1) / trust.sum(1).clamp_min(1e-6)


def main() -> None:
    started = time.monotonic()
    asset_root = exact_dir(
        ASSET_ROOT_CANDIDATES, CHECKPOINT_RELATIVE.as_posix(), "checkpoint asset dataset"
    )
    source_root = exact_dir(
        SOURCE_ROOT_CANDIDATES, "scripts/train_cnn_topk_attention_sliceorder.py",
        "plane-split model source",
    )
    checkpoint_path = asset_root / CHECKPOINT_RELATIVE
    summary_path = asset_root / SUMMARY_RELATIVE
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_sha256 != EXPECTED_SHA256:
        raise ValueError(f"Pinned checkpoint SHA-256 mismatch: {checkpoint_sha256}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary.get("best", {})
    if best.get("epoch") != EXPECTED_EPOCH or abs(best.get("macro_auc", -1) - EXPECTED_AUC) > 1e-10:
        raise ValueError("Run summary does not match the report-validation-best checkpoint")
    if (summary.get("train_studies"), summary.get("validation_studies")) != (3524, 825):
        raise ValueError("Unexpected training/validation split in the pinned run")

    data_root = exact_dir(
        COMPETITION_ROOT_CANDIDATES, "sample_submission.csv", "competition test data"
    )
    required = (
        source_root / "src/features/dicom.py",
        source_root / "src/images/dataset.py",
        source_root / "src/images/prepare.py",
        source_root / "src/reports/schema.py",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required inference source is missing: {path}")

    sys.path.insert(0, str(source_root))
    from src.features.dicom import decode_pixels, read_series, series_bounds
    from src.images.dataset import stack_channels
    from src.images.prepare import (
        physical_crop_with_support, pixel_spacing, select_series,
        select_slice_indices, to_uint8,
    )
    from src.reports.schema import TARGETS

    trainer_path = source_root / "scripts/train_cnn_topk_attention_sliceorder.py"
    spec = importlib.util.spec_from_file_location("plane_split_model_source", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the pinned plane-split model implementation")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    if tuple(trainer.PLANES) != PLANES:
        raise ValueError(f"Model source plane order differs from expected: {trainer.PLANES}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    cache_config = config.get("cache", {})
    if checkpoint.get("epoch") != EXPECTED_EPOCH or config.get("version") != EXPECTED_VERSION:
        raise ValueError("Checkpoint epoch or experiment version is not the pinned run")
    if (model_config.get("backbone"), model_config.get("plane_networks"),
            model_config.get("plane_min_k")) != ("resnet34", "split_layer3", 2) \
            or model_config.get("input_channels") not in (None, 6):
        raise ValueError(f"Unexpected checkpoint model configuration: {model_config}")
    expected_cache = {
        "field_of_view_mm": 140.0,
        "image_size": 224,
        "series_slots": 4,
        "slices_per_slot": 40,
        "central_band": [0.1, 0.9],
        "clip_percentiles": [1.0, 99.0],
        "all_slices_in_band": True,
    }
    for key, value in expected_cache.items():
        if cache_config.get(key) != value:
            raise ValueError(f"Checkpoint preprocessing mismatch for {key}: {cache_config.get(key)}")

    sample_path = data_root / "sample_submission.csv"
    sample_rows = read_csv(sample_path)
    if not sample_rows or list(sample_rows[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("sample_submission.csv columns differ from the model target schema")
    study_ids = [row["StudyInstanceUID"] for row in sample_rows]
    if len(set(study_ids)) != len(study_ids):
        raise ValueError("sample_submission.csv contains duplicate study IDs")
    fallback = {
        row["StudyInstanceUID"]: [float(row[target]) for target in TARGETS]
        for row in sample_rows
    }
    series_by_study: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(data_root / "test_series.csv"):
        series_by_study.setdefault(row["StudyInstanceUID"], []).append(row)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = trainer.SliceModel(
        model_config["backbone"], False, float(model_config["dropout"]), len(TARGETS),
        float(model_config["top_k_fraction"]), model_config["plane_networks"],
        int(model_config["plane_min_k"]),
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint/model mismatch: {incompatible}")
    model = model.to(device).eval()
    print(
        f"Pinned checkpoint: epoch={EXPECTED_EPOCH}, report-fold macro AUC={EXPECTED_AUC:.6f}, "
        f"sha256={checkpoint_sha256}", flush=True
    )
    print(
        f"Model=ResNet34 plane-split(layer3+), input=6x{cache_config['image_size']}x"
        f"{cache_config['image_size']}, device={device}; test studies={len(study_ids)}",
        flush=True,
    )

    predictions: dict[str, list[float]] = {}
    stats = {"studies": len(study_ids), "windows": 0, "decode_failures": 0,
             "empty_studies": 0, "unknown_plane_windows": 0}
    preprocess_seconds = 0.0
    inference_seconds = 0.0
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for study_number, study_id in enumerate(study_ids, start=1):
            prep_started = time.monotonic()
            images: list[np.ndarray] = []
            supports: list[np.ndarray] = []
            slots: list[int] = []
            plane_of_slot: dict[int, int] = {}
            for slot, row in enumerate(select_series(
                    series_by_study.get(study_id, []), cache_config["series_slots"])):
                plane_of_slot[slot] = (PLANES.index(row["Anatomical_Plane"])
                                       if row.get("Anatomical_Plane") in PLANES else -1)
                ordered = read_series(data_root / "test_series" / study_id / row["SeriesInstanceUID"])
                if ordered.order_source in cache_config["exclude_order_sources"] or not ordered.files:
                    continue
                decoded: list[tuple[np.ndarray, bool, tuple[float, float] | None]] = []
                positions = select_slice_indices(
                    len(ordered.files), cache_config["slices_per_slot"],
                    *cache_config["central_band"],
                )
                for position in positions:
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
                        print(f"DICOM decode failed ({study_id}/{row['SeriesInstanceUID']}): "
                              f"{type(error).__name__}: {error}", flush=True)
                if not decoded:
                    continue
                low, high = series_bounds(
                    [pixels for pixels, _, _ in decoded], *cache_config["clip_percentiles"]
                )
                spacing = next((value for _, _, value in decoded if value is not None), None)
                for pixels, inverted, slice_spacing in decoded:
                    support = np.isfinite(pixels)
                    image = to_uint8(pixels, low, high, inverted)
                    image[~support] = 0
                    cropped, cropped_support = physical_crop_with_support(
                        image.astype(np.float32), support,
                        slice_spacing if slice_spacing is not None else spacing,
                        cache_config["field_of_view_mm"], cache_config["image_size"],
                    )
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
            batch = torch.from_numpy(np.ascontiguousarray(model_input)).to(
                device=device, dtype=torch.float32
            )
            batch[:, :3] = (batch[:, :3] / 255.0 - float(trainer.IMAGENET_MEAN)) / float(trainer.IMAGENET_STD)
            batch[:, 3:] /= 255.0
            planes = torch.tensor([plane_of_slot[slot] for slot in slots],
                                  dtype=torch.long, device=device)
            stats["unknown_plane_windows"] += int((planes < 0).sum())
            preprocess_seconds += time.monotonic() - prep_started
            infer_started = time.monotonic()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = chunked_study_logits(model, batch, planes, trainer).float()[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.monotonic() - infer_started
            probability = torch.sigmoid(logits).cpu().tolist()
            if len(probability) != len(TARGETS) or not np.isfinite(probability).all():
                raise ValueError(f"Non-finite or wrong-sized prediction for {study_id}")
            predictions[study_id] = probability
            stats["windows"] += len(batch)
            print(f"Study {study_number}/{len(study_ids)}: {len(batch)} windows", flush=True)

    output_path = Path("/kaggle/working/submission.csv")
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["StudyInstanceUID", *TARGETS])
        for study_id in study_ids:
            writer.writerow([study_id, *[f"{value:.8f}" for value in predictions[study_id]]])

    written = read_csv(output_path)
    if len(written) != len(sample_rows) or not written:
        raise ValueError("Submission row count does not match sample_submission.csv")
    if list(written[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("Submission columns do not match sample_submission.csv")
    if [row["StudyInstanceUID"] for row in written] != study_ids:
        raise ValueError("Submission study IDs/order changed while writing")
    values = np.asarray([[float(row[target]) for target in TARGETS] for row in written])
    if values.shape != (len(study_ids), len(TARGETS)) or not np.isfinite(values).all() \
            or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Submission contains invalid probabilities")

    summary = {
        **stats,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": EXPECTED_EPOCH,
        "report_validation_macro_auc": EXPECTED_AUC,
        "device": str(device),
        "seconds_preprocess": round(preprocess_seconds, 2),
        "seconds_inference": round(inference_seconds, 2),
        "seconds_total": round(time.monotonic() - started, 2),
        "submission_path": str(output_path),
    }
    Path("/kaggle/working/inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
