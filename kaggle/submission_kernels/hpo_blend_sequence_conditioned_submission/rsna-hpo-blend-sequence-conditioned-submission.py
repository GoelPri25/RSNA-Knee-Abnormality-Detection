"""Fast inference for the sequence-conditioned support-aware ResNet-34."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


SOURCE = Path("/kaggle/input/datasets/tumis001/rsna-cnn-topk-support-source-v1/skanee")
CHECKPOINT = Path(
    "/kaggle/input/datasets/tumis001/rsna-hpo-blend-sequence-from-scratch-fold0/"
    "checkpoint_best.pt"
)
EXPECTED_SHA256 = "2be182d8cbd3d30b755fc8ab32325e129295dd0fbb814b19d5010100c7506a1f"
EXPECTED_EPOCH = 17
IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226


def sequence_category(plane: str, fluid_sensitive: str) -> int:
    """Match the training mapping: plane × fluid-sensitive, or unknown."""
    normalized_plane = str(plane or "").strip().lower()
    plane_index = (
        0 if normalized_plane.startswith("sag") else
        1 if normalized_plane.startswith("cor") else
        2 if normalized_plane.startswith(("axi", "trans")) else None
    )
    fluid_value = str(fluid_sensitive or "").strip().lower()
    fluid = (
        1 if fluid_value in {"1", "true", "yes"} else
        0 if fluid_value in {"0", "false", "no"} else None
    )
    return 6 if plane_index is None or fluid is None else plane_index * 2 + fluid


class SequenceConditionedSliceModel(torch.nn.Module):
    """The trained six-channel ResNet plus its learned sequence-logit branch."""

    def __init__(self, backbone: str, dropout: float, targets: int,
                 top_k_fraction: float) -> None:
        super().__init__()
        import torchvision

        self.backbone = getattr(torchvision.models, backbone)(weights=None)
        old_conv = self.backbone.conv1
        new_conv = torch.nn.Conv2d(
            6, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        self.backbone.conv1 = new_conv
        features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(dropout), torch.nn.Linear(features, targets)
        )
        self.sequence_correction = torch.nn.Linear(7, targets, bias=False)
        self.top_k_fraction = float(top_k_fraction)

    def forward(self, slices: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        encoded = self.backbone(slices)
        return self.head(encoded) + self.sequence_correction(metadata.to(encoded.dtype))


def target_topk_mean(window_logits: torch.Tensor, fraction: float) -> torch.Tensor:
    """Pool the top fraction independently for each finding, as in training."""
    k = max(1, math.ceil(len(window_logits) * fraction))
    return window_logits.topk(k, dim=0).values.mean(dim=0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    started = time.monotonic()
    for required in (SOURCE / "src/features/dicom.py", SOURCE / "src/images/prepare.py", CHECKPOINT):
        if not required.is_file():
            raise FileNotFoundError(f"Required submission input is missing: {required}")
    data_root = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
    if not (data_root / "sample_submission.csv").is_file():
        raise FileNotFoundError(f"Competition files are missing under {data_root}")

    sys.path.insert(0, str(SOURCE))
    from src.features.dicom import decode_pixels, read_series, series_bounds
    from src.images.dataset import stack_channels
    from src.images.prepare import (
        physical_crop_with_support, pixel_spacing, select_series,
        select_slice_indices, to_uint8,
    )
    from src.reports.schema import TARGETS

    checkpoint_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if checkpoint_hash != EXPECTED_SHA256:
        raise ValueError(f"Unexpected checkpoint SHA-256: {checkpoint_hash}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != EXPECTED_EPOCH:
        raise ValueError(f"Expected epoch {EXPECTED_EPOCH}, got {checkpoint.get('epoch')}")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("version") != "hpo_v1_blend_sequence_conditioned_from_scratch":
        raise ValueError("Unexpected checkpoint configuration")
    cache_config, model_config = config["cache"], config["model"]
    if (cache_config["image_size"] != 224 or cache_config["field_of_view_mm"] != 140.0
            or cache_config["series_slots"] != 4 or cache_config["slices_per_slot"] != 12
            or model_config["backbone"] != "resnet34"
            or model_config["top_k_fraction"] != 0.1):
        raise ValueError("Checkpoint image/model settings differ from this inference implementation")

    sample_rows = read_csv(data_root / "sample_submission.csv")
    expected_columns = ["StudyInstanceUID", *TARGETS]
    if not sample_rows or list(sample_rows[0]) != expected_columns:
        raise ValueError("sample_submission.csv columns differ from the trained target order")
    study_ids = [row["StudyInstanceUID"] for row in sample_rows]
    if len(set(study_ids)) != len(study_ids):
        raise ValueError("sample_submission.csv contains duplicate study IDs")
    fallback = {
        row["StudyInstanceUID"]: [float(row[target]) for target in TARGETS]
        for row in sample_rows
    }
    series_by_study: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(data_root / "test_series.csv"):
        required_fields = {"Anatomical_Plane", "Fluid_Sensitive", "SeriesInstanceUID"}
        if not required_fields.issubset(row):
            raise ValueError(f"test_series.csv is missing sequence fields: {required_fields - set(row)}")
        series_by_study.setdefault(row["StudyInstanceUID"], []).append(row)

    model = SequenceConditionedSliceModel(
        model_config["backbone"], model_config["dropout"], len(TARGETS),
        model_config["top_k_fraction"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    use_amp = device.type == "cuda"
    print(
        f"Checkpoint epoch={EXPECTED_EPOCH} sha256={checkpoint_hash}; "
        f"device={device}; test studies={len(study_ids)}", flush=True
    )

    predictions: dict[str, list[float]] = {}
    stats = {"studies": len(study_ids), "windows": 0,
             "decode_failures": 0, "empty_studies": 0}
    with torch.inference_mode():
        for study_number, study_id in enumerate(study_ids, start=1):
            images: list[np.ndarray] = []
            supports: list[np.ndarray] = []
            slots: list[int] = []
            metadata: list[np.ndarray] = []
            selected = select_series(
                series_by_study.get(study_id, []), cache_config["series_slots"]
            )
            for slot, row in enumerate(selected):
                ordered = read_series(
                    data_root / "test_series" / study_id / row["SeriesInstanceUID"]
                )
                if ordered.order_source in cache_config["exclude_order_sources"] or not ordered.files:
                    continue
                decoded: list[tuple[np.ndarray, bool, tuple[float, float] | None]] = []
                for position in select_slice_indices(
                    len(ordered.files), cache_config["slices_per_slot"],
                    *cache_config["central_band"],
                ):
                    try:
                        dicom = __import__("pydicom").dcmread(ordered.files[position])
                        pixels = decode_pixels(dicom)
                        if pixels.ndim != 2:
                            raise ValueError(f"Expected a 2D slice, got {pixels.shape}")
                        decoded.append((
                            pixels,
                            getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1",
                            pixel_spacing(dicom),
                        ))
                    except Exception as error:
                        stats["decode_failures"] += 1
                        print(
                            f"Decode failure {study_id}/{row['SeriesInstanceUID']}: "
                            f"{type(error).__name__}: {error}", flush=True,
                        )
                if not decoded:
                    continue
                low, high = series_bounds(
                    [pixels for pixels, _, _ in decoded], *cache_config["clip_percentiles"]
                )
                spacing = next((item[2] for item in decoded if item[2] is not None), None)
                category = sequence_category(row["Anatomical_Plane"], row["Fluid_Sensitive"])
                one_hot = np.zeros(7, dtype=np.float32)
                one_hot[category] = 1.0
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
                    metadata.append(one_hot)

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
            batch[:, :3] = (batch[:, :3] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            batch[:, 3:] /= 255.0
            sequence = torch.from_numpy(np.stack(metadata)).to(device=device)

            logits = []
            for begin in range(0, len(batch), 64):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits.append(model(batch[begin:begin + 64], sequence[begin:begin + 64]).float())
            study_logits = target_topk_mean(torch.cat(logits), model.top_k_fraction)
            probability = torch.sigmoid(study_logits).cpu().numpy()
            if probability.shape != (len(TARGETS),) or not np.isfinite(probability).all():
                raise ValueError(f"Invalid predictions for study {study_id}")
            if np.any(probability < 0) or np.any(probability > 1):
                raise ValueError(f"Probabilities out of bounds for study {study_id}")
            predictions[study_id] = probability.tolist()
            stats["windows"] += len(batch)
            print(
                f"Study {study_number}/{len(study_ids)}: {len(batch)} windows; "
                f"{time.monotonic() - started:.1f}s elapsed", flush=True,
            )

    output_path = Path("/kaggle/working/submission.csv")
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(expected_columns)
        for study_id in study_ids:
            writer.writerow([study_id, *[f"{value:.8f}" for value in predictions[study_id]]])

    written = read_csv(output_path)
    values = np.asarray([[float(row[target]) for target in TARGETS] for row in written])
    if (len(written) != len(sample_rows)
            or [row["StudyInstanceUID"] for row in written] != study_ids
            or list(written[0]) != expected_columns
            or values.shape != (len(study_ids), len(TARGETS))
            or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Generated submission failed schema, order, or probability checks")
    summary = {
        **stats, "device": str(device), "checkpoint_epoch": EXPECTED_EPOCH,
        "checkpoint_sha256": checkpoint_hash,
        "training_label_source": "HPO v1 blended report labels",
        "report_validation_macro_auc": 0.8893,
        "config_version": config["version"],
        "pooling": "per-target top-10% mean of window logits",
        "seconds": round(time.monotonic() - started, 1),
    }
    Path("/kaggle/working/inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
