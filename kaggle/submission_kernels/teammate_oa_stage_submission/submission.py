"""Submit the r34 AUC-control model with the OA presence x location stage (fold 0, epoch 57).

Model: shared ResNet-34 (0.918 recipe: 4 series x 12 slices, 140 mm crop at 224 px, 2.5D windows with
acquisition-support masks, per-finding top-10% pooling), trained from ImageNet on HPO v1 blend labels
(+281 train-only split) with the AUC-control controller (LR scheduler and early stopping on fold-0 macro
AUC, validated on HPO v1 labels) and contralateral mirror augmentation. OA is predicted in two stages:
P(Medial / Lateral / PF OA) = P(any OA) x P(compartment | OA). Fold-0 macro AUC 0.9107 (HPO labels),
gold macro 0.8809.

Inputs (attach all three, plus the competition):
  tumis001/rsna-cnn-topk-support-source-v1             preprocessing package (src/*), as used to build the cache
  priyanjaliml/rsna-hpo-topk-attention-sliceorder-source  model code (scripts/train_cnn_topk_attention_sliceorder.py)
  priyanjaliml/rsna-r34-auc-control-oa-stage-fold0     checkpoint_best.pt + run_summary.json

Preprocessing is the 0.918 submission's, unchanged: the same code built the training cache.

tumis001 copy of priyanjaliml/rsna-r34-oa-stage-submission: identical inference; inputs are read from fixed
dataset paths instead of a recursive search, and the checkpoint is pinned by SHA-256.
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


EXPECTED_EPOCH = 57
EXPECTED_REPORT_VALIDATION_AUC = 0.9107295250691672
EXPECTED_CONFIG_VERSION = "cnn_hpo_v1_blend_add281_auc_control_oa_stage"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    started = time.monotonic()
    def pick(*parts: str) -> Path:
        # Fixed dataset paths only (both Kaggle mount layouts); no recursive search.
        owner, slug, rel = parts
        found = [p for p in (Path("/kaggle/input/datasets") / owner / slug / rel, Path("/kaggle/input") / slug / rel)
                 if p.exists()]
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one input for {owner}/{slug}/{rel}; found {found}")
        return found[0]

    source = pick("tumis001", "rsna-cnn-topk-support-source-v1", "skanee")       # 0.918 preprocessing package
    trainer_path = pick("priyanjaliml", "rsna-hpo-topk-attention-sliceorder-source", "scripts/train_cnn_topk_attention_sliceorder.py")
    checkpoint_path = pick("priyanjaliml", "rsna-r34-auc-control-oa-stage-fold0", "checkpoint_best.pt")
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_hash != "972e01836821f26f17f1906be1f568864cdc31fb248da27e33b0eb5dee7844a6":
        raise ValueError(f"Unexpected checkpoint SHA-256: {checkpoint_hash}")
    run_summary = json.loads((checkpoint_path.parent / "run_summary.json").read_text(encoding="utf-8"))
    if run_summary.get("best", {}).get("epoch") != EXPECTED_EPOCH \
            or abs(run_summary["best"]["macro_auc"] - EXPECTED_REPORT_VALIDATION_AUC) > 1e-9:
        raise ValueError("Run summary does not describe the pinned checkpoint")
    if (run_summary.get("train_studies"), run_summary.get("validation_studies")) != (3524, 825):
        raise ValueError("Run summary does not describe the intended +281/fixed-validation split")
    data_root = next((path for path in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ) if (path / "sample_submission.csv").is_file()), None)
    if data_root is None:
        raise FileNotFoundError("Competition sample_submission.csv was not attached")

    sys.path.insert(0, str(source))
    from src.features.dicom import decode_pixels, read_series, series_bounds
    from src.images.dataset import stack_channels
    from src.images.prepare import (physical_crop_with_support, pixel_spacing, select_series,
                                    select_slice_indices, to_uint8)
    from src.reports.schema import TARGETS
    spec = importlib.util.spec_from_file_location("oa_stage_trainer", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)        # reuses the src.* modules imported above

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = config["model"]
    if checkpoint.get("epoch") != EXPECTED_EPOCH or config.get("version") != EXPECTED_CONFIG_VERSION \
            or not model_config.get("oa_stage") \
            or model_config.get("plane_networks", "shared") != "shared":
        raise ValueError(f"Checkpoint is not the pinned model: epoch {checkpoint.get('epoch')}, "
                         f"version {config.get('version')}")
    cache_config = config["cache"]

    sample_rows = read_csv(data_root / "sample_submission.csv")
    if not sample_rows or list(sample_rows[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("sample submission columns differ from the model target order")
    study_ids = [row["StudyInstanceUID"] for row in sample_rows]
    if len(set(study_ids)) != len(study_ids):
        raise ValueError("sample_submission.csv contains duplicate study IDs")
    fallback = {row["StudyInstanceUID"]: [float(row[t]) for t in TARGETS] for row in sample_rows}
    series_by_study: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(data_root / "test_series.csv"):
        series_by_study.setdefault(row["StudyInstanceUID"], []).append(row)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 12 target columns + presence + 3 OA locations; compose_oa_logits turns them into the 12 final logits.
    model = trainer.SliceModel(model_config["backbone"], False, model_config["dropout"], len(TARGETS),
                               top_k_fraction=model_config.get("top_k_fraction", 0.1), plane_networks="shared",
                               extra_outputs=trainer.extra_output_count(model_config))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    use_amp = device.type == "cuda"
    print(f"Source: {source}\nTrainer: {trainer_path}\nCheckpoint: {checkpoint_path}\nDevice: {device}", flush=True)

    predictions: dict[str, list[float]] = {}
    oa_parts: dict[str, list[float]] = {}
    stats = {"studies": len(study_ids), "windows": 0, "decode_failures": 0, "empty_studies": 0}
    mean, std = float(trainer.IMAGENET_MEAN), float(trainer.IMAGENET_STD)
    with torch.inference_mode():
        for number, study_id in enumerate(study_ids, start=1):
            images, supports, slots = [], [], []
            for slot, row in enumerate(select_series(series_by_study.get(study_id, []),
                                                     cache_config["series_slots"])):
                ordered = read_series(data_root / "test_series" / study_id / row["SeriesInstanceUID"])
                if ordered.order_source in cache_config["exclude_order_sources"] or not ordered.files:
                    continue
                decoded = []
                for position in select_slice_indices(len(ordered.files), cache_config["slices_per_slot"],
                                                     *cache_config["central_band"]):
                    try:
                        dicom = __import__("pydicom").dcmread(ordered.files[position])
                        pixels = decode_pixels(dicom)
                        if pixels.ndim != 2:
                            raise ValueError(f"Expected 2D slice, got {pixels.shape}")
                        decoded.append((pixels, getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1",
                                        pixel_spacing(dicom)))
                    except Exception as error:
                        stats["decode_failures"] += 1
                        print(f"Decode failure in {study_id}: {type(error).__name__}: {error}", flush=True)
                if not decoded:
                    continue
                low, high = series_bounds([p for p, _, _ in decoded], *cache_config["clip_percentiles"])
                spacing = next((v for _, _, v in decoded if v is not None), None)
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
            model_input = np.concatenate((stack_channels(np.stack(images), np.asarray(slots)),
                                          stack_channels(np.stack(supports), np.asarray(slots))), axis=1)
            batch = torch.from_numpy(np.ascontiguousarray(model_input)).to(device=device, dtype=torch.float32)
            batch[:, :3] = (batch[:, :3] / 255.0 - mean) / std
            batch[:, 3:] /= 255.0
            owner = torch.zeros(len(batch), dtype=torch.long, device=device)
            # All of a study's windows in one call, so top-k sees the whole study.
            with torch.autocast(device_type=device.type, enabled=use_amp):
                raw = model(batch, owner, 1).float()
            final = trainer.compose_oa_logits(raw, list(TARGETS))[0]
            probability = torch.sigmoid(final).cpu().tolist()
            if len(probability) != len(TARGETS) or not np.isfinite(probability).all():
                raise ValueError(f"Invalid predictions for study {study_id}")
            predictions[study_id] = probability
            oa_parts[study_id] = torch.sigmoid(raw[0, len(TARGETS):]).cpu().tolist()   # presence, M, L, PF
            stats["windows"] += len(batch)
            if number % 100 == 0 or number == len(study_ids):
                print(f"{number}/{len(study_ids)} studies; {time.monotonic() - started:.1f}s", flush=True)

    output_path = Path("/kaggle/working/submission.csv")
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["StudyInstanceUID", *TARGETS])
        for study_id in study_ids:
            writer.writerow([study_id, *[f"{value:.8f}" for value in predictions[study_id]]])
    written = read_csv(output_path)
    if [r["StudyInstanceUID"] for r in written] != study_ids or list(written[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("Saved submission rows or columns do not match sample_submission.csv")
    values = np.asarray([[float(r[t]) for t in TARGETS] for r in written])
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Saved submission contains invalid probabilities")
    # Readable OA breakdown (not part of the submission): presence and location per study.
    with Path("/kaggle/working/oa_presence_location.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["StudyInstanceUID", "p_any_oa", "p_medial_given_oa", "p_lateral_given_oa", "p_pf_given_oa"])
        for study_id, parts in oa_parts.items():
            writer.writerow([study_id, *[f"{v:.6f}" for v in parts]])
    summary = {**stats, "device": str(device), "checkpoint_epoch": EXPECTED_EPOCH,
               "report_validation_macro_auc_hpo_labels": EXPECTED_REPORT_VALIDATION_AUC,
               "gold_macro_auc": run_summary.get("gold_evaluation", {}).get("macro_auc"),
               "config_version": config.get("version"), "checkpoint_sha256": checkpoint_hash,
               "seconds": round(time.monotonic() - started, 1), "submission": str(output_path)}
    Path("/kaggle/working/inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
