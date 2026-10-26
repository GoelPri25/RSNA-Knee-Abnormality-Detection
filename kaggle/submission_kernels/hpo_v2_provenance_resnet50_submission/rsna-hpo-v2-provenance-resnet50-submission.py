"""Submit the best checkpoint from the HPO-v2 provenance ResNet-50 run."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


EXPECTED_CHECKPOINT_SHA256 = "e5edb5ffa08a24037d6141cd9a4bcba45c3ab05b3082c6935533368822c6f076"
EXPECTED_CHECKPOINT_EPOCH = 15
EXPECTED_REPORT_VALIDATION_AUC = 0.8897419150793966
EXPECTED_TRAIN_STUDIES = 3524
EXPECTED_VALIDATION_STUDIES = 825
EXPECTED_SOURCE_SHA256 = {
    "model/scripts/train_cnn_baseline.py": "b8f8302319afbcf9970f0cf5b806d8fb78e72ef6e169b276274ddf7cd5e0ba85",
    "model/src/images/dataset.py": "c497fab72c90516153621363f1e6d8f00595aceb90d23cfafa7301d4bae560bd",
    "model/src/reports/schema.py": "9e766595576b8cda917fe693419ebf5f5ae4ade3418f01b669d5133fc8cc4cca",
    "preprocess/src/features/dicom.py": "5786bae8b32cdc52c6f4002fe1286240fca7efbae5d4b79eda713a9860ac78c1",
    "preprocess/src/images/prepare.py": "e77a9b4f6faedcdcef15dc85d66ad0dfaef129d03b66cf3de94586db88e77353",
    "preprocess/src/images/dataset.py": "c497fab72c90516153621363f1e6d8f00595aceb90d23cfafa7301d4bae560bd",
    "preprocess/src/reports/schema.py": "9e766595576b8cda917fe693419ebf5f5ae4ade3418f01b669d5133fc8cc4cca",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_fixed_root(candidates: list[Path], required: tuple[str, ...],
                       description: str) -> Path:
    """Resolve one of the documented Kaggle mount aliases; never search recursively."""
    matches = [root for root in candidates if all((root / item).exists() for item in required)]
    if not matches:
        detail = {
            str(root): [item for item in required if not (root / item).exists()]
            for root in candidates
        }
        raise FileNotFoundError(f"No fixed {description} root contains all required files: {detail}")
    if len(matches) > 1:
        signatures = [tuple(sha256_file(root / item) if (root / item).is_file() else "<directory>"
                            for item in required) for root in matches]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise RuntimeError(f"Multiple non-identical fixed roots found for {description}: {matches}")
    return matches[0]


def main() -> None:
    started = time.monotonic()
    input_root = Path("/kaggle/input")
    # The mounted HPO source contains the exact model/trainer but not the DICOM
    # preprocessing modules. The top-k source contains the preprocessing code;
    # its dataset.py and schema.py are byte-identical to the HPO source copies.
    model_source = resolve_fixed_root([
        input_root / "datasets/tumis001/rsna-cnn-hpo-blend-xla-source-v1/skanee",
        input_root / "rsna-cnn-hpo-blend-xla-source-v1/skanee",
    ], ("scripts/train_cnn_baseline.py", "src/images/dataset.py", "src/reports/schema.py"),
       "HPO model source")
    preprocess_source = resolve_fixed_root([
        input_root / "datasets/tumis001/rsna-cnn-topk-support-source-v1/skanee",
        input_root / "rsna-cnn-topk-support-source-v1/skanee",
    ], ("src/features/dicom.py", "src/images/prepare.py",
        "src/images/dataset.py", "src/reports/schema.py"), "preprocessing source")
    artifact_dir = resolve_fixed_root([
        input_root / "datasets/tumis001/rsna-hpo-v2-provenance-resnet50-fold0-best",
        input_root / "rsna-hpo-v2-provenance-resnet50-fold0-best",
    ], ("checkpoint_best.pt", "run_summary.json"), "checkpoint artifact")
    checkpoint_path = artifact_dir / "checkpoint_best.pt"
    run_summary_path = artifact_dir / "run_summary.json"
    for label, path in (
        ("model/scripts/train_cnn_baseline.py", model_source / "scripts/train_cnn_baseline.py"),
        ("model/src/images/dataset.py", model_source / "src/images/dataset.py"),
        ("model/src/reports/schema.py", model_source / "src/reports/schema.py"),
        ("preprocess/src/features/dicom.py", preprocess_source / "src/features/dicom.py"),
        ("preprocess/src/images/prepare.py", preprocess_source / "src/images/prepare.py"),
        ("preprocess/src/images/dataset.py", preprocess_source / "src/images/dataset.py"),
        ("preprocess/src/reports/schema.py", preprocess_source / "src/reports/schema.py"),
    ):
        actual = sha256_file(path)
        if actual != EXPECTED_SOURCE_SHA256[label]:
            raise ValueError(f"Unexpected {label} SHA-256 at {path}: {actual}")
    print(f"Verified model source: {model_source}\nVerified preprocessing source: "
          f"{preprocess_source}\nVerified checkpoint artifact: {artifact_dir}", flush=True)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    best_record = run_summary.get("best", {})
    if best_record.get("epoch") != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError("Run summary does not select the pinned checkpoint epoch")
    if abs(best_record.get("macro_auc", -1) - EXPECTED_REPORT_VALIDATION_AUC) > 1e-12:
        raise ValueError("Published best AUC differs from the pinned submission checkpoint")
    if (run_summary.get("train_studies"), run_summary.get("validation_studies")) != (
            EXPECTED_TRAIN_STUDIES, EXPECTED_VALIDATION_STUDIES):
        raise ValueError("Run summary does not match the pinned training/validation split")
    data_root = resolve_fixed_root([
        input_root / "competitions/rsna-knee-abnormality-detection",
        input_root / "rsna-knee-abnormality-detection",
    ], ("sample_submission.csv", "test_series.csv", "test_series"), "competition data")

    # Bind the shared `src` package to the verified preprocessing tree. The
    # HPO trainer imports dataset.py/schema.py, verified byte-identical above.
    stale_src_modules = [module_name for module_name in tuple(sys.modules)
                         if module_name == "src" or module_name.startswith("src.")]
    for module_name in stale_src_modules:
        del sys.modules[module_name]
    print(f"Cleared {len(stale_src_modules)} preloaded src module(s) before binding verified code.",
          flush=True)
    package_path = preprocess_source / "src"
    package_spec = importlib.util.spec_from_file_location(
        "src", package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    if package_spec is None or package_spec.loader is None:
        raise ImportError(f"Could not load verified source package: {package_path}")
    source_package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_spec.name] = source_package
    package_spec.loader.exec_module(source_package)
    sys.path.insert(0, str(preprocess_source))
    from src.features.dicom import (
        OrderedSeries, decode_pixels, order_headers, series_bounds,
    )
    from src.images.dataset import stack_channels
    from src.images.prepare import (
        physical_crop_with_support, pixel_spacing, select_series,
        select_slice_indices, to_uint8,
    )
    from src.reports.schema import TARGETS
    trainer_path = model_source / "scripts/train_cnn_baseline.py"
    trainer_spec = importlib.util.spec_from_file_location("hpo_v2_train_cnn_baseline", trainer_path)
    if trainer_spec is None or trainer_spec.loader is None:
        raise ImportError(f"Could not load the exact training model source: {trainer_path}")
    trainer_module = importlib.util.module_from_spec(trainer_spec)
    sys.modules[trainer_spec.name] = trainer_module
    trainer_spec.loader.exec_module(trainer_module)
    IMAGENET_MEAN = trainer_module.IMAGENET_MEAN
    IMAGENET_STD = trainer_module.IMAGENET_STD
    SliceModel = trainer_module.SliceModel
    import pydicom

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain config and state_dict")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected best-checkpoint SHA-256: {checkpoint_hash}")
    if checkpoint.get("epoch") != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError(f"Expected selected checkpoint epoch {EXPECTED_CHECKPOINT_EPOCH}; "
                         f"got {checkpoint.get('epoch')}")
    if checkpoint["epoch"] != best_record["epoch"]:
        raise ValueError("Checkpoint epoch differs from the best epoch in published history")
    config = checkpoint["config"]
    if config.get("version") != "hpo_v2_provenance_resnet50_fold0":
        raise ValueError(f"Unexpected checkpoint config: {config.get('version')}")
    if config.get("model", {}).get("backbone") != "resnet50":
        raise ValueError("Pinned checkpoint is not a ResNet-50 model")
    cache_config = config["cache"]
    model_config = config["model"]
    if (int(cache_config["series_slots"]) * int(cache_config["slices_per_slot"]) > 64):
        raise ValueError("Inference chunk size would split study-level top-k pooling")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SliceModel(model_config["backbone"], False, model_config["dropout"], len(TARGETS),
                       model_config.get("top_k_fraction", 0.1))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    use_amp = device.type == "cuda"
    if use_amp:
        # NHWC convolution kernels are faster on T4 for this fixed 224px input.
        # Keep autotuning off because the three studies can have different batch
        # sizes, and tuning overhead can exceed the work saved in this short run.
        torch.backends.cudnn.benchmark = False
        model.to(memory_format=torch.channels_last)
    print(f"Model source: {model_source}\nPreprocessing source: {preprocess_source}\n"
          f"Checkpoint: {checkpoint_path}\nData: {data_root}", flush=True)
    print(f"Studies: {len(study_ids)} | checkpoint epoch: {checkpoint.get('epoch')} | "
          f"device: {device} | input: 6 x {cache_config['image_size']} x "
          f"{cache_config['image_size']}", flush=True)

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
            images: list[np.ndarray] = []
            supports: list[np.ndarray] = []
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
                # Match cache creation: the first available spacing is shared
                # by every selected slice in this series.
                spacing = next((value for _, _, value in decoded if value is not None), None)
                for pixels, inverted, _slice_spacing in decoded:
                    support = np.isfinite(pixels)
                    image = to_uint8(pixels, low, high, inverted)
                    image[~support] = 0
                    cropped, cropped_support = physical_crop_with_support(
                        image.astype(np.float32), support,
                        spacing,
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
            if len(batch) > 64:
                raise ValueError("Study exceeds the single-chunk limit for exact study-level pooling")
            batch[:, :3] = (batch[:, :3] / 255.0 - mean) / std
            batch[:, 3:] /= 255.0
            chunks = []
            timer = time.monotonic()
            for begin in range(0, len(batch), 64):
                chunk = batch[begin:begin + 64]
                owner = torch.zeros(len(chunk), dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    if use_amp:
                        chunk = chunk.contiguous(memory_format=torch.channels_last)
                    chunks.append(model(chunk, owner, 1).float() * len(chunk))
            logits = torch.cat(chunks).sum(dim=0) / len(batch)
            stage_seconds["model"] += time.monotonic() - timer
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
               "training_label_source": "HPO-v2 provenance candidate report labels",
               "report_validation_macro_auc": EXPECTED_REPORT_VALIDATION_AUC,
               "training_studies": EXPECTED_TRAIN_STUDIES,
               "validation_studies": EXPECTED_VALIDATION_STUDIES,
               "config_version": config.get("version"),
               "pooling": model_config.get("pooling"),
               "top_k_fraction": model_config.get("top_k_fraction", 0.1),
               "stage_seconds": {key: round(value, 3)
                                 for key, value in stage_seconds.items()},
               "seconds": round(time.monotonic() - started, 1),
               "submission": str(output_path)}
    Path("/kaggle/working/inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
