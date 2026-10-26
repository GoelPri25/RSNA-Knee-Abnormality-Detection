"""Offline dual-T4 TensorRT submission for the Steven v4 label-source checkpoint."""

from __future__ import annotations

import csv
import fnmatch
import functools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


ASSET_SLUGS = (
    "rsna-experiment1-baseline-source",
    "rsna-exp4-steven-label-run-v1",
    "medsiglip-448-vision-onnx",
    "medsiglip-448-onnx-runtime-benchmark",
    "tensorrt-10-16-slim-runtime-bundle",
)
STARTED = time.monotonic()


def log(message: str) -> None:
    print(f"[{time.monotonic() - STARTED:8.1f}s] {message}", flush=True)


def write_fallback_submission(sample_path: Path) -> None:
    """Leave a scoreable file behind even if inference later fails or times out."""
    with sample_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    if not columns or columns[0] != "StudyInstanceUID":
        raise ValueError(f"Unexpected sample submission columns: {columns}")
    with (WORK / "submission.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({columns[0]: row[columns[0]], **{name: 0.5 for name in columns[1:]}})
    log(f"Wrote initial fallback submission for {len(rows)} studies")


@functools.cache
def attached_asset_roots() -> tuple[Path, ...]:
    """Resolve only declared assets; never walk the competition DICOM tree."""
    roots: list[Path] = []
    for slug in ASSET_SLUGS:
        candidates = (
            INPUT / slug,
            INPUT / "datasets" / "tumis001" / slug,
            INPUT / "notebooks" / "tumis001" / slug,
            INPUT / "models" / "tumis001" / slug,
        )
        matches = [path for path in candidates if path.is_dir()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one mounted asset root for {slug}, found {matches}"
            )
        roots.append(matches[0])
    return tuple(roots)


@functools.cache
def attached_asset_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for root in attached_asset_roots():
        for directory, _, names in os.walk(root):
            files.extend(Path(directory) / name for name in names)
    log(f"Inventoried {len(files)} files in {len(attached_asset_roots())} attached assets")
    return tuple(files)


def find_one(name: str) -> Path:
    matches = [path for path in attached_asset_files()
               if fnmatch.fnmatchcase(path.name, name)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name}, found {matches}")
    return matches[0]


def main() -> None:
    data_root = INPUT / "competitions" / "rsna-knee-abnormality-detection"
    sample_path = data_root / "sample_submission.csv"
    if not sample_path.is_file():
        raise FileNotFoundError("Competition sample_submission.csv is not attached")
    write_fallback_submission(sample_path)

    discovery_started = time.monotonic()
    source = find_one("source_manifest.json").parent
    log(f"Resolved attached artifacts in {time.monotonic() - discovery_started:.1f}s")
    sys.path.insert(0, str(source))
    from src.mil.data import Example, load_cache
    from src.mil.train import build_model, predict
    from src.reports.schema import TARGETS

    checkpoint_path = find_one("best_checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint.get("feature_dim") != 1152:
        raise ValueError(f"Unexpected checkpoint feature dimension: {checkpoint.get('feature_dim')}")
    if checkpoint.get("label_version") != "steven_v4":
        raise ValueError(f"Unexpected label source in checkpoint: {checkpoint.get('label_version')}")

    onnx_path = find_one("medsiglip_448_vision_fp16.onnx")
    ort_wheel = find_one("onnxruntime_gpu-*.whl")
    engine_path = find_one("*.engine")
    runtime_manifest = find_one("TENSORRT_SLIM_RUNTIME.json")
    runtime_lib = runtime_manifest.parent / "lib"
    runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    if not runtime.get("fresh_process_validation", {}).get("finite"):
        raise ValueError("Attached TensorRT runtime did not pass fresh-process validation")
    install_started = time.monotonic()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", str(ort_wheel)],
        check=True,
    )
    log(f"Installed ONNX Runtime wheel in {time.monotonic() - install_started:.1f}s")
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        part for part in (str(runtime_lib), environment.get("LD_LIBRARY_PATH", "")) if part
    )
    feature_dir = WORK / "test_features"
    extraction_command = [
        sys.executable,
        str(source / "scripts" / "extract_features_dual.py"),
        "--data-root", str(data_root),
        "--split", "test",
        "--subset", "all",
        "--config", str(source / "configs" / "features_v1.json"),
        "--onnx-path", str(onnx_path),
        "--engine-dir", str(engine_path.parent),
        "--output-dir", str(feature_dir),
        "--workers-per-gpu", "2",
        "--time-budget-hours", "7.5",
    ]
    extraction_started = time.monotonic()
    subprocess.run(extraction_command, env=environment, check=True)
    log(f"Feature extraction subprocess finished in {time.monotonic() - extraction_started:.1f}s")

    dual_summary = json.loads((feature_dir / "dual_feature_summary.json").read_text())
    feature_summaries = [
        json.loads((feature_dir / f"shard_{index}" / "feature_summary.json").read_text())
        for index in range(2)
    ]
    if dual_summary["decode_errors"] != 0:
        log(f"WARNING: skipped {dual_summary['decode_errors']} undecodable DICOM slices")
    observed_dims = {summary["feature_dim"] for summary in feature_summaries
                     if summary.get("feature_dim") is not None}
    if observed_dims and observed_dims != {checkpoint["feature_dim"]}:
        raise ValueError("Test feature dimension differs from the trained checkpoint")
    if {summary.get("backend") for summary in feature_summaries} != {"onnx_tensorrt"}:
        raise ValueError("A feature shard did not use the TensorRT backend")

    cache, cache_summary = load_cache(
        [feature_dir / "shard_0", feature_dir / "shard_1"],
        config.get("exclude_order_sources", ("unordered",)),
    )
    if cache_summary["studies"] != dual_summary["studies_completed"]:
        log(f"WARNING: some completed studies were excluded while loading: {cache_summary}")

    with sample_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        sample_rows = list(reader)
        sample_columns = list(reader.fieldnames or [])
    expected_columns = ["StudyInstanceUID", *TARGETS]
    if sample_columns != expected_columns:
        raise ValueError(
            f"Submission columns differ from the model target contract: {sample_columns}"
        )
    sample_ids = [row["StudyInstanceUID"] for row in sample_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_submission.csv contains duplicate study IDs")
    extra_ids = sorted(set(cache) - set(sample_ids))
    if extra_ids:
        raise ValueError(f"Test cache contains unknown study IDs: {extra_ids[:20]}")

    usable_ids = []
    for study_id in sample_ids:
        if study_id not in cache:
            continue
        if any(len(np.asarray(series.embeddings)) for series in cache[study_id]):
            usable_ids.append(study_id)
    usable_set = set(usable_ids)
    fallback_ids = [study_id for study_id in sample_ids if study_id not in usable_set]
    log(
        f"Usable studies: {len(usable_ids)}/{len(sample_ids)}; "
        f"fallback studies: {len(fallback_ids)}"
    )

    dummy = np.zeros(len(TARGETS), dtype=np.float32)
    examples = [Example(study_id, cache[study_id], dummy, dummy, "test")
                for study_id in usable_ids]
    model = build_model(config, checkpoint["feature_dim"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = predict(model, examples, config, device) if examples else []
    by_id = {row["study_id"]: row for row in predictions}

    probabilities = np.full((len(sample_ids), len(TARGETS)), 0.5, dtype=np.float32)
    embeddings = np.zeros(
        (len(sample_ids), int(config["model"]["z_dim"])), dtype=np.float32
    )
    if usable_ids:
        usable_mil = np.asarray(
            [by_id[study_id]["probabilities"] for study_id in usable_ids], dtype=np.float32
        )
        sample_position = {study_id: index for index, study_id in enumerate(sample_ids)}
        positions = np.asarray([sample_position[study_id] for study_id in usable_ids])
        probabilities[positions] = usable_mil
        embeddings[positions] = np.asarray(
            [by_id[study_id]["z_study"] for study_id in usable_ids], dtype=np.float32
        )
    if probabilities.shape != (len(sample_ids), len(TARGETS)):
        raise ValueError(f"Unexpected probability shape {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Steven v4 model produced non-finite probabilities")

    submission_path = WORK / "submission.csv"
    with submission_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=expected_columns)
        writer.writeheader()
        for row_index, study_id in enumerate(sample_ids):
            writer.writerow({
                "StudyInstanceUID": study_id,
                **dict(zip(TARGETS, probabilities[row_index])),
            })

    np.savez_compressed(
        WORK / "test_study_embeddings.npz",
        study_ids=np.asarray(sample_ids),
        probabilities=probabilities,
        z_study=embeddings,
    )
    (WORK / "test_predictions.json").write_text(
        json.dumps(predictions) + "\n", encoding="utf-8"
    )
    run_summary = {
        "selected_label_arm": "steven_v4",
        "selection_metric": "29-study development macro ROC AUC",
        "development_macro_auc": 0.7974766066251293,
        "checkpoint": str(checkpoint_path),
        "test_feature_summary": dual_summary,
        "feature_shards": feature_summaries,
        "cache_summary": cache_summary,
        "studies": len(sample_ids),
        "model_studies": len(usable_ids),
        "fallback_studies": len(fallback_ids),
        "fallback_study_examples": fallback_ids[:50],
        "probability_shape": list(probabilities.shape),
        "z_study_shape": list(embeddings.shape),
        "probability_range": [float(probabilities.min()), float(probabilities.max())],
        "device": str(device),
        "feature_backend": "ONNX Runtime TensorRT FP16, cached T4 engine",
        "tensorrt_runtime": runtime,
        "submission": str(submission_path),
    }
    (WORK / "inference_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2), flush=True)
    print("Submission contract verified.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "elapsed_seconds": round(time.monotonic() - STARTED, 1),
            "fallback_submission_exists": (WORK / "submission.csv").is_file(),
        }
        (WORK / "submission_failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        log(f"Inference failed; preserving fallback submission: {failure}")
