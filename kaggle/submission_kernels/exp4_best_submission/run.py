"""Run the frozen MedSigLIP MIL checkpoint on the complete competition test set."""

from __future__ import annotations

import csv
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def find_one(name: str) -> Path:
    matches = list(INPUT.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name}, found {matches}")
    return matches[0]


def main() -> None:
    source = find_one("scripts/train_mil.py").parents[1]
    sys.path.insert(0, str(source))
    from src.mil.data import Example, load_cache
    from src.mil.train import build_model, predict
    from src.reports.schema import TARGETS

    data_root = INPUT / "competitions" / "rsna-knee-abnormality-detection"
    sample_path = data_root / "sample_submission.csv"
    if not sample_path.is_file():
        raise FileNotFoundError("Competition sample_submission.csv is not attached")

    # Select the point-estimate winner using only the prior 29-study
    # development-gold comparison; evaluation gold never enters selection.
    selected_arm = "steven_v4"
    scoring_summary = {
        "candidate_selected_by_dev_macro_auc": selected_arm,
        "development_studies": 29,
        "macro_auc": 0.7974766066251293,
        "confidence_v2_control_macro_auc": 0.7677101959576049,
        "flight_v4_macro_auc": 0.7757763579517852,
        "evaluation_studies_scored": 0,
    }
    artifact_slugs = {
        "confidence_v2_control": "rsna-exp4-report-label-comparison",
        "flight_v4": "rsna-exp4-alternate-label-runs-v1",
        "steven_v4": "rsna-exp4-steven-label-run-v1",
    }
    if selected_arm not in artifact_slugs:
        raise ValueError(f"Unexpected candidate selected by development AUC: {selected_arm}")
    candidate_dir = INPUT / "datasets/tumis001" / artifact_slugs[selected_arm]
    checkpoint_hits = list(candidate_dir.rglob("best_checkpoint.pt")) if candidate_dir.is_dir() else []
    if len(checkpoint_hits) != 1:
        raise FileNotFoundError(f"Expected one best checkpoint for {selected_arm} beneath {candidate_dir}, found {checkpoint_hits}")
    checkpoint_path = checkpoint_hits[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    provenance_path = find_one("MODEL_PROVENANCE.json")
    model_dir = provenance_path.parent
    feature_dir = WORK / "test_features"
    sys.argv = [
        str(source / "scripts" / "extract_features.py"),
        "--data-root", str(data_root),
        "--split", "test",
        "--subset", "all",
        "--shard-index", "0",
        "--shard-count", "1",
        "--config", str(source / "configs" / "features_v1.json"),
        "--model-dir", str(model_dir),
        "--output-dir", str(feature_dir),
        "--time-budget-hours", "11",
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")

    summary_path = feature_dir / "feature_summary.json"
    feature_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if feature_summary["shard"] != [0, 1]:
        raise ValueError(f"Test cache is partial: {feature_summary['shard']}")
    if feature_summary["studies_in_manifest"] != feature_summary["studies_requested_this_run"]:
        raise ValueError("Test feature manifest does not cover every requested study")
    if feature_summary["run_stats"]["decode_errors"] != 0:
        raise ValueError("Test feature cache contains DICOM decode errors")
    if feature_summary["feature_dim"] != checkpoint["feature_dim"]:
        raise ValueError("Test feature dimension differs from the trained checkpoint")

    cache, cache_summary = load_cache(
        [feature_dir], config.get("exclude_order_sources", ("unordered",))
    )
    if cache_summary["studies"] != feature_summary["studies_in_manifest"]:
        raise ValueError(f"Incomplete usable test cache: {cache_summary}")

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
    if set(sample_ids) != set(cache):
        raise ValueError(
            f"Test feature IDs do not match submission IDs: "
            f"missing={sorted(set(sample_ids) - set(cache))}, "
            f"extra={sorted(set(cache) - set(sample_ids))}"
        )

    dummy = np.zeros(len(TARGETS), dtype=np.float32)
    examples = [Example(study_id, cache[study_id], dummy, dummy, "test")
                for study_id in sample_ids]
    model = build_model(config, checkpoint["feature_dim"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = predict(model, examples, config, device)
    by_id = {row["study_id"]: row for row in predictions}

    submission_path = WORK / "submission.csv"
    with submission_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=expected_columns)
        writer.writeheader()
        for study_id in sample_ids:
            probabilities = by_id[study_id]["probabilities"]
            if len(probabilities) != len(TARGETS) or not np.isfinite(probabilities).all():
                raise ValueError(f"Invalid probabilities for {study_id}")
            writer.writerow({"StudyInstanceUID": study_id, **dict(zip(TARGETS, probabilities))})

    probabilities = np.asarray([by_id[s]["probabilities"] for s in sample_ids], dtype=np.float32)
    embeddings = np.asarray([by_id[s]["z_study"] for s in sample_ids], dtype=np.float32)
    if probabilities.shape != (len(sample_ids), len(TARGETS)):
        raise ValueError(f"Unexpected probability shape {probabilities.shape}")
    np.savez_compressed(
        WORK / "test_study_embeddings.npz",
        study_ids=np.asarray(sample_ids), probabilities=probabilities, z_study=embeddings,
    )
    (WORK / "test_predictions.json").write_text(
        json.dumps(predictions) + "\n", encoding="utf-8"
    )
    run_summary = {
        "selected_label_arm": selected_arm,
        "selection_metric": "29-study development macro ROC AUC",
        "selection_note": "Steven v4 selected from the private three-way Kaggle score kernel output.",
        "selection_summary": scoring_summary,
        "checkpoint": str(checkpoint_path),
        "test_feature_summary": feature_summary,
        "cache_summary": cache_summary,
        "studies": len(sample_ids),
        "probability_shape": list(probabilities.shape),
        "z_study_shape": list(embeddings.shape),
        "probability_range": [float(probabilities.min()), float(probabilities.max())],
        "device": str(device),
        "submission": str(submission_path),
    }
    (WORK / "inference_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2), flush=True)
    print("Submission contract verified.", flush=True)


if __name__ == "__main__":
    main()
