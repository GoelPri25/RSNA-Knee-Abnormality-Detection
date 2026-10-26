#!/usr/bin/env python3
"""One controlled ResNet run: HPO v1 blend training, Steven-v4 fold-0 scoring."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def exact_path(candidates: list[Path], description: str) -> Path:
    matches = [path for path in candidates if path.exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {description}; found {matches}")
    return matches[0]


def main() -> None:
    input_root = Path("/kaggle/input")
    source = exact_path([
        input_root / "datasets/tumis001/rsna-cnn-hpo-blend-xla-source-v1/skanee",
        input_root / "rsna-cnn-hpo-blend-xla-source-v1/skanee",
    ], "XLA training source")
    labels_path = exact_path([
        input_root / "datasets/tumis001/rsna-knee-hpo-v1-blended-report-labels/soft_labels_hpo_v1_blend.json",
        input_root / "rsna-knee-hpo-v1-blended-report-labels/soft_labels_hpo_v1_blend.json",
    ], "HPO v1 blend labels")
    steven_csv = exact_path([
        input_root / "datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        input_root / "rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    ], "Steven-v4 reference labels")
    folds_path = source / "configs/report_folds_v5.csv"
    gold_split_path = source / "configs/gold_split_v1.csv"
    config_path = source / "configs/cnn_hpo_v1_blend.json"
    trainer = source / "scripts/train_cnn_baseline.py"
    targets = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture",
    ]

    # Exact, known cache mount points. This intentionally avoids recursive scans.
    cache_dirs = [
        input_root / f"notebooks/tumis001/rsna-cnn-top-k-support-cache-shard-{index}/slice_cache"
        for index in range(3)
    ]
    missing_caches = [str(path) for path in cache_dirs if not (path / "cache_summary.json").is_file()]
    if missing_caches:
        raise FileNotFoundError(f"Missing expected cache shards: {missing_caches}")
    for path in (folds_path, gold_split_path, config_path, trainer):
        if not path.is_file():
            raise FileNotFoundError(f"Missing source package file: {path}")

    source_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if source_labels.get("targets_in_order") != targets:
        raise ValueError("HPO blend target order differs from the CNN target schema")
    hpo_ids = {row["study_id"] for row in source_labels["labels"]}
    if len(hpo_ids) != 4349:
        raise ValueError(f"Expected 4,349 report-only HPO labels, found {len(hpo_ids)}")
    with gold_split_path.open(newline="", encoding="utf-8") as stream:
        gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    if hpo_ids & gold_ids:
        raise ValueError("HPO training label file contains gold IDs")

    with folds_path.open(newline="", encoding="utf-8") as stream:
        fold_rows = list(csv.DictReader(stream))
    fold_map = {row["study_id"]: int(row["fold"]) for row in fold_rows}
    validation_ids = {study_id for study_id, fold in fold_map.items() if fold == 0}
    if not validation_ids or not validation_ids.issubset(hpo_ids):
        raise ValueError("Fold-0 report studies are not fully covered by HPO labels")
    if validation_ids & gold_ids:
        raise ValueError("A locked gold study appears in report-fold validation")

    # Keep validation anchored to the exact Steven-v4 labels used in the 0.906
    # reference. Only report-fold IDs enter this validation file; gold stays out.
    reference_items = []
    seen = set()
    with steven_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"StudyInstanceUID", *targets}.issubset(reader.fieldnames or []):
            raise ValueError("Steven-v4 file does not match the expected 12-target schema")
        for row in reader:
            study_id = row["StudyInstanceUID"]
            if study_id not in validation_ids:
                continue
            if study_id in seen:
                raise ValueError(f"Duplicate Steven-v4 row: {study_id}")
            seen.add(study_id)
            probabilities, weights = [], []
            for target in targets:
                try:
                    value = float(row[target])
                except (TypeError, ValueError):
                    value = float("nan")
                valid = value == value and 0.0 <= value <= 1.0
                probabilities.append(value if valid else None)
                weights.append(float(valid))
            reference_items.append({"study_id": study_id,
                                    "probabilities": probabilities, "weights": weights})
    if seen != validation_ids:
        raise ValueError(f"Steven-v4 is missing {len(validation_ids - seen)} fold-0 rows")

    work = Path("/kaggle/working/hpo_v1_blend_fold0")
    work.mkdir(parents=True, exist_ok=True)
    validation_path = work / "steven_v4_fold0_reference.json"
    validation_path.write_text(json.dumps({
        "version": "steven_v4_fold0_validation_only",
        "targets_in_order": targets,
        "labels": reference_items,
    }, allow_nan=False) + "\n", encoding="utf-8")

    print("Experiment: HPO v1 blend labels on the existing support-aware ResNet-34", flush=True)
    print("Accelerator request: Kaggle TPU v5e-8 via PyTorch/XLA (single XLA device)", flush=True)
    print(f"HPO report-only labels: {len(hpo_ids)}; locked gold IDs: {len(gold_ids)}", flush=True)
    print(f"Fold 0 validation: {len(validation_ids)} studies, scored against Steven v4", flush=True)
    print("Training recipe: ResNet-34; 24 windows/study; study batch=8; AdamW; "
          "BCE; plateau-60 schedule; AUC checkpointing/early stop", flush=True)
    print(f"Cache shards: {[str(path) for path in cache_dirs]}", flush=True)
    print(f"Output: {work}", flush=True)

    command = [
        sys.executable, "-u", str(trainer),
        "--cache", *map(str, cache_dirs),
        "--soft-labels", str(labels_path),
        "--validation-soft-labels", str(validation_path),
        "--folds", str(folds_path),
        "--config", str(config_path),
        "--output-dir", str(work / "run"),
        "--validation-fold", "0",
        "--train-slices-per-study", "24",
        "--xla",
    ]
    print("Running: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=source, check=True)
    summary_path = work / "run/run_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Training process ended without run_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps({key: summary.get(key) for key in
                      ("device", "train_studies", "validation_studies", "best", "epochs_run",
                       "stopped_early", "seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
