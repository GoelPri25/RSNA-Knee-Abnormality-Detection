#!/usr/bin/env python3
"""Balanced plane-split ResNet-34 on the wide-band cache, 0.918 recipe otherwise.

Every slice inside the central 10-90% of each stack (cap 40) instead of 12 spaced over 20-80%.
Same model and settings as the balanced plane-split run; Steven-v4 fold-0 validation.
Capped at 16 epochs with the schedule unchanged, and the 58 gold studies are scored once at
the end from the wide-band gold cache.

With MCL_FROM_STEVEN_V4 on, MCL training labels that the HPO blend leaves unsure (0.3-0.7)
take Steven v4's value instead: those reports describe sprains and partial injuries, which
gold and Steven v4 count as MCL-positive. Every other label is left as it is.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


MCL_FROM_STEVEN_V4 = False

def exact_path(candidates: list[Path], description: str) -> Path:
    matches = [path for path in candidates if path.exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {description}; found {matches}")
    return matches[0]


def main() -> None:
    input_root = Path("/kaggle/input")
    # Kaggle mounts datasets at different depths; locate the package by its trainer file.
    trainers = sorted(input_root.rglob("scripts/train_cnn_topk_attention_sliceorder.py"))
    if len(trainers) != 1:
        raise FileNotFoundError(f"Expected one train_cnn_topk_attention_sliceorder.py under {input_root}; found {trainers}")
    source = trainers[0].parents[1]
    labels_path = exact_path([
        input_root / "datasets/tumis001/rsna-knee-hpo-v1-blended-report-labels/soft_labels_hpo_v1_blend.json",
        input_root / "rsna-knee-hpo-v1-blended-report-labels/soft_labels_hpo_v1_blend.json",
    ], "HPO v1 blend labels")
    steven_csv = exact_path([
        input_root / "datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        input_root / "rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    ], "Steven-v4 reference labels")
    folds_path = source / "configs/report_folds_hpo_add281_trainonly.csv"
    gold_split_path = source / "configs/gold_split_v1.csv"
    config_path = source / "configs/cnn_hpo_v1_blend_plane_split_balanced_wideband_e16.json"
    trainer = source / "scripts/train_cnn_topk_attention_sliceorder.py"
    targets = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture",
    ]

    # The wide-band cache kernels (every slice in the central 10-90% band).
    cache_dirs = []
    for index in range(4):
        found = sorted([*input_root.glob(f"notebooks/*/rsna-knee-wideband-cache-shard-{index}/**/cache_summary.json"),
                        *input_root.glob(f"rsna-knee-wideband-cache-shard-{index}/**/cache_summary.json")])
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one wide-band cache shard {index}; found {found}")
        summary = json.loads(found[0].read_text())
        if summary["cache"].get("all_slices_in_band") is not True or summary["cache"].get("central_band") != [0.1, 0.9]:
            raise ValueError(f"{found[0]} is not the wide-band cache")
        cache_dirs.append(found[0].parent)
    gold_found = sorted([*input_root.glob("notebooks/*/rsna-knee-wideband-cache-gold/**/cache_summary.json"),
                         *input_root.glob("rsna-knee-wideband-cache-gold/**/cache_summary.json")])
    if len(gold_found) != 1:
        raise FileNotFoundError(f"Expected one wide-band gold cache; found {gold_found}")
    gold_settings = json.loads(gold_found[0].read_text())["cache"]
    geometry = ("field_of_view_mm", "image_size", "series_slots", "slices_per_slot",
                "central_band", "all_slices_in_band", "clip_percentiles")
    if any(gold_settings.get(key) != summary["cache"].get(key) for key in geometry):
        raise ValueError("The gold cache was built with different settings from the training cache")
    gold_cache_dir = gold_found[0].parent
    train_csv = exact_path([
        input_root / "competitions/rsna-knee-abnormality-detection/train.csv",
        input_root / "rsna-knee-abnormality-detection/train.csv",
    ], "competition train.csv")
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

    work = Path("/kaggle/working/hpo_v1_plane_split_balanced_wideband"
                + ("_mcl_v4" if MCL_FROM_STEVEN_V4 else "") + "_fold0")
    work.mkdir(parents=True, exist_ok=True)
    if MCL_FROM_STEVEN_V4:
        mcl = targets.index("MCL")
        with steven_csv.open(newline="", encoding="utf-8-sig") as stream:
            steven_mcl = {row["StudyInstanceUID"]: float(row["MCL"]) for row in csv.DictReader(stream)
                          if row.get("MCL") not in (None, "")}
        moved = {"positive": 0, "unsure": 0, "negative": 0}
        for row in source_labels["labels"]:
            value = row["probabilities"][mcl]
            if 0.3 < value < 0.7 and row["study_id"] in steven_mcl:
                new = steven_mcl[row["study_id"]]
                row["probabilities"][mcl] = new
                moved["positive" if new >= 0.7 else "negative" if new <= 0.3 else "unsure"] += 1
        # Measured offline on these exact files: 552 unsure MCL labels, 485 of them positive in v4.
        if sum(moved.values()) != 552 or moved["positive"] != 485:
            raise ValueError(f"MCL relabelling differs from the offline check: {moved}")
        source_labels["version"] = source_labels.get("version", "hpo_v1_blend") + "_mcl_from_steven_v4"
        labels_path = work / "soft_labels_hpo_v1_blend_mcl_v4.json"
        labels_path.write_text(json.dumps(source_labels) + "\n", encoding="utf-8")
        print(f"MCL labels taken from Steven v4 where HPO was unsure: {moved}", flush=True)

    train_ids = {study_id for study_id, fold in fold_map.items() if fold != 0}
    if (len(validation_ids), len(train_ids)) != (825, 3524):
        raise ValueError(f"Expected the 0.918 split (825 validation, 3,524 train); got "
                         f"{len(validation_ids)} / {len(train_ids)}")
    validation_path = work / "steven_v4_fold0_reference.json"
    validation_path.write_text(json.dumps({
        "version": "steven_v4_fold0_validation_only",
        "targets_in_order": targets,
        "labels": reference_items,
    }, allow_nan=False) + "\n", encoding="utf-8")

    print("Experiment: 0.918 recipe (HPO v1 blend +281 train-only) with balanced plane-split on the wide-band cache", flush=True)
    print("Accelerator: CUDA GPU; trained from ImageNet, no warm start", flush=True)
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
        "--gold-cache", str(gold_cache_dir),
        "--gold-train-csv", str(train_csv),
    ]
    print("Running: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=source, check=True)
    summary_path = work / "run/run_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Training process ended without run_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps({key: summary.get(key) for key in
                      ("device", "train_studies", "validation_studies", "best", "epochs_run",
                       "stopped_early", "seconds", "plane_trust", "gold_evaluation")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
