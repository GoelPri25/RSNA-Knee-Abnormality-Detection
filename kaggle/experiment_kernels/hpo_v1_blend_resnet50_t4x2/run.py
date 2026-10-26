#!/usr/bin/env python3
"""Train the fixed HPO-v1 +281 ResNet-50 experiment across both Kaggle T4s.

DataParallel is applied only to the per-slice ResNet trunk. Each GPU receives
half of the flattened slice windows; study grouping, target-specific pooling,
the global batch of eight studies, loss, validation, and checkpoint selection
remain on the primary GPU and retain the original recipe.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def exact_path(candidates: list[Path], description: str) -> Path:
    found = [path for path in candidates if path.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {description}; found {found}")
    return found[0]


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"This run requires both Kaggle T4 GPUs; found {torch.cuda.device_count()} GPU(s)."
        )
    torch.cuda.set_device(0)
    print("Accelerator: two T4 GPUs; using DataParallel on the slice ResNet only.", flush=True)
    print("GPU 0:", torch.cuda.get_device_name(0), flush=True)
    print("GPU 1:", torch.cuda.get_device_name(1), flush=True)

    input_root = Path("/kaggle/input")
    source = exact_path([
        input_root / "datasets/tumis001/rsna-cnn-hpo-blend-xla-source-v1/skanee",
        input_root / "rsna-cnn-hpo-blend-xla-source-v1/skanee",
    ], "HPO ResNet source")
    labels_path = exact_path([
        input_root / "datasets/tumis001/rsna-knee-hpo-v1-blended-report-labels/soft_labels_hpo_v1_blend.json",
        input_root / "rsna-knee-hpo-v1-blended-report-labels/soft_labels_hpo_v1_blend.json",
    ], "HPO-v1 blend labels")
    steven_csv = exact_path([
        input_root / "datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        input_root / "rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    ], "Steven-v4 validation labels")
    cache_dirs = [
        input_root / f"notebooks/tumis001/rsna-cnn-top-k-support-cache-shard-{i}/slice_cache"
        for i in range(3)
    ]
    missing = [str(path) for path in cache_dirs if not (path / "cache_summary.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing expected cache shards: {missing}")

    baseline_folds_path = source / "configs/report_folds_v5.csv"
    gold_split_path = source / "configs/gold_split_v1.csv"
    trainer_path = source / "scripts/train_cnn_baseline.py"
    base_config_path = source / "configs/cnn_hpo_v1_blend.json"
    for path in (baseline_folds_path, gold_split_path, trainer_path, base_config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing expected source file: {path}")

    with baseline_folds_path.open(newline="", encoding="utf-8") as stream:
        baseline_folds = {row["study_id"]: int(row["fold"]) for row in csv.DictReader(stream)}
    if (len(baseline_folds) != 4068
            or hashlib.sha256(baseline_folds_path.read_bytes()).hexdigest()
            != "2ed70719f9423d3a0c3b9ac424841f257d57dda1b9dfd72b26805de8eb4bb6ba"):
        raise ValueError("Original 4,068-study fold file differs from the fixed reference")
    validation_ids = {study_id for study_id, fold in baseline_folds.items() if fold == 0}
    if len(validation_ids) != 825:
        raise ValueError(f"Expected the fixed 825-study validation fold; found {len(validation_ids)}")
    with gold_split_path.open(newline="", encoding="utf-8") as stream:
        gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    if len(gold_ids) != 58 or gold_ids & set(baseline_folds):
        raise ValueError("Gold lock failed: expected 58 excluded gold IDs and no fold overlap")

    with labels_path.open(encoding="utf-8") as stream:
        hpo_labels = json.load(stream)
    hpo_ids = {row["study_id"] for row in hpo_labels["labels"]}
    if (hpo_labels.get("targets_in_order") != TARGETS or len(hpo_ids) != 4349
            or hashlib.sha256(labels_path.read_bytes()).hexdigest()
            != "ffdffcb2a68e7dbdfbac14c97b1017c28ab7161628cf067040287fa9356c608f"):
        raise ValueError("HPO-v1 label schema or study count changed")
    added_ids = hpo_ids - set(baseline_folds)
    if (len(added_ids) != 281 or validation_ids & gold_ids or not validation_ids <= hpo_ids
            or added_ids & gold_ids):
        raise ValueError("Validation IDs are missing from report labels or overlap gold")

    # The canonical expanded split assigns these exact 281 added IDs to fold 5.
    # Recreate its training/validation membership from the attached source and
    # label files, avoiding dependence on an extra Kaggle input file.
    folds = dict(baseline_folds)
    folds.update({study_id: 5 for study_id in added_ids})

    reference = []
    seen = set()
    with steven_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"StudyInstanceUID", *TARGETS}.issubset(reader.fieldnames or []):
            raise ValueError("Steven-v4 label columns do not match the 12 targets")
        for row in reader:
            study_id = row["StudyInstanceUID"]
            if study_id not in validation_ids:
                continue
            if study_id in seen:
                raise ValueError(f"Duplicate Steven-v4 row: {study_id}")
            seen.add(study_id)
            values, weights = [], []
            for target in TARGETS:
                try:
                    value = float(row[target])
                except (TypeError, ValueError):
                    value = float("nan")
                valid = value == value and 0.0 <= value <= 1.0
                values.append(value if valid else None)
                weights.append(float(valid))
            reference.append({"study_id": study_id, "probabilities": values, "weights": weights})
    if seen != validation_ids or len(validation_ids) != 825:
        raise ValueError(f"Expected exact 825-study validation coverage; found {len(seen)}")

    work = Path("/kaggle/working/hpo_v1_blend_resnet50_t4x2")
    run_dir = work / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    folds_path = work / "report_folds_hpo_add281_trainonly.csv"
    with folds_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("study_id", "fold"))
        writer.writerows(sorted(folds.items()))
    if sum(fold != 0 for fold in folds.values()) != 3524:
        raise ValueError("Expanded split does not contain exactly 3,524 training studies")
    validation_path = work / "steven_v4_fold0_reference.json"
    validation_path.write_text(json.dumps({
        "version": "steven_v4_fold0_validation_only",
        "targets_in_order": TARGETS,
        "labels": reference,
    }, allow_nan=False) + "\n", encoding="utf-8")

    config = json.loads(base_config_path.read_text(encoding="utf-8"))
    if config["model"]["backbone"] != "resnet34":
        raise ValueError("Expected the ResNet-34 reference recipe before changing the backbone")
    config["model"]["backbone"] = "resnet50"
    config["version"] = "cnn_topk_support_v1_resnet50_t4x2"
    config["epochs"] = 60
    config["batch_studies"] = 8
    config_path = work / "cnn_hpo_v1_blend_resnet50_t4x2.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Import the existing, audited trainer and wrap only its independent
    # per-window trunk. This avoids scattering study IDs/labels and preserves
    # the training recipe and portable single-GPU checkpoint key names.
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("train_cnn_baseline", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer: {trainer_path}")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    BaseSliceModel = trainer.SliceModel

    class TwoT4SliceModel(BaseSliceModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backbone = torch.nn.DataParallel(
                self.backbone, device_ids=[0, 1], output_device=0
            )

        def state_dict(self, *args, **kwargs):
            state = super().state_dict(*args, **kwargs)
            portable = OrderedDict()
            for key, value in state.items():
                key = key.replace("backbone.module.", "backbone.", 1)
                portable[key] = value
            return portable

        def load_state_dict(self, state_dict, *args, **kwargs):
            parallel_state = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith("backbone.") and not key.startswith("backbone.module."):
                    key = key.replace("backbone.", "backbone.module.", 1)
                parallel_state[key] = value
            return super().load_state_dict(parallel_state, *args, **kwargs)

    trainer.SliceModel = TwoT4SliceModel
    print(f"Source: {source}", flush=True)
    print(f"Train labels: {len(hpo_ids)} HPO-v1 studies; fold-0 validation: {len(validation_ids)} Steven-v4 studies", flush=True)
    print("Fold assignment: 3,524 train / fixed 825 validation; 281 additions are training-only; gold excluded.", flush=True)
    print("Global batch: 8 studies (DataParallel splits the slice windows over both T4s).", flush=True)
    print("Max epochs: 60; AUC checkpoint/early stopping; BCE ReduceLROnPlateau; checkpoint each epoch.", flush=True)
    print(f"Caches: {[str(path) for path in cache_dirs]}", flush=True)
    print(f"Outputs: {run_dir}", flush=True)

    sys.argv = [
        str(trainer_path), "--cache", *map(str, cache_dirs),
        "--soft-labels", str(labels_path),
        "--validation-soft-labels", str(validation_path),
        "--folds", str(folds_path),
        "--config", str(config_path),
        "--output-dir", str(run_dir),
        "--validation-fold", "0",
        "--train-slices-per-study", "24",
    ]
    trainer.main()

    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Training ended without run_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("FINAL_RUN_SUMMARY", json.dumps({
        key: summary.get(key) for key in
        ("device", "train_studies", "validation_studies", "best", "epochs_run", "stopped_early", "seconds")
    }), flush=True)


if __name__ == "__main__":
    main()
