#!/usr/bin/env python3
"""Kaggle runner for the HPO-v2 label comparison.

It keeps the HPO-v1 +281 ResNet-50 recipe fixed and changes only the train
label probabilities. Fold-0 validation remains the same 825 Steven-v4 studies.
The global study batch stays at 8; when Kaggle supplies two T4s, only the
per-slice ResNet trunk is split across them.
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
LABEL_SHA256 = "6ac9374b9b6d3e9f49a33f30529ca00bf1b50cb1d12f5d60fa5d20a2995dcfc9"
CONFIG_SHA256 = "31125adb466de6490c0797687401f9608b7b92bf769f0d6b1a40b4b690c7568c"
FOLDS_SHA256 = "831e1cc71639c3d5cd59cf61da2546c1f42a2ff644864bf6ac778fbc87af08ad"
BASE_FOLDS_SHA256 = "2ed70719f9423d3a0c3b9ac424841f257d57dda1b9dfd72b26805de8eb4bb6ba"


def exact_path(candidates: list[Path], description: str) -> Path:
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one {description}; found {found}")
    return found[0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    input_root = Path("/kaggle/input")
    trainer_input = exact_path([
        input_root / "datasets/tumis001/rsna-cnn-hpo-blend-xla-source-v1/skanee/scripts/train_cnn_baseline.py",
        input_root / "rsna-cnn-hpo-blend-xla-source-v1/skanee/scripts/train_cnn_baseline.py",
    ], "HPO ResNet training script")
    source = trainer_input.parents[1]

    labels_path = exact_path([
        input_root / "datasets/tumis001/rsna-knee-hpo-v2-provenance-candidate/soft_labels_hpo_v2_provenance_candidate.json",
        input_root / "rsna-knee-hpo-v2-provenance-candidate/soft_labels_hpo_v2_provenance_candidate.json",
    ], "HPO-v2 provenance candidate labels")
    steven_csv = exact_path([
        input_root / "datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        input_root / "rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    ], "Steven-v4 validation labels")
    cache_dirs = [
        input_root / f"notebooks/tumis001/rsna-cnn-top-k-support-cache-shard-{i}/slice_cache"
        for i in range(3)
    ]
    missing = [str(path / "cache_summary.json") for path in cache_dirs
               if not (path / "cache_summary.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Expected attached cache shards are missing: {missing}")

    baseline_folds_path = source / "configs/report_folds_v5.csv"
    folds_path = source / "configs/report_folds_hpo_add281_trainonly.csv"
    gold_split_path = source / "configs/gold_split_v1.csv"
    base_config_path = source / "configs/cnn_hpo_v1_blend.json"
    trainer_path = source / "scripts/train_cnn_baseline.py"
    for path in (baseline_folds_path, folds_path, gold_split_path, base_config_path, trainer_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required experiment input: {path}")

    if sha256(labels_path) != LABEL_SHA256:
        raise ValueError("Attached HPO-v2 labels do not match the audited candidate")
    if sha256(base_config_path) != CONFIG_SHA256:
        raise ValueError("Attached training config differs from the fixed HPO-v1 recipe")
    if sha256(folds_path) != FOLDS_SHA256 or sha256(baseline_folds_path) != BASE_FOLDS_SHA256:
        raise ValueError("Attached fold assignments differ from the fixed experiment split")

    with labels_path.open(encoding="utf-8") as stream:
        labels = json.load(stream)
    if labels.get("targets_in_order") != TARGETS or len(labels.get("labels", [])) != 4349:
        raise ValueError("HPO-v2 schema/coverage check failed")
    label_ids = {row["study_id"] for row in labels["labels"]}
    if len(label_ids) != 4349:
        raise ValueError("HPO-v2 labels contain duplicate study IDs")

    def read_fold_ids(path: Path) -> dict[str, int]:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        mapping = {row["study_id"]: int(row["fold"]) for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"Duplicate study IDs in {path}")
        return mapping

    base_folds = read_fold_ids(baseline_folds_path)
    folds = read_fold_ids(folds_path)
    validation_ids = {study_id for study_id, fold in folds.items() if fold == 0}
    added_ids = label_ids - set(base_folds)
    if (len(base_folds), len(folds), len(validation_ids), len(added_ids),
            sum(fold != 0 for fold in folds.values())) != (4068, 4349, 825, 281, 3524):
        raise ValueError("Expected 3,524 train / fixed 825 validation / 281 train-only IDs")
    if set(folds) != label_ids or validation_ids != {
            study_id for study_id, fold in base_folds.items() if fold == 0}:
        raise ValueError("The fixed validation membership or label coverage changed")

    with gold_split_path.open(newline="", encoding="utf-8") as stream:
        gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    if len(gold_ids) != 58 or (gold_ids & label_ids) or (gold_ids & set(folds)):
        raise ValueError("Gold exclusion lock failed")

    # Construct a validation-only Steven-v4 file. Gold labels are never loaded.
    validation_rows = []
    seen = set()
    with steven_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(TARGETS + ["StudyInstanceUID"]).issubset(reader.fieldnames or []):
            raise ValueError("Steven-v4 columns do not match the fixed 12-target schema")
        for row in reader:
            study_id = row["StudyInstanceUID"]
            if study_id not in validation_ids:
                continue
            if study_id in seen:
                raise ValueError(f"Duplicate Steven-v4 validation row: {study_id}")
            seen.add(study_id)
            probs, weights = [], []
            for target in TARGETS:
                try:
                    value = float(row[target])
                except (TypeError, ValueError):
                    value = float("nan")
                valid = value == value and 0.0 <= value <= 1.0
                probs.append(value if valid else None)
                weights.append(float(valid))
            validation_rows.append({"study_id": study_id, "probabilities": probs, "weights": weights})
    if seen != validation_ids:
        raise ValueError(f"Steven-v4 validation coverage mismatch: {len(validation_ids - seen)} missing")

    work = Path("/kaggle/working/hpo_v2_provenance_resnet50_fold0")
    run_dir = work / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    validation_path = work / "steven_v4_fold0_reference.json"
    validation_path.write_text(json.dumps({
        "version": "steven_v4_fold0_validation_only",
        "targets_in_order": TARGETS,
        "labels": validation_rows,
    }, allow_nan=False) + "\n", encoding="utf-8")
    config = json.loads(base_config_path.read_text(encoding="utf-8"))
    if config["model"].get("backbone") != "resnet34" or config["batch_studies"] != 8:
        raise ValueError("Unexpected reference backbone or batch size")
    config["model"]["backbone"] = "resnet50"
    config["version"] = "hpo_v2_provenance_resnet50_fold0"
    config_path = work / "cnn_hpo_v2_provenance_resnet50.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; this Kaggle run is configured for GPU training")
    torch.cuda.set_device(0)
    gpu_count = torch.cuda.device_count()
    print(f"Accelerator: {gpu_count} GPU(s); primary device {torch.cuda.get_device_name(0)}", flush=True)

    # Keep the global batch at 8. On a two-T4 Kaggle instance only the
    # per-window backbone is data-parallel; this does not change optimization.
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("train_cnn_baseline", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer: {trainer_path}")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    BaseSliceModel = trainer.SliceModel

    if gpu_count >= 2:
        class TwoT4SliceModel(BaseSliceModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[0, 1], output_device=0)

            def state_dict(self, *args, **kwargs):
                state = super().state_dict(*args, **kwargs)
                portable = OrderedDict()
                for key, value in state.items():
                    portable[key.replace("backbone.module.", "backbone.", 1)] = value
                return portable

            def load_state_dict(self, state_dict, *args, **kwargs):
                parallel_state = OrderedDict()
                for key, value in state_dict.items():
                    if key.startswith("backbone.") and not key.startswith("backbone.module."):
                        key = key.replace("backbone.", "backbone.module.", 1)
                    parallel_state[key] = value
                return super().load_state_dict(parallel_state, *args, **kwargs)

        trainer.SliceModel = TwoT4SliceModel
        print("Using both T4s for the slice backbone; global batch remains 8.", flush=True)
    else:
        print("Using one T4; global batch remains 8.", flush=True)

    print("Experiment: HPO-v2 provenance labels; ResNet-50/ImageNet; 24 ordered windows; batch 8.", flush=True)
    print("Split: 3,524 train (includes 281 train-only studies), fixed 825 Steven-v4 validation; 58 gold excluded.", flush=True)
    print("Controls: max 60 epochs; AUC checkpoint/early stop (patience 8, min delta 0.0005); BCE ReduceLROnPlateau.", flush=True)
    print("Checkpoints/history are saved to /kaggle/working after every epoch.", flush=True)
    print(f"Inputs: labels={labels_path}; source={source}; cache_shards={len(cache_dirs)}", flush=True)
    print(f"Output: {run_dir}", flush=True)

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
    print("FINAL_RUN_SUMMARY " + json.dumps({
        key: summary.get(key) for key in
        ("device", "train_studies", "validation_studies", "best", "epochs_run", "stopped_early", "seconds")
    }), flush=True)


if __name__ == "__main__":
    main()
