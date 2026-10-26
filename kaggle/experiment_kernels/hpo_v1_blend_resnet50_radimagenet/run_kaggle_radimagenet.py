from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


ASSET_ROOT = Path(os.environ.get(
    "RSNA_RADIMAGENET_ASSET_ROOT",
    "/kaggle/working/radimagenet_assets",
))
HPO_ROOT = Path(
    "/kaggle/input/datasets/tumis001/rsna-knee-hpo-v1-blended-report-labels"
)
STEVEN_ROOT = Path(
    "/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels"
)
COMPETITION_ROOT = Path(
    "/kaggle/input/competitions/rsna-knee-abnormality-detection"
)
CACHE_ROOTS = [
    Path(f"/kaggle/input/notebooks/tumis001/rsna-cnn-top-k-support-cache-shard-{i}/slice_cache")
    for i in range(3)
]
OUTPUT_SNAPSHOT_DIR = Path("/kaggle/working/radimagenet_checkpoint_snapshots")
RUN_DIR = Path("/kaggle/working/radimagenet_resnet50_fold0")
EXPECTED_WEIGHTS_SHA256 = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
EXPECTED_LABEL_SHA256 = "ffdffcb2a68e7dbdfbac14c97b1017c28ab7161628cf067040287fa9356c608f"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
SNAPSHOT_EVERY_EPOCHS = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files(paths: list[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fixed {label} inputs: {missing}")


def write_steven_validation(steven_csv: Path, folds_path: Path, gold_path: Path,
                            output_path: Path) -> tuple[int, int]:
    with gold_path.open(newline="", encoding="utf-8") as stream:
        gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    with folds_path.open(newline="", encoding="utf-8") as stream:
        fold_map = {row["study_id"]: int(row["fold"]) for row in csv.DictReader(stream)}
    validation_ids = {study_id for study_id, fold in fold_map.items() if fold == 0}
    if len(gold_ids) != 58 or len(validation_ids) != 825 or gold_ids & validation_ids:
        raise RuntimeError("Gold lock or fixed fold-0 validation dimensions failed")

    seen: set[str] = set()
    labels = []
    with steven_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"StudyInstanceUID", *TARGETS}.issubset(reader.fieldnames or []):
            raise ValueError("Steven v4 validation label schema mismatch")
        for row in reader:
            study_id = row["StudyInstanceUID"]
            if study_id not in validation_ids:
                continue
            if study_id in seen:
                raise ValueError(f"Duplicate validation label: {study_id}")
            seen.add(study_id)
            probabilities, weights = [], []
            for target in TARGETS:
                try:
                    value = float(row[target])
                except (TypeError, ValueError):
                    value = float("nan")
                valid = bool(np.isfinite(value) and 0.0 <= value <= 1.0)
                probabilities.append(value if valid else None)
                weights.append(float(valid))
            labels.append({
                "study_id": study_id,
                "probabilities": probabilities,
                "weights": weights,
            })
    if seen != validation_ids:
        raise RuntimeError(f"Steven v4 validation coverage mismatch: {len(validation_ids - seen)} missing")
    output_path.write_text(json.dumps({
        "version": "steven_v4_fold0_validation_only",
        "targets_in_order": TARGETS,
        "labels": labels,
    }, allow_nan=False) + "\n", encoding="utf-8")
    return len(gold_ids), len(validation_ids)


def verify_caches() -> tuple[list[Path], set[str]]:
    counts = []
    all_studies: set[str] = set()
    for cache in CACHE_ROOTS:
        manifest = cache / "manifest.jsonl"
        summary_path = cache / "cache_summary.json"
        if not manifest.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing fixed cache manifest/summary in {cache}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        studies = set()
        with manifest.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    studies.add(json.loads(line)["study_id"])
        expected = int(summary["studies_in_manifest"])
        if len(studies) != expected:
            raise RuntimeError(f"Cache summary mismatch at {cache}: {len(studies)} != {expected}")
        overlap = all_studies & studies
        if overlap:
            raise RuntimeError(f"Study IDs overlap between cache shards: {sorted(overlap)[:5]}")
        counts.append(len(studies))
        all_studies.update(studies)
    if counts != [1456, 1466, 1427] or len(all_studies) != 4349:
        raise RuntimeError(f"Unexpected fixed cache coverage: {counts}, total={len(all_studies)}")
    return CACHE_ROOTS, all_studies


def save_output_snapshot(epoch: int, status: str, run_manifest: dict, log_path: Path) -> None:
    OUTPUT_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "checkpoint_best.pt", "checkpoint_latest.pt", "training_history.csv",
        "training_history.json", "run_summary.json", "learning_curve.png",
    ):
        source = RUN_DIR / name
        if source.is_file():
            shutil.copy2(source, OUTPUT_SNAPSHOT_DIR / name)
    if log_path.is_file():
        shutil.copy2(log_path, OUTPUT_SNAPSHOT_DIR / "training.log")
    run_manifest["status"] = status
    run_manifest["published_epoch"] = epoch
    summary_path = RUN_DIR / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_manifest["best_epoch"] = summary.get("best", {}).get("epoch")
        run_manifest["best_report_validation_auc"] = summary.get("best", {}).get("macro_auc")
        run_manifest["epochs_run"] = summary.get("epochs_run")
    (OUTPUT_SNAPSHOT_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Local Kaggle output snapshot saved at epoch {epoch}: {OUTPUT_SNAPSHOT_DIR} "
        "(Kaggle exposes it when this kernel completes).",
        flush=True,
    )


def main() -> None:
    source = ASSET_ROOT / "source/skanee"
    trainer = source / "scripts/train_cnn_baseline.py"
    weights = ASSET_ROOT / "weights/radimagenet_resnet50.pt"
    hpo_labels = HPO_ROOT / "soft_labels_hpo_v1_blend.json"
    steven_labels = STEVEN_ROOT / "llm_labels_v4_blend.csv"
    folds = source / "configs/report_folds_hpo_add281_trainonly.csv"
    base_folds = source / "configs/report_folds_v5.csv"
    gold_split = source / "configs/gold_split_v1.csv"
    base_config = source / "configs/cnn_hpo_v1_blend.json"
    require_files(
        [trainer, weights, hpo_labels, steven_labels, folds, base_folds,
         gold_split, base_config, *[p / "manifest.jsonl" for p in CACHE_ROOTS]],
        "experiment",
    )
    if not (COMPETITION_ROOT / "train.csv").is_file():
        raise FileNotFoundError(f"Competition input missing at {COMPETITION_ROOT}")
    weight_hash = sha256(weights)
    if weight_hash != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(f"RadImageNet weights checksum mismatch: {weight_hash}")
    label_hash = sha256(hpo_labels)
    if label_hash != EXPECTED_LABEL_SHA256:
        raise RuntimeError(f"HPO-v1 label checksum mismatch: {label_hash}")

    cache_dirs, cached_ids = verify_caches()
    payload = json.loads(hpo_labels.read_text(encoding="utf-8"))
    if payload.get("targets_in_order") != TARGETS or len(payload.get("labels", [])) != 4349:
        raise RuntimeError("HPO-v1 label schema/coverage mismatch")
    label_ids = {record["study_id"] for record in payload["labels"]}
    if label_ids != cached_ids:
        raise RuntimeError("HPO-v1 label IDs do not exactly match cached report-only studies")

    with base_folds.open(newline="", encoding="utf-8") as stream:
        base_map = {row["study_id"]: int(row["fold"]) for row in csv.DictReader(stream)}
    with folds.open(newline="", encoding="utf-8") as stream:
        fold_rows = list(csv.DictReader(stream))
    fold_map = {row["study_id"]: int(row["fold"]) for row in fold_rows}
    if len(fold_map) != len(fold_rows) or set(fold_map) != cached_ids:
        raise RuntimeError("Expanded fold map does not exactly cover the 4,349 cached studies")
    validation_ids = {study_id for study_id, fold in fold_map.items() if fold == 0}
    added_ids = cached_ids - set(base_map)
    if (len(base_map), len(validation_ids), len(added_ids),
            sum(fold != 0 for fold in fold_map.values())) != (4068, 825, 281, 3524):
        raise RuntimeError("Unexpected +281 train-only split dimensions")
    if validation_ids != {k for k, v in base_map.items() if v == 0}:
        raise RuntimeError("Fold-0 validation changed from the 0.918 reference")
    if any(fold_map[study_id] == 0 for study_id in added_ids):
        raise RuntimeError("At least one of the 281 additions entered validation")
    with gold_split.open(newline="", encoding="utf-8") as stream:
        gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    if len(gold_ids) != 58 or gold_ids & cached_ids:
        raise RuntimeError("Gold studies are not fully locked away from this run")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    validation_labels = Path("/kaggle/working/steven_v4_fold0_reference.json")
    gold_count, validation_count = write_steven_validation(
        steven_labels, folds, gold_split, validation_labels
    )
    config = json.loads(base_config.read_text(encoding="utf-8"))
    config["model"]["backbone"] = "resnet50"
    config["model"]["pretrained"] = False
    config["model"]["radimagenet_checkpoint"] = str(weights)
    config["version"] = "cnn_topk_support_v1_resnet50_radimagenet"
    config_path = Path("/kaggle/working/cnn_hpo_v1_blend_resnet50_radimagenet.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Kaggle may expose two T4 devices. This matched reference run deliberately
    # uses a single T4 and effective study batch 8, as in the Modal control.
    print(f"Visible GPUs: {torch.cuda.device_count()}; selected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is unavailable; refusing to start a slow CPU training run")
    torch.cuda.set_device(0)
    print("Controlled change: RadImageNet trunk initialization only. Fresh task head and optimizer.", flush=True)
    print("Fixed: HPO-v1 labels, +281 train-only, original 825-study fold-0 validation, ResNet-50, 24 windows, batch 8, loss, optimizer, scheduler, early stopping.", flush=True)
    print(f"Verified data: cache={len(cached_ids)}, train=3524, validation={validation_count}, gold locked={gold_count}, additions={len(added_ids)}", flush=True)

    run_manifest = {
        "run_id": "hpo_v1_blend_add281_resnet50_radimagenet_fold0_kaggle",
        "status": "training",
        "experiment": "RadImageNet-initialized ResNet-50; HPO-v1 labels; +281 train-only",
        "initialization": "official RadImageNet trunk; fresh task head and optimizer",
        "initialization_checkpoint_sha256": weight_hash,
        "label_sha256": label_hash,
        "validation": "same fixed 825 fold-0 report-only studies scored against Steven v4",
        "train_studies": 3524,
        "validation_studies": validation_count,
        "locked_gold_studies": gold_count,
        "gold_labels_loaded": False,
        "raw_dicoms_transferred": False,
        "gpu": torch.cuda.get_device_name(0),
        "visible_gpus": torch.cuda.device_count(),
        "effective_batch_studies": 8,
        "train_windows_per_study": 24,
        "checkpoint_snapshot_interval_epochs": SNAPSHOT_EVERY_EPOCHS,
        "early_stopping": {"metric": "report-validation macro AUC", "patience": 8, "min_delta": 0.0005},
        "scheduler": {"type": "ReduceLROnPlateau", "metric": "report-validation BCE", "factor": 0.5, "patience": 2, "cooldown": 1, "min_lr": 1e-6},
    }
    config_sha = sha256(config_path)
    run_manifest["config_sha256"] = config_sha
    log_path = RUN_DIR / "training.log"
    command = [
        sys.executable, "-u", str(trainer),
        "--cache", *map(str, cache_dirs),
        "--soft-labels", str(hpo_labels),
        "--validation-soft-labels", str(validation_labels),
        "--folds", str(folds),
        "--config", str(config_path),
        "--output-dir", str(RUN_DIR),
        "--validation-fold", "0",
        "--train-slices-per-study", "24",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ASSET_ROOT / "source")
    print("Command:", " ".join(command), flush=True)
    process = subprocess.Popen(
        command, cwd=source, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, bufsize=1,
    )
    marker = re.compile(r"^CHECKPOINTS_READY epoch=(\d+)$")
    last_published = -1
    assert process.stdout is not None
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                match = marker.match(line.strip())
                if not match:
                    continue
                epoch = int(match.group(1))
                if epoch == 0 or epoch % SNAPSHOT_EVERY_EPOCHS == 0:
                    if epoch != last_published:
                        save_output_snapshot(epoch, "training", run_manifest, log_path)
                        last_published = epoch
        code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    if code:
        save_output_snapshot(-1, f"interrupted_exit_{code}", run_manifest, log_path)
        raise RuntimeError(f"Training process exited with code {code}; local checkpoint snapshots were saved in /kaggle/working when possible")
    if not (RUN_DIR / "run_summary.json").is_file():
        raise RuntimeError("Training exited without a run_summary.json")
    save_output_snapshot(int(json.loads((RUN_DIR / "run_summary.json").read_text())["epochs_run"]) - 1,
                         "completed", run_manifest, log_path)
    print("Training complete. Summary:", RUN_DIR / "run_summary.json", flush=True)


if __name__ == "__main__":
    main()
