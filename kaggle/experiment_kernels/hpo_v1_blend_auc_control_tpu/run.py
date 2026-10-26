#!/usr/bin/env python3
"""One fixed-input Kaggle TPU run with AUC-aligned LR and stopping control."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


EXPERIMENT = "hpo_v1_blend_add281_auc_control_tpu_fold0"
ARTIFACT_DATASET_ID = "tumis001/rsna-hpo-blend-auc-control-tpu-fold0"
PUBLISH_EVERY_EPOCHS = 2
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_path(candidates: list[Path], description: str) -> Path:
    matches = [path for path in candidates if path.exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {description}; found {matches}")
    return matches[0]


def kaggle_cli(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["kaggle", *args], text=True, capture_output=True, check=False)
    if check and result.returncode:
        detail = (result.stderr or result.stdout)[-1600:]
        raise RuntimeError(f"Kaggle CLI failed ({result.returncode}): {detail}")
    return result


def configure_kaggle_api() -> None:
    """Require working Kaggle credentials before spending TPU time."""
    if shutil.which("kaggle") is None:
        raise RuntimeError("Kaggle CLI is unavailable; refusing to begin an uncheckpointed run.")
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as error:
        raise RuntimeError("Kaggle Secrets client is unavailable.") from error
    secrets = UserSecretsClient()
    token = None
    try:
        token = secrets.get_secret("KAGGLE_API_TOKEN")
    except Exception:
        pass
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    else:
        try:
            os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
        except Exception as error:
            raise RuntimeError(
                "Add Kaggle Secret KAGGLE_API_TOKEN, or KAGGLE_USERNAME and KAGGLE_KEY, "
                "before running. Credentials are checked before TPU training starts."
            ) from error
    check = kaggle_cli(["datasets", "list", "--mine", "--csv"], check=False)
    if check.returncode:
        raise RuntimeError(
            "Kaggle API authentication failed before TPU training. Check the Kaggle secret "
            "and notebook internet setting."
        )
    print("Kaggle API credentials verified; interval checkpoint publishing is enabled.", flush=True)


def main() -> None:
    configure_kaggle_api()
    input_root = Path("/kaggle/input")
    source_input = exact_path([
        input_root / "datasets/tumis001/rsna-cnn-hpo-auc-control-xla-source-v1",
        input_root / "rsna-cnn-hpo-auc-control-xla-source-v1",
    ], "AUC-control source dataset")
    source = source_input / "skanee"
    if not source.is_dir():
        package_zip = exact_path([source_input / "skanee.zip"], "packaged AUC-control source")
        extraction_root = Path("/kaggle/working") / EXPERIMENT / "source"
        extraction_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(package_zip) as archive:
            archive.extractall(extraction_root)
        source = extraction_root / "skanee"
    if not source.is_dir():
        raise FileNotFoundError(f"Source package did not extract to the expected path: {source}")
    labels_path = exact_path([
        input_root / "datasets/tumis001/rsna-knee-hpo-v1-blended-report-labels/soft_labels_hpo_v1_blend.json",
        input_root / "rsna-knee-hpo-v1-blended-report-labels/soft_labels_hpo_v1_blend.json",
    ], "HPO v1 blended labels")
    steven_csv = exact_path([
        input_root / "datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        input_root / "rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    ], "Steven-v4 reference labels")
    folds_path = source / "configs/report_folds_hpo_add281_trainonly.csv"
    gold_split_path = source / "configs/gold_split_v1.csv"
    config_path = source / "configs/cnn_hpo_v1_blend_add281_auc_control.json"
    trainer = source / "scripts/train_cnn_baseline.py"
    cache_dirs = [
        input_root / f"notebooks/tumis001/rsna-cnn-top-k-support-cache-shard-{index}/slice_cache"
        for index in range(3)
    ]
    missing_caches = [str(path) for path in cache_dirs if not (path / "cache_summary.json").is_file()]
    if missing_caches:
        raise FileNotFoundError(f"Missing expected cache shards: {missing_caches}")
    for path in (folds_path, gold_split_path, config_path, trainer):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required source file: {path}")

    labels_doc = json.loads(labels_path.read_text(encoding="utf-8"))
    if labels_doc.get("targets_in_order") != TARGETS:
        raise ValueError("HPO v1 label target order does not match the model schema")
    hpo_ids = [row["study_id"] for row in labels_doc["labels"]]
    if len(hpo_ids) != 4349 or len(set(hpo_ids)) != len(hpo_ids):
        raise ValueError(f"Expected 4,349 unique HPO labels, found {len(hpo_ids)} rows")
    with gold_split_path.open(newline="", encoding="utf-8") as stream:
        gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    if set(hpo_ids) & gold_ids:
        raise ValueError("HPO labels contain one or more locked gold studies")
    with folds_path.open(newline="", encoding="utf-8") as stream:
        fold_rows = list(csv.DictReader(stream))
    fold_by_id = {row["study_id"]: int(row["fold"]) for row in fold_rows}
    if set(fold_by_id) != set(hpo_ids):
        raise ValueError("The +281 fold file must cover the same 4,349 labeled reports")
    validation_ids = {study_id for study_id, fold in fold_by_id.items() if fold == 0}
    train_ids = set(hpo_ids) - validation_ids
    if len(validation_ids) != 825 or len(train_ids) != 3524:
        raise ValueError(f"Unexpected fixed split: {len(train_ids)} train, {len(validation_ids)} validation")
    with folds_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    validation_groups = {row["report_group_sha256"] for row in rows if int(row["fold"]) == 0}
    train_groups = {row["report_group_sha256"] for row in rows if int(row["fold"]) != 0}
    if validation_groups & train_groups:
        raise ValueError("Report duplicate group leakage between training and validation")
    if validation_ids & gold_ids:
        raise ValueError("A locked gold study appears in report validation")

    reference_items, seen = [], set()
    with steven_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"StudyInstanceUID", *TARGETS}.issubset(reader.fieldnames or []):
            raise ValueError("Steven-v4 file does not match the expected target schema")
        for row in reader:
            study_id = row["StudyInstanceUID"]
            if study_id not in validation_ids:
                continue
            if study_id in seen:
                raise ValueError(f"Duplicate Steven-v4 validation row: {study_id}")
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
            reference_items.append({"study_id": study_id, "probabilities": values, "weights": weights})
    if seen != validation_ids:
        raise ValueError(f"Steven-v4 is missing {len(validation_ids - seen)} validation studies")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["scheduler"].get("metric") != "validation_macro_auc" or config["scheduler"].get("mode") != "max":
        raise ValueError("LR scheduler must monitor validation macro AUC in max mode")
    if config.get("early_stopping_metric") != "validation_macro_auc":
        raise ValueError("Early stopping must monitor the same validation macro AUC")
    if config.get("xla_eval_slices") != 48:
        raise ValueError("Use 48-window full-series evaluation, matching the Modal comparison")

    work = Path("/kaggle/working") / EXPERIMENT
    work.mkdir(parents=True, exist_ok=True)
    validation_path = work / "steven_v4_fold0_reference.json"
    validation_path.write_text(json.dumps({
        "version": "steven_v4_fold0_validation_only",
        "targets_in_order": TARGETS,
        "labels": reference_items,
    }, allow_nan=False) + "\n", encoding="utf-8")
    run_manifest = {
        "run_id": EXPERIMENT,
        "experiment": "same HPO v1 +281 training split with AUC-aligned LR/early stopping",
        "train_studies": len(train_ids), "validation_studies": len(validation_ids),
        "validation_ids_unchanged_from_modal_add281": True,
        "locked_gold_studies": len(gold_ids), "gold_labels_loaded": False,
        "training_label_sha256": sha256_file(labels_path),
        "validation_label_source": "Steven v4; fold 0 only",
        "validation_label_sha256": sha256_file(validation_path),
        "fold_sha256": sha256_file(folds_path), "config_sha256": sha256_file(config_path),
        "cache_kernels": [f"tumis001/rsna-cnn-top-k-support-cache-shard-{i}" for i in range(3)],
        "model": "ResNet-34 support-aware 2.5D; 24 train windows/study; full 48-window eval",
        "controller": config["scheduler"],
        "early_stopping": {"metric": config["early_stopping_metric"],
                           "patience": config["early_stopping_patience"],
                           "min_delta": config["early_stopping_min_delta"]},
        "checkpoint_publish_interval_epochs": PUBLISH_EVERY_EPOCHS,
    }
    (work / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    print("Run preflight passed. No recursive filesystem search was used.", flush=True)
    print(f"Split: {len(train_ids)} train / {len(validation_ids)} fixed report-validation; locked gold={len(gold_ids)}", flush=True)
    print(f"Labels: HPO v1 blend for training ({len(hpo_ids)} studies); Steven v4 for validation only", flush=True)
    print("Controller: ReduceLROnPlateau and early stopping both watch validation macro AUC.", flush=True)
    print(f"TPU validation uses 48 windows; training uses 24 windows. Output: {work}", flush=True)

    command = [
        sys.executable, "-u", str(trainer), "--cache", *map(str, cache_dirs),
        "--soft-labels", str(labels_path), "--validation-soft-labels", str(validation_path),
        "--folds", str(folds_path), "--config", str(config_path),
        "--output-dir", str(work / "run"), "--validation-fold", "0",
        "--train-slices-per-study", "24", "--xla",
    ]

    metadata = {
        "title": "RSNA HPO Blend AUC Control TPU Fold 0 Checkpoints",
        "id": ARTIFACT_DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }
    upload_root = work / "checkpoint_upload"
    upload_root.mkdir(parents=True, exist_ok=True)
    (upload_root / "datasets-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(work / "run_manifest.json", upload_root / "run_manifest.json")
    mine = kaggle_cli(["datasets", "list", "--mine", "--csv"])
    exists = ARTIFACT_DATASET_ID.lower() in mine.stdout.lower()
    if exists:
        kaggle_cli(["datasets", "version", "-p", str(upload_root), "-m", "Initialize AUC-control TPU run manifest"])
    else:
        kaggle_cli(["datasets", "create", "-p", str(upload_root)])
    print(f"Private checkpoint dataset ready: {ARTIFACT_DATASET_ID}", flush=True)

    def publish(epoch: int | str, final: bool = False) -> None:
        run_dir = work / "run"
        names = ["checkpoint_best.pt", "checkpoint_latest.pt", "training_history.csv",
                 "training_history.json", "learning_curve.png", "run_progress.json",
                 "run_manifest.json", "run_summary.json"]
        for name in names:
            src = run_dir / name if name != "run_manifest.json" else work / name
            dst = upload_root / name
            if src.is_file():
                shutil.copy2(src, dst)
            elif dst.exists():
                dst.unlink()
        marker = {"last_published_epoch": epoch, "final_snapshot": final}
        (upload_root / "checkpoint_status.json").write_text(json.dumps(marker, indent=2) + "\n")
        kaggle_cli(["datasets", "version", "-p", str(upload_root), "-m",
                    f"{EXPERIMENT}: {'final' if final else 'epoch ' + str(epoch)} checkpoint"])
        print(f"Durable Kaggle checkpoint published at {ARTIFACT_DATASET_ID} (epoch {epoch}).", flush=True)

    process = subprocess.Popen(command, cwd=source, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_published = None
    last_epoch = None
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("CHECKPOINTS_READY epoch="):
                last_epoch = int(line.rsplit("=", 1)[1])
                if (last_epoch + 1) % PUBLISH_EVERY_EPOCHS == 0:
                    publish(last_epoch)
                    last_published = last_epoch
        return_code = process.wait()
        if return_code:
            if last_epoch is not None and (last_published != last_epoch):
                publish(last_epoch, final=True)
            raise RuntimeError(f"Trainer exited with code {return_code}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise

    summary_path = work / "run/run_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Training ended without run_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final_epoch = summary["history"][-1]["epoch"]
    publish(final_epoch, final=True)
    print(json.dumps({key: summary.get(key) for key in
                      ("device", "train_studies", "validation_studies", "best", "epochs_run",
                       "stopped_early", "early_stopping_metric", "lr_scheduler", "seconds")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
