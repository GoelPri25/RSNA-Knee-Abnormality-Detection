"""Train the isolated sequence-conditioned classifier on one personal Modal T4.

Inputs are the existing derived Kaggle slice caches and report labels. Raw
DICOMs and gold labels are never downloaded. A Modal Volume caches only the
derived slice shards for reuse; checkpoints, metrics, and logs are published
to a private Kaggle dataset.
"""

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

import modal


APP_NAME = "rsna-hpo-blend-sequence-conditioned-fold0"
ARTIFACT_DATASET = "tumis001/rsna-hpo-blend-sequence-conditioned-fold0"
SCRATCH_ARTIFACT_DATASET = "tumis001/rsna-hpo-blend-sequence-from-scratch-fold0"
HPO_LABEL_DATASET = "tumis001/rsna-knee-hpo-v1-blended-report-labels"
STEVEN_LABEL_DATASET = "stevenleehans/rsna-knee-llm-report-labels"
CACHE_KERNELS = [
    "tumis001/rsna-cnn-top-k-support-cache-shard-0",
    "tumis001/rsna-cnn-top-k-support-cache-shard-1",
    "tumis001/rsna-cnn-top-k-support-cache-shard-2",
]
REMOTE_ROOT = Path("/root/rsna/data/hpo_v1_blend_sequence_conditioned_fold0")
SCRATCH_REMOTE_ROOT = Path("/root/rsna/data/hpo_v1_blend_sequence_from_scratch_fold0")
CACHE_VOLUME_ROOT = Path("/root/rsna/persistent_cache")
CACHE_VOLUME_NAME = "rsna-cnn-topk-support-cache-v1"
SOURCE_DIR = Path(__file__).resolve().parents[1] / "kaggle/experiment_kernels/hpo_v1_sequence_conditioned/source/skanee"
INITIAL_CHECKPOINT = Path(__file__).resolve().parents[1] / "data/experiment_hpo_add281/artifacts/checkpoint_best.pt"
INITIAL_CHECKPOINT_SHA256 = "f26823f9bf63c759f298cec085dbedc8792ec7a3e893f4a98bf025117f4e0a93"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
PUBLISH_INTERVAL = 2

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "numpy", "scikit-learn",
        "matplotlib", "kaggle==2.0.2",
    )
    .add_local_dir(str(SOURCE_DIR), "/root/rsna/source")
    .add_local_file(str(INITIAL_CHECKPOINT), "/root/rsna/checkpoint_0918.pt")
)

if modal.is_local():
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint is missing: {INITIAL_CHECKPOINT}")
    local_checkpoint_hash = hashlib.sha256(INITIAL_CHECKPOINT.read_bytes()).hexdigest()
    if local_checkpoint_hash != INITIAL_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected warm-start checkpoint checksum: {local_checkpoint_hash}")
    kaggle_token_path = Path.home() / ".kaggle/access_token"
    if not kaggle_token_path.is_file():
        raise FileNotFoundError(f"Kaggle API access token not found at {kaggle_token_path}")
    kaggle_secret = modal.Secret.from_dict({
        "KAGGLE_ACCESS_TOKEN": kaggle_token_path.read_text(encoding="utf-8").strip()
    })
else:
    kaggle_secret = modal.Secret.from_dict({})

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


def write_dataset_metadata(path: Path, scratch: bool = False) -> None:
    subtitle = ("ResNet-34 ImageNet start with plane/fluid sequence conditioning" if scratch
                else "ResNet-34 warm start with plane/fluid metadata conditioning")
    if not 20 <= len(subtitle) <= 80:
        raise ValueError(f"Kaggle dataset subtitle must be 20–80 characters; got {len(subtitle)}")
    metadata = {
        "id": SCRATCH_ARTIFACT_DATASET if scratch else ARTIFACT_DATASET,
        "title": ("RSNA HPO Blend Sequence From Scratch Fold 0" if scratch
                  else "RSNA HPO Blend Sequence Conditioned Fold 0"),
        "subtitle": subtitle,
        "description": (
            "Private experiment artifacts only: checkpoints, training history, "
            "learning curves, predictions, logs, and run metadata. No raw DICOMs "
            "or gold labels are included. "
            + ("This trains from the ResNet-34 ImageNet initialization and adds a zero-initialized, "
               "per-target logit correction from plane and fluid-sensitive metadata. " if scratch else
               "This warm-starts the HPO v1 +281 checkpoint and adds a zero-initialized, per-target "
               "logit correction from plane and fluid-sensitive metadata. ")
            + "The original sampling, validation split, labels, and training controls are retained."
        ),
        "licenses": [{"name": "other"}],
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def validate_local_source_bundle() -> None:
    """Catch missing or mismatched source inputs before any Modal function starts."""
    configs = SOURCE_DIR / "configs"
    required = (
        configs / "report_folds_v5.csv",
        configs / "report_folds_hpo_add281_trainonly.csv",
        configs / "gold_split_v1.csv",
        configs / "cnn_hpo_v1_blend_baseline.json",
        configs / "cnn_hpo_v1_blend_sequence_conditioned.json",
        configs / "cnn_hpo_v1_blend_sequence_conditioned_from_scratch.json",
        SOURCE_DIR / "scripts/train_cnn_sequence_conditioned.py",
        INITIAL_CHECKPOINT,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Local Modal source bundle is incomplete: {missing}")

    expected_hashes = {
        configs / "report_folds_v5.csv":
            "2ed70719f9423d3a0c3b9ac424841f257d57dda1b9dfd72b26805de8eb4bb6ba",
        configs / "report_folds_hpo_add281_trainonly.csv":
            "831e1cc71639c3d5cd59cf61da2546c1f42a2ff644864bf6ac778fbc87af08ad",
        configs / "cnn_hpo_v1_blend_baseline.json":
            "31125adb466de6490c0797687401f9608b7b92bf769f0d6b1a40b4b690c7568c",
        INITIAL_CHECKPOINT: INITIAL_CHECKPOINT_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Local experiment input checksum mismatch: {path.name}")

    baseline_cfg = json.loads((configs / "cnn_hpo_v1_blend_baseline.json").read_text())
    scratch_cfg = json.loads(
        (configs / "cnn_hpo_v1_blend_sequence_conditioned_from_scratch.json").read_text()
    )
    for cfg in (baseline_cfg, scratch_cfg):
        cfg.pop("version", None)
        cfg.pop("note", None)
    if baseline_cfg != scratch_cfg:
        raise ValueError("From-scratch sequence config differs from the 0.918 baseline controls")

    with (configs / "report_folds_v5.csv").open(newline="", encoding="utf-8") as stream:
        baseline_rows = list(csv.DictReader(stream))
    with (configs / "report_folds_hpo_add281_trainonly.csv").open(
            newline="", encoding="utf-8") as stream:
        expanded_rows = list(csv.DictReader(stream))
    with (configs / "gold_split_v1.csv").open(newline="", encoding="utf-8") as stream:
        gold_rows = list(csv.DictReader(stream))
    baseline = {row["study_id"]: int(row["fold"]) for row in baseline_rows}
    expanded = {row["study_id"]: int(row["fold"]) for row in expanded_rows}
    baseline_validation = {study for study, fold in baseline.items() if fold == 0}
    expanded_validation = {study for study, fold in expanded.items() if fold == 0}
    additions = set(expanded) - set(baseline)
    gold_ids = {row["StudyInstanceUID"] for row in gold_rows}
    if (len(baseline) != 4068 or len(baseline_validation) != 825
            or len(expanded) != 4349 or len(additions) != 281
            or expanded_validation != baseline_validation or len(gold_ids) != 58
            or any(expanded[study] == 0 for study in additions)
            or additions & gold_ids):
        raise ValueError("Local fold or gold-lock inputs failed fixed coverage checks")
    print("Local source preflight passed: fixed folds, 281 train-only additions, 58 locked gold IDs, "
          "baseline controls, and checkpoint hash.", flush=True)


def configure_kaggle() -> str:
    token = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Kaggle API token secret was not injected into Modal")
    config_dir = Path.home() / ".kaggle"
    config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    token_file = config_dir / "access_token"
    token_file.write_text(token + "\n", encoding="utf-8")
    token_file.chmod(0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(config_dir)
    os.environ["KAGGLE_API_TOKEN"] = token
    return token


def command(args: list[str], *, capture: bool = False, cwd: Path | None = None) -> str:
    print("$", " ".join(args), flush=True)
    result = subprocess.run(
        args, cwd=cwd, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    output = result.stdout or "" if capture else ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}\n{output[-2500:]}")
    return output


@app.function(image=image, cpu=1, memory=2048, timeout=3_600, secrets=[kaggle_secret],
              volumes={str(CACHE_VOLUME_ROOT): cache_volume})
def verify_access(scratch: bool = False) -> str:
    """Verify cache, labels, and artifact writes on CPU before allocating a GPU."""
    configure_kaggle()
    for kernel in CACHE_KERNELS:
        listing = command(["kaggle", "kernels", "files", kernel, "--page-size", "200"], capture=True)
        expected = ("manifest.jsonl", "slices_000.npy", "support_000.npy", "cache_summary.json")
        if any(name not in listing for name in expected):
            raise RuntimeError(f"Unexpected output files in cache kernel {kernel}")
    for dataset, expected_file in (
        (HPO_LABEL_DATASET, "soft_labels_hpo_v1_blend.json"),
        (STEVEN_LABEL_DATASET, "llm_labels_v4_blend.csv"),
    ):
        listing = command(["kaggle", "datasets", "files", dataset, "--page-size", "200"], capture=True)
        if expected_file not in listing:
            raise RuntimeError(f"Expected {expected_file} is unavailable in {dataset}")

    artifact_dataset = SCRATCH_ARTIFACT_DATASET if scratch else ARTIFACT_DATASET
    artifact_root = SCRATCH_REMOTE_ROOT if scratch else REMOTE_ROOT
    artifact_dir = artifact_root / "kaggle_upload"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_metadata(artifact_dir / "dataset-metadata.json", scratch)
    manifest = {
        "run_id": ("hpo_v1_blend_sequence_from_scratch_fold0" if scratch
                   else "hpo_v1_blend_sequence_conditioned_fold0"),
        "status": "initialized",
        "experiment": ("ImageNet-initialized ResNet-34; HPO-v1 labels; +281 train-only; "
                        "plane/fluid sequence conditioning" if scratch else
                        "Warm-started ResNet-34; HPO-v1 labels; +281 train-only; plane/fluid sequence conditioning only"),
        "validation": "same fixed 825 fold-0 report-only studies, scored against Steven v4",
        "locked_gold_studies": 58,
        "gold_labels_loaded": False,
        "raw_dicoms_transferred": False,
        "gpu": "Modal T4",
        "warm_start_checkpoint_sha256": None if scratch else INITIAL_CHECKPOINT_SHA256,
        "baseline_public_score": 0.918,
        "sequence_conditioning": "zero-initialized target-specific logit correction over plane × fluid-sensitive categories",
        "training_sampling": "original 0.918 baseline: 24 windows uniformly sampled across all study windows and sorted",
        "cache_kernels": CACHE_KERNELS,
        "input_cache_volume": CACHE_VOLUME_NAME,
        "artifacts_destination": artifact_dataset,
        "checkpoint_publish_interval_epochs": PUBLISH_INTERVAL,
        "early_stopping": {"metric": "report-validation macro AUC", "patience": 8, "min_delta": 0.0005},
        "scheduler": {"type": "ReduceLROnPlateau", "metric": "report-validation BCE", "factor": 0.5, "patience": 2},
    }
    (artifact_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    probe = subprocess.run(
        ["kaggle", "datasets", "files", artifact_dataset, "--page-size", "1"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if probe.returncode:
        command(["kaggle", "datasets", "create", "-p", str(artifact_dir),
                 "--dir-mode", "skip", "--keep-tabular"])
        print("Created private Kaggle artifact dataset before GPU allocation.", flush=True)
    else:
        print("Kaggle artifact dataset already exists; checkpoints will version it.", flush=True)
    cache_volume.reload()
    cache_dirs = [read_kernel_files(kernel, CACHE_VOLUME_ROOT / f"shard_{i}")
                  for i, kernel in enumerate(CACHE_KERNELS)]
    all_study_ids: set[str] = set()
    expected_counts = [1456, 1466, 1427]
    counts = []
    series_rows = 0
    known_metadata_rows = 0
    for directory in cache_dirs:
        summary = json.loads((directory / "cache_summary.json").read_text(encoding="utf-8"))
        with (directory / "manifest.jsonl").open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        shard_study_ids = {record["study_id"] for record in records}
        series_rows += len(records)
        for record in records:
            plane = str(record.get("plane") or "").strip().lower()
            fluid = str(record.get("fluid_sensitive") or "").strip().lower()
            plane_known = plane.startswith(("sag", "cor", "axi", "trans"))
            fluid_known = fluid in {"1", "true", "yes", "y", "0", "false", "no", "n"}
            known_metadata_rows += int(plane_known and fluid_known)
        expected = int(summary["studies_in_manifest"])
        if len(shard_study_ids) != expected or all_study_ids & shard_study_ids:
            raise RuntimeError(f"CPU preflight found malformed or overlapping cache shard: {directory}")
        counts.append(expected)
        all_study_ids.update(shard_study_ids)
    if counts != expected_counts or len(all_study_ids) != 4349:
        raise RuntimeError(f"CPU preflight cache coverage mismatch: {counts}; total={len(all_study_ids)}")
    metadata_coverage = known_metadata_rows / max(series_rows, 1)
    if series_rows == 0 or metadata_coverage < 0.95:
        raise RuntimeError(
            f"Sequence metadata is incomplete: {known_metadata_rows}/{series_rows} series "
            f"({metadata_coverage:.1%}) have recognized plane and fluid-sensitive fields"
        )
    print(f"CPU preflight verified Modal cache volume: {counts}; 4,349 studies; "
          f"sequence metadata {known_metadata_rows}/{series_rows} series ({metadata_coverage:.1%}).", flush=True)
    return "Kaggle inputs, cache volume, and private artifact destination verified on CPU."


def read_kernel_files(kernel: str, destination: Path) -> Path:
    import numpy as np

    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / "slice_cache"
    names = ["cache_summary.json", "manifest.jsonl"]
    names += [f"slices_{i:03d}.npy" for i in range(4)]
    names += [f"support_{i:03d}.npy" for i in range(4)]
    # Fetch one output at a time. A single `kernels output` call downloads all
    # multi-GB arrays in one request; Kaggle can reset that connection and
    # discard the entire shard. Per-file retries preserve completed files and
    # bound a retry to only the file that failed.
    def validate_file(path: Path, name: str) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Cache file is missing or empty: {path}")
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_bytes = array.offset + array.nbytes
            del array
            if path.stat().st_size < expected_bytes:
                raise RuntimeError(f"Truncated NumPy cache file: {path}")
        elif name == "manifest.jsonl":
            count = 0
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        json.loads(line)
                        count += 1
            if count == 0:
                raise RuntimeError(f"Empty cache manifest: {path}")
        else:
            json.loads(path.read_text(encoding="utf-8"))

    for index, name in enumerate(names, start=1):
        target = cache / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size > 0:
            try:
                validate_file(target, name)
                print(f"Cache hit {kernel}: {name}", flush=True)
                continue
            except Exception as error:
                print(f"Cached file is incomplete or invalid ({target}): {error}; "
                      "downloading it again.", flush=True)
                target.unlink(missing_ok=True)
        for attempt in range(1, 5):
            try:
                print(f"Cache transfer {kernel}: file {index}/{len(names)} {name} "
                      f"(attempt {attempt}/4)", flush=True)
                command([
                    "kaggle", "kernels", "output", kernel,
                    "-p", str(destination),
                    "--file-pattern", f"{re.escape(name)}$",
                    "--force",
                ])
                validate_file(target, name)
                # Persist each complete file so a later task can resume at
                # the next missing/corrupt file instead of fetching the shard
                # again. The Kaggle dataset remains the durable source of truth.
                cache_volume.commit()
                break
            except Exception as error:
                if attempt == 4:
                    raise RuntimeError(
                        f"Could not download {kernel}/{name} after {attempt} attempts"
                    ) from error
                delay = (10, 30, 60)[attempt - 1]
                print(f"Cache transfer failed for {name}: {type(error).__name__}: {error}; "
                      f"retrying in {delay}s", flush=True)
                target.unlink(missing_ok=True)
                time.sleep(delay)
    required = [cache / name for name in names]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete cache output from {kernel}: {missing}")
    cache_volume.commit()
    return cache


def prepare_steven_validation(steven_csv: Path, folds_path: Path, gold_split_path: Path,
                              output_path: Path) -> tuple[int, int]:
    import numpy as np

    with gold_split_path.open(newline="", encoding="utf-8") as stream:
        gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    with folds_path.open(newline="", encoding="utf-8") as stream:
        fold_map = {row["study_id"]: int(row["fold"]) for row in csv.DictReader(stream)}
    validation_ids = {study_id for study_id, fold in fold_map.items() if fold == 0}
    if len(gold_ids) != 58 or len(validation_ids) != 825 or gold_ids & validation_ids:
        raise RuntimeError("Fold/gold split dimensions or separation check failed")

    records = []
    seen: set[str] = set()
    with steven_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"StudyInstanceUID", *TARGETS}.issubset(reader.fieldnames or []):
            raise ValueError("Steven v4 label file has an unexpected target schema")
        for row in reader:
            study_id = row["StudyInstanceUID"]
            if study_id not in validation_ids:
                continue
            if study_id in seen:
                raise ValueError(f"Duplicate Steven v4 validation row: {study_id}")
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
            records.append({"study_id": study_id, "probabilities": probabilities, "weights": weights})
    if seen != validation_ids:
        raise RuntimeError(f"Steven v4 validation coverage mismatch: {len(validation_ids - seen)} missing")
    output_path.write_text(json.dumps({
        "version": "steven_v4_fold0_validation_only",
        "targets_in_order": TARGETS,
        "labels": records,
    }, allow_nan=False) + "\n", encoding="utf-8")
    return len(gold_ids), len(validation_ids)


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    # Modal reported the first from-scratch worker exceeded its 16 GiB request
    # before epoch 0. Keep the experiment/data settings fixed and give this
    # GPU worker 32 GiB of host memory so batch materialization can complete.
    memory=32_768,
    timeout=14_400,
    secrets=[kaggle_secret],
    volumes={str(CACHE_VOLUME_ROOT): cache_volume},
    scaledown_window=60,
)
def train_one_fold(scratch: bool = False) -> dict:
    import numpy as np
    import torch

    configure_kaggle()
    root = SCRATCH_REMOTE_ROOT if scratch else REMOTE_ROOT
    artifact_dataset = SCRATCH_ARTIFACT_DATASET if scratch else ARTIFACT_DATASET
    input_dir = root / "inputs"
    labels_dir = input_dir / "labels"
    upload_dir = root / "kaggle_upload"
    run_dir = root / "run"
    for path in (input_dir, labels_dir, upload_dir, run_dir):
        path.mkdir(parents=True, exist_ok=True)

    cache_volume.reload()
    cache_root = CACHE_VOLUME_ROOT
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    cache_dirs = [read_kernel_files(kernel, cache_root / f"shard_{i}")
                  for i, kernel in enumerate(CACHE_KERNELS)]
    summary_counts = []
    manifest_ids: set[str] = set()
    for directory in cache_dirs:
        summary = json.loads((directory / "cache_summary.json").read_text(encoding="utf-8"))
        expected_studies = int(summary["studies_in_manifest"])
        shard_ids: set[str] = set()
        with (directory / "manifest.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    # The manifest is series-level: one study normally has
                    # several rows within its own shard. Only cross-shard
                    # duplication would indicate overlapping partitions.
                    shard_ids.add(record["study_id"])
        if len(shard_ids) != expected_studies:
            raise RuntimeError(
                f"Cache summary/manifest mismatch in {directory}: "
                f"summary={expected_studies}, manifest unique studies={len(shard_ids)}"
            )
        overlap = manifest_ids & shard_ids
        if overlap:
            raise RuntimeError(f"Studies appear in multiple cache shards: {sorted(overlap)[:5]}")
        summary_counts.append(expected_studies)
        manifest_ids.update(shard_ids)
    if summary_counts != [1456, 1466, 1427] or len(manifest_ids) != 4349:
        raise RuntimeError(f"Unexpected cache coverage: shard studies={summary_counts}, total={len(manifest_ids)}")
    print(f"Cache verified: {summary_counts} studies; {len(manifest_ids)} total; raw DICOMs not fetched.", flush=True)

    hpo_dir = labels_dir / "hpo"
    steven_dir = labels_dir / "steven"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    steven_dir.mkdir(parents=True, exist_ok=True)
    command(["kaggle", "datasets", "download", "-d", HPO_LABEL_DATASET,
             "-f", "soft_labels_hpo_v1_blend.json", "-p", str(hpo_dir), "--unzip", "--quiet"])
    command(["kaggle", "datasets", "download", "-d", STEVEN_LABEL_DATASET,
             "-f", "llm_labels_v4_blend.csv", "-p", str(steven_dir), "--unzip", "--quiet"])
    hpo_labels = hpo_dir / "soft_labels_hpo_v1_blend.json"
    steven_labels = steven_dir / "llm_labels_v4_blend.csv"

    source = Path("/root/rsna/source")
    baseline_folds_path = source / "configs/report_folds_v5.csv"
    baseline_config_path = source / "configs/cnn_hpo_v1_blend_baseline.json"
    folds_path = source / "configs/report_folds_hpo_add281_trainonly.csv"
    gold_split_path = source / "configs/gold_split_v1.csv"
    config_path = source / (
        "configs/cnn_hpo_v1_blend_sequence_conditioned_from_scratch.json"
        if scratch else "configs/cnn_hpo_v1_blend_sequence_conditioned.json"
    )
    trainer = source / "scripts/train_cnn_sequence_conditioned.py"
    initial_checkpoint_path = Path("/root/rsna/checkpoint_0918.pt")
    required_paths = [hpo_labels, steven_labels, baseline_folds_path, baseline_config_path,
                      folds_path, gold_split_path, config_path, trainer]
    if not scratch:
        required_paths.append(initial_checkpoint_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing fixed experiment input: {path}")
    import hashlib
    checkpoint_sha256 = None
    if not scratch:
        checkpoint_sha256 = hashlib.sha256(initial_checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_sha256 != INITIAL_CHECKPOINT_SHA256:
            raise ValueError(f"Warm-start checkpoint hash mismatch: {checkpoint_sha256}")
    payload = json.loads(hpo_labels.read_text(encoding="utf-8"))
    if payload.get("targets_in_order") != TARGETS or len(payload.get("labels", [])) != 4349:
        raise ValueError("HPO label schema/coverage is not the prepared 4,349-study set")
    hpo_label_sha256 = __import__("hashlib").sha256(hpo_labels.read_bytes()).hexdigest()
    if hpo_label_sha256 != "ffdffcb2a68e7dbdfbac14c97b1017c28ab7161628cf067040287fa9356c608f":
        raise ValueError("HPO-v1 labels differ from the exact label artifact used by the 0.918 baseline")
    if __import__("hashlib").sha256(baseline_config_path.read_bytes()).hexdigest() != (
        "31125adb466de6490c0797687401f9608b7b92bf769f0d6b1a40b4b690c7568c"
    ):
        raise ValueError("Baseline training config differs from the 0.918 HPO-v1 run")
    if __import__("hashlib").sha256(baseline_folds_path.read_bytes()).hexdigest() != (
        "2ed70719f9423d3a0c3b9ac424841f257d57dda1b9dfd72b26805de8eb4bb6ba"
    ):
        raise ValueError("Original fold assignment differs from the 0.918 HPO-v1 run")
    label_ids = {row["study_id"] for row in payload["labels"]}
    if label_ids != manifest_ids:
        raise ValueError("HPO label IDs do not exactly match the three cached report-only shards")
    with baseline_folds_path.open(newline="", encoding="utf-8") as stream:
        baseline_rows = list(csv.DictReader(stream))
    baseline_map = {row["study_id"]: int(row["fold"]) for row in baseline_rows}
    with folds_path.open(newline="", encoding="utf-8") as stream:
        expanded_rows = list(csv.DictReader(stream))
    fold_map = {row["study_id"]: int(row["fold"]) for row in expanded_rows}
    group_by_id = {row["study_id"]: row["report_group_sha256"] for row in expanded_rows}
    if len(fold_map) != len(expanded_rows):
        raise ValueError("Expanded fold file has duplicate study IDs")
    if not set(fold_map).issubset(label_ids) or set(fold_map) != label_ids:
        raise ValueError("Expanded fold file must cover exactly the 4,349 HPO-labeled studies")
    added_ids = label_ids - set(baseline_map)
    baseline_validation_ids = {study_id for study_id, fold in baseline_map.items() if fold == 0}
    validation_ids = {study_id for study_id, fold in fold_map.items() if fold == 0}
    train_count = sum(fold != 0 for fold in fold_map.values())
    validation_count_from_folds = len(validation_ids)
    if (len(baseline_map), len(baseline_validation_ids), len(added_ids),
            len(fold_map), train_count, validation_count_from_folds) != (4068, 825, 281, 4349, 3524, 825):
        raise RuntimeError(
            "Unexpected expanded split coverage: "
            f"base_folds={len(baseline_map)}, added={len(added_ids)}, "
            f"folds={len(fold_map)}, train={train_count}, validation={validation_count_from_folds}"
        )
    if validation_ids != baseline_validation_ids:
        raise RuntimeError("The expanded fold file changed the original validation IDs")
    if any(fold_map[study_id] == 0 for study_id in added_ids):
        raise RuntimeError("An added study was assigned to validation instead of training")
    validation_hashes = {group_by_id[study_id] for study_id in validation_ids}
    overlapping_report_groups = {
        study_id for study_id in added_ids if group_by_id[study_id] in validation_hashes
    }
    if overlapping_report_groups:
        raise RuntimeError(
            f"Added studies share report groups with validation: {len(overlapping_report_groups)}"
        )
    with gold_split_path.open(newline="", encoding="utf-8") as stream:
        gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    if len(gold_ids) != 58 or added_ids & gold_ids:
        raise RuntimeError("Gold lock failed for the added studies")
    validation_path = input_dir / "steven_v4_fold0_reference.json"
    gold_count, validation_count = prepare_steven_validation(
        steven_labels, folds_path, gold_split_path, validation_path)
    print(f"Labels/split verified: HPO labels={len(label_ids)}; fixed-fold eligible={len(fold_map)}; "
          f"actual training={train_count}; Steven v4 validation={validation_count}; "
          f"new training-only additions={len(added_ids)}; report-group overlap with validation=0; "
          f"gold IDs locked={gold_count}.", flush=True)

    metadata_path = upload_dir / "dataset-metadata.json"
    write_dataset_metadata(metadata_path, scratch)
    run_manifest = {
        "run_id": ("hpo_v1_blend_sequence_from_scratch_fold0" if scratch
                   else "hpo_v1_blend_sequence_conditioned_fold0"),
        "status": "training",
        "experiment": ("ImageNet-initialized HPO v1 +281 ResNet-34 with plane/fluid metadata conditioning"
                        if scratch else
                        "Warm-started HPO v1 +281 ResNet-34 with plane/fluid metadata conditioning only"),
        "validation": "same fixed 825 fold-0 report-only studies scored against Steven v4",
        "label_sha256": hpo_label_sha256,
        "locked_gold_studies": 58,
        "gold_labels_loaded": False,
        "raw_dicoms_transferred": False,
        "cache_studies": len(manifest_ids),
        "hpo_label_rows": len(label_ids),
        "fold_eligible_studies": len(fold_map),
        "baseline_training_studies": 3243,
        "added_training_studies": len(added_ids),
        "train_studies_expected": train_count,
        "validation_studies_expected": validation_count,
        "hpo_rows_without_fixed_fold_added_to_train": len(added_ids),
        "fixed_validation_ids_unchanged": True,
        "added_report_groups_overlapping_validation": len(overlapping_report_groups),
        "folds_sha256": __import__("hashlib").sha256(folds_path.read_bytes()).hexdigest(),
        "cache_kernels": CACHE_KERNELS,
        "input_cache_volume": CACHE_VOLUME_NAME,
        "artifacts_destination": artifact_dataset,
        "gpu": "Modal T4",
        "host_memory_gib": 32,
        "warm_start_checkpoint_sha256": checkpoint_sha256,
        "baseline_public_score": 0.918,
        "sequence_conditioning": "zero-initialized target-specific logit correction over six plane × fluid-sensitive categories plus unknown",
        "training_sampling": "original 0.918 baseline: 24 windows uniformly sampled across all study windows and sorted",
        "max_epochs": 60,
        "checkpoint_publish_interval_epochs": PUBLISH_INTERVAL,
        "early_stopping": {"metric": "report-validation macro AUC", "patience": 8, "min_delta": 0.0005},
        "scheduler": {"type": "ReduceLROnPlateau", "metric": "report-validation BCE", "factor": 0.5, "patience": 2},
    }
    (upload_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")

    created_probe = subprocess.run(
        ["kaggle", "datasets", "files", artifact_dataset, "--page-size", "1"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    created = created_probe.returncode == 0
    if not created:
        command(["kaggle", "datasets", "create", "-p", str(upload_dir),
                 "--dir-mode", "skip", "--keep-tabular"])
        created = True

    def publish(note: str) -> None:
        for item in run_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, upload_dir / item.name)
        (upload_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
        command(["kaggle", "datasets", "version", "-p", str(upload_dir), "-m", note,
                 "--dir-mode", "skip", "--keep-tabular"])

    command_args = [
        sys.executable, "-u", str(trainer),
        "--cache", *map(str, cache_dirs),
        "--soft-labels", str(hpo_labels),
        "--validation-soft-labels", str(validation_path),
        "--folds", str(folds_path),
        "--config", str(config_path),
        "--output-dir", str(run_dir),
        "--validation-fold", "0",
        "--train-slices-per-study", "24",
    ]
    if not scratch:
        command_args.extend(["--init-checkpoint", str(initial_checkpoint_path)])
    print(("Train the sequence-conditioned ResNet-34 from the baseline's ImageNet initialization; "
           "the zero-initialized plane × fluid-sensitive correction is the only architecture addition."
           if scratch else
           "Single change: warm-start the 0.918 baseline and add a zero-initialized, target-specific "
           "plane × fluid-sensitive logit correction."), flush=True)
    print("Original 24-window sampling, labels, +281 training-only studies, fixed 825 validation studies, optimizer, LR schedule and early stopping stay fixed.", flush=True)
    print("Gold labels remain locked. As in the 0.918 run, ReduceLROnPlateau watches validation BCE; checkpoint selection and early stopping watch report-fold AUC.", flush=True)
    print("Running:", " ".join(command_args), flush=True)
    training_env = os.environ.copy()
    training_env["PYTHONPATH"] = str(source)
    process = subprocess.Popen(
        command_args, cwd=source, env=training_env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    log_path = run_dir / "training.log"
    marker = re.compile(r"^CHECKPOINTS_READY epoch=(\d+)$")
    last_published = -1
    return_code = -1
    try:
        assert process.stdout is not None
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                found = marker.match(line.strip())
                if found:
                    epoch = int(found.group(1))
                    if (epoch == 0 or epoch % PUBLISH_INTERVAL == 0) and epoch != last_published:
                        note = "after first epoch" if epoch == 0 else f"after epoch {epoch + 1}"
                        run_label = "Sequence-from-scratch" if scratch else "Sequence-conditioned"
                        publish(f"{run_label} fold 0: checkpoint {note}")
                        last_published = epoch
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    if return_code != 0 and run_dir.exists():
        run_manifest["status"] = "failed_before_completion"
        run_manifest["failure_exit_code"] = return_code
        sync = [item for item in run_dir.iterdir() if item.is_file()]
        if sync:
            for item in sync:
                shutil.copy2(item, upload_dir / item.name)
            (upload_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
            command(["kaggle", "datasets", "version", "-p", str(upload_dir),
                     "-m", f"Sequence-conditioned fold 0: final failed state (exit {return_code})",
                     "--dir-mode", "skip", "--keep-tabular"])
    if return_code:
        raise RuntimeError(f"Training process exited with code {return_code}")
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Training ended without run_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_manifest.update({
        "status": "completed",
        "best_epoch": summary.get("best", {}).get("epoch"),
        "best_report_validation_auc": summary.get("best", {}).get("macro_auc"),
        "epochs_run": summary.get("epochs_run"),
        "stopped_early": summary.get("stopped_early"),
        "runtime_seconds": summary.get("seconds"),
    })
    (upload_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    publish(("Sequence-from-scratch fold 0: completed" if scratch
             else "Sequence-conditioned fold 0: completed"))
    return summary


@app.local_entrypoint()
def main(scratch: bool = False) -> None:
    validate_local_source_bundle()
    print(verify_access.remote(scratch), flush=True)
    call = train_one_fold.spawn(scratch)
    print(f"Detached training function started: {call.object_id}", flush=True)
