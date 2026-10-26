"""Five-fold regional Pillar-0 reader experiment on personal Modal T4.

Raw competition DICOMs remain on Kaggle. Modal receives only the compressed
precomputed Pillar-0 token cache and small report-label/fold files; outputs are
published at intervals to a private Kaggle dataset.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_ROOT = Path("/root/rsna")
REMOTE_ROOT = Path("/tmp/rsna_pillar0_regional_reader")
APP_NAME = "rsna-pillar0-regional-reader-cv"
CACHE_DATASET = "tumis001/rsna-pillar0-spatial-token-cache-v1"
LABEL_DATASET = "stevenleehans/rsna-knee-llm-report-labels"
SOURCE_DATASET = "tumis001/rsna-cnn-topk-support-source-v1"
ARTIFACT_DATASET = "tumis001/rsna-pillar0-regional-reader-cv-v1"
CHECKPOINT_REVISION = "be5050ba482607f90e607e044f03a2ebc38f98c3"
EXPECTED_SERIES = 24_035
PUBLISH_EVERY_EPOCHS = 5
TRAINER_REMOTE = str(REMOTE_PROJECT_ROOT / "scripts/train_pillar0_regional_reader.py")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "scikit-learn", "matplotlib", "pandas", "kaggle==2.0.2")
    .add_local_file(str(PROJECT_ROOT / "scripts/train_pillar0_regional_reader.py"), TRAINER_REMOTE)
    .add_local_file(str(PROJECT_ROOT / "scripts/train_pillar0_reader.py"),
                    str(REMOTE_PROJECT_ROOT / "scripts/train_pillar0_reader.py"))
    .add_local_dir(str(PROJECT_ROOT / "src/rsna_knee"), str(REMOTE_PROJECT_ROOT / "src/rsna_knee"))
)

if modal.is_local():
    kaggle_token_path = Path.home() / ".kaggle/access_token"
    if not kaggle_token_path.is_file():
        raise FileNotFoundError(f"Kaggle API access token not found at {kaggle_token_path}")
    kaggle_secret = modal.Secret.from_dict({
        "KAGGLE_ACCESS_TOKEN": kaggle_token_path.read_text(encoding="utf-8").strip()
    })
else:
    kaggle_secret = modal.Secret.from_dict({})

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16_384,
    timeout=28_800,
    secrets=[kaggle_secret],
    scaledown_window=60,
)
def train_regional_reader_cv() -> dict:
    import sys

    root = REMOTE_ROOT
    inputs = root / "inputs"
    scratch = root / "scratch"
    run_id = "pillar0_regional_reader_cv5_rerun_20260929"
    run_dir = root / "runs" / run_id
    upload = root / "kaggle_upload"
    for directory in (inputs, scratch, run_dir, upload):
        directory.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Personal Kaggle API token was not injected into Modal")
    os.environ["KAGGLE_API_TOKEN"] = token
    kaggle_config = Path.home() / ".kaggle"
    kaggle_config.mkdir(mode=0o700, parents=True, exist_ok=True)
    (kaggle_config / "access_token").write_text(token + "\n", encoding="utf-8")
    (kaggle_config / "access_token").chmod(0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_config)

    def run(command: list[str], *, capture: bool = False) -> str:
        print("$", " ".join(command), flush=True)
        result = subprocess.run(command, text=True,
                                stdout=subprocess.PIPE if capture else None,
                                stderr=subprocess.STDOUT if capture else None,
                                check=False)
        output = result.stdout or "" if capture else ""
        if output:
            print(output, end="" if output.endswith("\n") else "\n", flush=True)
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{output[-2500:]}")
        return output

    def dataset_files(dataset: str) -> list[str]:
        names, page_token = [], None
        while True:
            command = ["kaggle", "datasets", "files", dataset, "--page-size", "200", "--csv"]
            if page_token:
                command.extend(("--page-token", page_token))
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"Cannot list {dataset}: {(result.stdout or '')[-1200:]}")
            output = result.stdout or ""
            match = re.search(r"(?m)^Next Page Token = (.+)$", output)
            page_token = match.group(1).strip() if match else None
            start = output.find("name,")
            if start < 0:
                raise RuntimeError(f"Unrecognized Kaggle file listing for {dataset}")
            names.extend(row["name"] for row in csv.DictReader(io.StringIO(output[start:])))
            if not page_token:
                return names

    def download_files(dataset: str, destination: Path, filenames: list[str]) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        pending = [name for name in filenames if not (destination / Path(name).name).is_file()]
        for completed, filename in enumerate(pending, 1):
            last_error = ""
            for attempt, wait_seconds in enumerate((10, 20, 40, 80, 120)):
                result = subprocess.run(
                    ["kaggle", "datasets", "download", "-d", dataset, "-f", filename,
                     "-p", str(destination), "--unzip", "--quiet"],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    check=False, timeout=180,
                )
                local = destination / Path(filename).name
                if result.returncode == 0 and local.is_file():
                    break
                last_error = (result.stdout or "")[-1000:]
                if attempt < 4:
                    time.sleep(wait_seconds)
            else:
                raise RuntimeError(f"Download failed for {filename}: {last_error}")
            if completed % 16 == 0 or completed == len(pending):
                print(f"Downloaded {completed}/{len(pending)} files from {dataset}", flush=True)
            time.sleep(0.2)

    def find_file(directory: Path, filename: str) -> Path:
        found = list(directory.rglob(filename))
        if len(found) != 1:
            raise RuntimeError(f"Expected one {filename} below {directory}, found {len(found)}")
        return found[0]

    cache_dir = inputs / "pillar0_cache"
    cache_files = dataset_files(CACHE_DATASET)
    chunks = sorted(n for n in cache_files if n.startswith("spatial_batch_") and n.endswith(".npz"))
    if len(chunks) != 376 or "run_config.json" not in cache_files:
        raise RuntimeError(f"Unexpected cache manifest: {len(chunks)} batches")
    download_files(CACHE_DATASET, cache_dir, ["run_config.json", *chunks])
    cache_config = json.loads(find_file(cache_dir, "run_config.json").read_text(encoding="utf-8"))
    if cache_config.get("checkpoint_revision") != CHECKPOINT_REVISION:
        raise RuntimeError("Pillar-0 cache encoder revision does not match the verified run")
    if cache_config.get("encoder_token_gating_applied") is not False:
        raise RuntimeError("Expected ungated baseline contextual tokens")
    if cache_config.get("support_map_saved") is not True:
        raise RuntimeError("Required fractional support map is missing")
    print(f"Verified {EXPECTED_SERIES:,} cached series; encoder revision {CHECKPOINT_REVISION}", flush=True)

    label_dir = inputs / "labels"
    label_files = dataset_files(LABEL_DATASET)
    if "llm_labels_v4_blend.csv" not in label_files:
        raise RuntimeError("Steven v4 label file is missing")
    download_files(LABEL_DATASET, label_dir, ["llm_labels_v4_blend.csv"])
    label_path = find_file(label_dir, "llm_labels_v4_blend.csv")

    source_dir = inputs / "source"
    source_files = dataset_files(SOURCE_DATASET)
    fold_file = "skanee/configs/report_folds_v5.csv"
    if fold_file not in source_files:
        raise RuntimeError("Report fold configuration is missing")
    download_files(SOURCE_DATASET, source_dir, [fold_file])
    folds_path = find_file(source_dir, "report_folds_v5.csv")

    metadata = {
        "id": ARTIFACT_DATASET,
        "title": "RSNA Pillar0 Regional Reader CV V1",
        "subtitle": "Private five-fold checkpoint and evaluation artifacts",
        "description": (
            "Inference/training artifacts for the frozen Pillar-0 scale-aligned regional reader. "
            "Uses report labels only; excludes official gold labels and raw DICOM."
        ),
        "licenses": [{"name": "other"}],
    }
    (upload / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {
        "run_id": run_id,
        "architecture": "scale_aligned_regional_target_query",
        "cache": CACHE_DATASET,
        "cache_revision": CHECKPOINT_REVISION,
        "labels": LABEL_DATASET,
        "folds": SOURCE_DATASET,
        "validation": "five-fold report-only CV; no ensemble submission",
        "gold_labels_loaded": False,
        "raw_dicom_transferred": False,
        "gpu": "Modal T4",
        "max_epochs": 60,
        "batch_studies": 4,
        "learning_rate": 1e-4,
        "selection_metric": "raw report-validation macro AUC",
        "scheduler": {"metric": "3-epoch median macro AUC", "factor": 0.5,
                      "patience": 4, "cooldown": 2},
        "early_stopping": {"metric": "report-validation macro AUC", "patience": 12,
                           "min_delta": 0.0005, "min_epochs_after_lr_drop": 4},
        "checkpoint_publication_interval_epochs": PUBLISH_EVERY_EPOCHS,
    }
    (upload / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    existing = subprocess.run(["kaggle", "datasets", "files", ARTIFACT_DATASET],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, check=False).returncode == 0

    def sync_outputs() -> None:
        # Kaggle's --dir-mode skip ignores fold directories. Keep the
        # directory structure on Modal, but publish each fold as one flat,
        # root-level archive so checkpoints survive container shutdown.
        for fold_dir in sorted(path for path in run_dir.glob("fold_*") if path.is_dir()):
            archive = upload / f"{fold_dir.name}_artifacts.zip"
            temporary = archive.with_suffix(".zip.partial")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED,
                                 allowZip64=True) as bundle:
                for path in sorted(fold_dir.rglob("*")):
                    if path.is_file():
                        bundle.write(path, path.relative_to(fold_dir))
            temporary.replace(archive)
        for path in run_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, upload / path.name)

    def publish(note: str) -> None:
        nonlocal existing
        sync_outputs()
        command = (["kaggle", "datasets", "version", "-p", str(upload), "-m", note,
                    "--dir-mode", "skip", "--keep-tabular"] if existing else
                   ["kaggle", "datasets", "create", "-p", str(upload),
                    "--dir-mode", "skip", "--keep-tabular"])
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, check=False)
        print(result.stdout or "", end="", flush=True)
        if result.returncode and not existing:
            retry = subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(upload), "-m", note,
                 "--dir-mode", "skip", "--keep-tabular"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
            )
            print(retry.stdout or "", end="", flush=True)
            if retry.returncode == 0:
                existing = True
                return
        if result.returncode:
            raise RuntimeError(f"Could not publish Kaggle checkpoint: {(result.stdout or '')[-2500:]}")
        existing = True

    publish(f"{run_id}: initialized; gold labels excluded")
    output = run_dir
    command = [
        sys.executable, "-u", TRAINER_REMOTE,
        "--cache", str(cache_dir), "--labels", str(label_path),
        "--folds", str(folds_path), "--scratch", str(scratch / "memmaps"),
        "--output", str(output), "--all-folds", "--epochs", "60",
        "--batch-studies", "4", "--lr", "0.0001", "--weight-decay", "0.03",
        "--patience", "12", "--min-delta", "0.0005",
    ]
    print("Starting five-fold regional reader training on Modal T4:", " ".join(command), flush=True)
    log_path = run_dir / "training.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_PROJECT_ROOT / 'src'}:{REMOTE_PROJECT_ROOT / 'scripts'}"
    process = subprocess.Popen(command, cwd=REMOTE_PROJECT_ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    epoch_re = re.compile(r"^P0 regional fold (\d+) epoch (\d+)/(\d+):")
    fold_re = re.compile(r"^P0 regional fold (\d+) complete:")
    published_epochs: set[tuple[int, int]] = set()
    return_code = -1
    try:
        assert process.stdout is not None
        with log_path.open("a", encoding="utf-8", buffering=1) as stream:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                match = epoch_re.match(line.strip())
                fold_match = fold_re.match(line.strip())
                if match:
                    fold, epoch = int(match.group(1)), int(match.group(2))
                    key = (fold, epoch)
                    if epoch % PUBLISH_EVERY_EPOCHS == 0 and key not in published_epochs:
                        publish(f"{run_id}: fold {fold}, epoch {epoch}")
                        published_epochs.add(key)
                elif fold_match:
                    publish(f"{run_id}: fold {int(fold_match.group(1))} completed")
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        sync_outputs()
        publish(f"{run_id}: finished with exit code {return_code}")
    if return_code:
        raise RuntimeError(f"Regional reader training exited with code {return_code}")
    return json.loads((run_dir / "cross_validation_summary.json").read_text(encoding="utf-8"))


@app.local_entrypoint()
def main():
    result = train_regional_reader_cv.remote()
    print(json.dumps(result, indent=2, allow_nan=True))
