"""Run one report-fold Pillar-0 reader experiment on a personal Modal T4.

The cache is fetched from private Kaggle storage into temporary Modal scratch.
Only checkpoints, metrics, plots, and logs are written back to Kaggle.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import csv
import io
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_ROOT = Path("/root/rsna")
APP_NAME = "rsna-pillar0-multiscale-target-reader"
CACHE_DATASET = "tumis001/rsna-pillar0-spatial-token-cache-v1"
LABEL_DATASET = "stevenleehans/rsna-knee-llm-report-labels"
SOURCE_DATASET = "tumis001/rsna-cnn-topk-support-source-v1"
ARTIFACT_DATASET = "tumis001/rsna-pillar0-target-query-reader-v1"
REMOTE_ROOT = Path("/tmp/rsna_pillar0_target_reader")
TRAINER_REMOTE = str(REMOTE_PROJECT_ROOT / "scripts/train_pillar0_reader.py")
CHECKPOINT_REVISION = "be5050ba482607f90e607e044f03a2ebc38f98c3"
EXPECTED_CACHE_SERIES = 24_035
PUBLISH_EVERY_EPOCHS = 5

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Kaggle CLI 2.2.2 currently lists the private cache but its full-download
    # endpoint returns 404. Version 2.0.2 successfully handles both operations.
    .pip_install("torch==2.5.1", "numpy", "scikit-learn", "matplotlib", "kaggle==2.0.2")
    .add_local_file(str(PROJECT_ROOT / "scripts/train_pillar0_reader.py"), TRAINER_REMOTE)
    .add_local_dir(str(PROJECT_ROOT / "src/rsna_knee"), "/root/rsna/src/rsna_knee")
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
def train_pillar0_reader() -> dict:
    import sys

    root = REMOTE_ROOT
    inputs = root / "inputs"
    scratch = root / "scratch"
    run_id = "pillar0_target_query_fold0"
    run_dir = root / "runs" / run_id
    upload = root / "kaggle_upload"
    for directory in (inputs, scratch, run_dir, upload):
        directory.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Personal Kaggle API token was not injected into Modal")
    # Kaggle CLI 2.x documents KAGGLE_API_TOKEN as the environment auth source.
    # Setting it explicitly avoids CLI-version differences around token-file paths.
    os.environ["KAGGLE_API_TOKEN"] = token
    kaggle_config = Path.home() / ".kaggle"
    kaggle_config.mkdir(mode=0o700, parents=True, exist_ok=True)
    (kaggle_config / "access_token").write_text(token + "\n", encoding="utf-8")
    (kaggle_config / "access_token").chmod(0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_config)

    def run(command: list[str], *, capture: bool = False) -> str:
        print("$", " ".join(command), flush=True)
        completed = subprocess.run(
            command, text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
            check=False,
        )
        output = completed.stdout or "" if capture else ""
        if output:
            print(output, end="" if output.endswith("\n") else "\n", flush=True)
        if completed.returncode:
            raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{output[-2500:]}")
        return output

    def list_dataset_files(dataset: str) -> list[str]:
        names: list[str] = []
        page_token: str | None = None
        while True:
            command = ["kaggle", "datasets", "files", dataset,
                       "--page-size", "200", "--csv"]
            if page_token:
                command.extend(("--page-token", page_token))
            response = subprocess.run(
                command, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False,
            )
            if response.returncode:
                raise RuntimeError(f"Could not list Kaggle files for {dataset}: {(response.stdout or '')[-1200:]}")
            output = response.stdout or ""
            match = re.search(r"(?m)^Next Page Token = (.+)$", output)
            page_token = match.group(1).strip() if match else None
            csv_start = output.find("name,")
            if csv_start < 0:
                raise RuntimeError(f"Kaggle returned an unrecognized file listing for {dataset}")
            rows = csv.DictReader(io.StringIO(output[csv_start:]))
            names.extend(row["name"] for row in rows)
            if not page_token:
                return names

    def download_files(dataset: str, destination: Path, filenames: list[str]) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        local_path = lambda name: destination / Path(name).name
        pending = [name for name in filenames if not local_path(name).is_file()]
        if not pending:
            return

        def download_one(filename: str) -> str:
            last_error = ""
            for attempt, wait_seconds in enumerate((10, 20, 40, 80, 120)):
                try:
                    result = subprocess.run(
                        ["kaggle", "datasets", "download", "-d", dataset,
                         "-f", filename, "-p", str(destination), "--unzip", "--quiet"],
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        check=False, timeout=180,
                    )
                except subprocess.TimeoutExpired as error:
                    last_error = f"download command timed out after 180 seconds: {error}"
                    result = None
                if result is not None and result.returncode == 0 and local_path(filename).is_file():
                    return filename
                if result is not None:
                    last_error = (result.stdout or "")[-1000:]
                if attempt < 4:
                    if "429" in last_error or "Too Many Requests" in last_error:
                        print(f"Kaggle rate limit on {filename}; retrying in {wait_seconds}s", flush=True)
                    time.sleep(wait_seconds)
            raise RuntimeError(f"Failed downloading {filename} from {dataset}: {last_error}")

        print(f"Downloading {len(pending):,} individual Kaggle files from {dataset}", flush=True)
        for completed, filename in enumerate(pending, start=1):
            download_one(filename)
            if completed % 8 == 0 or completed == len(pending):
                print(f"Downloaded {completed:,}/{len(pending):,} files from {dataset}", flush=True)
            time.sleep(0.25)

    def find_one(directory: Path, filename: str) -> Path:
        matches = list(directory.rglob(filename))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {filename} below {directory}; found {matches}")
        return matches[0]

    cache_dir = inputs / "pillar0_cache"
    cache_names = list_dataset_files(CACHE_DATASET)
    cache_chunks = sorted(name for name in cache_names if name.startswith("spatial_batch_") and name.endswith(".npz"))
    if len(cache_chunks) != 376 or "run_config.json" not in cache_names:
        raise RuntimeError(f"Unexpected Pillar-0 cache contents: {len(cache_chunks)} chunks")
    download_files(CACHE_DATASET, cache_dir, ["run_config.json", *cache_chunks])
    cache_config_path = find_one(cache_dir, "run_config.json")
    cache_config = json.loads(cache_config_path.read_text(encoding="utf-8"))
    if cache_config.get("checkpoint_revision") != CHECKPOINT_REVISION:
        raise RuntimeError("Pillar-0 cache checkpoint revision differs from the verified model")
    if cache_config.get("encoder_token_gating_applied") is not False:
        raise RuntimeError("Expected ungated baseline contextual tokens in this cache")
    cache_chunks = list(cache_dir.rglob("spatial_batch_*.npz"))
    if len(cache_chunks) != 376:
        raise RuntimeError(f"Expected 376 cache chunks, found {len(cache_chunks)}")
    print(f"Pillar-0 cache downloaded: {len(cache_chunks)} batches; {EXPECTED_CACHE_SERIES:,} series", flush=True)

    label_dir = inputs / "steven_v4_labels"
    label_names = list_dataset_files(LABEL_DATASET)
    if "llm_labels_v4_blend.csv" not in label_names:
        raise RuntimeError("Steven v4 dataset does not contain llm_labels_v4_blend.csv")
    download_files(LABEL_DATASET, label_dir, ["llm_labels_v4_blend.csv"])
    label_path = find_one(label_dir, "llm_labels_v4_blend.csv")

    source_dir = inputs / "cnn_support_source"
    source_names = list_dataset_files(SOURCE_DATASET)
    fold_file = "skanee/configs/report_folds_v5.csv"
    if fold_file not in source_names:
        raise RuntimeError("CNN support-source dataset does not contain report_folds_v5.csv")
    download_files(SOURCE_DATASET, source_dir, [fold_file])
    folds_path = find_one(source_dir, "report_folds_v5.csv")

    created = subprocess.run(
        ["kaggle", "datasets", "files", ARTIFACT_DATASET],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
    ).returncode == 0
    metadata = {
        "id": ARTIFACT_DATASET,
        "title": "RSNA Pillar-0 Multiscale Target-Query Reader",
        "subtitle": "Single frozen-encoder, support-aware spatial-token reader",
        "description": (
            "Private training artifacts only: report-fold checkpoints, history, learning curves, "
            "and run metadata. The 58 official-label studies and raw DICOMs are excluded."
        ),
        "licenses": [{"name": "other"}],
    }
    (upload / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {
        "run_id": run_id,
        "experiment": "Pillar-0 frozen multiscale token reader, one target-query cross-attention layer",
        "cache_dataset": CACHE_DATASET,
        "cache_checkpoint_revision": CHECKPOINT_REVISION,
        "labels": LABEL_DATASET,
        "report_fold": 0,
        "gold_labels_loaded": False,
        "gold_studies_extracted": 0,
        "gpu": "T4",
        "max_epochs": 60,
        "early_stopping": {"metric": "report-validation macro AUC", "patience": 8, "min_delta": 0.0005},
        "scheduler": {"type": "ReduceLROnPlateau", "monitor": "report-validation BCE", "patience": 2, "factor": 0.5},
    }
    (upload / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def sync_outputs() -> None:
        for path in run_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, upload / path.name)

    def publish(note: str) -> None:
        nonlocal created
        sync_outputs()
        command = (
            ["kaggle", "datasets", "version", "-p", str(upload), "-m", note,
             "--dir-mode", "skip", "--keep-tabular"] if created else
            # Kaggle CLI 2.0.2 creates datasets privately by default; only
            # `--public` changes visibility, so omit unsupported `--private`.
            ["kaggle", "datasets", "create", "-p", str(upload),
             "--dir-mode", "skip", "--keep-tabular"]
        )
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, check=False)
        print(result.stdout or "", end="", flush=True)
        if result.returncode and not created:
            retry = subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(upload), "-m", note,
                 "--dir-mode", "skip", "--keep-tabular"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
            )
            print(retry.stdout or "", end="", flush=True)
            if retry.returncode == 0:
                created = True
                return
        if result.returncode:
            raise RuntimeError(f"Kaggle artifact publish failed: {(result.stdout or '')[-2500:]}")
        created = True

    publish(f"{run_id}: initialized; gold labels excluded")
    output = run_dir
    command = [
        sys.executable, TRAINER_REMOTE,
        "--cache", str(cache_dir), "--labels", str(label_path),
        "--folds", str(folds_path), "--scratch", str(scratch / "memmaps"),
        "--output", str(output), "--validation-fold", "0", "--epochs", "60",
        "--batch-studies", "8", "--lr", "0.0003", "--weight-decay", "0.03",
        "--patience", "8", "--min-delta", "0.0005",
    ]
    print("Starting one-fold training on a single Modal T4:", " ".join(command), flush=True)
    log_path = run_dir / "training.log"
    training_env = os.environ.copy()
    training_env["PYTHONPATH"] = str(REMOTE_PROJECT_ROOT / "src")
    process = subprocess.Popen(command, cwd=REMOTE_PROJECT_ROOT, env=training_env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_published_epoch = 0
    epoch_pattern = re.compile(r"^P0 epoch (\d+)/(\d+):")
    return_code = -1
    try:
        assert process.stdout is not None
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                match = epoch_pattern.match(line.strip())
                if match:
                    epoch = int(match.group(1))
                    if epoch % PUBLISH_EVERY_EPOCHS == 0 and epoch != last_published_epoch:
                        publish(f"{run_id}: epoch {epoch}")
                        last_published_epoch = epoch
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        sync_outputs()
        publish(f"{run_id}: completed with exit code {return_code}")
    if return_code:
        raise RuntimeError(f"Pillar-0 reader training exited with code {return_code}")
    return json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))


@app.function(
    image=image,
    cpu=0.25,
    memory=512,
    timeout=120,
    secrets=[kaggle_secret],
)
def verify_kaggle_cache_access() -> str:
    """Check private-cache listing and tiny-file download without using a GPU."""
    token = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Personal Kaggle API token was not injected into Modal")
    environment = os.environ.copy()
    environment["KAGGLE_API_TOKEN"] = token
    result = subprocess.run(
        ["kaggle", "datasets", "files", CACHE_DATASET],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=environment, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "Modal's Kaggle credential cannot read the private Pillar-0 cache: "
            + (result.stdout or "")[-1200:]
        )
    listing = result.stdout or ""
    if "run_config.json" not in listing or "spatial_batch_00001.npz" not in listing:
        raise RuntimeError("Kaggle access worked, but the expected token-cache files were not listed")
    probe_dir = REMOTE_ROOT / "kaggle_auth_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    download = subprocess.run(
        ["kaggle", "datasets", "download", "-d", CACHE_DATASET,
         "-f", "run_config.json", "-p", str(probe_dir)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=environment, check=False,
    )
    if download.returncode:
        raise RuntimeError(
            "Kaggle file listing worked, but the cache download endpoint failed: "
            + (download.stdout or "")[-1200:]
        )
    return "Modal Kaggle listing and small-file download verified for the private cache."


@app.local_entrypoint()
def main() -> None:
    print(verify_kaggle_cache_access.remote(), flush=True)
    result = train_pillar0_reader.remote()
    print(json.dumps(result, indent=2))
