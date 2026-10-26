"""Run the paired Steven v4 temporal-CNN vs Experiment 4 comparison on Modal."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "rsna-steven-compact-temporal-cnn-reader"
ARTIFACT_DATASET = "tumis001/rsna-steven-compact-temporal-cnn-reader"
SOURCE_DATASET = "tumis001/rsna-experiment1-baseline-source"
LABEL_DATASET = "stevenleehans/rsna-knee-llm-report-labels"
FEATURE_KERNELS = [
    "tumis001/rsna-knee-medsiglip-features-shard-0",
    "tumis001/rsna-knee-medsiglip-features-shard-1",
    "tumis001/rsna-knee-medsiglip-features-shard-2",
    "tumis001/rsna-knee-medsiglip-gold-features",
]
REPORT_AUDIT_KERNEL = "tumis001/report-label-audit"
ARTIFACT_ROOT = Path("/tmp/rsna_steven_temporal_cnn")
TRAINER_REMOTE = "/root/rsna/scripts/train_steven_temporal_cnn.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "pandas", "scikit-learn",
                 "matplotlib", "kaggle==2.2.2")
    .add_local_file(str(PROJECT_ROOT / "scripts/train_steven_temporal_cnn.py"), TRAINER_REMOTE)
    .add_local_dir(str(PROJECT_ROOT / "src"), "/root/rsna/src")
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


@app.function(image=image, gpu="T4", timeout=28_800, secrets=[kaggle_secret],
              scaledown_window=60)
def train_paired_readers() -> dict:
    import datetime as dt
    import sys

    root = ARTIFACT_ROOT
    inputs = root / "inputs"
    run_root = root / "runs"
    upload = root / "kaggle_upload"
    for directory in (inputs, run_root, upload):
        directory.mkdir(parents=True, exist_ok=True)

    kaggle_directory = Path.home() / ".kaggle"
    kaggle_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Personal Kaggle API token was not injected into Modal")
    (kaggle_directory / "access_token").write_text(token + "\n", encoding="utf-8")
    (kaggle_directory / "access_token").chmod(0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_directory)

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

    def kernel_output(kernel: str, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        if not any(destination.iterdir()):
            run(["kaggle", "kernels", "output", kernel, "-p", str(destination)])

    def find_one(directory: Path, filename: str) -> Path:
        matches = list(directory.rglob(filename))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {filename} under {directory}, found {matches}")
        return matches[0]

    source_directory = inputs / "experiment4_source"
    source_directory.mkdir(exist_ok=True)
    if not list(source_directory.rglob("scripts/train_experiment4.py")):
        run(["kaggle", "datasets", "download", "-d", SOURCE_DATASET,
             "-p", str(source_directory), "--unzip"])
    source_hits = list(source_directory.rglob("scripts/train_experiment4.py"))
    if len(source_hits) != 1:
        raise RuntimeError(f"Expected one Experiment 4 source tree, found {source_hits}")
    source_root = source_hits[0].parents[1]

    cache_root = root / "cache"
    for index, kernel in enumerate(FEATURE_KERNELS):
        destination = inputs / kernel.replace("/", "__")
        kernel_output(kernel, destination)
        manifests = list(destination.rglob("manifest.jsonl"))
        feature_directories = [path.parent for path in manifests
                               if (path.parent / "feature_summary.json").is_file()]
        if len(feature_directories) != 1:
            raise RuntimeError(f"Expected one feature cache from {kernel}; got {feature_directories}")
        target = cache_root / (f"shard_{index}" if index < 3 else "gold") / "features"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(feature_directories[0], target_is_directory=True)

    audit_directory = inputs / "report_audit"
    kernel_output(REPORT_AUDIT_KERNEL, audit_directory)
    gold_split = find_one(audit_directory, "report_gold_split.csv")

    label_directory = inputs / "steven_v4_labels"
    label_directory.mkdir(exist_ok=True)
    if not list(label_directory.rglob("llm_labels_v4_blend.csv")):
        run(["kaggle", "datasets", "download", "-d", LABEL_DATASET,
             "-p", str(label_directory), "--unzip"])
    labels_path = find_one(label_directory, "llm_labels_v4_blend.csv")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"steven_v4_compact_temporal_cnn_{timestamp}"
    output = run_root / run_id
    output.mkdir(parents=True)
    (output / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id,
        "experiment": "paired Experiment 4 vs compact regularized slice-preserving temporal CNN",
        "reader_profile": "compact",
        "label_source": LABEL_DATASET,
        "feature_cache_kernels": FEATURE_KERNELS,
        "gold_labels_downloaded": False,
        "gold_ids_used_only_for_exclusion": True,
        "folds": [0, 1, 2, 3, 4],
    }, indent=2), encoding="utf-8")

    metadata = {
        "id": ARTIFACT_DATASET,
        "title": "RSNA Steven v4 Compact Temporal CNN Reader",
        "subtitle": "Regularized slice-preserving reader paired against Experiment 4",
        "description": (
            "Private derived training artifacts: model checkpoints, histories, "
            "learning curves, and report-only out-of-fold predictions. No DICOMs "
            "or official evaluation labels are included."
        ),
        "licenses": [{"name": "other"}],
    }
    (upload / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    dataset_exists = subprocess.run(
        ["kaggle", "datasets", "files", ARTIFACT_DATASET],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    ).returncode == 0
    if dataset_exists:
        previous = inputs / "previous_artifacts"
        previous.mkdir(exist_ok=True)
        run(["kaggle", "datasets", "download", "-d", ARTIFACT_DATASET,
             "-p", str(previous), "--unzip"])
        for item in previous.rglob("*"):
            if item.is_file() and item.name != "dataset-metadata.json":
                shutil.copy2(item, upload / f"archive__{item.name}")

    created = dataset_exists

    def sync_outputs() -> None:
        for item in output.rglob("*"):
            if item.is_file():
                relative = item.relative_to(output)
                flat_name = "__".join((run_id, *relative.parts))
                shutil.copy2(item, upload / flat_name)

    def publish(note: str) -> None:
        nonlocal created
        command = (["kaggle", "datasets", "version", "-p", str(upload), "-m", note,
                    "--dir-mode", "skip", "--keep-tabular"] if created else
                   ["kaggle", "datasets", "create", "-p", str(upload),
                    "--dir-mode", "skip", "--keep-tabular"])
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, check=False)
        print(completed.stdout or "", end="", flush=True)
        if completed.returncode and not created:
            retry = subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(upload), "-m", note,
                 "--dir-mode", "skip", "--keep-tabular"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            print(retry.stdout or "", end="", flush=True)
            if retry.returncode == 0:
                created = True
                return
        if completed.returncode:
            raise RuntimeError(f"Kaggle artifact publish failed: {(completed.stdout or '')[-2000:]}")
        created = True

    sync_outputs()
    publish(f"{run_id}: started; gold labels remain excluded")

    command = [
        sys.executable, TRAINER_REMOTE,
        "--source-root", str(source_root),
        "--cache-root", str(cache_root),
        "--steven-labels", str(labels_path),
        "--gold-split", str(gold_split),
        "--output", str(output),
        "--folds", "0", "1", "2", "3", "4",
        "--epochs", "60", "--seed", "2026",
        "--reader-profile", "compact",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        ["/root/rsna/src", str(source_root), environment.get("PYTHONPATH", "")]
    )
    print("\nLaunching paired five-fold training:\n" + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=source_root, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    interval = 15
    epoch_pattern = re.compile(r"^epoch (\d+)/(\d+) \| (\w+) \| fold (\d+) \|")
    published_epochs: set[tuple[str, int, int]] = set()
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            match = epoch_pattern.match(line.strip())
            if match:
                epoch, _, arm, fold = match.groups()
                epoch, fold = int(epoch), int(fold)
                key = (arm, fold, epoch)
                if epoch % interval == 0 and key not in published_epochs:
                    sync_outputs()
                    publish(f"{run_id}: {arm}, fold {fold}, epoch {epoch}")
                    published_epochs.add(key)
    finally:
        return_code = process.wait()
        sync_outputs()
        publish(f"{run_id}: training process ended with code {return_code}")
    if return_code:
        raise RuntimeError(f"Paired training subprocess failed with exit code {return_code}")
    return json.loads((output / "paired_summary.json").read_text(encoding="utf-8"))


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(train_paired_readers.remote(), indent=2))
