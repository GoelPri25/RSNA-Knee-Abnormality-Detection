"""Run a matched BCE-vs-contrastive Experiment 4 comparison on Modal T4.

Only derived MedSigLIP embeddings, report labels, and split metadata leave
Kaggle. Gold labels and DICOMs are never downloaded. Artifacts are checkpointed
to a private Kaggle dataset every 15 epochs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "rsna-steven-contrastive-medsiglip"
ARTIFACT_DATASET = "tumis001/rsna-steven-v4-contrastive-medsiglip"
KAGGLE_KERNELS = [
    "tumis001/rsna-knee-medsiglip-features-shard-0",
    "tumis001/rsna-knee-medsiglip-features-shard-1",
    "tumis001/rsna-knee-medsiglip-features-shard-2",
    "tumis001/rsna-knee-medsiglip-gold-features",
    "tumis001/report-label-audit",
]
LABEL_DATASET = "stevenleehans/rsna-knee-llm-report-labels"
SOURCE_DATASET = "tumis001/rsna-experiment1-baseline-source"
ARTIFACT_ROOT = Path("/tmp/rsna_steven_contrastive")
TRAINER_REMOTE = "/root/rsna/scripts/train_steven_contrastive.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "pandas", "scikit-learn",
                 "matplotlib", "kaggle==2.2.2")
    .add_local_file(str(PROJECT_ROOT / "scripts/train_steven_contrastive.py"),
                    TRAINER_REMOTE)
    .add_local_dir(str(PROJECT_ROOT / "src"), "/root/rsna/src")
)

if modal.is_local():
    kaggle_token_file = Path.home() / ".kaggle/access_token"
    if not kaggle_token_file.is_file():
        raise FileNotFoundError(f"Kaggle API token not found at {kaggle_token_file}")
    kaggle_secret = modal.Secret.from_dict(
        {"KAGGLE_ACCESS_TOKEN": kaggle_token_file.read_text(encoding="utf-8").strip()}
    )
else:
    kaggle_secret = modal.Secret.from_dict({})

app = modal.App(APP_NAME)


@app.function(image=image, gpu="T4", timeout=28_800, secrets=[kaggle_secret],
              scaledown_window=60)
def run_experiment() -> dict:
    import csv
    import datetime as dt
    import subprocess
    import sys
    from pathlib import Path

    root = ARTIFACT_ROOT
    inputs = root / "inputs"
    runs = root / "runs"
    upload = root / "kaggle_upload"
    inputs.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    upload.mkdir(parents=True, exist_ok=True)

    kaggle_config = Path.home() / ".kaggle"
    kaggle_config.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Personal Kaggle API token was not injected into Modal")
    (kaggle_config / "access_token").write_text(token + "\n", encoding="utf-8")
    (kaggle_config / "access_token").chmod(0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_config)

    def run(command: list[str], *, capture: bool = False) -> str:
        print("$", " ".join(command), flush=True)
        completed = subprocess.run(command, text=True,
                                   stdout=subprocess.PIPE if capture else None,
                                   stderr=subprocess.STDOUT if capture else None,
                                   check=False)
        result = completed.stdout or "" if capture else ""
        if result:
            print(result, end="" if result.endswith("\n") else "\n", flush=True)
        if completed.returncode:
            raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{result[-2500:]}")
        return result

    def output_kernel(kernel: str, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        if not any(destination.iterdir()):
            run(["kaggle", "kernels", "output", kernel, "-p", str(destination)])

    def find_one(directory: Path, filename: str) -> Path:
        matches = list(directory.rglob(filename))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {filename} under {directory}; found {matches}")
        return matches[0]

    source_download = inputs / "baseline_source"
    source_download.mkdir(exist_ok=True)
    if not list(source_download.rglob("scripts/train_experiment4.py")):
        run(["kaggle", "datasets", "download", "-d", SOURCE_DATASET,
             "-p", str(source_download), "--unzip"])
    source_hits = list(source_download.rglob("scripts/train_experiment4.py"))
    if len(source_hits) != 1:
        raise RuntimeError(f"Expected one Experiment 4 source tree; found {source_hits}")
    source_root = source_hits[0].parents[1]

    cache_root = root / "cache"
    for index, kernel in enumerate(KAGGLE_KERNELS[:4]):
        kernel_dir = inputs / kernel.replace("/", "__")
        output_kernel(kernel, kernel_dir)
        manifests = list(kernel_dir.rglob("manifest.jsonl"))
        feature_dirs = [path.parent for path in manifests
                        if (path.parent / "feature_summary.json").is_file()]
        if len(feature_dirs) != 1:
            raise RuntimeError(f"Expected one feature cache from {kernel}; found {feature_dirs}")
        destination = cache_root / (f"shard_{index}" if index < 3 else "gold") / "features"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(feature_dirs[0], target_is_directory=True)

    audit_dir = inputs / "report_audit"
    output_kernel(KAGGLE_KERNELS[4], audit_dir)
    gold_split = find_one(audit_dir, "report_gold_split.csv")

    label_dir = inputs / "steven_labels"
    label_dir.mkdir(exist_ok=True)
    if not list(label_dir.rglob("llm_labels_v4_blend.csv")):
        run(["kaggle", "datasets", "download", "-d", LABEL_DATASET,
             "-p", str(label_dir), "--unzip"])
    labels_path = find_one(label_dir, "llm_labels_v4_blend.csv")

    # Verify output indexing and the report-only development IDs without
    # downloading any gold labels. Create a unique, resumable artifact prefix.
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"steven_v4_contrastive_{timestamp}"
    run_root = runs / run_id
    run_root.mkdir(parents=True)
    metadata_path = upload / "dataset-metadata.json"
    metadata_path.write_text(json.dumps({
        "id": ARTIFACT_DATASET,
        "title": "RSNA Steven v4 Contrastive MedSigLIP Training",
        "subtitle": "Matched BCE control and target-aware contrastive reader",
        "description": (
            "Private training artifacts only: best/latest checkpoints, epoch logs, "
            "learning curves, and development-only predictions. No DICOMs, "
            "official labels, or locked evaluation labels are included."
        ),
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")
    created = subprocess.run(["kaggle", "datasets", "files", ARTIFACT_DATASET],
                             text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False).returncode == 0
    if created:
        prior = inputs / "prior_artifacts"
        prior.mkdir(exist_ok=True)
        run(["kaggle", "datasets", "download", "-d", ARTIFACT_DATASET,
             "-p", str(prior), "--unzip"])
        for item in prior.rglob("*"):
            if item.is_file() and item.name not in {
                    "dataset-metadata.json", "preflight.txt", "version_preflight.txt"}:
                relative = item.relative_to(prior)
                flat_name = "__".join(relative.parts)
                shutil.copy2(item, upload / flat_name)

    def publish(note: str) -> None:
        nonlocal created
        command = (["kaggle", "datasets", "version", "-p", str(upload),
                    "-m", note, "--dir-mode", "skip", "--keep-tabular"]
                   if created else
                   ["kaggle", "datasets", "create", "-p", str(upload),
                    "--dir-mode", "skip", "--keep-tabular"])
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            # A create can race with a prior attempt that completed remotely.
            if not created:
                retry = subprocess.run(
                    ["kaggle", "datasets", "version", "-p", str(upload),
                     "-m", note, "--dir-mode", "skip", "--keep-tabular"],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    check=False,
                )
                print(retry.stdout, end="", flush=True)
                if retry.returncode == 0:
                    created = True
                    return
            raise RuntimeError(f"Kaggle artifact publish failed: {result.stdout[-2000:]}")
        created = True

    def sync_run_artifacts(arm_name: str, arm_output: Path) -> None:
        # Kaggle's dataset uploader was configured with --dir-mode skip. Keep
        # every artifact at the dataset root so interval versions contain the
        # actual checkpoints instead of silently omitting nested run folders.
        for item in arm_output.iterdir():
            if item.is_file():
                flat_name = f"{run_id}__{arm_name}__{item.name}"
                shutil.copy2(item, upload / flat_name)

    def execute_arm(arm_name: str, contrastive_weight: float) -> dict:
        arm_output = run_root / arm_name
        command = [
            sys.executable, TRAINER_REMOTE,
            "--source-root", str(source_root),
            "--cache-root", str(cache_root),
            "--steven-labels", str(labels_path),
            "--gold-split", str(gold_split),
            "--output", str(arm_output),
            "--validation-fold", "0",
            "--epochs", "60",
            "--contrastive-weight", str(contrastive_weight),
            "--contrastive-objective", "absolute_mass_v2",
            "--temperature", "0.10",
            "--memory-capacity", "512",
            "--seed", "2026",
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            ["/root/rsna/src", str(source_root), environment.get("PYTHONPATH", "")]
        )
        print(f"\n=== {arm_name}: contrastive weight {contrastive_weight:g} ===", flush=True)
        process = subprocess.Popen(command, cwd=source_root, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        epoch_pattern = re.compile(r"^epoch (\d+)/60:")
        for line in process.stdout:
            print(line, end="", flush=True)
            match = epoch_pattern.match(line.strip())
            if match and int(match.group(1)) % 15 == 0:
                sync_run_artifacts(arm_name, arm_output)
                publish(f"{run_id}: {arm_name}, completed epoch {match.group(1)}")
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"{arm_name} training subprocess failed ({return_code})")
        sync_run_artifacts(arm_name, arm_output)
        summary = json.loads((arm_output / "run_summary.json").read_text())
        return summary

    results = {
        "experiment": "matched Steven v4 MedSigLIP BCE vs absolute-mass soft contrastive",
        "run_id": run_id,
        "feature_cache_source": KAGGLE_KERNELS[:4],
        "label_source": LABEL_DATASET,
        "gold_evaluation_labels_loaded": False,
        "arms": {},
    }
    results["arms"]["bce_control"] = execute_arm("bce_control", 0.0)
    results["arms"]["target_contrastive"] = execute_arm("target_contrastive", 0.05)
    summary_path = run_root / "paired_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copy2(summary_path, upload / f"{run_id}__paired_summary.json")
    publish(f"{run_id}: completed matched experiment")
    return results


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run_experiment.remote(), indent=2))
