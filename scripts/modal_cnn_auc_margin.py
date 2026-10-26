"""Detached T4 AUC refinement; persistent caches and full resumable state."""
from pathlib import Path
import os
import subprocess
import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/root/experiment")
source = ROOT / "kaggle/experiment_kernels/hpo_v1_blend_resnet50_warmstart/source/skanee"
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "scikit-learn", "matplotlib", "kaggle==2.0.2", "modal")
    .add_local_dir(str(source), str(REMOTE / "source"))
    .add_local_file(str(ROOT / "scripts/train_cnn_auc_margin.py"), str(REMOTE / "scripts/train_cnn_auc_margin.py"))
    .add_local_dir(str(ROOT / "src/rsna_knee"), str(REMOTE / "src/rsna_knee"))
    .add_local_dir(str(ROOT / "data/hpo_v1_resnet50_auc_margin_fold0/inputs"), str(REMOTE / "inputs"))
    .add_local_file(str(ROOT / "data/label_datasets/hpo_v1_blend/soft_labels_hpo_v1_blend.json"), str(REMOTE / "labels.json"))
    .add_local_file(str(ROOT / "data/hpo_blend_resnet50_submission/artifact/checkpoint_best.pt"), str(REMOTE / "initial.pt")))
app = modal.App("rsna-hpo-v1-resnet50-auc-margin-fold0")
cache = modal.Volume.from_name("rsna-cnn-topk-support-cache-v1")
outputs = modal.Volume.from_name("rsna-cnn-auc-margin-artifacts-v1", create_if_missing=True)

@app.function(image=image, gpu="T4", cpu=8, memory=32768, timeout=43200,
              retries=modal.Retries(max_retries=3, initial_delay=2),
              volumes={"/root/rsna/persistent_cache": cache, "/root/auc_outputs": outputs})
def train():
    outputs.reload()
    paths = [Path(f"/root/rsna/persistent_cache/shard_{i}/slice_cache") for i in range(3)]
    for path in paths:
        if not path.is_dir() or not any(path.glob("*.npy")):
            raise RuntimeError(f"Existing cache volume missing expected files: {path}; no GPU download attempted")
    command = ["python", "-u", str(REMOTE / "scripts/train_cnn_auc_margin.py"),
        "--source", str(REMOTE / "source"), "--soft-labels", str(REMOTE / "labels.json"),
        "--validation-soft-labels", str(REMOTE / "inputs/validation.json"),
        "--evidence", str(REMOTE / "inputs/evidence.json"),
        "--folds", str(REMOTE / "source/configs/report_folds_hpo_add281_trainonly.csv"),
        "--init-checkpoint", str(REMOTE / "initial.pt"), "--output-dir", "/root/auc_outputs/fold0/run",
        "--commit-volume", "--cache", *map(str, paths)]
    subprocess.run(command, check=True)

@app.local_entrypoint()
def main():
    train.remote()
