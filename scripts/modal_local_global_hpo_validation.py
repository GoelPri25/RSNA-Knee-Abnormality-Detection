"""Detached local+global HPO-validation comparison; reuse all four native shards."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parents[1]
VOLUME_NAME = "rsna-cnn-native-triplets-224px-v1"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = (modal.Secret.from_dict({"KAGGLE_ACCESS_TOKEN": (Path.home() / ".kaggle/access_token").read_text().strip()})
          if modal.is_local() else modal.Secret.from_dict({}))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "scikit-learn", "matplotlib", "kaggle==2.2.2", "modal")
         .add_local_dir(str(LOCAL / "data/kaggle_reference/hpo_xla_actual/skanee"), "/root/source")
         .add_local_dir(str(LOCAL / "data/local_global_hpo_validation/assets"), "/root/assets"))
for name in ["efficientnet_v31_runner.py", "local_global_runner.py", "local_global_model.py", "local_global_data.py", "local_global_speed.py", "build_native_triplet_cache.py"]:
    image = image.add_local_file(str(LOCAL / "scripts" / name), f"/root/{name}")
app = modal.App("rsna-local-global-hpo-validation")
ROOT = Path("/root/cache")


def authenticate():
    token = os.environ["KAGGLE_ACCESS_TOKEN"]
    os.environ["KAGGLE_API_TOKEN"] = token
    folder = Path.home() / ".kaggle"; folder.mkdir(exist_ok=True)
    path = folder / "access_token"; path.write_text(token); path.chmod(0o600)


@app.function(image=image, gpu="T4", cpu=4, memory=16384, timeout=43200, retries=3, max_containers=1, volumes={"/root/cache": volume})
def train():
    volume.reload()
    print("EXPERIMENT: local+global ResNet50/ImageNet; HPO-v3.1-D2 train AND validation; Steven audit-only.", flush=True)
    print("CONTROLS: HPO BCE LR plateau; HPO AUC checkpoint/early-stop patience=8 delta=.0005; max60 epochs.", flush=True)
    import sys
    sys.path.insert(0, "/root/source")
    from src.images.dataset import load_manifest
    from local_global_data import NativeStudyCache
    paths = [ROOT / f"shard_{i}/slice_cache" for i in range(4)]
    for p in paths:
        if not (p / "transfer_complete.json").is_file(): raise ValueError(f"Transfer not complete: {p}")
    manifest = load_manifest(paths, ["unordered"]); NativeStudyCache(manifest)
    labels = json.loads(Path("/root/assets/labels.json").read_text())
    if set(manifest) != {r["study_id"] for r in labels["labels"]} or len(manifest) != 4349:
        raise ValueError("Cache does not match all 4349 report studies")
    out = ROOT / "experiments/local_global_v31_trainval_d2_seed2026"; out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env["PYTHONPATH"] = "/root/source:/root"; env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "4"; env["MKL_NUM_THREADS"] = "4"
    process = subprocess.Popen(["python", "-u", "/root/local_global_runner.py", "--experiment", "local_global_v31_trainval_d2_seed2026"], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (out / "training.log").open("a", buffering=1) as f:
        for line in process.stdout: print(line, end="", flush=True); f.write(line)
    code = process.wait(); volume.commit()
    if code: raise RuntimeError(f"Training exited {code}; committed recovery retained")


@app.function(image=image, cpu=1, memory=2048, timeout=3600, secrets=[secret], volumes={"/root/cache": volume})
def publish():
    volume.reload(); authenticate()
    run = ROOT / "experiments/local_global_v31_trainval_d2_seed2026/run"
    output = Path("/root/publish"); output.mkdir(exist_ok=True)
    for name in ["checkpoint_best.pt", "checkpoint_latest.pt", "run_summary.json", "training_history.json",
                 "training_history.csv", "oof_predictions.json", "learning_curve.png", "execution.json"]:
        shutil.copy2(run / name, output / name)
    for name in ["local_global_model.py", "local_global_data.py", "build_native_triplet_cache.py"]:
        shutil.copy2(Path("/root") / name, output / name)
    (output / "dataset-metadata.json").write_text(json.dumps({"id": "tumis001/rsna-local-global-resnet50-v31-trainval",
        "title": "RSNA Local Global ResNet50 V31 Trainval", "licenses": [{"name": "other"}]}))
    subprocess.run(["kaggle", "datasets", "create", "-p", str(output), "-r", "zip"], check=True)


@app.function(image=image, cpu=.25, memory=512, timeout=86400, max_containers=1)
def coordinate_train():
    train.remote()
    try: publish.remote()
    except Exception as error: print(f"PUBLICATION_FAILED {error}; checkpoint is safe on persistent volume", flush=True)


@app.local_entrypoint()
def main():
    job = modal.Function.from_name(app.name, "coordinate_train").spawn()
    receipt = {"app_name": app.name, "function_call_id": job.object_id,
               "mode": "deployed background job; no local result waiter"}
    (LOCAL / "data/local_global_hpo_validation/latest_launch.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print("BACKGROUND_JOB_SUBMITTED " + json.dumps(receipt), flush=True)
