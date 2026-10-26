"""Explicit CPU transfer or detached single-GPU training; no automatic duplicate run."""
import hashlib
import json
import os
import re
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
         .add_local_dir(str(LOCAL / "data/local_global/assets"), "/root/assets"))
for name in ["efficientnet_v31_runner.py", "local_global_runner.py", "local_global_model.py", "local_global_data.py", "local_global_speed.py", "build_native_triplet_cache.py"]:
    image = image.add_local_file(str(LOCAL / "scripts" / name), f"/root/{name}")
app = modal.App("rsna-local-global-native-triplets")
ROOT = Path("/root/cache")


def authenticate():
    token = os.environ["KAGGLE_ACCESS_TOKEN"]
    os.environ["KAGGLE_API_TOKEN"] = token
    folder = Path.home() / ".kaggle"; folder.mkdir(exist_ok=True)
    path = folder / "access_token"; path.write_text(token); path.chmod(0o600)


@app.function(image=image, cpu=2, memory=4096, timeout=43200, retries=2, secrets=[secret], volumes={"/root/cache": volume})
def transfer(shard: int):
    if shard not in range(4): raise ValueError("Expected shard 0..3")
    volume.reload(); authenticate()
    target = ROOT / f"shard_{shard}/slice_cache"
    complete = target / "transfer_complete.json"
    if complete.is_file():
        print(f"TRANSFER_REUSED shard={shard}", flush=True); return
    # Reuse any previous downloads on the volume across retries. Only fixed
    # output directories; no recursive file discovery or duplicate data cache.
    download = ROOT / f"downloads/shard_{shard}"; download.mkdir(parents=True, exist_ok=True)
    slug = f"tumis001/rsna-cnn-native-triplet-cache-shard-{shard}"
    subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(download),
                    "--file-pattern", r"(?:manifest\.jsonl|cache_summary\.json|identity\.json)$"], check=True)
    source = download / f"native_triplets/shard_{shard}/slice_cache"
    summary = json.loads((source / "cache_summary.json").read_text())
    if not summary["complete"] or summary["schema"] != "native_triplets_v1": raise ValueError("Incomplete/wrong cache")
    target.mkdir(parents=True, exist_ok=True)
    for name in ["manifest.jsonl", "cache_summary.json", "identity.json"]:
        shutil.copy2(source / name, target / name)
    records = [json.loads(s) for s in (target / "manifest.jsonl").read_text().splitlines() if s]
    names = {r[k] for r in records for k in ("part", "support_part")}
    for name in sorted(names):
        dest = target / name
        import numpy as np
        rows = max(r["row_start"] + r["stored_count"] for r in records if r["part"] == name or r["support_part"] == name)
        if dest.exists():
            array = np.load(dest, mmap_mode="r")
            valid = array.shape == (rows, 224, 224) and array.dtype == np.uint8
            del array
            if valid: continue
            raise ValueError(f"Existing transferred array differs: {dest}")
        subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(download),
                        "--file-pattern", re.escape(f"native_triplets/shard_{shard}/slice_cache/{name}") + "$"], check=True)
        array = np.load(source / name, mmap_mode="r")
        if array.shape != (rows, 224, 224) or array.dtype != np.uint8: raise ValueError(f"Wrong downloaded array: {name}")
        del array
        (source / name).replace(dest)
        volume.commit(); print(f"TRANSFER_PART shard={shard} file={name}", flush=True)
    (target / "transfer_complete.json").write_text(json.dumps({"shard": shard, "summary": summary}) + "\n")
    volume.commit()


@app.function(image=image, gpu="T4", cpu=4, memory=16384, timeout=43200, retries=3, max_containers=1, volumes={"/root/cache": volume})
def train():
    volume.reload()
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
    out = ROOT / "experiments/local_global_v31_d2_seed2026"; out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env["PYTHONPATH"] = "/root/source:/root"; env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "4"; env["MKL_NUM_THREADS"] = "4"
    process = subprocess.Popen(["python", "-u", "/root/local_global_runner.py"], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (out / "training.log").open("a", buffering=1) as f:
        for line in process.stdout: print(line, end="", flush=True); f.write(line)
    code = process.wait(); volume.commit()
    if code: raise RuntimeError(f"Training exited {code}; committed recovery retained")


@app.function(image=image, cpu=1, memory=2048, timeout=3600, secrets=[secret], volumes={"/root/cache": volume})
def publish():
    volume.reload(); authenticate()
    run = ROOT / "experiments/local_global_v31_d2_seed2026/run"
    output = Path("/root/publish"); output.mkdir(exist_ok=True)
    for name in ["checkpoint_best.pt", "checkpoint_latest.pt", "run_summary.json", "training_history.json",
                 "training_history.csv", "oof_predictions.json", "learning_curve.png", "execution.json"]:
        shutil.copy2(run / name, output / name)
    for name in ["local_global_model.py", "local_global_data.py", "build_native_triplet_cache.py"]:
        shutil.copy2(Path("/root") / name, output / name)
    (output / "dataset-metadata.json").write_text(json.dumps({"id": "tumis001/rsna-local-global-resnet50-v31-d2",
        "title": "RSNA Local Global ResNet50 V31 D2", "licenses": [{"name": "other"}]}))
    subprocess.run(["kaggle", "datasets", "create", "-p", str(output), "-r", "zip"], check=True)


@app.function(image=image, cpu=.25, memory=512, timeout=86400, max_containers=1)
def coordinate_train():
    train.remote()
    try: publish.remote()
    except Exception as error: print(f"PUBLICATION_FAILED {error}; checkpoint is safe on persistent volume", flush=True)


@app.function(image=image, cpu=.25, memory=512, timeout=86400)
def coordinate_transfer():
    # The detached remote coordinator survives the local CLI disconnecting.
    # Serial transfers keep download RAM/CPU bounded and reuse committed files.
    for shard in range(4):
        transfer.remote(shard)
        print(f"SHARD_TRANSFER_COMPLETE {shard}/4", flush=True)
    print("ALL_CACHE_TRANSFERS_COMPLETE", flush=True)


@app.local_entrypoint()
def main(stage: str = "transfer", shard: int = 0):
    # Launch the deployed job without a local synchronous result waiter.
    if stage == "transfer":
        job = modal.Function.from_name(app.name, "transfer").spawn(shard)
    elif stage == "transfer-all":
        job = modal.Function.from_name(app.name, "coordinate_transfer").spawn()
    elif stage == "train":
        job = modal.Function.from_name(app.name, "coordinate_train").spawn()
    else: raise ValueError("Stage must be transfer, transfer-all or train")
    receipt = {"app_name": app.name, "stage": stage, "function_call_id": job.object_id,
               "mode": "deployed background job; no local result waiter"}
    (LOCAL / "data/local_global/latest_launch.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print("BACKGROUND_JOB_SUBMITTED " + json.dumps(receipt), flush=True)
