"""Model B3 ablations on Kaggle (free T4 x2): scripts/swin3d_d96n_runner.py (SWIN_PHASE=swin3d_b2) on the d96n cache
(kernel outputs tumis001/rsna-d96-native-cache-shard-{0..5}), one arm per GPU, fp16 + GradScaler (T4 has no bf16).

MODE "profile": preflight + timing per arm (no training). MODE "train": each arm trains on the fold-0 split and
writes its run folder (val_epochNN.json, val_swa.json, history.json, ...) to /kaggle/working/<arm>/; weights except
the SWA average are deleted at the end to keep the output small. Arms and the schedule are pre-registered in
docs/experiment_ledger.md (row B3-abl). Kinetics-400 Swin3D-T weights are downloaded from torchvision (internet on;
no competition data leaves Kaggle) and checked against torchvision's hash-in-name.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

MODE = "train"
SCHEDULE = {"epochs": 6, "swa_epochs": [5, 6], "eval_epochs": [2, 4, 5, 6]}   # fixed from the v1 / v2 profiles (ledger B3-abl)
ARMS = [                                               # (name, overrides, gpu); one batch per entry of BATCHES
    ("A0_frozen", {}, 0),
    ("A1_unfrozen_ckpt", {"freeze": 0, "ckpt": True}, 1),   # v1 profile: A1 without checkpointing OOM on 16 GB
]
BATCHES = [[0, 1]]
COMMON = {"workers": 2, "kill": None}

WORK = Path("/kaggle/working"); TMP = Path("/tmp/b3abl")
CODE = next(p.parent for p in Path("/kaggle/input").rglob("swin3d_d96n_runner.py"))
print(f"code + assets {CODE}", flush=True)

# ---- d96n cache: symlink the six shard outputs as shard_0..5
cache = TMP / "d96n"; cache.mkdir(parents=True, exist_ok=True)
found = {}
for s in Path("/kaggle/input").rglob("d96n/cache_summary.json"):
    i = json.loads(s.read_text())["shard"][0]
    assert i not in found, (i, s); found[i] = s.parent
assert sorted(found) == list(range(6)), found
for i, d in found.items():
    (cache / f"shard_{i}").symlink_to(d)
print("cache shards", {i: str(d) for i, d in sorted(found.items())}, flush=True)

# ---- code to a writable dir (python imports the runner + its IO module from there)
code = TMP / "code"; code.mkdir(exist_ok=True)
for f in ("swin3d_d96n_runner.py", "coatnet_d96_runner.py", "d96_positions.py"):
    shutil.copy(CODE / f, code / f)

# ---- Kinetics-400 weights (torchvision), hash-in-name check
from torchvision.models.video import Swin3D_T_Weights
url = Swin3D_T_Weights.KINETICS400_V1.url
wpath = TMP / "swin3d_t_k400.pth"
urllib.request.urlretrieve(url, wpath)
sha = hashlib.sha256(wpath.read_bytes()).hexdigest()
assert sha.startswith(url.rsplit("-", 1)[1].split(".")[0]), (url, sha)
print(f"weights {url} sha256 {sha}", flush=True)

import torch
print(f"torch {torch.__version__} | GPUs {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]} "
      f"| CPUs {os.cpu_count()}", flush=True)


def run_arm(k, results):
    name, over, gpu = ARMS[k]
    out = WORK / name; out.mkdir(parents=True, exist_ok=True)
    cfg = {**COMMON, **SCHEDULE, **over}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "D96_CACHE": str(cache), "D96_ASSETS": str(CODE),
           "D96_OUT": str(out), "SWIN_PHASE": "swin3d_b2", "SWIN_MODE": MODE, "SWIN_AMP": "fp16",
           "SWIN_CFG": json.dumps(cfg), "SWIN_WEIGHTS": str(wpath), "PYTHONUNBUFFERED": "1"}
    t0 = time.monotonic()
    with (out / f"{MODE}.log").open("w") as log:
        p = subprocess.Popen([sys.executable, "-u", str(code / "swin3d_d96n_runner.py")], env=env, cwd=str(code),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line); log.flush()
            if not line.startswith("  epoch"):
                print(f"[{name}] {line}", end="", flush=True)
        p.wait()
    results[name] = {"returncode": p.returncode, "minutes": round((time.monotonic() - t0) / 60, 1), "cfg": cfg}
    if MODE == "train":
        for f in out.glob("weights_epoch*.pt"):
            f.unlink()
        (out / "recovery.pt").unlink(missing_ok=True)


results = {}
if torch.cuda.device_count() < 2:                     # one GPU: arms run one after another
    ARMS = [(n, o, 0) for n, o, _ in ARMS]; BATCHES = [[k] for b in BATCHES for k in b]
for batch in BATCHES:
    th = [threading.Thread(target=run_arm, args=(k, results)) for k in batch]
    [t.start() for t in th]; [t.join() for t in th]
    print("BATCH_DONE " + json.dumps({ARMS[k][0]: results[ARMS[k][0]] for k in batch}), flush=True)
(WORK / f"{MODE}_results.json").write_text(json.dumps(results, indent=2) + "\n")
print("ALL_DONE " + json.dumps(results), flush=True)
