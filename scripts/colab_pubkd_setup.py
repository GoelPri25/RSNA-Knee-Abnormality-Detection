"""PUB-KD lanes on Colab (Colab CLI): VM-side setup, run with `colab exec -f` after the bundle and token are uploaded.

Same inputs and checks as scripts/modal_pubkd.py (MeluXina's pubkd.sbatch on Modal): torch 2.5.1 + cu124 / timm 1.0.29 /
numpy 2.4.6 on Python 3.11 (= rsna-mlx-train:v1) in a uv venv; the d96 cache from our Kaggle shard kernel outputs
(tumis001/rsna-coatnet-d96-cache-shard-0..3, the source of the Modal volume), checked against MeluXina's manifests; the
bundle (runner files, assets, MS1 crops, target files, env files, MeluXina's resnet50.tv_in1k HF cache) unpacked to /work.
MeluXina's paths as links: /data/d96, /opt/hf, /work/out. The Kaggle token (/content/kaggle.env) is deleted once the
shards are in. Prints SETUP_OK when every pinned check passes.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

C = Path("/content"); D96 = C / "d96"; WORK = Path("/work"); VENV = C / "venv"
KERNEL = "tumis001/rsna-coatnet-d96-cache-shard-"
D96_MANIFESTS = ("a8945d545d19a4cb", "2a830ad175c78d08", "875e9f04bc17c72f", "a562a6369e632044")   # MeluXina ~/rsna/d96
PINNED = {"assets/rule_states.csv": "ce4473c71f3ea100db021805750c6bb9038bc7b0e56a143c81516edf2d29e77b",   # = pubkd.sbatch
          "assets/soft_labels_final_per_target_latD2.json": "dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d",
          "assets/folds.csv": "2dd1044adcc54ca4d9bcb9c9fd3b910a6784bd2b99e792bbeaf84a42d417f06a",
          "pubkd/pubkd_targets_fullorder_rapvalues_mix73.json": "f07f202b846b99272b6fe34bcb1d30d4d71e18a1870bcc7d3ce61559737ccd34",
          "pubkd/pubkd_targets_raptor_pure.json": "6ed9225434bf8cf77cc2a3ded2c47a8443fc0967a1a215c081da482e1f3bfa49"}
R50_SAFETENSORS = "5d061a3c593d795bfe682d9b152bafbcf550579873492def3515b46db1189888"


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# 1. environment: the container's versions in an isolated venv (Colab's own torch is left alone)
if not (VENV / "bin/python").exists():
    sh("pip install -q uv")
    sh(f"uv venv -q --python 3.11 {VENV}")
    # one resolve, as Modal's pip_install: installed separately, timm's torchvision dependency pulls the newest torchvision,
    # which replaced torch 2.5.1 by 2.14.1+cu130 on the first VM (10-21)
    sh(f"uv pip install -q --python {VENV}/bin/python torch==2.5.1 torchvision==0.20.1 timm==1.0.29 numpy==2.4.6 "
       f"--index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple "
       f"--index-strategy unsafe-best-match")
sh(f"{VENV}/bin/python -c \"import sys, torch, timm, numpy; print(sys.version.split()[0], torch.__version__, "
   f"torch.version.cuda, timm.__version__, numpy.__version__, torch.cuda.get_device_name(0)); "
   f"assert torch.__version__.startswith('2.5.1') and torch.version.cuda == '12.4', 'torch pin lost'\"")

# 2. the d96 cache from Kaggle, then the token is removed
tok = C / "kaggle.env"
if any(not (D96 / f"shard_{i}/cache_summary.json").is_file() for i in range(4)):
    assert tok.is_file(), "upload the Kaggle token first"
    env = {**os.environ, "KAGGLE_API_TOKEN": tok.read_text().strip().split("=", 1)[1]}
    sh("pip install -q kaggle==2.2.2")
    (Path.home() / ".config/kaggle").mkdir(parents=True, exist_ok=True)   # 4 parallel CLIs race to create it

    def fetch(i):
        dest = D96 / f"shard_{i}"
        if (dest / "cache_summary.json").is_file():
            return
        tmp = C / f"_download_{i}"
        for _ in range(3):                                   # Kaggle downloads sometimes break mid-file
            shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
            if subprocess.run(["kaggle", "kernels", "output", f"{KERNEL}{i}", "-p", str(tmp)], env=env,
                              stdout=subprocess.DEVNULL).returncode == 0:
                break
        else:
            raise RuntimeError(f"shard {i}: download failed 3 times")
        D96.mkdir(exist_ok=True); shutil.rmtree(dest, ignore_errors=True)
        (tmp / "d96").rename(dest); shutil.rmtree(tmp)
        print(f"shard {i} downloaded", flush=True)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(fetch, range(4)))
tok.unlink(missing_ok=True)
assert not tok.exists()
for i, want in enumerate(D96_MANIFESTS):
    assert sha(D96 / f"shard_{i}/manifest.jsonl")[:16] == want, f"d96 shard {i} manifest differs from MeluXina's"
    rows = [json.loads(x) for x in (D96 / f"shard_{i}/manifest.jsonl").read_text().splitlines()]
    for part in sorted({r["part"] for r in rows}):           # npy header vs size (truncated downloads)
        p = D96 / f"shard_{i}" / part
        with open(p, "rb") as f:
            import numpy.lib.format as fmt
            fmt.read_magic(f); shape, _, _ = fmt.read_array_header_1_0(f); off = f.tell()
        n = sum(1 for r in rows if r["part"] == part) * 96
        assert shape == (n, 384, 384) and p.stat().st_size == off + n * 384 * 384, (i, part, shape)
print("d96 cache verified (manifests = MeluXina's, all parts complete)", flush=True)

# 3. the bundle, MeluXina's paths, pinned inputs
if not (WORK / "scripts/r50_gl4_runner.py").exists():
    with tarfile.open(C / "bundle.tgz") as t:
        t.extractall("/", filter="data")
(WORK / "out").mkdir(exist_ok=True)
for link, target in (("/data/d96", str(D96)), ("/opt/hf", str(WORK / "hf"))):
    Path(link).parent.mkdir(parents=True, exist_ok=True)
    if not Path(link).exists():
        os.symlink(target, link)
for rel, want in PINNED.items():
    assert sha(WORK / rel) == want, f"pinned input differs: {rel}"
st = list(Path("/opt/hf/hub/models--timm--resnet50.tv_in1k/snapshots").glob("*/model.safetensors"))
assert len(st) == 1 and sha(st[0]) == R50_SAFETENSORS, "resnet50.tv_in1k weights differ from MeluXina's"
for f in ("r50_gl4_runner.py", "r50_gl_runner.py", "coatnet_d96_runner.py"):
    print(f"{sha(WORK / 'scripts' / f)}  {f}", flush=True)
sh("df -h /content | tail -1; free -g | head -2; nproc")
print("SETUP_OK", flush=True)
