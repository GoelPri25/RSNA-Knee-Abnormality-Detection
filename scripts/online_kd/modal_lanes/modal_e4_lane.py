"""E4-recipe lanes on Modal (a personal Modal workspace), the Colab pipeline ported: same inputs, pins and runner.

prepare (CPU): the d96 cache from the Kaggle shard kernels tumis001/rsna-coatnet-d96-cache-shard-0..3 (4 in parallel),
the teammate kit tumis001/rsna-pubkd-teammate-kit file by file (parallel per-file downloads hit HTTP 429), checks
(MeluXina's d96 manifests, every npy part complete, the pinned assets / targets / resnet50 weights of
colab_pubkd_setup.py), then the ImageNet weights of the extra backbones into the kit's HF cache. Kaggle token: Modal
secret "kaggle" (KAGGLE_API_TOKEN), never written to the volume.
train (H100): the lane's env = the kit's pubkd_kd_e4.env + V4_BB (+ V4_VAL_FOLD, D96_SEED); the runner copied to the
container and patched there (r101 / cnxs / val-fold); the 32-step smoke first, the lane only after SMOKE_DONE; the volume
committed every 5 minutes, so a restarted container resumes from recovery.pt.
Usage: MODAL_PROFILE=<profile> modal deploy modal_e4_lane.py, then spawn.py (detached; never cancel a running lane).
"""
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
OKD = HERE.parent
app = modal.App(os.environ.get("E4_APP", "rsna-e4-lanes"))      # a second deploy name leaves running calls alone
vol = modal.Volume.from_name("rsna-e4-lanes", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("unzip")
         .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("timm==1.0.29", "numpy==2.4.6", "kaggle==2.2.2")
         .add_local_file(str(OKD / "apply_r101_patch.py"), "/root/patches/apply_r101_patch.py")
         .add_local_file(str(OKD / "apply_cnxs_patch.py"), "/root/patches/apply_cnxs_patch.py")
         .add_local_file(str(OKD / "apply_bit101_patch.py"), "/root/patches/apply_bit101_patch.py")
         .add_local_file(str(OKD / "apply_valfold_patch.py"), "/root/patches/apply_valfold_patch.py"))
V = Path("/vol"); KIT = V / "kit"; D96 = V / "d96"; OUT = V / "out"
KIT_DS, SHARD = "tumis001/rsna-pubkd-teammate-kit", "tumis001/rsna-coatnet-d96-cache-shard-"
D96_MANIFESTS = ("a8945d545d19a4cb", "2a830ad175c78d08", "875e9f04bc17c72f", "a562a6369e632044")   # MeluXina ~/rsna/d96
PINNED = {"assets/rule_states.csv": "ce4473c71f3ea100db021805750c6bb9038bc7b0e56a143c81516edf2d29e77b",   # = pubkd.sbatch
          "assets/soft_labels_final_per_target_latD2.json": "dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d",
          "assets/folds.csv": "2dd1044adcc54ca4d9bcb9c9fd3b910a6784bd2b99e792bbeaf84a42d417f06a",
          "pubkd/pubkd_targets_fullorder_rapvalues_mix73.json": "f07f202b846b99272b6fe34bcb1d30d4d71e18a1870bcc7d3ce61559737ccd34"}
R50_SAFETENSORS = "5d061a3c593d795bfe682d9b152bafbcf550579873492def3515b46db1189888"
EXTRA_ARCHS = {"r101": "resnet101.tv_in1k", "cnxs": "convnext_small.fb_in22k_ft_in1k",
               "bit101": "resnetv2_101x1_bit.goog_in21k_ft_in1k"}
PATCH = {"r101": "apply_r101_patch.py", "cnxs": "apply_cnxs_patch.py", "bit101": "apply_bit101_patch.py"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def kaggle(*args, capture=True):
    return subprocess.run(["kaggle", *args], capture_output=capture, text=True)


@app.function(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("kaggle")], cpu=8, memory=16384,
              timeout=4 * 3600)
def prepare():
    D96.mkdir(parents=True, exist_ok=True)

    def fetch(i):                                            # one d96 shard (Kaggle downloads sometimes break mid-file)
        dest = D96 / f"shard_{i}"
        if (dest / "cache_summary.json").is_file():
            return f"shard {i} present"
        tmp = V / f"_dl_{i}"
        for t in range(3):
            shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
            r = kaggle("kernels", "output", f"{SHARD}{i}", "-p", str(tmp))
            if r.returncode == 0 and (tmp / "d96/cache_summary.json").is_file():
                break
            log(f"shard {i} try {t}: {(r.stdout + r.stderr)[-200:]}")
        else:
            raise RuntimeError(f"shard {i}: download failed 3 times")
        shutil.rmtree(dest, ignore_errors=True); (tmp / "d96").rename(dest); shutil.rmtree(tmp)
        return f"shard {i} downloaded"

    def kit():                                               # sequential, rate-limit aware (kit_seq.py)
        lst = kaggle("datasets", "files", KIT_DS, "--page-size", "1000", "--csv")
        files = [l.split(",")[0] for l in lst.stdout.splitlines()[1:] if l.strip()]
        assert files, f"kit file list empty: {lst.stderr[-300:]}"
        for f in files:
            d = KIT / os.path.dirname(f); d.mkdir(parents=True, exist_ok=True)
            for t in range(8):
                if (KIT / f).exists():
                    break
                r = kaggle("datasets", "download", KIT_DS, "-f", f, "-p", str(d), "-q")
                z = d / (os.path.basename(f) + ".zip")
                if z.exists():
                    subprocess.run(["unzip", "-oq", str(z), "-d", str(d)]); z.unlink()
                if not (KIT / f).exists():
                    time.sleep(60 if "429" in r.stdout + r.stderr else 10)
            time.sleep(1.5)
        left = [f for f in files if not (KIT / f).exists()]
        assert not left, f"kit files missing: {left[:5]}"
        return f"kit {len(files)} files"

    t0 = time.monotonic()
    with ThreadPoolExecutor(5) as pool:
        futs = [pool.submit(fetch, i) for i in range(4)] + [pool.submit(kit)]
        for fu in as_completed(futs):
            log(f"{fu.result()} ({time.monotonic() - t0:.0f}s)")
            vol.commit()
    import numpy.lib.format as fmt
    for i, want in enumerate(D96_MANIFESTS):                 # manifests = MeluXina's; npy header vs size (truncation)
        assert sha(D96 / f"shard_{i}/manifest.jsonl")[:16] == want, f"d96 shard {i} manifest differs"
        rows = [json.loads(x) for x in (D96 / f"shard_{i}/manifest.jsonl").read_text().splitlines()]
        for part in sorted({r["part"] for r in rows}):
            p = D96 / f"shard_{i}" / part
            with open(p, "rb") as f:
                fmt.read_magic(f); shape, _, _ = fmt.read_array_header_1_0(f); off = f.tell()
            n = sum(1 for r in rows if r["part"] == part) * 96
            assert shape == (n, 384, 384) and p.stat().st_size == off + n * 384 * 384, (i, part, shape)
    log("d96 cache verified")
    W = KIT / "work"
    for rel, want in PINNED.items():
        assert sha(W / rel) == want, f"pinned input differs: {rel}"
    st = list((W / "hf/hub/models--timm--resnet50.tv_in1k/snapshots").glob("*/model.safetensors"))
    assert len(st) == 1 and sha(st[0]) == R50_SAFETENSORS, "resnet50.tv_in1k weights differ"
    env = {**os.environ, "HF_HOME": str(W / "hf"), "HF_HUB_OFFLINE": "0"}
    for arch in EXTRA_ARCHS.values():
        subprocess.run(["python", "-c", f"import timm; m = timm.create_model('{arch}', pretrained=True); "
                        f"print('{arch}', round(sum(p.numel() for p in m.parameters()) / 1e6, 1), 'M params')"],
                       env=env, check=True)
    (V / "PREPARE_OK").write_text(time.strftime("%FT%TZ") + "\n")
    vol.commit()
    log(f"PREPARE_OK ({time.monotonic() - t0:.0f}s)")


def lane_env(name, bb, val_fold, seed, smoke):
    src = KIT / "work/env" / ("pubkd_smoke_kd_e4.env" if smoke else "pubkd_kd_e4.env")
    env = dict(l.split("=", 1) for l in src.read_text().splitlines() if l.strip() and not l.startswith("#"))
    env.update({"GL_OUT": f"/work/out/{name}", "PYTHONPATH": "/work/scripts", "D96_SEED": seed})
    if bb != "r50":
        env["V4_BB"] = bb
    if val_fold != "0":
        env["V4_VAL_FOLD"] = val_fold
    return env


@app.function(image=image, volumes={"/vol": vol}, gpu="H100", cpu=16, memory=65536, timeout=10 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def train(lane: str, bb: str, val_fold: str = "0", seed: str = "2026"):
    assert (V / "PREPARE_OK").is_file(), "run prepare first"
    import torch
    import timm
    assert torch.__version__.startswith("2.5.1") and timm.__version__ == "1.0.29", (torch.__version__, timm.__version__)
    W, w = KIT / "work", Path("/work")
    w.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    for name in ("assets", "ms1_crops_slim", "pubkd", "env", "hf"):
        if not (w / name).exists():
            os.symlink(W / name, w / name)
    for link, target in ((w / "out", OUT), (Path("/data/d96"), D96), (Path("/opt/hf"), W / "hf")):
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            os.symlink(target, link)
    shutil.rmtree(w / "scripts", ignore_errors=True); shutil.copytree(W / "scripts", w / "scripts")
    runner = w / "scripts/r50_gl4_runner.py"
    if bb in PATCH:
        subprocess.run(["python", f"/root/patches/{PATCH[bb]}", str(runner)], check=True)
    if val_fold != "0":
        subprocess.run(["python", "/root/patches/apply_valfold_patch.py", str(runner)], check=True)
    if bb in EXTRA_ARCHS:                                    # ImageNet weights into the kit's HF cache if missing
        subprocess.run(["python", "-c", f"import timm; timm.create_model('{EXTRA_ARCHS[bb]}', pretrained=True)"],
                       env={**os.environ, "HF_HOME": str(W / "hf"), "HF_HUB_OFFLINE": "0"}, check=True)
    log(f"lane {lane}: bb {bb}, validation fold {val_fold}, seed {seed}, GPU {torch.cuda.get_device_name(0)}; "
        f"runner sha {sha(runner)[:12]}")

    stop = threading.Event()

    def committer():
        while not stop.wait(300):
            try:
                vol.commit()
            except Exception as e:                           # noqa: BLE001 (keep training; next commit retries)
                log(f"commit failed: {e}")
    threading.Thread(target=committer, daemon=True).start()

    def run(name, smoke):
        o = OUT / name; o.mkdir(exist_ok=True)
        env = {"PATH": os.environ["PATH"], "HOME": "/root", **lane_env(name, bb, val_fold, seed, smoke)}
        (o / "lane_env.json").write_text(json.dumps(env, indent=1))
        with open(o / "train.log", "a") as f:                # also to stdout, so Modal's log view shows training
            pr = subprocess.Popen(["python", "-u", str(runner)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1)
            for line in pr.stdout:
                f.write(line); f.flush(); print(line, end="", flush=True)
            rc = pr.wait()
        log(f"{name} exit {rc}")
        return rc

    try:
        main = f"pubkd_{lane}"
        if (OUT / main / "run_summary.json").exists():
            log(f"{main} already finished"); return
        if not (OUT / main / "recovery.pt").exists():
            smoke = f"pubkd_smoke_{lane}"
            shutil.rmtree(OUT / smoke, ignore_errors=True)
            if run(smoke, True) != 0 or "SMOKE_DONE" not in (OUT / smoke / "train.log").read_text():
                raise RuntimeError(f"smoke failed: {(OUT / smoke / 'train.log').read_text()[-2000:]}")
        if run(main, False) != 0:
            raise RuntimeError(f"lane failed: {(OUT / main / 'train.log').read_text()[-2000:]}")
        for p in (OUT / main).glob("weights_epoch*.pt"):
            p.unlink()
        (OUT / main / "recovery.pt").unlink(missing_ok=True)
        log(f"{main} DONE")
    finally:
        stop.set()
        vol.commit()
