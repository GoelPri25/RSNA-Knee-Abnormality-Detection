"""PUB-KD / KD-CV lanes on Modal: the MeluXina job scripts/meluxina/pubkd.sbatch, one lane per H100 container.

MeluXina's GPU allowance is at 0 (2026-10-21, AssocGrpGRESMinutes), so the lanes run here instead, unchanged: the same
env files (scripts/meluxina/env/pubkd_<lane>.env, pubkd_smoke_<lane>.env), runner (scripts/r50_gl4_runner.py), image
versions (torch 2.5.1, timm 1.0.29, numpy 2.4.6 = rsna-mlx-train:v1) and inputs. The MeluXina paths the env files use are
recreated as links: /data/d96 = the volume's d96 cache (manifests = MeluXina's), /opt/hf = MeluXina's HF cache
(resnet50.tv_in1k, uploaded to /hf_mlx), /work/{scripts,assets,ms1_crops_slim,pubkd,env} = this image, /work/out = the
volume's experiments/pubkd_modal. Pinned inputs are checked as in pubkd.sbatch before anything trains.
Per lane: the 32-step smoke (GL_SMOKE=1) first, the lane only if SMOKE_DONE; the volume is committed every 5 minutes
(recovery.pt each epoch, so a restart resumes); after exit 0 with SWA weights, per-epoch weights and recovery.pt are
deleted, as on MeluXina.
Usage: source ~/.config/rsna/modal-personal.env && modal run --detach scripts/modal_pubkd.py --lanes cv1
"""
import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parents[1]
D = LOCAL / "data/coatnet_d96_fold0"
TARGETS = ("pubkd_targets_fullorder_rapvalues_mix73.json", "pubkd_targets_raptor_pure.json")
volume = modal.Volume.from_name("rsna-coatnet-d96-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "timm==1.0.29", "numpy==2.4.6")
         .add_local_dir(str(D / "assets"), "/root/work/assets")
         .add_local_dir(str(D / "ms1_crops_slim"), "/root/work/ms1_crops_slim")
         .add_local_dir(str(LOCAL / "scripts/meluxina/env"), "/root/work/env"))
for f in TARGETS:
    image = image.add_local_file(str(D / "pubkd" / f), f"/root/work/pubkd/{f}")
# RUN-C (2026-10-22): label v3a, the Hydra member targets and the fastMRI holdout at /work/fbf; the fastMRI d96 cache
# (scripts/modal_fmcache.py) and the R7 soup / Raptor-384 checkpoints are read from the volume (/root/vol/fm, /root/vol/pubckpt).
for src, f in (("data/label_v3a/v3a_targets_mix73.json", "v3a_targets_mix73.json"),
               ("data/label_v3a/hydra_members.json", "hydra_members.json"),
               ("data/surpass_f/meluxina_copy/fastmri_targets/fm_holdout.json", "fm_holdout.json")):
    image = image.add_local_file(str(LOCAL / src), f"/root/work/fbf/{f}")
for f in ("r50_gl4_runner.py", "r50_gl_runner.py", "coatnet_d96_runner.py", "d96_positions.py"):
    image = image.add_local_file(str(LOCAL / "scripts" / f), f"/root/work/scripts/{f}")
app = modal.App("rsna-pubkd", image=image)
VOL = Path("/root/vol")
OUT = VOL / "experiments/pubkd_modal"
PINNED = {"assets/rule_states.csv": "ce4473c71f3ea100db021805750c6bb9038bc7b0e56a143c81516edf2d29e77b",   # = pubkd.sbatch
          "assets/soft_labels_final_per_target_latD2.json": "dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d",
          "assets/folds.csv": "2dd1044adcc54ca4d9bcb9c9fd3b910a6784bd2b99e792bbeaf84a42d417f06a",
          "pubkd/" + TARGETS[0]: "f07f202b846b99272b6fe34bcb1d30d4d71e18a1870bcc7d3ce61559737ccd34",
          "pubkd/" + TARGETS[1]: "6ed9225434bf8cf77cc2a3ded2c47a8443fc0967a1a215c081da482e1f3bfa49"}
D96_MANIFESTS = ("a8945d545d19a4cb", "2a830ad175c78d08", "875e9f04bc17c72f", "a562a6369e632044")   # MeluXina ~/rsna/d96
R50_SAFETENSORS = "5d061a3c593d795bfe682d9b152bafbcf550579873492def3515b46db1189888"           # MeluXina ~/rsna/hf


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def setup():
    """MeluXina's paths as links, then the pinned-input checks (any mismatch stops before training)."""
    OUT.mkdir(parents=True, exist_ok=True)
    for link, target in (("/work", "/root/work"), ("/data/d96", str(VOL / "d96")), ("/opt/hf", str(VOL / "hf_mlx"))):
        Path(link).parent.mkdir(parents=True, exist_ok=True)
        if not Path(link).exists():
            os.symlink(target, link)
    if not Path("/work/out").exists():
        os.symlink(str(OUT), "/work/out")
    for rel, want in PINNED.items():
        assert sha(f"/work/{rel}") == want, f"pinned input differs: {rel}"
    for k, want in enumerate(D96_MANIFESTS):
        assert sha(f"/data/d96/shard_{k}/manifest.jsonl")[:16] == want, f"d96 shard {k} manifest differs from MeluXina's"
    st = list(Path("/opt/hf/hub/models--timm--resnet50.tv_in1k/snapshots").glob("*/model.safetensors"))
    assert len(st) == 1 and sha(st[0]) == R50_SAFETENSORS, "resnet50.tv_in1k weights differ from MeluXina's"
    if (VOL / "fm").is_dir():                                # RUN-C: the fastMRI cache as r7's (8 shards, manifests)
        for k in range(8):
            print(f"fm shard {k} manifest {sha(VOL / f'fm/shard_{k}/manifest.jsonl')[:16]}", flush=True)
    for f in ("r50_gl4_runner.py", "r50_gl_runner.py", "coatnet_d96_runner.py"):
        print(f"{sha('/work/scripts/' + f)}  {f}", flush=True)


EXTRA_ENV = {}   # per-container overrides (KF-CNXS: a container-local HF cache with the ConvNeXt-S 22k weights)


def fetch_cnxs():
    """KF-CNXS (2026-10-22): convnext_small.fb_in22k_ft_in1k ImageNet weights into a container-local HF cache (the
    volume's MeluXina cache has resnet50 only; 5 containers writing the shared volume at once would race)."""
    code = ("import timm; m = timm.create_model('convnext_small.fb_in22k_ft_in1k', pretrained=True, num_classes=0); "
            "print('convnext_small.fb_in22k_ft_in1k', sum(p.numel() for p in m.parameters()), 'params', flush=True)")
    rc = subprocess.run(["python", "-c", code], env={**os.environ, "HF_HOME": "/root/hfl", "HF_HUB_OFFLINE": "0",
                                                     "CUDA_VISIBLE_DEVICES": ""}).returncode
    assert rc == 0, "convnext_small weights download failed"
    EXTRA_ENV["HF_HOME"] = "/root/hfl"


def env_of(name):
    """The lane's env file as pubkd.sbatch passes it to apptainer (--cleanenv: only PATH-like basics + the file)."""
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES",
                                      "NVIDIA_DRIVER_CAPABILITIES") if k in os.environ}
    for line in Path(f"/work/env/{name}.env").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            k, v = line.split("=", 1); env[k] = v
    env.update(EXTRA_ENV)
    return env


def run(name):
    """One runner process with its log at /work/out/<name>/train.log (appended, as on MeluXina); returns the exit code."""
    out = Path(f"/work/out/{name}"); out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with (out / "train.log").open("a", buffering=1) as log:
        proc = subprocess.Popen(["python", "-u", "/work/scripts/r50_gl4_runner.py"], env=env_of(name), cwd="/work/scripts",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line); print(f"[{name}] {line}", end="", flush=True)
        rc = proc.wait()
    print(f"{name} exit {rc} after {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    if rc == 0 and list(out.glob("weights_swa*.pt")):
        for p in [*out.glob("weights_epoch*.pt"), out / "recovery.pt"]:
            p.unlink(missing_ok=True)
    volume.commit()
    return rc


@app.function(gpu="H100", cpu=12, memory=65536, timeout=10 * 3600, volumes={str(VOL): volume})
def lane(name: str):
    volume.reload(); setup()
    assert not Path(f"/work/out/pubkd_{name}/run_summary.json").exists(), f"pubkd_{name} already finished"
    if "V4_BB=cnxs" in Path(f"/work/env/pubkd_{name}.env").read_text():
        fetch_cnxs()
    stop = threading.Event()

    def commit_loop():
        while not stop.wait(300):
            try:
                volume.commit()
            except Exception as e:                       # a failed periodic commit must not kill the run
                print(f"volume commit failed: {e}", flush=True)
    threading.Thread(target=commit_loop, daemon=True).start()
    smoke = f"pubkd_smoke_{name}"
    if not Path(f"/work/out/pubkd_{name}/recovery.pt").exists():   # a resumed lane skips its smoke
        subprocess.run(["rm", "-rf", f"/work/out/{smoke}"], check=True)
        if run(smoke) or "SMOKE_DONE" not in Path(f"/work/out/{smoke}/train.log").read_text():
            stop.set(); raise RuntimeError(f"smoke failed: {name}")
    rc = run(f"pubkd_{name}")
    stop.set(); volume.commit()
    if rc:
        raise RuntimeError(f"lane {name} exited {rc}")


@app.function(cpu=2, memory=8192, timeout=900, volumes={str(VOL): volume})
def check(name: str = "cv1"):
    """CPU only: links + pinned inputs, then the lane's config as the runner builds it (import only, nothing trains)."""
    volume.reload(); setup()
    code = ("import r50_gl4_runner as L; c = L.CFG; print('CONFIG', c['version'], c['train_windows'], c['train_res'], "
            "c.get('train_crop'), c['aug'], c['epochs'], c['swa_epochs'][0], c['swa_epochs'][-1]); import timm; "
            "m = timm.create_model('resnet50.tv_in1k', pretrained=True); print('PRETRAINED OK', sum(p.numel() for p in m.parameters()))")
    rc = subprocess.run(["python", "-c", code], env={**env_of(f"pubkd_{name}"), "CUDA_VISIBLE_DEVICES": ""},
                        cwd="/work/scripts").returncode
    assert rc == 0, rc


@app.local_entrypoint()
def main(lanes: str = "cv1", check_only: bool = False):
    if check_only:
        for n in lanes.split(","):
            check.remote(n)
        return
    for n in lanes.split(","):
        assert (LOCAL / f"scripts/meluxina/env/pubkd_{n}.env").is_file() and \
            (LOCAL / f"scripts/meluxina/env/pubkd_smoke_{n}.env").is_file(), n
    calls = [lane.spawn(n) for n in lanes.split(",")]
    for n, c in zip(lanes.split(","), calls):
        print(f"SPAWNED {n} {c.object_id}", flush=True)
