"""OrthoDiffusion go/no-go probe on Modal: frozen OrthoDiffusion + ImageNet r50 features for all d96 cache shards
(scripts/od_probe_features.py) on one H100, written to the cache volume and downloaded with `modal volume get`.

Run with the personal Modal profile (personal_env() in scripts/run_modal_personal_resnet50_warmstart.py):
  modal run scripts/modal_od_probe.py
then: modal volume get rsna-coatnet-d96-cache experiments/od_probe data/orthodiffusion_probe/modal_output
"""
import os
import subprocess
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parents[1]
OD = LOCAL.parent / "data/orthodiffusion/dataset"
volume = modal.Volume.from_name("rsna-coatnet-d96-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy")
         .add_local_dir(str(OD), "/root/od", ignore=["dataset-metadata.json", "README.md"])
         .add_local_file(str(LOCAL / "data/ssl_probe/backbones_download/tv_in1k.pth"), "/root/tv_in1k.pth")
         .add_local_file(str(LOCAL / "scripts/od_probe_features.py"), "/root/od_probe_features.py"))
app = modal.App("rsna-od-probe")


@app.function(image=image, gpu="H100", cpu=10, memory=49152, timeout=2 * 3600, volumes={"/root/vol": volume})
def features():
    volume.reload()
    env = {**os.environ, "OD_CACHE": "/root/vol/d96", "OD_WEIGHTS": "/root/od", "OD_TV": "/root/tv_in1k.pth",
           "OD_OUT": "/root/vol/experiments/od_probe"}
    subprocess.run(["python", "-I", "/root/od_probe_features.py"], env=env, check=True)
    volume.commit()


@app.local_entrypoint()
def main():
    features.remote()
