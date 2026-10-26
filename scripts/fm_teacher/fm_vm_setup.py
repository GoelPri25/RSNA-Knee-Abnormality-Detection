"""FB-F teacher VM setup (Colab CLI VM, run as root with the system python3): the public R6 + 2.5D-reader teacher's Kaggle
environment, rebuilt so scripts/fm_teacher/teacher_fm.py runs unchanged.

  venv   /content/fmv: Python 3.12 (the reader's pylibjpeg wheels are cp311 / cp312; Colab's own python is 3.13),
         torch 2.11.0+cu128 (= the R6 receipt's Kaggle torch) + torchvision from the same resolve, numpy / pandas / scipy /
         opencv-headless / safetensors / huggingface_hub / pyyaml, pylibjpeg + openjpeg + libjpeg (fastMRI exams are
         JPEG 2000 lossless). timm 1.0.25 and pydicom 3.0.1 are installed by the teacher from the code dataset's wheels.
  inputs /content/kin/<slug> (Colab mounts its own read-only /kaggle/input) for the three public datasets the Kaggle teacher kernel mounted (kernel-metadata.json of
         kaggle/submission_kernels/pubkd_r6_teacher/s0), downloaded here with the Kaggle token, which is deleted afterwards.
         The competition mount (rsna-knee-abnormality-detection) is a symlink the driver points at each chunk of exams.
Usage (VM): python3 /content/fm/fm_vm_setup.py        (token at /content/fm/kaggle.env, mode 600)
"""
import os
import subprocess
import sys
from pathlib import Path

C = Path("/content/fm")
VENV = Path("/content/fmv")
INP = Path("/content/kin")
DATASETS = ["rostigor/rsna-knee-r6-fe055406", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
            "goodpjw2008/rsna-knee-2-5d-convnext-reader"]
CHECK = {"rsna-knee-r6-fe055406": "manifest.json", "rsna-orthofoundation-fold0-epoch7-20261004": "receipt.json",
         "rsna-knee-2-5d-convnext-reader": "cnxt_v0_fold0.pt"}


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


# 1. venv
py = VENV / "bin/python"
if not (VENV / ".ok").exists():
    sh("pip install -q uv")
    sh(f"uv venv -q --seed --python 3.12 {VENV}")
    sh(f"uv pip install -q --python {py} torch==2.11.0 torchvision 'numpy<2.4' 'pandas>=2.2,<3' scipy 'opencv-python-headless<5' safetensors "
       f"huggingface_hub pyyaml pillow pylibjpeg pylibjpeg-openjpeg pylibjpeg-libjpeg kaggle==2.2.2 "
       f"--index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple "
       f"--index-strategy unsafe-best-match")
    (VENV / ".ok").write_text("ok\n")
sh(f"{py} -c \"import sys, torch, torchvision, numpy, pandas, cv2; print(sys.version.split()[0], torch.__version__, "
   f"torchvision.__version__, numpy.__version__, pandas.__version__, cv2.__version__, torch.cuda.get_device_name(0)); "
   f"assert torch.__version__.startswith('2.11.0') and torch.version.cuda == '12.8', 'torch pin lost'\"")

# 2. public datasets -> /content/kin/<slug>; token removed afterwards
tok = C / "kaggle.env"
todo = [d for d in DATASETS if not (INP / d.split("/")[1] / CHECK[d.split("/")[1]]).exists()]
if todo:
    assert tok.is_file(), "upload the Kaggle token first"
    env = {**os.environ, "KAGGLE_API_TOKEN": tok.read_text().strip().split("=", 1)[1]}
    for d in todo:
        dest = INP / d.split("/")[1]
        for _ in range(3):
            if subprocess.run([str(VENV / "bin/kaggle"), "datasets", "download", "-d", d, "-p", str(dest), "--unzip", "-q"],
                              env=env, stdout=subprocess.DEVNULL).returncode == 0 and (dest / CHECK[dest.name]).exists():
                break
        else:
            raise RuntimeError(f"{d}: download failed 3 times")
        print("downloaded", d, flush=True)
tok.unlink(missing_ok=True)
assert not tok.exists()
for slug, f in CHECK.items():
    assert (INP / slug / f).exists(), slug
Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
# the driver reads series headers itself: the code dataset's pinned pydicom wheel (the teacher reinstalls it, no-op)
whl = sorted((INP / "rsna-orthofoundation-fold0-epoch7-20261004").rglob("pydicom-3.0.1-py3-none-any.whl"))
assert whl, "pydicom wheel missing"
sh([str(py), "-m", "pip", "install", "-q", "--no-index", "--no-deps", str(whl[0])])
sh("du -sh /content/kin/*; df -h /content | tail -1; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
print("SETUP_OK", flush=True)
