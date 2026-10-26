"""MeluXina preflight for the r50-GL v4 job (first step of the allocation, ~2-5 min on one GPU; exit != 0 stops the job):
offline pretrained weights, label / fold / validation hashes (both phases), cache layout (manifest rows vs vol arrays),
MS1 slim table coverage, and one bf16 forward + backward of each v4 arm on random input on the GPU."""
import os
import subprocess
import sys
import time

import numpy as np
import torch

t0 = time.monotonic()
print(f"host {os.uname().nodename} torch {torch.__version__} cuda {torch.cuda.is_available()} "
      f"devices {torch.cuda.device_count()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
assert torch.cuda.is_available()
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, flush=True)
if True:                                                  # the r50_lv2 phase config (separate process: CFG at import)
    e = {**os.environ, "D96_PHASE": "r50_lv2", "D96_SEED": "2027"}
    code = subprocess.run([sys.executable, "-c", "import coatnet_d96_runner as r; c = r.CFG; "
                           "assert c['version'] == 'd96_r50_lv2_v1_seed2027' and 'eval_epochs' not in c; "
                           "r.load_labels(r.ASSETS / r.LABELS_FILE, 'labels'); print('LV2_CFG_OK')"], env=e).returncode
    assert code == 0, "r50_lv2 config check failed"

import coatnet_d96_runner as r                            # D96_PHASE=r50_gl (v4)
import r50_gl4_runner as m

import timm
w = timm.create_model("resnet50.tv_in1k", pretrained=True, num_classes=0)
print(f"offline weights ok ({sum(p.numel() for p in w.parameters())} params)", flush=True)
labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
assert r.sha(r.ASSETS / "folds.csv") == r.CFG["sha256"]["folds"]
meta = r.read_manifest()
files = {}
for s, v in meta.items():
    files.setdefault(v["file"], []).append(v["row"])
for f, rows in sorted(files.items()):
    a = np.load(f, mmap_mode="r")
    assert a.shape[1:] == (r.CFG["img"], r.CFG["img"]) and a.dtype == np.uint8, (f, a.shape, a.dtype)
    assert max(rows) + r.CFG["maxs"] <= a.shape[0], (f, max(rows), a.shape)
    _ = int(a[max(rows)].sum()) + int(a[0].sum())         # touch first / last used rows
tab = m.v2.crop_table(meta)
assert set(labels) <= set(meta) and set(labels) <= set(tab), "labelled studies missing from cache / crop table"
print(f"cache ok: {len(meta)} studies in {len(files)} arrays; labels {len(labels)}, validation {len(ref)}, "
      f"crop table {len(tab)}", flush=True)
dev = torch.device("cuda")
for arm in ("shared", "adapter", "l4copy"):
    m.ARM = arm
    if arm == "adapter":
        m.G4.update({"lora_rank": 32, "lora_alpha": 32, "local_lr": 5e-4})
    model, opt, clip = m.build(dev, pretrained=True)
    model.train(); b = 2
    xg = torch.randn(b * 4, 3, 384, 384, device=dev).contiguous(memory_format=torch.channels_last)
    xc = torch.randn(b * 10, 3, 384, 384, device=dev).contiguous(memory_format=torch.channels_last)
    fid = torch.as_tensor(m.CF, device=dev).repeat(b, 1); wc = torch.ones(b, 10, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        g, l = model(xg, b, xc, fid, wc)
    (g.float().mean() + l.float().mean()).backward()
    opt.step()
    print(f"{arm} forward/backward ok, mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    del model, opt; torch.cuda.empty_cache()
print(f"PREFLIGHT_OK {time.monotonic() - t0:.0f}s", flush=True)
