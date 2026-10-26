"""Experiment 1 GPU choice: CoAtNet-2 training/eval throughput on Modal T4 / L4 / A100-80GB / H100.

Includes data loading: a synthetic 96-slice uint8 corpus (384 px) is memory-mapped from local disk and
16 random 3-slice windows per study are cut in DataLoader workers, as the real trainer will do.
Random init; no competition data. Prints one JSON line per configuration and returns all results.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "timm==1.0.29", "numpy")
app = modal.App("rsna-coatnet-gpu-bench")

STUDIES, MAXS, IMG, TRAIN_W, EVAL_W = 160, 96, 384, 16, 94


def run(gpu_name: str) -> list[dict]:
    import numpy as np
    import timm
    import torch
    import torch.nn as nn

    torch.backends.cudnn.benchmark = True
    corpus = Path("/tmp/corpus.npy")
    rng = np.random.default_rng(0)
    arr = np.lib.format.open_memmap(corpus, mode="w+", dtype=np.uint8, shape=(STUDIES * MAXS, IMG, IMG))
    for i in range(0, STUDIES * MAXS, 960):
        arr[i:i + 960] = rng.integers(0, 255, size=(min(960, STUDIES * MAXS - i), IMG, IMG), dtype=np.uint8)
    arr.flush(); del arr

    class Windows(torch.utils.data.Dataset):
        def __len__(self):
            return 10_000

        def __getitem__(self, i):
            vol = np.load(corpus, mmap_mode="r")
            s = np.random.randint(STUDIES) * MAXS
            c = np.sort(np.random.choice(np.arange(1, MAXS - 1), TRAIN_W, replace=False))
            return torch.from_numpy(np.stack([vol[s + c - 1], vol[s + c], vol[s + c + 1]], 1))  # [K,3,H,W] uint8

    bf16 = torch.cuda.is_bf16_supported() and "T4" not in gpu_name
    dtype = torch.bfloat16 if bf16 else torch.float16
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    results = []
    for res, studies_per_step in ((384, 1), (384, 2), (320, 1), (320, 2)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        out = {"gpu": gpu_name, "res": res, "studies_per_step": studies_per_step,
               "dtype": "bf16" if bf16 else "fp16", "device": torch.cuda.get_device_name(0)}
        try:
            enc = timm.create_model("coatnet_rmlp_2_rw_384", pretrained=False, num_classes=0, img_size=res)
            enc = enc.cuda().to(memory_format=torch.channels_last)
            head = nn.Linear(enc.num_features, 12).cuda()
            opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-4)
            scaler = torch.amp.GradScaler(enabled=not bf16)
            loader = torch.utils.data.DataLoader(Windows(), batch_size=studies_per_step, num_workers=4,
                                                 pin_memory=True, persistent_workers=True)
            it = iter(loader)
            enc.train()
            steps, warm = 12, 3
            for step in range(warm + steps):
                if step == warm:
                    torch.cuda.synchronize(); t0 = time.time()
                x = next(it).cuda(non_blocking=True).flatten(0, 1).float().div_(255)
                lo = (IMG - res) // 2
                x = ((x[..., lo:lo + res, lo:lo + res] - mean) / std).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=dtype):
                    f = enc(x).view(studies_per_step, TRAIN_W, -1).mean(1)
                    loss = nn.functional.binary_cross_entropy_with_logits(head(f).float(),
                                                                          torch.rand(studies_per_step, 12, device="cuda"))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / steps
            out["train_s_per_study"] = round(dt / studies_per_step, 4)
            out["train_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            enc.eval()
            xe = torch.randn(EVAL_W, 3, res, res, device="cuda").to(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                for step in range(2 + 4):
                    if step == 2:
                        torch.cuda.synchronize(); t0 = time.time()
                    torch.cat([enc(c) for c in xe.split(32)])
            torch.cuda.synchronize()
            out["eval_s_per_study_94w"] = round((time.time() - t0) / 4, 4)
            del loader, it
        except torch.OutOfMemoryError:
            out["error"] = "OOM"
        except Exception as error:
            out["error"] = f"{type(error).__name__}: {error}"[:300]
        print(json.dumps(out), flush=True)
        results.append(out)
    return results


@app.function(image=image, gpu="T4", cpu=4, memory=16384, timeout=1800)
def bench_t4():
    return run("T4")


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=1800)
def bench_l4():
    return run("L4")


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=32768, timeout=1800)
def bench_a100():
    return run("A100-80GB")


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=1800)
def bench_h100():
    return run("H100")


@app.local_entrypoint()
def main():
    calls = [f.spawn() for f in (bench_t4, bench_l4, bench_a100, bench_h100)]
    results = []
    for call in calls:
        try:
            results += call.get()
        except Exception as error:
            results.append({"error": f"{type(error).__name__}: {error}"[:300]})
    out = Path(__file__).resolve().parents[1] / "data/coatnet_bench/modal_gpu_bench.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
