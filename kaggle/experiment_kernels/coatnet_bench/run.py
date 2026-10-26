"""Throughput benchmark for experiment 1 (their CoAtNet recipe on our data), Kaggle T4.

Measures, per backbone and input size, on a single T4:
  train: one study = 16 windows of 3 x res x res, forward + backward + AdamW step, fp16 autocast,
         with and without gradient checkpointing -> windows/s and peak memory;
  eval:  one study = 94 windows, forward only in chunks of 16 -> studies/s.
Then repeats the main candidate on both GPUs at once to check that two-GPU scaling is real.
Random init (no weights needed for timing). Writes /kaggle/working/bench.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

TRAIN_WINDOWS = 16
EVAL_WINDOWS = 94
WARMUP, STEPS = 3, 12
CANDIDATES = [
    ("coatnet_rmlp_2_rw_384", (256, 320, 384)),   # the public 0.949 backbone
    ("coatnet_rmlp_1_rw2_224", (256, 320)),       # smaller CoAtNet fallback
    ("convnext_small", (256, 320)),               # conv control
]


class Head(nn.Module):
    """Per-finding attention pooling over the windows of one study (the public recipe's head)."""

    def __init__(self, dim: int, n: int = 12):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Linear(256, n))
        self.w = nn.Parameter(torch.zeros(n, dim))
        self.b = nn.Parameter(torch.zeros(n))

    def forward(self, f):                       # f [B, K, D]
        h = self.norm(f)
        a = self.att(h).softmax(1)              # [B, K, n]
        return (torch.einsum("bkn,bkd->bnd", a, h) * self.w).sum(-1) + self.b


def build(arch: str, res: int, ckpt: bool):
    import timm
    kw = dict(pretrained=False, num_classes=0)
    if arch.startswith("coatnet"):
        kw["img_size"] = res
    enc = timm.create_model(arch, **kw)
    enc.set_grad_checkpointing(ckpt)
    return enc, Head(enc.num_features)


def bench_one(arch: str, res: int, ckpt: bool, device: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    out = {"arch": arch, "res": res, "grad_ckpt": ckpt}
    try:
        enc, head = build(arch, res, ckpt)
        enc, head = enc.to(device).to(memory_format=torch.channels_last), head.to(device)
        out["params_m"] = round(sum(p.numel() for p in enc.parameters()) / 1e6, 1)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-4)
        scaler = torch.amp.GradScaler()
        x = torch.randn(TRAIN_WINDOWS, 3, res, res, device=device).to(memory_format=torch.channels_last)
        y = torch.rand(1, 12, device=device)
        enc.train(); head.train()
        for step in range(WARMUP + STEPS):
            if step == WARMUP:
                torch.cuda.synchronize(device); t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = head(enc(x).unsqueeze(0))
                loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        torch.cuda.synchronize(device)
        dt = (time.time() - t0) / STEPS
        out["train_s_per_study"] = round(dt, 4)
        out["train_windows_per_s"] = round(TRAIN_WINDOWS / dt, 1)
        out["train_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        del opt, scaler
        enc.eval(); head.eval()
        xe = torch.randn(EVAL_WINDOWS, 3, res, res, device=device).to(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for step in range(2 + 5):
                if step == 2:
                    torch.cuda.synchronize(device); t0 = time.time()
                f = torch.cat([enc(c) for c in xe.split(16)])
                head(f.unsqueeze(0))
        torch.cuda.synchronize(device)
        out["eval_s_per_study_94w"] = round((time.time() - t0) / 5, 4)
    except torch.OutOfMemoryError:
        out["error"] = "OOM"
    except Exception as error:                  # an arch missing from this timm version, etc.
        out["error"] = f"{type(error).__name__}: {error}"[:300]
    return out


def worker(device: str, path: str) -> None:
    """Child process for the two-GPU check: the main candidate on one GPU."""
    Path(path).write_text(json.dumps(bench_one("coatnet_rmlp_2_rw_384", 320, False, device)))


def main() -> None:
    if len(sys.argv) == 3:
        worker(sys.argv[1], sys.argv[2]); return
    import timm
    torch.backends.cudnn.benchmark = True
    env = {"torch": torch.__version__, "timm": timm.__version__, "gpus": torch.cuda.device_count(),
           "gpu_name": torch.cuda.get_device_name(0)}
    print(env, flush=True)
    results = []
    for arch, sizes in CANDIDATES:
        for res in sizes:
            for ckpt in (False, True):
                r = bench_one(arch, res, ckpt, "cuda:0")
                print(json.dumps(r), flush=True)
                results.append(r)
    concurrent = []
    if torch.cuda.device_count() >= 2:
        torch.cuda.empty_cache()
        paths = [f"/kaggle/working/_dual_{i}.json" for i in range(2)]
        procs = [subprocess.Popen([sys.executable, __file__, f"cuda:{i}", paths[i]]) for i in range(2)]
        for p in procs:
            p.wait()
        concurrent = [json.loads(Path(p).read_text()) for p in paths if Path(p).is_file()]
        print("concurrent two-GPU:", json.dumps(concurrent), flush=True)
    Path("/kaggle/working/bench.json").write_text(json.dumps(
        {"env": env, "train_windows": TRAIN_WINDOWS, "eval_windows": EVAL_WINDOWS,
         "single_gpu": results, "concurrent_two_gpu": concurrent}, indent=2))


if __name__ == "__main__":
    main()
