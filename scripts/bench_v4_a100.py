"""r50-GL v4 capacity bench on MeluXina's GPU type (A100-40GB, 16 CPUs): is a run data-bound or GPU-bound, and how
much GPU memory is left (larger batches / two runs per GPU)?
1. Loader: training DataLoader alone (real cache, 2 studies per batch), 8 vs 15 workers -> s/study the CPUs can supply.
2. GPU: per arm, training step (prepare_gl + forward + backward + AdamW) on a fixed real batch -> s/study, peak memory.
3. Memory limit (shared + adapter): studies per step 2, 4, 8, 12, 16 with real shapes -> peak memory or OOM.
4. Eval: one fold-0 study (94 windows + 40 crops) -> peak memory.
Env as the v4 smoke (D96_PHASE=r50_gl, GL_CROPS, GL_OUT, D96_CACHE)."""
import json
import os
import time

import numpy as np
import torch

import coatnet_d96_runner as r
import r50_gl4_runner as m

CFG = r.CFG
dev = torch.device("cuda")
res = {"gpu": torch.cuda.get_device_name(), "cpus": os.cpu_count(), "sched_cpus": len(os.sched_getaffinity(0))}
labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
meta = r.read_manifest(); tab = m.v2.crop_table(meta)
ids = sorted(s for s in labels if s in tab)
rng = np.random.default_rng(0); train = [ids[i] for i in rng.permutation(len(ids))[:400]]
cmask = torch.from_numpy(m.CMASK.astype(np.float32)).to(dev)

# 1. loader throughput
for nw in (8, 15):
    ds = m.v2.GLStudies(train, meta, labels, CFG["train_windows"], True, tab)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=nw, pin_memory=True)
    it = iter(dl); n = 0
    for _ in range(2 * nw):                                   # warm-up: every worker has produced
        next(it)
    t0 = time.monotonic()
    for _ in range(60):
        next(it); n += 2
    res[f"loader_s_per_study_w{nw}"] = round((time.monotonic() - t0) / n, 4)
    print(f"LOADER workers {nw}: {res[f'loader_s_per_study_w{nw}']} s/study", flush=True)
    del it, dl

batch = next(iter(torch.utils.data.DataLoader(m.v2.GLStudies(train[:2], meta, labels, CFG["train_windows"], True, tab),
                                              batch_size=2)))


def make(b):
    """a real-shaped batch of b studies (tiles the real 2-study batch)"""
    rep = (b + 1) // 2
    return [t.repeat(rep, *([1] * (t.dim() - 1)))[:b] if torch.is_tensor(t) else t for t in batch]


def step_time(arm, b, steps):
    m.ARM = arm
    if arm == "adapter":
        m.G4.update({"lora_rank": 32, "lora_alpha": 32, "local_lr": 5e-4})
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model, opt, clip = m.build(dev, pretrained=False); model.train()
    bt = make(b)
    try:
        for i in range(steps + 3):
            if i == 3:
                torch.cuda.synchronize(); t0 = time.monotonic()
            parts = m.train_step_loss(model, bt, dev, cmask)
            (parts[0] + 0.5 * parts[1]).backward()
            for c in clip:
                torch.nn.utils.clip_grad_norm_(c, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        out = {"s_per_study": round((time.monotonic() - t0) / (steps * b), 4),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    except torch.cuda.OutOfMemoryError:
        out = {"oom": True}
    del model, opt; torch.cuda.empty_cache()
    return out


# 2. GPU-only training speed per arm (2 studies per step, as the recipe)
for arm in ("shared", "adapter", "l4copy"):
    res[f"gpu_{arm}"] = step_time(arm, 2, 20)
    print(f"GPU {arm}: {res[f'gpu_{arm}']}", flush=True)
# 3. memory limit
for arm in ("shared", "adapter"):
    for b in (4, 8, 12, 16):
        o = step_time(arm, b, 4); res[f"mem_{arm}_b{b}"] = o
        print(f"MEM {arm} studies/step {b}: {o}", flush=True)
        if o.get("oom"):
            break
# 4. eval memory, one study at 94 windows + 40 crops
m.ARM = "shared"; torch.cuda.reset_peak_memory_stats()
model = m.build(dev, pretrained=False)[0]
t0 = time.monotonic(); m.predict(model, ids[:4], meta, tab, dev)
res["eval_s_per_study"] = round((time.monotonic() - t0) / 4, 3)
res["eval_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
res["total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
print("BENCH " + json.dumps(res), flush=True)
