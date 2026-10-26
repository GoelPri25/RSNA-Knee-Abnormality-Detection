"""Submission A' (kaggle/submission_kernels/v4_l4copy_loc_submission) vs the L4C-LOC scoring arrays: on the first N fold-0
studies of the d96 export, predict_study(loc=...) must reproduce sigmoid of the patched wp fusion in xai_arrays_s2026.npz
(MeluXina, A100 fp16) up to fp16-vs-fp32 noise. CPU, localhost/rsna-mlx-train:v1, --network=none.
Mounts: /sub (submission dir), /c (d96 fold-0 shard root), /w (v4 weights), /l (locator.pt), /x (xai_arrays_s2026.npz).
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("sub", "/sub/submission.py"); sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
N = int(os.environ.get("N", "8")); dev = torch.device("cpu")
model = sub.load_model("/w", dev, sub.WEIGHTS["sha256"]); loc = sub.load_locator("/l", dev, sub.LOCATOR["sha256"])
z = np.load("/x"); zi = {str(s): i for i, s in enumerate(z["ids"])}
GO = [sub.TARGETS.index(t) for t in sub.GLOBAL_ONLY]
meta = {}
for root in sorted(Path("/c").glob("shard_*")):
    for line in (root / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line); meta[r["study_id"]] = (root / r["part"], r["row_start"], np.array(r["mask"], bool))
ids = [s for s in sorted(meta) if s in zi][:N]
worst = 0.0; flips = 0
with torch.no_grad():
    for s in ids:
        f, row, mask = meta[s]; vol = np.ascontiguousarray(np.load(f, mmap_mode="r")[row:row + sub.MAXS])
        p = sub.predict_study(model, vol, mask, dev, amp=False, loc=loc)
        i = zi[s]; gl = z["glob"][i]; f_ = gl.copy(); f_[sub.CF] = 0.5 * (gl[sub.CF] + z["loc_wp_100"][i][sub.CF]); f_[GO] = gl[GO]
        ref = 1 / (1 + np.exp(-f_)); d = np.abs(p[0] - ref).max(); worst = max(worst, d); flips += d > 0.01
        print(s[-12:], f"max|d| {d:.2e}", flush=True)
print(f"worst {worst:.2e} studies>0.01 {flips}/{len(ids)}")
assert worst < 0.02 and flips <= 1, "submission locator path differs from the scored wp"
print("SUBMISSION_EQUIV_OK")
