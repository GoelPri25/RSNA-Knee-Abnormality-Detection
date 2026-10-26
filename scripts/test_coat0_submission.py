"""BB-CMP coat0 submission (kaggle/submission_kernels/v4_coat0_submission) vs the MeluXina fold-0 SWA predictions: on the
first N fold-0 studies of the d96 export, predict_study with MS1's crop table (table hook) must reproduce
val_swa_fused.json (A100 fp16) up to fp16-vs-fp32 noise; the self-located pass is printed next to it (descriptive).
CPU, localhost/rsna-mlx-train:v1, --network=none. Mounts: /sub (submission dir), /c (d96 fold-0 shard root),
/w (weights_swa21_25.pt), /v (val_swa_fused.json), /cp (crop_probe cp_*.npz).
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("sub", "/sub/submission.py"); sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
N = int(os.environ.get("N", "6")); dev = torch.device("cpu")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
model = sub.load_model("/w", dev, sub.WEIGHTS["sha256"])
ref = {p["study_id"]: np.array(p["probabilities"]) for p in json.loads(Path("/v").read_text())["predictions"]}
meta = {}
for root in sorted(Path("/c").glob("shard_*")):
    for line in (root / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line); meta[r["study_id"]] = (root / r["part"], r["row_start"], np.array(r["mask"], bool))
cp = {}
for f in sorted(Path("/cp").glob("cp_*.npz")):
    z = np.load(f)
    for s, cen, at in zip(z["ids"], z["centres"], z["a_top"]):
        cp[str(s)] = (cen, at.astype(np.float32))
ids = [s for s in sorted(meta) if s in ref and s in cp][:N]
assert len(ids) == N, len(ids)
worst, flips = 0.0, 0
with torch.no_grad():
    for s in ids:
        f, row, mask = meta[s]; vol = np.ascontiguousarray(np.load(f, mmap_mode="r")[row:row + sub.MAXS])
        cen, at = cp[s]; c = sub.uniform_centres(mask)
        tab = tuple(a[sub.CF] for a in (c[cen[..., 0].astype(int)], cen[..., 1].astype(int), cen[..., 2].astype(int), at))
        p = sub.predict_study(model, vol, mask, dev, amp=False, table=tab)
        q = sub.predict_study(model, vol, mask, dev, amp=False)
        d = np.abs(p[0] - ref[s]).max(); worst = max(worst, d); flips += d > 0.01
        print(s[-12:], f"MS1-table max|d| {d:.2e} | selfloc - MS1 fused max|d| {np.abs(q[0] - p[0]).max():.3f}", flush=True)
print(f"worst {worst:.2e} studies>0.01 {flips}/{len(ids)}")
assert worst < 0.02 and flips <= 1, "submission model differs from the trained coat0"
print("SUBMISSION_EQUIV_OK")
