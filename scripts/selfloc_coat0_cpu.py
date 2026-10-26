"""BB-CMP coat0 self-location cost on fold 0 (descriptive, for reading its LB score): the submission kernel's own
predict_study (self-located crops, fp32, CPU) on every fold-0 study of the d96 export, shard SHARD of NSHARD.
Writes /out/self_{SHARD}.json {study: [fused, global, local] x 12}. Compare with val_swa_fused.json (MS1 crops, same
weights) with scripts/audit_selfloc_coat0.py. Mounts as scripts/test_coat0_submission.py (/sub, /c, /w) + /out.
"""
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("sub", "/sub/submission.py"); sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
SHARD, NSHARD = int(os.environ["SHARD"]), int(os.environ["NSHARD"])
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8"))); dev = torch.device("cpu")
model = sub.load_model("/w", dev, sub.WEIGHTS["sha256"])
meta = {}
for root in sorted(Path("/c").glob("shard_*")):
    for line in (root / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line); meta[r["study_id"]] = (root / r["part"], r["row_start"], np.array(r["mask"], bool))
ids = sorted(meta)[SHARD::NSHARD]; out, t0 = {}, time.monotonic()
with torch.no_grad():
    for i, s in enumerate(ids, 1):
        f, row, mask = meta[s]; vol = np.ascontiguousarray(np.load(f, mmap_mode="r")[row:row + sub.MAXS])
        p = sub.predict_study(model, vol, mask, dev, amp=False)
        assert np.isfinite(p).all(), s
        out[s] = p.round(7).tolist()
        if i % 25 == 0 or i == len(ids):
            print(f"shard {SHARD}: {i}/{len(ids)} {(time.monotonic() - t0) / i:.1f}s/study", flush=True)
Path(f"/out/self_{SHARD}.json").write_text(json.dumps(out))
print("DONE", len(out), flush=True)
