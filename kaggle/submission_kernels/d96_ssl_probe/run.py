"""Exp 5a diagnostic: frozen-feature probe of the starting / self-supervised ResNet-50 backbones (T4, inference only).

For every study of the d96 training cache (attached shard kernel outputs; gold excluded) and each backbone in the
probe dataset (tv_in1k = 1e's start; Barlow Twins ImageNet = 5a's start; 5a SSL epochs 2 / 5 / 10), the frozen r50
features (2048, after global pooling) of 32 evenly spaced 3-slice windows (the runner's eval centres, centre 320 crop,
ImageNet normalisation) are mean-pooled per study: over all windows and per plane (sagittal slots 0-1, coronal 2-3,
axial 4). Output feats_<name>.npy float16 [S, 4, 2048] (all, sagittal, coronal, axial; zeros if a plane is absent),
study_ids.json, backbones.json. No labels here; the probe (logistic regression, training folds -> fold 0) is fitted
locally by scripts/audit_ssl_probe.py.
"""
from __future__ import annotations

import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

SLOT_OF = np.repeat(np.arange(5), [26, 22, 18, 12, 18])
PLANE = np.array([0, 0, 1, 1, 2])[SLOT_OF]                                  # 0 sagittal, 1 coronal, 2 axial
MAXS, K, RES = 96, 32, 320
NAMES = ["tv_in1k", "bt_imagenet", "ssl_e02", "ssl_e05", "ssl_e10"]
OUT = Path("/kaggle/working")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, pattern
    return hits


def manifest():
    studies = {}
    for m in find("/kaggle/input/**/manifest.jsonl"):
        root = Path(m).parent
        if not (root / "cache_summary.json").is_file():
            continue
        for line in Path(m).read_text().splitlines():
            r = json.loads(line)
            if r["gold"]:
                continue
            assert r["study_id"] not in studies
            studies[r["study_id"]] = {"file": str(root / r["part"]), "row": r["row_start"], "mask": np.array(r["mask"], bool)}
    return studies


def centres(mask):
    valid = np.where(mask)[0]
    lo, hi = (int(valid.min()), int(valid.max())) if len(valid) >= 3 else (0, 2)
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([1])
    return cs[np.linspace(0, len(cs) - 1, K).round().astype(int)]


class Windows(torch.utils.data.Dataset):
    def __init__(self, ids, meta):
        self.ids, self.meta, self._mm = ids, meta, {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        m = self.meta[self.ids[i]]
        if m["file"] not in self._mm:
            self._mm[m["file"]] = np.load(m["file"], mmap_mode="r")
        c = np.clip(centres(m["mask"]), 1, MAXS - 2)
        rows = np.stack([c - 1, c, c + 1], 1)
        uniq, inv = np.unique(rows, return_inverse=True)
        block = np.asarray(self._mm[m["file"]][m["row"] + uniq])
        lo = (block.shape[-1] - RES) // 2
        win = block[inv.reshape(rows.shape)][..., lo:lo + RES, lo:lo + RES]   # [K, 3, 320, 320]
        return torch.from_numpy(np.ascontiguousarray(win)), torch.from_numpy(PLANE[c])


def backbone(path):
    m = torchvision.models.resnet50()
    m.fc = nn.Identity()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    m.load_state_dict(sd, strict=True)
    return m.eval().half().cuda().to(memory_format=torch.channels_last)


def main():
    t0 = time.monotonic()
    meta = manifest(); ids = sorted(meta)
    print(f"{len(ids)} non-gold studies from {len(set(m['file'] for m in meta.values()))} cache parts", flush=True)
    src = {n: find(f"/kaggle/input/**/{n}.pt*")[0] for n in NAMES}
    info = {n: {"file": Path(p).name, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for n, p in src.items()}
    print(json.dumps(info), flush=True)
    models = {n: backbone(p) for n, p in src.items()}
    feats = {n: np.zeros((len(ids), 4, 2048), np.float16) for n in NAMES}
    dl = torch.utils.data.DataLoader(Windows(ids, meta), batch_size=1, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for i, (x, plane) in enumerate(dl):
            x = x[0].cuda(non_blocking=True).half().div_(255)
            x = ((x - MEAN.cuda().half()) / STD.cuda().half()).contiguous(memory_format=torch.channels_last)
            plane = plane[0].numpy()
            for n, m in models.items():
                f = m(x).float().cpu().numpy()                                  # [K, 2048]
                feats[n][i, 0] = f.mean(0)
                for p in range(3):
                    if (plane == p).any():
                        feats[n][i, p + 1] = f[plane == p].mean(0)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(ids)} studies, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    for n in NAMES:
        np.save(OUT / f"feats_{n}.npy", feats[n])
    (OUT / "study_ids.json").write_text(json.dumps(ids) + "\n")
    (OUT / "backbones.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"PROBE_FEATURES_DONE {len(ids)} studies x {len(NAMES)} backbones, {(time.monotonic() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
