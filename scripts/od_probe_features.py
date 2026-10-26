"""OrthoDiffusion go/no-go, feature extraction (Modal H100; replaces the slower Kaggle T4 kernel
kaggle/submission_kernels/d96_od_probe, which covered cache shard 0 only).

All d96 cache shards, gold excluded. Per study:
  - OrthoDiffusion (arXiv 2602.20752, MIT; EMA U-Nets re-saved as plain tensors, data/orthodiffusion/dataset): one
    series per plane (sagittal slot 0 else 1, coronal slot 2 else 3, axial slot 4), 16 central valid rows, 384 -> 256,
    per-volume min-max to [-1, 1]; 2 in-plane orientations (as cached / transposed) x t in {50, 150, 250} in one
    batch, fixed noise per study; middle blocks mid_0 / mid_2 [256, 1, 16, 16] -> global mean (256) + 2x2 mean (1024).
    Denoising error mean((eps_hat - eps)^2) at t = 150 for both orientations (label-free domain / orientation check).
  - ImageNet r50 (tv_in1k, 1e's start) reference exactly as the Exp 5a probe: 32 evenly spaced 3-slice windows,
    centre 320 crop, ImageNet normalisation, 2048-d features mean-pooled (all, sagittal, coronal, axial).
bf16 autocast (H100). Outputs in OD_OUT: od_feats.npy float16 [S, 2, 3, 3, 2, 1280], od_present.npy [S, 3],
od_eps.npy [S, 2, 3], feats_tv_in1k.npy float16 [S, 4, 2048], study_ids.json, run_info.json.
Scored locally by scripts/audit_od_probe.py.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CACHE, WEIGHTS, OUT = Path(os.environ["OD_CACHE"]), Path(os.environ["OD_WEIGHTS"]), Path(os.environ["OD_OUT"])
TV = Path(os.environ["OD_TV"])
SLOTS = [26, 22, 18, 12, 18]
OFF = np.cumsum([0] + SLOTS)
SLOT_OF = np.repeat(np.arange(5), SLOTS)
PLANE_OF_ROW = np.array([0, 0, 1, 1, 2])[SLOT_OF]
PLANE_SLOTS = [(0, 1), (2, 3), (4,)]
PLANES = ["sagittal", "coronal", "axial"]
TS, BLOCKS, DEPTH, SIZE = [50, 150, 250], ["mid_0", "mid_2"], 16, 256
K, RES = 32, 320
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def manifest():
    studies = {}
    for m in sorted(glob.glob(str(CACHE / "**/manifest.jsonl"), recursive=True)):
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


def plane_rows(mask, plane):
    for s in PLANE_SLOTS[plane]:
        rows = np.arange(OFF[s], OFF[s + 1])[mask[OFF[s]:OFF[s + 1]]]
        if len(rows) >= 3:
            c = len(rows) // 2
            return rows[np.clip(np.arange(c - DEPTH // 2, c + DEPTH // 2), 0, len(rows) - 1)]
    return None


def centres(mask):
    valid = np.where(mask)[0]
    lo, hi = (int(valid.min()), int(valid.max())) if len(valid) >= 3 else (0, 2)
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([1])
    return cs[np.linspace(0, len(cs) - 1, K).round().astype(int)]


class Studies(torch.utils.data.Dataset):
    def __init__(self, ids, meta):
        self.ids, self.meta, self._mm = ids, meta, {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        m = self.meta[self.ids[i]]
        if m["file"] not in self._mm:
            self._mm[m["file"]] = np.load(m["file"], mmap_mode="r")
        mm = self._mm[m["file"]]
        vols, present = np.zeros((3, DEPTH, 384, 384), np.uint8), np.zeros(3, bool)
        for p in range(3):
            r = plane_rows(m["mask"], p)
            if r is not None:
                vols[p], present[p] = mm[m["row"] + r], True
        c = np.clip(centres(m["mask"]), 1, OFF[-1] - 2)
        rows = np.stack([c - 1, c, c + 1], 1)
        uniq, inv = np.unique(rows, return_inverse=True)
        block = np.asarray(mm[m["row"] + uniq])
        lo = (block.shape[-1] - RES) // 2
        win = block[inv.reshape(rows.shape)][..., lo:lo + RES, lo:lo + RES]
        return torch.from_numpy(vols), torch.from_numpy(present), torch.from_numpy(np.ascontiguousarray(win)), torch.from_numpy(PLANE_OF_ROW[c])


def main():
    t0 = time.monotonic()
    sys.path.insert(0, str(WEIGHTS / "od_code"))
    from diffusion_model.unet import create_model

    meta = manifest(); ids = sorted(meta)
    print(f"{len(ids)} non-gold studies from {len(set(m['file'] for m in meta.values()))} cache parts", flush=True)
    info, nets = {}, []
    for p in PLANES:
        f = WEIGHTS / f"od_{p}_ema.pt"
        info[p] = hashlib.sha256(f.read_bytes()).hexdigest()
        ck = torch.load(f, map_location="cpu", weights_only=True)
        m = create_model(SIZE, 64, 1, in_channels=1, out_channels=1)
        m.load_state_dict(ck["unet"], strict=True)
        nets.append(m.eval().cuda())
        a = ck["schedule"]["sqrt_alphas_cumprod"].cuda(); b = ck["schedule"]["sqrt_one_minus_alphas_cumprod"].cuda()
    info["tv_in1k"] = hashlib.sha256(TV.read_bytes()).hexdigest()
    r50 = torchvision.models.resnet50(); r50.fc = nn.Identity()
    sd = torch.load(TV, map_location="cpu", weights_only=True)
    r50.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=True)
    r50 = r50.eval().cuda().to(memory_format=torch.channels_last)
    print(json.dumps(info), flush=True)

    S = len(ids)
    feats = np.zeros((S, 2, 3, len(TS), len(BLOCKS), 1280), np.float16)
    eps = np.full((S, 2, 3), np.nan, np.float32)
    present = np.zeros((S, 3), bool)
    tv = np.zeros((S, 4, 2048), np.float16)
    tt = torch.tensor(TS * 2, device="cuda")                                   # [orient0 t50,150,250, orient1 ...]
    dl = torch.utils.data.DataLoader(Studies(ids, meta), batch_size=1, num_workers=8, pin_memory=True, prefetch_factor=4)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, (vols, pres, win, wplane) in enumerate(dl):
            g = torch.Generator(device="cuda").manual_seed(i)
            present[i] = pres[0].numpy()
            for p in range(3):
                if not present[i, p]:
                    continue
                x = vols[0, p].cuda(non_blocking=True).float()
                x = F.interpolate(x[None], size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
                x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 2 - 1
                x2 = torch.stack([x, x.transpose(1, 2)])[:, None]                # [2, 1, 16, 256, 256]
                x6 = x2.repeat_interleave(len(TS), 0)                           # [6, ...] orient-major
                noise = torch.randn(x6.shape, generator=g, device="cuda")
                xt = a[tt].view(-1, 1, 1, 1, 1) * x6 + b[tt].view(-1, 1, 1, 1, 1) * noise
                _, inf = nets[p](xt, tt, ret_mid=True)
                for j, name in enumerate(BLOCKS):
                    h = inf["mid"][name].float()[:, :, 0]                       # [6, 256, 16, 16]
                    v = torch.cat([h.mean((2, 3)), F.avg_pool2d(h, 8).flatten(1)], 1).view(2, len(TS), 1280)
                    feats[i, :, p, :, j] = v.cpu().numpy()
                sel = [TS.index(150), len(TS) + TS.index(150)]
                out, _ = nets[p](xt[sel], tt[sel], ret_out=True)               # skips are kept only with ret_out
                eps[i, :, p] = ((out.float() - noise[sel]) ** 2).mean((1, 2, 3, 4)).cpu().numpy()
            w = win[0].cuda(non_blocking=True).float().div_(255)
            w = ((w - MEAN.cuda()) / STD.cuda()).contiguous(memory_format=torch.channels_last)
            f = r50(w).float().cpu().numpy(); wp = wplane[0].numpy()
            tv[i, 0] = f.mean(0)
            for p in range(3):
                if (wp == p).any():
                    tv[i, p + 1] = f[wp == p].mean(0)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{S} studies, {(time.monotonic() - t0) / 60:.1f} min; eps t150 [orient][plane] "
                      f"{np.nanmean(eps[:i + 1], 0).round(4).tolist()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "od_feats.npy", feats); np.save(OUT / "od_eps.npy", eps); np.save(OUT / "od_present.npy", present)
    np.save(OUT / "feats_tv_in1k.npy", tv)
    (OUT / "study_ids.json").write_text(json.dumps(ids) + "\n")
    (OUT / "run_info.json").write_text(json.dumps({"sha256": info, "ts": TS, "blocks": BLOCKS, "studies": S,
                                                   "eps_t150_mean": np.nanmean(eps, 0).tolist(),
                                                   "minutes": (time.monotonic() - t0) / 60}, indent=1) + "\n")
    print(f"OD_PROBE_DONE {S} studies, present {present.mean(0).round(3).tolist()}, {(time.monotonic() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
