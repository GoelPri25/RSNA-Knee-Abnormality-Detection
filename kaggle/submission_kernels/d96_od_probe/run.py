"""OrthoDiffusion go/no-go: frozen-feature probe of the public OrthoDiffusion knee MRI diffusion backbones (T4, inference only).

OrthoDiffusion (arXiv 2602.20752, MIT; weights lanstat0123/orthodiffusion, EMA state dicts re-saved as plain tensors)
= three 3D U-Net denoisers (sagittal / coronal / axial, 34.5M params each) pretrained on 15,948 unlabelled knee MRIs.
Input as in the paper: one series per plane, 16 central slices, 256x256, per-volume min-max to [-1, 1]. From the d96
cache (shard 0, gold excluded; 384 px, 140 mm crop): sagittal slot 0 (fluid-sensitive), else slot 1; coronal slot 2,
else 3; axial slot 4; valid rows only, centre 16 (edge-repeated if fewer), resized 384 -> 256.
Per study, plane, in-plane orientation (as cached / transposed), timestep t in {50, 150, 250} (fixed noise per study):
  - middle-block features mid_0 and mid_2 [256, 1, 16, 16] -> global mean (256) and 2x2 mean (1024);
  - denoising error mean((eps_hat - eps)^2): a label-free check of domain fit and of which orientation matches.
Outputs: od_feats.npy float16 [S, orient 2, plane 3, t 3, block 2, 1280], od_present.npy [S, 3], od_eps.npy [S, 2, 3, 3],
study_ids.json, run_info.json. The probe itself is fitted locally (scripts/audit_od_probe.py).
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
import torch.nn.functional as F

SLOTS = [26, 22, 18, 12, 18]
OFF = np.cumsum([0] + SLOTS)
PLANE_SLOTS = [(0, 1), (2, 3), (4,)]
PLANES = ["sagittal", "coronal", "axial"]
TS, BLOCKS, DEPTH, SIZE = [50, 150, 250], ["mid_0", "mid_2"], 16, 256
OUT = Path("/kaggle/working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"  # CPU only for the local smoke test


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


def volume(block, mask, plane):
    """[16, 384, 384] uint8 rows of the plane's first non-empty slot (centre 16 valid rows), or None."""
    for s in PLANE_SLOTS[plane]:
        rows = np.arange(OFF[s], OFF[s + 1])[mask[OFF[s]:OFF[s + 1]]]
        if len(rows) >= 3:
            c = len(rows) // 2
            idx = np.clip(np.arange(c - DEPTH // 2, c + DEPTH // 2), 0, len(rows) - 1)
            return block[rows[idx]]
    return None


def main():
    t0 = time.monotonic()
    code = Path(find("/kaggle/input/**/od_code/diffusion_model/unet.py")[0]).parent.parent
    sys.path.insert(0, str(code))
    from diffusion_model.unet import create_model

    meta = manifest(); ids = sorted(meta)[:int(os.environ.get("OD_LIMIT", "0")) or None]  # OD_LIMIT: smoke test only
    print(f"{len(ids)} non-gold studies", flush=True)
    nets, info = [], {}
    for p in PLANES:
        f = find(f"/kaggle/input/**/od_{p}_ema.pt")[0]
        info[p] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        ck = torch.load(f, map_location="cpu", weights_only=True)
        m = create_model(SIZE, 64, 1, in_channels=1, out_channels=1)
        m.load_state_dict(ck["unet"], strict=True)
        nets.append(m.eval().to(DEV))
        a = ck["schedule"]["sqrt_alphas_cumprod"].to(DEV); b = ck["schedule"]["sqrt_one_minus_alphas_cumprod"].to(DEV)
    print(json.dumps(info), flush=True)

    S = len(ids)
    feats = np.zeros((S, 2, 3, len(TS), len(BLOCKS), 1280), np.float16)
    eps = np.full((S, 2, 3, len(TS)), np.nan, np.float32)
    present = np.zeros((S, 3), bool)
    mm = {}
    with torch.no_grad():
        for i, sid in enumerate(ids):
            m = meta[sid]
            if m["file"] not in mm:
                mm[m["file"]] = np.load(m["file"], mmap_mode="r")
            block = mm[m["file"]][m["row"]:m["row"] + OFF[-1]]
            g = torch.Generator(device=DEV).manual_seed(i)
            for p in range(3):
                v = volume(block, m["mask"], p)
                if v is None:
                    continue
                present[i, p] = True
                x = torch.from_numpy(np.ascontiguousarray(v)).to(DEV).float()          # [16, 384, 384]
                x = F.interpolate(x[None], size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
                x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 2 - 1
                for o, xo in enumerate((x, x.transpose(1, 2))):
                    x5 = xo[None, None].contiguous()                                  # [1, 1, 16, 256, 256]
                    for k, t in enumerate(TS):
                        noise = torch.randn(x5.shape, generator=g, device=DEV)
                        xt = a[t] * x5 + b[t] * noise
                        tt = torch.tensor([t], device=DEV)
                        out, inf = nets[p](xt, tt, ret_mid=True, ret_out=True)
                        eps[i, o, p, k] = F.mse_loss(out.float(), noise).item()
                        for j, name in enumerate(BLOCKS):
                            h = inf["mid"][name].float()[0, :, 0]                       # [256, 16, 16]
                            feats[i, o, p, k, j] = torch.cat([h.mean((1, 2)), F.avg_pool2d(h, 8).flatten()]).cpu().numpy()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{S} studies, {(time.monotonic() - t0) / 60:.1f} min; eps (orient x t, sagittal) "
                      f"{np.nanmean(eps[:i + 1, :, 0], 0).round(4).tolist()}", flush=True)
    np.save(OUT / "od_feats.npy", feats); np.save(OUT / "od_eps.npy", eps); np.save(OUT / "od_present.npy", present)
    (OUT / "study_ids.json").write_text(json.dumps(ids) + "\n")
    (OUT / "run_info.json").write_text(json.dumps({"weights_sha256": info, "ts": TS, "blocks": BLOCKS,
                                                    "eps_mean": np.nanmean(eps, 0).tolist()}, indent=1) + "\n")
    print(f"OD_PROBE_DONE {S} studies, present {present.mean(0).round(3).tolist()}, {(time.monotonic() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
