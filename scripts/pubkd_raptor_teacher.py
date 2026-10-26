"""PUB-KD teacher, Raptor slot: the two public nartaa CoAtNet checkpoints on every study of our d96 cache.
Reproduces their inference (goodpjw2008 0.958 notebook, Parts A1 / A1b = nartaa's 0.949 / 0.945 scripts):
  windows  _eval_centers over the rows that are not all zero (their mask), 94 centres by linspace, triplet c-1..c+1
  384      model built at img_size 320, 320 centre crop of the 384 px cache rows, ImageNet normalisation
  224      model built at 224, centre crop 0.8 of the field (307 px), bilinear resize to 224, then normalisation
  TTA      anatomical mirror (sagittal: reverse the 3 slice channels; others: flip width), probabilities averaged
  precision fp16 autocast (theirs)
Our d96 cache was built from their corpus code (Exp 1), so the per-study volume is their [96, 384, 384] uint8 input.
Outputs (OUT/raptor{384,224}_{SHARD}.json): per study plain / mirror probabilities [12], TARGETS order.
Usage: PUBKD_MODEL=384|224 PUBKD_SHARD=k PUBKD_NSHARD=n PUBKD_CKPT=<pt> PUBKD_OUT=<dir> python pubkd_raptor_teacher.py
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coatnet_d96_runner as r

if os.environ.get("PUBKD_CACHE_SHARDS"):                          # CPU test on the 1-shard smoke cache
    r.CFG["shards"] = int(os.environ["PUBKD_CACHE_SHARDS"])

SHA = {"384": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef",
       "224": "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"}
ARCH = {"384": "coatnet_rmlp_2_rw_384", "224": "coatnet_rmlp_2_rw_224"}
MODEL = os.environ["PUBKD_MODEL"]; assert MODEL in SHA
SHARD, NSHARD = int(os.environ.get("PUBKD_SHARD", 0)), int(os.environ.get("PUBKD_NSHARD", 1))
CKPT, OUT = Path(os.environ["PUBKD_CKPT"]), Path(os.environ["PUBKD_OUT"])
LIMIT = int(os.environ.get("PUBKD_LIMIT", 0))                     # CPU test: first n studies of the shard
K, IMG, MAXS = 94, 384, 96
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SLOT = np.repeat(np.arange(5), [26, 22, 18, 12, 18])


class RaptorClassifier(nn.Module):
    """Their head: LayerNorm, per-finding attention (Linear-Tanh-Dropout-Linear), softmax over windows, clsW / clsb."""

    def __init__(self, backbone, d, n=12):
        super().__init__()
        self.backbone, self.norm = backbone, nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.0), nn.Linear(256, n))
        self.clsW, self.clsb = nn.Parameter(torch.zeros(n, d)), nn.Parameter(torch.zeros(n))

    def forward(self, x):                                         # x [K, 3, H, W] -> logits [12]
        f = torch.cat([self.backbone(c) for c in x.split(16)])
        h = self.norm(f)
        a = torch.softmax(self.att(h).float(), 0).to(h.dtype)
        return (torch.einsum("kn,kf->nf", a, h) * self.clsW).sum(-1) + self.clsb


def eval_centers(mask):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, MAXS))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, MAXS - 2))]
    return np.array([max(1, min(cs[i], MAXS - 2)) for i in np.linspace(0, len(cs) - 1, K).round().astype(int)])


def windows(vol, dev, mirror):
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    cs = eval_centers(mask)
    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # [K, 3, 384, 384]
    if mirror:
        sag = torch.from_numpy(np.isin(SLOT[cs], (0, 1))).to(dev).view(-1, 1, 1, 1)
        tri = torch.where(sag, tri.flip(1), tri.flip(-1))
    x = tri.float() / 255.0
    if MODEL == "384":
        lo = (IMG - 320) // 2
        x = x[..., lo:lo + 320, lo:lo + 320]
    else:
        c = int(round(IMG * 0.8)); o = (IMG - c) // 2
        x = F.interpolate(x[..., o:o + c, o:o + c], size=(224, 224), mode="bilinear", align_corners=False)
    return (x - MEAN.to(dev)) / STD.to(dev)


def main():
    import timm
    assert hashlib.sha256(CKPT.read_bytes()).hexdigest() == SHA[MODEL], "checkpoint hash differs"
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert ck["lab"] == r.TARGETS, ck["lab"]
    kw = {"img_size": 320} if MODEL == "384" else {}
    bb = timm.create_model(ARCH[MODEL], pretrained=False, num_classes=0, **kw)
    model = RaptorClassifier(bb, bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)
    meta = r.read_manifest()
    ids = sorted(meta)[SHARD::NSHARD]
    if LIMIT:
        ids = ids[:LIMIT]
    out, mm, t0 = {}, {}, time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, s in enumerate(ids):
            m = meta[s]
            if m["file"] not in mm:
                mm[m["file"]] = np.load(m["file"], mmap_mode="r")
            vol = np.asarray(mm[m["file"]][m["row"]:m["row"] + MAXS])
            assert vol.shape == (MAXS, IMG, IMG), vol.shape
            rec = {}
            for nm, mir in (("plain", False), ("mirror", True)):
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    z = model(windows(vol, dev, mir)).float()
                assert torch.isfinite(z).all(), s
                rec[nm] = torch.sigmoid(z).cpu().numpy().round(6).tolist()
            out[s] = rec
            if (i + 1) % 200 == 0 or i + 1 == len(ids):
                print(f"raptor{MODEL} shard {SHARD}/{NSHARD}: {i + 1}/{len(ids)} {(time.monotonic() - t0) / (i + 1):.3f}s/study", flush=True)
    path = OUT / f"raptor{MODEL}_{SHARD}.json"
    path.write_text(json.dumps({"model": MODEL, "sha256": SHA[MODEL], "targets": r.TARGETS, "shard": [SHARD, NSHARD],
                                "n": len(out), "seconds": round(time.monotonic() - t0, 1), "pred": out}))
    print(f"DONE {path} {len(out)} studies", flush=True)


if __name__ == "__main__":
    main()
