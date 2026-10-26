"""Two inference-only audits of the d96 r50 SWA models (Exp 1c and Exp 1e) on the 825 fold-0 studies (train_series
DICOMs, same d96 builder), scored offline against Steven-v4.

1. Series ablation (eval view: centre 320 crop, 94 windows): the backbone runs once; the head is re-applied with the
   windows of one series (window-centre slot) or one plane removed from the attention softmax. How much does each
   series actually add to AUC (vs attention share, which is not contribution)?
2. Shift TTA: the 320 eval crop moved by +-24 px (6% of 384) in 8 directions, no resampling. Shift TTA gave +0.003
   to +0.004 on single 224 px r50s (ledger, TTA row); untested on d96.
Output per model: logits [n, variants, 12] (float32) and variant names.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ARCH = "resnet50.tv_in1k"
IMG, CROP_MM, EVAL_RES, EVAL_WINDOWS = 384, 140.0, 320, 94
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(k for _, _, k in SLOTS)
SLOT_OF = np.repeat(np.arange(5), [k for _, _, k in SLOTS])
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def pick(paths):
    found = [p for p in paths if p.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one input among {paths}; found {found}")
    return found[0]


DATA = pick([Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
             Path("/kaggle/input/rsna-knee-abnormality-detection")])
MODELS = {"1c": ("rsna-d96-r50-exp1c-fold0", "dec52d6009d59d1f309f001e9a89376a8b61717131ba1f434458c3715cc328f7"),
          "1e": ("rsna-d96-r50-exp1e-pertarget-fold0", "cb2382ecfc05153be1024ef54f2cd1bdb55b3cd10558edb561fc6debcb37bf85")}
ART = {k: pick([Path(f"/kaggle/input/datasets/tumis001/{d}"), Path(f"/kaggle/input/{d}")]) for k, (d, _) in MODELS.items()}
IDS = pick([Path("/kaggle/input/datasets/tumis001/rsna-d96-fold0-ids"),
            Path("/kaggle/input/rsna-d96-fold0-ids")]) / "fold0_ids.json"
OUT = Path("/kaggle/working")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------- d96 builder (same as the training cache)
def header(path):
    import pydicom
    try:
        h = pydicom.dcmread(path, stop_before_pixels=True)
        iop = getattr(h, "ImageOrientationPatient", None)
        ipp = getattr(h, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r, c = np.array(iop[:3], float), np.array(iop[3:], float)
            pos = float(np.dot(np.array(ipp, float), np.cross(r, c)))
        else:
            pos = float(getattr(h, "InstanceNumber", 0) or 0)
        ps = getattr(h, "PixelSpacing", None)
        return pos, path, (float(ps[0]) if ps is not None else 0.5), True
    except Exception:
        return 0.0, path, 0.5, False


def pixels(path):
    import pydicom
    from pydicom.pixels import apply_modality_lut
    d = pydicom.dcmread(path)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def crop_resize(a, ps):
    import cv2
    h, w = a.shape
    cpx = min(int(round(CROP_MM / max(ps, 1e-3))), h, w)
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (IMG, IMG), interpolation=cv2.INTER_AREA)


def build_study(job):
    study_id, rows = job
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx, used, errors = 0, set(), 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for plane, fluid, k in SLOTS:
            cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
            pref = [r for r in cands if fluid in (0, 1) and int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
            row = (pref or cands or [None])[0]
            if row is None:
                idx += k
                continue
            used.add(row["SeriesInstanceUID"])
            files = sorted((DATA / "train_series" / study_id / row["SeriesInstanceUID"]).glob("*.dcm"))
            heads = list(pool.map(header, files))
            ok_ps = [h[2] for h in heads if h[3]]
            med_ps = float(np.median(ok_ps)) if ok_ps else 0.5
            heads.sort(key=lambda h: h[0])
            n = len(heads)
            if n == 0:
                idx += k
                continue
            lo, hi = int(n * 0.02), int(n * 0.98) - 1
            hi = max(hi, lo)
            picks = [int(min(p, n - 1)) for p in (np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k)]

            def load(p):
                try:
                    return pixels(heads[p][1]), heads[p][2]
                except Exception:
                    return None, med_ps
            got = list(pool.map(load, picks))
            valid = [a for a, _ in got if a is not None]
            errors += len(got) - len(valid)
            loq, hiq = (np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0]) if valid else (0.0, 1.0))
            for a, ps in got:
                if a is not None:
                    aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                    vol[idx] = (crop_resize(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8)
                idx += 1
    mask = vol.reshape(MAXS, -1).max(1) > 0
    return study_id, vol, mask, errors


def windows(vol, mask):
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, MAXS - 2))])
    c = np.clip(cs[np.linspace(0, len(cs) - 1, EVAL_WINDOWS).round().astype(int)], 1, MAXS - 2)
    return np.ascontiguousarray(vol[np.stack([c - 1, c, c + 1], 1)])           # [94, 3, 384, 384]


# ---------------------------------------------------------------- model
SHIFT = 24
SHIFTS = [(-SHIFT, 0), (SHIFT, 0), (0, -SHIFT), (0, SHIFT), (-SHIFT, -SHIFT), (-SHIFT, SHIFT), (SHIFT, -SHIFT), (SHIFT, SHIFT)]
PLANE_OF_SLOT = np.array([0, 0, 1, 1, 2])
SLOT_NAMES = ["sag_fluid", "sag_nonfluid", "cor_fluid", "cor_nonfluid", "axial"]
DROPS = [("drop_" + n, [j]) for j, n in enumerate(SLOT_NAMES)] + \
        [("drop_sagittal", [0, 1]), ("drop_coronal", [2, 3]), ("drop_axial", [4]), ("only_sagittal", [2, 3, 4])]
VARIANTS = ["full"] + [n for n, _ in DROPS] + [f"shift_{dy}_{dx}" for dy, dx in SHIFTS]


def ensure_timm():
    try:
        import timm
        if timm.__version__ == "1.0.29":
            return
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "-q",
                    str(ART["1c"] / "timm-1.0.29-py3-none-any.whl")], check=True)


def build_model(key, device):
    import timm
    import torch
    import torch.nn as nn

    class Raptor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(ARCH, pretrained=False, num_classes=0)
            d = self.backbone.num_features
            self.norm = nn.LayerNorm(d)
            self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
            self.clsW = nn.Parameter(torch.zeros(12, d))
            self.clsb = nn.Parameter(torch.zeros(12))

        def features(self, x):
            return self.norm(self.backbone(x).float())                         # [K, d]

        def head(self, h, keep=None):
            s = self.att(h).float()                                            # [K, 12]
            if keep is not None:
                s = s.masked_fill(~keep[:, None], float("-inf"))
            a = torch.softmax(s, dim=0)
            return (torch.einsum("kn,kf->nf", a, h) * self.clsW).sum(-1) + self.clsb

    path = ART[key] / "weights_swa21_25.pt"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != MODELS[key][1]:
        raise ValueError(f"Unexpected weights SHA-256 for {key}: {got}")
    model = Raptor()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval().to(memory_format=torch.channels_last)


def main():
    started = time.monotonic()
    ensure_timm()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {k: build_model(k, device) for k in MODELS}
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    ids = json.loads(IDS.read_text())
    by_study = {}
    for row in read_csv(DATA / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    print(f"Studies {len(ids)} | device {device} | variants {len(VARIANTS)}", flush=True)
    out = {k: np.full((len(ids), len(VARIANTS), 12), np.nan, np.float32) for k in MODELS}
    lo = (IMG - EVAL_RES) // 2
    with Pool(4) as procs, torch.inference_mode():
        for n, (s, vol, mask, errors) in enumerate(
                procs.imap(build_study, [(s, by_study.get(s, [])) for s in ids])):
            if not mask.any():
                continue
            valid = np.where(mask)[0]; vlo, vhi = int(valid.min()), int(valid.max())
            cs = np.arange(vlo + 1, vhi) if vhi - vlo >= 2 else np.array([max(1, min((vlo + vhi) // 2, MAXS - 2))])
            centre = np.clip(cs[np.linspace(0, len(cs) - 1, EVAL_WINDOWS).round().astype(int)], 1, MAXS - 2)
            slot = torch.from_numpy(SLOT_OF[centre]).to(device)
            x = torch.from_numpy(windows(vol, mask)).to(device).float().div_(255.0)
            views = [(0, 0)] + SHIFTS
            for key, model in models.items():
                for v, (dy, dx) in enumerate(views):
                    xi = ((x[..., lo + dy:lo + dy + EVAL_RES, lo + dx:lo + dx + EVAL_RES] - mean) / std).contiguous(
                        memory_format=torch.channels_last)
                    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        h = model.features(xi)
                    h = h.float()
                    if not torch.isfinite(h).all():
                        h = model.features(xi.float()).float()
                    if v == 0:
                        out[key][n, 0] = model.head(h).cpu().numpy()
                        for j, (_, drop) in enumerate(DROPS, 1):
                            keep = ~torch.isin(slot, torch.tensor(drop, device=device))
                            if keep.any():
                                out[key][n, j] = model.head(h, keep).cpu().numpy()
                    else:
                        out[key][n, len(DROPS) + v] = model.head(h).cpu().numpy()
            if (n + 1) % 50 == 0:
                print(f"{n + 1}/{len(ids)}; {time.monotonic() - started:.0f}s", flush=True)
    np.savez_compressed(OUT / "series_ablation_tta.npz", ids=np.array(ids), variants=np.array(VARIANTS),
                        **{f"logits_{k}": v for k, v in out.items()})
    summary = {"studies": len(ids), "variants": VARIANTS, "shift_px": SHIFT,
               "nan_cells": {k: int(np.isnan(v).sum()) for k, v in out.items()},
               "weights_sha256": {k: v[1] for k, v in MODELS.items()}, "seconds": round(time.monotonic() - started, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
