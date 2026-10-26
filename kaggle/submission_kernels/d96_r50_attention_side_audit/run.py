"""Attention-placement audit: does Exp 1e (best single d96 r50, SWA 21-25) look on the anatomically right side of the knee?
Same exact CAM decomposition and d96 builder as kaggle/submission_kernels/d96_r50_cam_centring_audit (Exp 1c), but kept
per window instead of summed per slot, plus the DICOM geometry needed to place each window:
  att [S, 94, 12]            softmax attention of each finding over the 94 eval windows
  centre [S, 94]             d96 row of each window's centre slice (slot = SLOT_OF[centre])
  camx / camy [S, 94, 12, 12]  positive CAM mass per window and finding, summed over rows (-> per column) / over columns
  ipp [S, 96, 3], iop [S, 5, 6]  ImagePositionPatient of every picked slice, ImageOrientationPatient per slot
  lat (json)                 Laterality / ImageLaterality / BodyPartExamined per slot
Full 384 input (12x12 grid, 140 mm). No labels are used here; analysed offline by scripts/audit_d96_attention_side.py.
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

WEIGHTS_SHA = "cb2382ecfc05153be1024ef54f2cd1bdb55b3cd10558edb561fc6debcb37bf85"
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
ART = pick([Path("/kaggle/input/datasets/tumis001/rsna-d96-r50-exp1e-pertarget-fold0"),
            Path("/kaggle/input/rsna-d96-r50-exp1e-pertarget-fold0")])
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
        geo = ([float(v) for v in ipp] if ipp is not None else None, [float(v) for v in iop] if iop is not None else None)
        lat = {k: str(getattr(h, k, "") or "") for k in ("Laterality", "ImageLaterality", "BodyPartExamined")}
        return pos, path, (float(ps[0]) if ps is not None else 0.5), True, geo, lat
    except Exception:
        return 0.0, path, 0.5, False, (None, None), {}


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
    ipp = np.full((MAXS, 3), np.nan, np.float32); iop = np.full((5, 6), np.nan, np.float32); lats = [{}] * 5
    idx, used, errors = 0, set(), 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for j_slot, (plane, fluid, k) in enumerate(SLOTS):
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
            for q, p in enumerate(picks):
                if heads[p][4][0] is not None:
                    ipp[idx + q] = heads[p][4][0]
            o = next((h[4][1] for h in heads if h[4][1] is not None), None)
            if o is not None:
                iop[j_slot] = o
            lats[j_slot] = next((h[5] for h in heads if any(h[5].values())), {})

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
    return study_id, vol, mask, errors, ipp, iop, lats


def windows(vol, mask):
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, MAXS - 2))])
    c = np.clip(cs[np.linspace(0, len(cs) - 1, EVAL_WINDOWS).round().astype(int)], 1, MAXS - 2)
    return np.ascontiguousarray(vol[np.stack([c - 1, c, c + 1], 1)])           # [94, 3, 384, 384]


# ---------------------------------------------------------------- model
def ensure_timm():
    try:
        import timm
        if timm.__version__ == "1.0.29":
            return
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "-q",
                    str(ART / "timm-1.0.29-py3-none-any.whl")], check=True)


def build_model(device):
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

        def cam(self, x):
            """x: [K,3,H,W] one study. Returns logits [12], cam [K,12,h*w] (exact decomposition), attention [K,12]."""
            F = self.backbone.forward_features(x).float()                       # [K, d, h, w]
            K, d, h, w = F.shape
            f = F.mean((2, 3))
            mu = f.mean(-1, keepdim=True)
            sig = torch.sqrt(f.var(-1, unbiased=False, keepdim=True) + self.norm.eps)
            hn = self.norm(f)
            a = torch.softmax(self.att(hn), dim=0)                               # [K, 12] softmax over windows
            logits = (torch.einsum("kn,kf->nf", a, hn) * self.clsW).sum(-1) + self.clsb
            V = self.clsW * self.norm.weight                                     # [12, d]
            Z = (F.view(K, d, h * w) - mu[..., None]) / sig[..., None]          # [K, d, hw]
            cam = torch.einsum("nf,kfp->knp", V, Z) * a[..., None]               # [K, 12, hw]
            const = (a * (self.clsW @ self.norm.bias)[None]).sum(0) + self.clsb
            recon = cam.mean(-1).sum(0) + const
            return logits, recon, cam, a

    path = ART / "weights_swa21_25.pt"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != WEIGHTS_SHA:
        raise ValueError(f"Unexpected weights SHA-256: {got}")
    model = Raptor()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def main():
    started = time.monotonic()
    ensure_timm()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    ids = json.loads(IDS.read_text())
    by_study = {}
    for row in read_csv(DATA / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    print(f"Studies {len(ids)} | device {device} | data {DATA}", flush=True)
    n_s = len(ids)
    ATT = np.zeros((n_s, EVAL_WINDOWS, 12), np.float16); CEN = np.zeros((n_s, EVAL_WINDOWS), np.int16)
    CAMX = np.zeros((n_s, EVAL_WINDOWS, 12, 12), np.float16); CAMY = np.zeros_like(CAMX)
    IPP = np.full((n_s, MAXS, 3), np.nan, np.float32); IOP = np.full((n_s, 5, 6), np.nan, np.float32); LAT = []
    LOG = np.zeros((n_s, 12), np.float32); ok = np.zeros(n_s, bool); max_err = 0.0
    with Pool(4) as procs, torch.inference_mode():
        for n, (s, vol, mask, errors, ipp, iop, lats) in enumerate(
                procs.imap(build_study, [(s, by_study.get(s, [])) for s in ids])):
            IPP[n], IOP[n] = ipp, iop; LAT.append(lats)
            if not mask.any():
                continue
            wv = windows(vol, mask)
            valid = np.where(mask)[0]; lo_, hi_ = int(valid.min()), int(valid.max())
            cs = np.arange(lo_ + 1, hi_) if hi_ - lo_ >= 2 else np.array([max(1, min((lo_ + hi_) // 2, MAXS - 2))])
            centre = np.clip(cs[np.linspace(0, len(cs) - 1, EVAL_WINDOWS).round().astype(int)], 1, MAXS - 2)
            x = torch.from_numpy(wv).to(device).float().div_(255.0)
            x = (x - mean) / std                                             # full 384 field (12x12 grid)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, recon, cam, a = model.cam(x)
            max_err = max(max_err, float((logits.float() - recon.float()).abs().max()))
            pc = cam.float().clamp(min=0).view(len(centre), 12, 12, 12)          # [K, finding, row, col]
            CAMX[n] = pc.sum(2).cpu().numpy().astype(np.float16)                 # per column
            CAMY[n] = pc.sum(3).cpu().numpy().astype(np.float16)                 # per row
            ATT[n] = a.float().cpu().numpy().astype(np.float16); CEN[n] = centre
            LOG[n] = logits.float().cpu().numpy(); ok[n] = True
            if (n + 1) % 50 == 0:
                print(f"{n + 1}/{n_s}; {time.monotonic() - started:.0f}s; max |logit - recon| {max_err:.2e}", flush=True)
    np.savez_compressed(OUT / "attention_side_fold0.npz", ids=np.array(ids), att=ATT, centre=CEN, camx=CAMX, camy=CAMY,
                        ipp=IPP, iop=IOP, logits=LOG, ok=ok)
    (OUT / "laterality.json").write_text(json.dumps(LAT))
    summary = {"studies": n_s, "ok": int(ok.sum()), "max_abs_logit_minus_recon": max_err, "grid": [12, 12],
               "field_mm": CROP_MM, "weights_sha256": WEIGHTS_SHA, "seconds": round(time.monotonic() - started, 1)}
    (OUT / "cam_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if max_err > 0.05:
        raise ValueError(f"CAM decomposition does not reproduce the logits: {max_err}")


if __name__ == "__main__":
    main()
