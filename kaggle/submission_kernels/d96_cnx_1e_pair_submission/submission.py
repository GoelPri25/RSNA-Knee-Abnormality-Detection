"""ConvNeXt partner probe (2026-10-14): mean probability of Exp 1e (r50, LB 0.936) and P1-cnx (ConvNeXt-T on the
exact 1e recipe; fold-0 partner check +0.0038 clear6 / +0.0041 10-target over the 1e seed pair, which scored 0.937).
Question: does ConvNeXt's diversity carry to LB? Built from the seed-average kernel; only model 2 changed.
Submit Exp 1e: ResNet-50 on the d96 recipe trained on the teammate per-target labels, SWA of epochs 21-25 (fold 0 Steven-v4 0.9037).
Same pipeline as the Exp 1b (CoAtNet) submission; only the backbone and weights differ.

Preprocessing is the d96 cache builder (kaggle/experiment_kernels/coatnet_d96_cache/run.py) applied to
test_series: 5 slots, 96 slices over 2..98%, per-series 2/98 window, 140 mm centre crop, INTER_AREA to 384.
Model and eval match scripts/coatnet_d96_runner.py: 94 windows of (c-1, c, c+1) over the valid slice range,
centre 320 crop of the 384 image, ImageNet normalisation, RaptorClassifier head, no TTA (the pre-registered
primary). Training ran in bf16 on an H100; a T4 has no fast bf16, so this uses fp16 autocast and falls back to
fp32 for any study whose logits are not finite.
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
WEIGHTS_SHA2 = "0125e766812169502ed70416811cef39ccc7ec15e17748b0a212f78b32919e86"   # P1-cnx ConvNeXt-T
ARCH2 = "convnext_tiny.fb_in22k_ft_in1k_384"
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
ART2 = pick([Path("/kaggle/input/datasets/tumis001/rsna-d96-cnxt-p1-pertarget-fold0"),
             Path("/kaggle/input/rsna-d96-cnxt-p1-pertarget-fold0")])
OUT = Path("/kaggle/working/submission.csv")


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
            files = sorted((DATA / "test_series" / study_id / row["SeriesInstanceUID"]).glob("*.dcm"))
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
def ensure_timm():
    try:
        import timm
        if timm.__version__ == "1.0.29":
            return
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "-q",
                    str(ART / "timm-1.0.29-py3-none-any.whl")], check=True)


def build_model(device, path, sha, arch=ARCH):
    import timm
    import torch
    import torch.nn as nn

    class Raptor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(arch, pretrained=False, num_classes=0)
            d = self.backbone.num_features
            self.norm = nn.LayerNorm(d)
            self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
            self.clsW = nn.Parameter(torch.zeros(12, d))
            self.clsb = nn.Parameter(torch.zeros(12))

        def forward(self, x, b):
            f = self.backbone(x).view(b, x.shape[0] // b, -1)
            h = self.norm(f)
            a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)
            return (torch.einsum("bkn,bkf->bnf", a, h) * self.clsW).sum(-1) + self.clsb

    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha:
        raise ValueError(f"Unexpected weights SHA-256: {got}")
    model = Raptor()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval().to(memory_format=torch.channels_last)


def main():
    started = time.monotonic()
    ensure_timm()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [build_model(device, ART / "weights_swa21_25.pt", WEIGHTS_SHA),
              build_model(device, ART2 / "weights_swa21_25.pt", WEIGHTS_SHA2, ARCH2)]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    sample = read_csv(DATA / "sample_submission.csv")
    if not sample or list(sample[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("sample_submission.csv columns differ from the model target order")
    ids = [r["StudyInstanceUID"] for r in sample]
    fallback = {r["StudyInstanceUID"]: [float(r[t]) for t in TARGETS] for r in sample}

    def write(preds):
        with OUT.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["StudyInstanceUID", *TARGETS])
            for s in ids:
                w.writerow([s, *[f"{v:.8f}" for v in preds[s]]])
    write(fallback)                                          # valid file even if inference is interrupted

    by_study = {}
    for row in read_csv(DATA / "test_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    print(f"Studies {len(ids)} | device {device} | data {DATA} | artifacts {ART}", flush=True)

    preds, stats = {}, {"studies": len(ids), "empty_studies": 0, "decode_errors": 0, "fp32_fallbacks": 0}
    use_amp = device.type == "cuda"
    with Pool(4) as procs, torch.inference_mode():
        for n, (s, vol, mask, errors) in enumerate(
                procs.imap(build_study, [(s, by_study.get(s, [])) for s in ids]), 1):
            stats["decode_errors"] += errors
            if not mask.any():
                stats["empty_studies"] += 1
                preds[s] = fallback[s]
                continue
            x = torch.from_numpy(windows(vol, mask)).to(device).float().div_(255.0)
            lo = (IMG - EVAL_RES) // 2
            x = ((x[..., lo:lo + EVAL_RES, lo:lo + EVAL_RES] - mean) / std).contiguous(memory_format=torch.channels_last)
            ps = []
            for model in models:
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x, 1).float()
                if not torch.isfinite(logits).all():
                    stats["fp32_fallbacks"] += 1
                    logits = model(x, 1).float()
                ps.append(logits.sigmoid()[0].cpu().numpy())
            p = np.mean(ps, 0)                                   # equal probability mean of 1e and ConvNeXt
            if p.shape != (12,) or not np.isfinite(p).all():
                raise ValueError(f"Invalid predictions for {s}")
            preds[s] = p.tolist()
            if n % 50 == 0 or n == len(ids):
                print(f"{n}/{len(ids)} studies; {time.monotonic() - started:.0f}s", flush=True)

    write(preds)
    written = read_csv(OUT)
    vals = np.array([[float(r[t]) for t in TARGETS] for r in written])
    if [r["StudyInstanceUID"] for r in written] != ids or not np.isfinite(vals).all() \
            or vals.min() < 0 or vals.max() > 1:
        raise ValueError("Saved submission is invalid")
    summary = {**stats, "device": str(device), "weights_sha256": [WEIGHTS_SHA, WEIGHTS_SHA2],
               "model": "mean of d96_r50_exp1e (seed 2026) + d96_cnxt_p1_pertarget (ConvNeXt-T), SWA 21-25, fold-0 training studies, per-target labels",
               "fold0_steven_v4_auc": 0.9037, "seconds": round(time.monotonic() - started, 1)}
    Path("/kaggle/working/inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
