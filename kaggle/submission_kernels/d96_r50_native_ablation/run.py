"""Exp 4b ablation on the 825 fold-0 studies (inference only): what does the native-capped model rely on, and what
caused its losses (PF OA -0.005, lateral OA -0.009 vs 1e)? Native d96n builder (same as the training cache kernels,
reading train_series), Exp 4b SWA 21-25, centre 320 crop, one window per slice (runner eval). The backbone runs once;
the attention head is re-applied with windows masked (slot = slot of the window's centre slice):
  full; drop each slot (sag fluid, sag non-fluid, cor fluid, axial); drop sagittal; only sagittal;
  thin_cor / thin_ax / thin_both: keep 18 evenly spaced windows of the coronal / axial slot (d96 window count; the
  windows' own slice spacing stays native) -> does the density gain need many eval windows?
Labels are not used here; scored offline by scripts/audit_d96_native_ablation.py (with Exp 1c / 1e ablation 2-abl).
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
WEIGHTS_SHA = "9a0c3a48eb5c792b24457a0be22a3ec3ba72d35c7e5862c57cc5a391f1a5b81e"
IMG, CROP_MM, CAP, MIN_ROWS, EVAL_RES = 384, 140.0, 40, 3, 320
SLOTS = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Axial", 1)]
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def pick(paths):
    found = [p for p in paths if p.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one input among {paths}; found {found}")
    return found[0]


DATA = pick([Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
             Path("/kaggle/input/rsna-knee-abnormality-detection")])
ART = pick([Path("/kaggle/input/datasets/tumis001/rsna-d96-r50-exp4b-native-pertarget-fold0"),
            Path("/kaggle/input/rsna-d96-r50-exp4b-native-pertarget-fold0")])
IDS = pick([Path("/kaggle/input/datasets/tumis001/rsna-d96-fold0-ids"),
            Path("/kaggle/input/rsna-d96-fold0-ids")]) / "fold0_ids.json"
OUT = Path("/kaggle/working")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


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
        geo = {"ipp": [float(v) for v in ipp] if ipp is not None else None,
               "iop": [float(v) for v in iop] if iop is not None else None,
               "ps": [float(v) for v in ps] if ps is not None else None}
        lat = {k: str(getattr(h, k, "") or "") for k in ("Laterality", "ImageLaterality", "BodyPartExamined")}
        return pos, path, (float(ps[0]) if ps is not None else 0.5), True, geo, lat
    except Exception:
        return 0.0, path, 0.5, False, None, None


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


def native_picks(n):
    """Every slice in 2%..98% (same bounds as d96); evenly resampled to CAP if more."""
    if n <= 1:
        return [0] * n
    lo, hi = int(n * 0.02), max(int(n * 0.98) - 1, int(n * 0.02))
    if hi - lo + 1 <= CAP:
        return list(range(lo, hi + 1))
    return [int(p) for p in np.linspace(lo, hi, CAP).round().astype(int)]


def build_study(job):
    study_id, rows = job
    slices, slot_of, used, slots_meta = [], [], set(), []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for j, (plane, fluid) in enumerate(SLOTS):
            cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
            pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
            row = (pref or cands or [None])[0]
            meta = {"plane": plane, "fluid_wanted": fluid, "row_start": len(slices), "k": 0, "series": None}
            slots_meta.append(meta)
            if row is None:
                continue
            used.add(row["SeriesInstanceUID"])
            files = sorted((DATA / "train_series" / study_id / row["SeriesInstanceUID"]).glob("*.dcm"))
            heads = list(pool.map(header, files))
            ok_ps = [h[2] for h in heads if h[3]]
            med_ps = float(np.median(ok_ps)) if ok_ps else 0.5
            heads.sort(key=lambda h: h[0])
            n = len(heads)
            meta.update(series=row["SeriesInstanceUID"], fluid=row.get("Fluid_Sensitive"),
                        fat=row.get("Fat_Suppression"), n_files=n,
                        laterality=next((h[5] for h in heads if h[5]), None))
            picks = native_picks(n)
            if not picks:
                continue
            meta.update(picks=picks, k=len(picks), geometry=[heads[p][4] for p in picks])

            def load(p):
                try:
                    return pixels(heads[p][1]), heads[p][2]
                except Exception:
                    return None, med_ps
            got = list(pool.map(load, picks))
            valid = [a for a, _ in got if a is not None]
            meta["decode_errors"] = len(got) - len(valid)
            loq, hiq = (np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0]) if valid else (0.0, 1.0))
            for a, ps in got:
                if a is None:
                    slices.append(np.zeros((IMG, IMG), np.uint8))
                else:
                    aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                    slices.append((crop_resize(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8))
                slot_of.append(j)
    while len(slices) < MIN_ROWS:                     # keep the window code valid for (near-)empty studies
        slices.append(np.zeros((IMG, IMG), np.uint8)); slot_of.append(-1)
    vol = np.stack(slices)
    mask = vol.reshape(len(vol), -1).max(1) > 0
    return study_id, vol, {"study_id": study_id, "n": len(vol), "slot_of": slot_of,
                           "mask": mask.astype(int).tolist(), "slots": slots_meta}



DROPS = [("drop_sag_fluid", [0]), ("drop_sag_nonfluid", [1]), ("drop_cor_fluid", [2]), ("drop_axial", [3]),
         ("drop_sagittal", [0, 1]), ("only_sagittal", [2, 3])]
THIN = [("thin_cor", [2]), ("thin_ax", [3]), ("thin_both", [2, 3])]
VARIANTS = ["full"] + [n for n, _ in DROPS] + [n for n, _ in THIN]


def thin_keep(slot, slots, k=18):
    keep = np.ones(len(slot), bool)
    for j in slots:
        idx = np.where(slot == j)[0]
        if len(idx) > k:
            keep[idx] = False
            keep[idx[np.linspace(0, len(idx) - 1, k).round().astype(int)]] = True
    return keep


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

        def features(self, x):
            return self.norm(self.backbone(x).float())                         # [K, d]

        def head(self, h, keep=None):
            s = self.att(h).float()                                            # [K, 12]
            if keep is not None:
                s = s.masked_fill(~keep[:, None], float("-inf"))
            a = torch.softmax(s, dim=0)
            return (torch.einsum("kn,kf->nf", a, h) * self.clsW).sum(-1) + self.clsb

    path = ART / "weights_swa21_25.pt"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != WEIGHTS_SHA:
        raise ValueError(f"Unexpected weights SHA-256 for {key}: {got}")
    model = Raptor()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval().to(memory_format=torch.channels_last)



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
    print(f"Studies {len(ids)} | device {device} | variants {len(VARIANTS)}", flush=True)
    out = np.full((len(ids), len(VARIANTS), 12), np.nan, np.float32)
    nwin = np.zeros((len(ids), 5), np.int32)                   # windows per slot (-1 padding, 0..3)
    lo = (IMG - EVAL_RES) // 2
    with Pool(4) as procs, torch.inference_mode():
        for n, (s, vol, rec) in enumerate(procs.imap(build_study, [(s, by_study.get(s, [])) for s in ids])):
            mask = np.array(rec["mask"], bool); slot_of = np.array(rec["slot_of"])
            valid = np.where(mask)[0]
            if len(valid) < 3:
                valid = np.arange(3)
            vlo, vhi = int(valid.min()), int(valid.max())
            cs = np.arange(vlo + 1, vhi) if vhi - vlo >= 2 else np.array([max(1, min((vlo + vhi) // 2, len(mask) - 2))])
            c = np.clip(cs, 1, len(mask) - 2)
            slot = slot_of[c]
            for j in range(-1, 4):
                nwin[n, j + 1] = int((slot == j).sum())
            x = torch.from_numpy(np.ascontiguousarray(vol[np.stack([c - 1, c, c + 1], 1)])).to(device).float().div_(255.0)
            xi = ((x[..., lo:lo + EVAL_RES, lo:lo + EVAL_RES] - mean) / std).contiguous(memory_format=torch.channels_last)
            h = []
            for b in range(0, len(xi), 64):
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    hb = model.features(xi[b:b + 64]).float()
                if not torch.isfinite(hb).all():
                    hb = model.features(xi[b:b + 64].float()).float()
                h.append(hb)
            h = torch.cat(h)
            st = torch.from_numpy(slot).to(device)
            out[n, 0] = model.head(h).cpu().numpy()
            for j, (_, drop) in enumerate(DROPS, 1):
                keep = ~torch.isin(st, torch.tensor(drop, device=device))
                if keep.any():
                    out[n, j] = model.head(h, keep).cpu().numpy()
            for j, (_, slots) in enumerate(THIN, 1 + len(DROPS)):
                out[n, j] = model.head(h, torch.from_numpy(thin_keep(slot, slots)).to(device)).cpu().numpy()
            if (n + 1) % 50 == 0:
                print(f"{n + 1}/{len(ids)}; {time.monotonic() - started:.0f}s", flush=True)
    np.savez_compressed(OUT / "native_ablation.npz", ids=np.array(ids), variants=np.array(VARIANTS), logits=out, nwin=nwin)
    summary = {"studies": len(ids), "variants": VARIANTS, "nan_cells": int(np.isnan(out).sum()),
               "windows_per_slot_median": [float(v) for v in np.median(nwin, 0)],
               "weights_sha256": WEIGHTS_SHA, "seconds": round(time.monotonic() - started, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
