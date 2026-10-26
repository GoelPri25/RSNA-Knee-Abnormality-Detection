"""E3 labelset for the teammate (user request 2026-10-17): the lean E3 kernel (LB 0.944, rsna-lean-e3-d96-probability-ensemble)
run unchanged on the TRAINING DICOMs instead of the test set. Same d96 builder, same 4 members and weights (sha-checked), same
windows, routing, fp16/fp32 fallback and equal probability mean. Only the study list (train_series.csv, shard SHARD of NSHARDS)
and the DICOM folder (train_series/) differ. Reads no reports and no labels.

All four members trained on folds 1-4 (validated on fold 0), so predictions on fold 1-4 studies are IN-SAMPLE; only fold 0 and
the 58 gold studies are out-of-fold. The fold flag is added locally when the shards are merged.
Outputs: e3_train_predictions.csv (E3 mean), individual_predictions.json (4 members), inference_summary.json; saved every 250
studies so a time-out keeps the finished part.
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
import torch

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IMG, CROP_MM, EVAL_RES, EVAL_WINDOWS = 384, 140.0, 320, 94
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(k for _, _, k in SLOTS)
SLOT_OF = np.repeat(np.arange(5), [k for _, _, k in SLOTS])
BOUNDS = np.concatenate([[0], np.cumsum([k for _, _, k in SLOTS])])

D96_MODELS = [
    {"name": "r50_d96_labelv2", "arch": "resnet50.tv_in1k", "dataset": "priyanjaliml/rsna-knee-labelv2-checkpoint",
     "file": "r50_labelv2_swa21_25.pt", "sha256": "1c6006b148250edee2050acd833cc0c9e2cb697feb00084fb477da2982490c49"},
    {"name": "r50_d96_pp4_v2", "arch": "resnet50.tv_in1k", "dataset": "priyanjaliml/rsna-knee-pp4-v2-checkpoint",
     "file": "r50_pp4_v2_swa21_25.pt", "sha256": "876a796fbcdacb9502432862cc2ca573b1eb10ca4e7470aebac5ebd177a46f86",
     "pp4": True},
    {"name": "coatnet_d96_1b", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
     "dataset": "tumis001/rsna-coatnet-d96-exp1b-fold0",
     "file": "weights_swa20_23.pt", "sha256": "3a36fe4e1ccb9cb4aa3e3990fa7f0f53ed4b893f013fd7d11c6e7509869331f7"},
    {"name": "cnx_d96_p1", "arch": "convnext_tiny.fb_in22k_ft_in1k_384", "dataset": "tumis001/rsna-d96-cnxt-p1-pertarget-fold0",
     "file": "weights_swa21_25.pt", "sha256": "0125e766812169502ed70416811cef39ccc7ec15e17748b0a212f78b32919e86"},
]

DATA = next(p for p in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                        Path("/kaggle/input/rsna-knee-abnormality-detection")) if p.exists())


def dataset_dir(ref):
    owner, slug = ref.split("/")
    return next(p for p in (Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input") / slug) if p.exists())


ART = dataset_dir("tumis001/rsna-coatnet-d96-exp1b-fold0")                 # holds the timm 1.0.29 wheel


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
    """E1 kernel's builder plus the teammate's per-slot Fat_Suppression flag (volume identical)."""
    study_id, rows = job
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx, used, errors = 0, set(), 0
    fat = np.zeros(len(SLOTS), np.int64)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for j, (plane, fluid, k) in enumerate(SLOTS):
            cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
            pref = [r for r in cands if fluid in (0, 1) and int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
            row = (pref or cands or [None])[0]
            if row is None:
                idx += k
                continue
            used.add(row["SeriesInstanceUID"])
            fat[j] = 1 if str(row.get("Fat_Suppression")) == "1" else 0     # training manifest's rule
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
    return study_id, vol, mask, errors, fat


def uniform_centres(mask):
    """Raptor members: 94 centres over the whole valid range (coatnet_d96_runner eval)."""
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, MAXS - 2))])
    return np.clip(cs[np.linspace(0, len(cs) - 1, EVAL_WINDOWS).round().astype(int)], 1, MAXS - 2)


def slot_centres(mask):
    """4-network member (teammate kernel): every interior centre of each slot with >= 3 valid slices, in slot order;
    a study with no such slot falls back to the 94 uniform centres."""
    out = []
    for s in range(len(SLOTS)):
        valid = np.where(mask[BOUNDS[s]:BOUNDS[s + 1]])[0] + BOUNDS[s]
        if len(valid) >= 3:
            out.append(np.arange(valid.min() + 1, valid.max()))
    if out:
        return np.clip(np.concatenate(out), 1, MAXS - 2)
    return uniform_centres(mask)


def route(c, fat):
    """Network of each window (teammate's training rule): sagittal by fat-sat flag, coronal 2, axial 3."""
    sl = SLOT_OF[c]
    return np.where(sl <= 1, np.where(fat[sl] == 1, 0, 1), np.where(sl <= 3, 2, 3))


# ---------------------------------------------------------------- models
def ensure_timm():
    try:
        import timm
        if timm.__version__ == "1.0.29":
            return
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "-q",
                    str(ART / "timm-1.0.29-py3-none-any.whl")], check=True)


def build_d96_model(spec, device):
    import timm
    import torch.nn as nn

    class Raptor(nn.Module):
        def __init__(self):
            super().__init__()
            size = {"img_size": EVAL_RES} if spec["arch"].startswith("coatnet") else {}
            self.backbone = timm.create_model(spec["arch"], pretrained=False, num_classes=0, **size)
            d = self.backbone.num_features
            self.norm = nn.LayerNorm(d)
            self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
            self.clsW = nn.Parameter(torch.zeros(12, d))
            self.clsb = nn.Parameter(torch.zeros(12))

        def forward(self, x, plane=None):
            f = self.backbone(x)[None]
            h = self.norm(f)
            a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)
            return (torch.einsum("bkn,bkf->bnf", a, h) * self.clsW).sum(-1) + self.clsb

    G = 4

    class RaptorPP(nn.Module):
        """Teammate's RaptorPP (phase r50_pp4_v2), inference path, copied from their kernel unchanged."""

        def __init__(self):
            super().__init__()
            self.backbone = nn.ModuleList([timm.create_model(spec["arch"], pretrained=False, num_classes=0)
                                           for _ in range(G)])
            d = self.backbone[0].num_features
            self.pnorm = nn.ModuleList([nn.LayerNorm(d) for _ in range(G)])
            self.patt = nn.ModuleList([nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
                                       for _ in range(G)])
            self.clsW = nn.Parameter(torch.zeros(12, d)); self.clsb = nn.Parameter(torch.zeros(12))
            self.auxW = nn.Parameter(torch.zeros(G, 12, d)); self.auxb = nn.Parameter(torch.zeros(G, 12))
            self.gate = nn.Parameter(torch.zeros(G, 12))

        def forward(self, x, plane):
            """x [k, 3, H, W] (one study), plane [k] network per window -> logits [1, 12]."""
            f, z, keep = None, [], []
            for p in range(G):
                idx = (plane == p).nonzero(as_tuple=True)[0]
                if len(idx):
                    fp = self.pnorm[p](self.backbone[p](x[idx]))
                    if f is None:
                        f = torch.zeros(x.shape[0], fp.shape[1], device=x.device, dtype=fp.dtype)
                    f[idx] = fp
            f = f[None]; plane = plane[None]                                       # [1, k, d], [1, k]
            for p in range(G):
                m = plane == p
                s = self.patt[p](f).float().masked_fill(~m[..., None], -1e4)
                a = (torch.softmax(s, dim=1) * m[..., None]).to(f.dtype)
                z.append(torch.einsum("bkn,bkf->bnf", a, f)); keep.append(m.any(1))
            z = torch.stack(z, 1); keep = torch.stack(keep, 1)                     # [1, G, 12, d], [1, G]
            g = self.gate.float()[None].masked_fill(~keep[..., None], -1e4)
            wgt = torch.softmax(g, dim=1).to(z.dtype)
            return ((wgt[..., None] * z).sum(1) * self.clsW).sum(-1) + self.clsb

    path = dataset_dir(spec["dataset"]) / spec["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Weights hash mismatch: {spec['name']}")
    model = RaptorPP() if spec.get("pp4") else Raptor()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval().to(memory_format=torch.channels_last)


SHARD, NSHARDS = 0, 2


def main():
    started = time.monotonic()
    work = Path("/kaggle/working")
    by_study = {}
    for row in read_csv(DATA / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    ids = sorted(by_study)[SHARD::NSHARDS]

    ensure_timm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [(s, build_d96_model(s, device)) for s in D96_MODELS]
    print(f"Shard {SHARD}/{NSHARDS}: studies {len(ids)} of {len(by_study)} | device {device} | "
          f"members {[s['name'] for s in D96_MODELS]}", flush=True)
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    members = {s["name"]: {} for s in D96_MODELS}
    stats = {"shard": SHARD, "nshards": NSHARDS, "studies": len(ids), "empty_studies": 0, "decode_errors": 0,
             "fp32_fallbacks": 0, "pp4_windows": 0, "pp4_windows_per_network": [0] * 4, "fat_sag_studies": 0}
    use_amp = device.type == "cuda"
    lo = (IMG - EVAL_RES) // 2

    def prep(c, vol):
        x = torch.from_numpy(np.ascontiguousarray(vol[np.stack([c - 1, c, c + 1], 1)])).to(device).float().div_(255.0)
        return ((x[..., lo:lo + EVAL_RES, lo:lo + EVAL_RES] - mean_t) / std_t).contiguous(memory_format=torch.channels_last)

    def save(done):
        final = {s: np.mean(np.array([m[s] for m in members.values() if s in m], np.float64), 0).tolist()
                 for s in ids if any(s in m for m in members.values())}
        with (work / "e3_train_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["StudyInstanceUID", *TARGETS, "n_members"])
            for s, p in final.items():
                w.writerow([s, *[f"{v:.8f}" for v in p], sum(s in m for m in members.values())])
        (work / "individual_predictions.json").write_text(json.dumps(members) + "\n")
        summary = {"members": list(members), "combination": "equal arithmetic mean of 4 members' probabilities (E3)",
                   "weights_sha256": {s["name"]: s["sha256"] for s in D96_MODELS}, "studies_done": done,
                   "studies_predicted": len(final), "d96": stats, "seconds": round(time.monotonic() - started, 1)}
        (work / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    with Pool(4) as procs, torch.inference_mode():
        for n, (sid, vol, mask, errors, fat) in enumerate(
                procs.imap(build_study, [(s, by_study[s]) for s in ids]), 1):
            stats["decode_errors"] += errors
            if not mask.any():
                stats["empty_studies"] += 1
                continue                                     # every member skipped for this study
            x_uni = prep(uniform_centres(mask), vol)
            c_pp = slot_centres(mask)
            r_pp = route(c_pp, fat)
            x_pp, plane_pp = prep(c_pp, vol), torch.from_numpy(r_pp).to(device)
            stats["pp4_windows"] += len(r_pp); stats["fat_sag_studies"] += int(fat[:2].any())
            for g in range(4):
                stats["pp4_windows_per_network"][g] += int((r_pp == g).sum())
            for spec, model in models:
                x, plane = (x_pp, plane_pp) if spec.get("pp4") else (x_uni, None)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x, plane).float()
                if not torch.isfinite(logits).all():
                    stats["fp32_fallbacks"] += 1
                    logits = model(x, plane).float()
                p = logits.sigmoid()[0].cpu().numpy()
                if p.shape != (12,) or not np.isfinite(p).all():
                    raise ValueError(f"Invalid prediction {sid} {spec['name']}")
                members[spec["name"]][sid] = p.tolist()
            if n % 50 == 0 or n == len(ids):
                print(f"d96 {n}/{len(ids)} studies; {time.monotonic() - started:.0f}s", flush=True)
            if n % 250 == 0:
                save(n)

    print(json.dumps(save(len(ids)), indent=2), flush=True)

if __name__ == "__main__":
    main()
