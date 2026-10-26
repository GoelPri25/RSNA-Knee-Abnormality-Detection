"""Submit Model B2 alone (user request; below the pre-registered submit gate, LB read only): Swin3D-T with frozen
patch embedding + stages 1-2, one stage-4 branch shared by all planes, per-finding attention over every stage-4 token,
label-v2, SWA of epochs 11-15 (fold 0: clear6 0.9542, 10-target 0.8941, val12 0.8911; ledger B2). Same as the B1
kernel except the model (one branch, token pooling) and the weights.

Preprocessing is the d96n builder (kaggle/submission_kernels/d96_r50_native_ablation/run.py, the same code as the
training cache kernels) applied to test_series: up to 4 series (sagittal fluid, sagittal non-fluid, coronal fluid,
axial fluid-preferred), every slice in 2..98% capped at 40 per series, per-series 2/98 window, 140 mm centre crop,
INTER_AREA to 384. Model and eval match scripts/swin3d_d96n_runner.py (SwinB, Prep eval path, predict): each series
one [1, 3, T, 320, 320] volume (centre 320 crop of 384, grey x3, Kinetics normalisation), no TTA. Training and the
fold-0 predictions ran in bf16 on an H100; this runs in fp32 (no T4 bf16) with torchvision's own attention, which
the runner's fused attention was benchmarked equal to (max |d logit| 7e-5 fp32).

Parity check first: 8 fold-0 studies rebuilt from train_series and compared with the saved Modal predictions; the
run stops if any probability differs by more than PARITY_TOL.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

WEIGHTS_SHA = "7853700f70726a33f174137b6e442a0d7064dbf0c0967f2d4a57849552acbe9a"
IMG, CROP_MM, CAP, MIN_ROWS, RES = 384, 140.0, 40, 3, 320
SLOTS = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Axial", 1)]
PARITY_TOL = 0.03
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def pick(paths):
    found = [p for p in paths if p.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one input among {paths}; found {found}")
    return found[0]


DATA = pick([Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
             Path("/kaggle/input/rsna-knee-abnormality-detection")])
ART = pick([Path("/kaggle/input/datasets/tumis001/rsna-d96n-swin3d-b2-fold0"),
            Path("/kaggle/input/rsna-d96n-swin3d-b2-fold0")])
OUT = Path("/kaggle/working/submission.csv")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------- d96n builder (same as the training cache)
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


def native_picks(n):
    """Every slice in 2%..98% (same bounds as d96); evenly resampled to CAP if more."""
    if n <= 1:
        return [0] * n
    lo, hi = int(n * 0.02), max(int(n * 0.98) - 1, int(n * 0.02))
    if hi - lo + 1 <= CAP:
        return list(range(lo, hi + 1))
    return [int(p) for p in np.linspace(lo, hi, CAP).round().astype(int)]


def build_study(job):
    split, study_id, rows = job
    slices, slot_of, used, errors = [], [], set(), 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for j, (plane, fluid) in enumerate(SLOTS):
            cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
            pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
            row = (pref or cands or [None])[0]
            if row is None:
                continue
            used.add(row["SeriesInstanceUID"])
            files = sorted((DATA / split / study_id / row["SeriesInstanceUID"]).glob("*.dcm"))
            heads = list(pool.map(header, files))
            ok_ps = [h[2] for h in heads if h[3]]
            med_ps = float(np.median(ok_ps)) if ok_ps else 0.5
            heads.sort(key=lambda h: h[0])
            picks = native_picks(len(heads))
            if not picks:
                continue

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
                if a is None:
                    slices.append(np.zeros((IMG, IMG), np.uint8))
                else:
                    aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                    slices.append((crop_resize(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8))
                slot_of.append(j)
    while len(slices) < MIN_ROWS:                     # same padding as the cache (slot -1 is ignored by the model)
        slices.append(np.zeros((IMG, IMG), np.uint8)); slot_of.append(-1)
    return study_id, np.stack(slices), np.array(slot_of), errors


# ---------------------------------------------------------------- model (scripts/swin3d_d96n_runner.py SwinB, eval)
def build_model(device):
    import torch
    import torch.nn as nn
    from torchvision.models.video import swin3d_t

    class SwinB(nn.Module):
        def __init__(self):
            super().__init__()
            net = swin3d_t(weights=None)
            f = net.features
            assert len(f) == 7 and type(f[5]).__name__ == "PatchMerging" and len(f[6]) == 2
            self.patch_embed, self.pos_drop = net.patch_embed, net.pos_drop
            self.shared = f[:5]
            self.branch = nn.ModuleList([nn.Sequential(copy.deepcopy(f[5]), copy.deepcopy(f[6]),
                                                       copy.deepcopy(net.norm)) for _ in range(1)])
            d = net.num_features
            self.norm = nn.LayerNorm(d)
            self.series = nn.Embedding(4, d)
            self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
            self.clsW = nn.Parameter(torch.zeros(12, d))
            self.clsb = nn.Parameter(torch.zeros(12))

        def encode(self, v, j):
            h = self.pos_drop(self.patch_embed(v))
            for m in self.shared:
                h = m(h)
            for m in self.branch[0]:                                 # one branch shared by all planes
                h = m(h)
            return h.flatten(0, 3)                                   # every stage-4 token: [T/2 * 10 * 10, 768]

        def forward(self, series):
            if not series:
                return self.clsb[None]
            toks = [self.encode(v, j) for j, v in series]
            ids = torch.cat([torch.full((len(t),), j, device=t.device) for t, (j, _) in zip(toks, series)])
            h = self.norm(torch.cat(toks)) + self.series(ids)
            a = torch.softmax(self.att(h).float(), dim=0).to(h.dtype)
            return ((torch.einsum("nk,nf->kf", a, h) * self.clsW).sum(-1) + self.clsb)[None]

    path = ART / "weights_swa11_15.pt"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != WEIGHTS_SHA:
        raise ValueError(f"Unexpected weights SHA-256: {got}")
    model = SwinB()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def kinetics_norm():
    from torchvision.models.video import Swin3D_T_Weights
    t = Swin3D_T_Weights.KINETICS400_V1.transforms()
    return list(t.mean), list(t.std)


def main():
    started = time.monotonic()
    import torch
    import torchvision
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    km, ks = kinetics_norm()
    mean = torch.tensor(km, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(ks, device=device).view(1, 3, 1, 1, 1)
    lo = (IMG - RES) // 2
    print(f"torch {torch.__version__} torchvision {torchvision.__version__} | device {device} | "
          f"kinetics mean {km} std {ks}", flush=True)

    def predict(vol, slot_of):
        x = torch.from_numpy(vol).to(device).float().div_(255.0)                  # [n, 384, 384]
        series = []
        for j in range(4):
            rows = np.where(slot_of == j)[0]
            if len(rows) == 0:
                continue
            v = x[int(rows[0]):int(rows[-1]) + 1, lo:lo + RES, lo:lo + RES]
            series.append((j, (v[None, None].expand(1, 3, *v.shape) - mean) / std))
        p = model(series).float().sigmoid()[0].cpu().numpy()
        if p.shape != (12,) or not np.isfinite(p).all():
            raise ValueError("Invalid predictions")
        return p

    # ---- parity: fold-0 studies from train_series vs the saved Modal predictions
    ref = json.loads((ART / "parity_fold0.json").read_text())
    order = [ref["targets_in_order"].index(t) for t in TARGETS]
    train_rows = {}
    for row in read_csv(DATA / "train_series.csv"):
        train_rows.setdefault(row["StudyInstanceUID"], []).append(row)
    diffs = []
    with Pool(4) as procs, torch.inference_mode():
        jobs = [("train_series", r["study_id"], train_rows.get(r["study_id"], [])) for r in ref["predictions"]]
        for (s, vol, slot_of, _), r in zip(procs.imap(build_study, jobs), ref["predictions"]):
            d = float(np.abs(predict(vol, slot_of) - np.array(r["probabilities"])[order]).max())
            diffs.append(d)
            print(f"parity {s[-12:]}: rows {len(vol)}, max |dp| {d:.4f}", flush=True)
    if max(diffs) > PARITY_TOL:
        raise ValueError(f"Parity check failed: max |dp| {max(diffs):.4f} > {PARITY_TOL}")

    # ---- submission
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
    print(f"Studies {len(ids)} | data {DATA} | artifacts {ART}", flush=True)
    preds, stats = {}, {"studies": len(ids), "empty_studies": 0, "decode_errors": 0}
    with Pool(4) as procs, torch.inference_mode():
        for n, (s, vol, slot_of, errors) in enumerate(
                procs.imap(build_study, [("test_series", s, by_study.get(s, [])) for s in ids]), 1):
            stats["decode_errors"] += errors
            if not (slot_of >= 0).any():
                stats["empty_studies"] += 1
            preds[s] = predict(vol, slot_of).tolist()
            if n % 50 == 0 or n == len(ids):
                print(f"{n}/{len(ids)} studies; {time.monotonic() - started:.0f}s", flush=True)

    write(preds)
    written = read_csv(OUT)
    vals = np.array([[float(r[t]) for t in TARGETS] for r in written])
    if [r["StudyInstanceUID"] for r in written] != ids or not np.isfinite(vals).all() \
            or vals.min() < 0 or vals.max() > 1:
        raise ValueError("Saved submission is invalid")
    summary = {**stats, "device": str(device), "weights_sha256": WEIGHTS_SHA, "precision": "fp32",
               "parity_max_abs_dp": round(max(diffs), 5), "parity_studies": len(diffs),
               "model": "Model B2 Swin3D-T frozen stages 1-2, shared stage 4, token attention, d96n, SWA 11-15, fold-0 training studies, label-v2",
               "fold0_val12": 0.8911, "fold0_clear6": 0.9542, "fold0_10target": 0.8941,
               "seconds": round(time.monotonic() - started, 1)}
    Path("/kaggle/working/inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
