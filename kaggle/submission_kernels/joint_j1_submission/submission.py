"""J1-sub (docs/experiment_ledger.md row J1-sub; post-hoc LB read of Exp J1, user decision 2026-10-16): the J1 joint
model's fused head as a single-model submission. One network: ResNet-50 branch (Exp 1e, d96 windows) + Swin3D-T branch
(Model B2, d96n volumes), per-finding attention over the union of both branches' tokens; label-v2, SWA 16-20.
Fold 0: clear6 0.9725 / 10-target 0.9084.

Preprocessing, both unchanged from kernels already used for submissions:
  * d96 (ResNet branch): the E1 kernel's builder; 94 windows (c-1, c, c+1) over the valid range, centre 320 crop,
    ImageNet normalisation; ResNet backbone in fp16 autocast (as E1), its LayerNorm and everything after in fp32.
  * d96n (Swin branch): the B2 kernel's builder and eval path; per series one [1, 3, T, 320, 320] volume, Kinetics
    normalisation, fp32 with TF32 off (as B2).
Fusion head as scripts/joint_r50_swin_runner.py JointNet.forward at eval (no branch dropout), in fp32. No TTA.
Training ran in bf16 on an H100, so a parity gate runs first: 8 fold-0 studies rebuilt from train_series must reproduce
the saved fused predictions within PARITY_TOL, else the run stops before any test inference.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

WEIGHTS_SHA = "d98dea2d7c52673121e9e604cb6dc1cbdaadf26b27dba1e3fda4f205e05b66ed"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IMG, CROP_MM, RES = 384, 140.0, 320
PARITY_TOL = 0.03
# d96 (ResNet branch)
EVAL_WINDOWS = 94
SLOTS96 = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(k for _, _, k in SLOTS96)
# d96n (Swin branch)
CAP, MIN_ROWS = 40, 3
SLOTSN = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Axial", 1)]


def pick(paths):
    found = [p for p in paths if p.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one input among {paths}; found {found}")
    return found[0]


DATA = pick([Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
             Path("/kaggle/input/rsna-knee-abnormality-detection")])
ART = pick([Path("/kaggle/input/datasets/tumis001/rsna-joint-j1-fold0"), Path("/kaggle/input/rsna-joint-j1-fold0")])
WHEEL = pick([Path("/kaggle/input/datasets/tumis001/rsna-coatnet-d96-exp1b-fold0"),
              Path("/kaggle/input/rsna-coatnet-d96-exp1b-fold0")]) / "timm-1.0.29-py3-none-any.whl"
OUT = Path("/kaggle/working/submission.csv")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------- shared DICOM helpers (both cache builders)
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


# ---------------------------------------------------------------- d96 builder (E1 kernel, split-aware)
def build_d96(split, study_id, rows, pool):
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx, used, errors = 0, set(), 0
    for plane, fluid, k in SLOTS96:
        cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
        pref = [r for r in cands if fluid in (0, 1) and int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        row = (pref or cands or [None])[0]
        if row is None:
            idx += k
            continue
        used.add(row["SeriesInstanceUID"])
        files = sorted((DATA / split / study_id / row["SeriesInstanceUID"]).glob("*.dcm"))
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
    return vol, mask, errors


def windows(vol, mask):
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, MAXS - 2))])
    c = np.clip(cs[np.linspace(0, len(cs) - 1, EVAL_WINDOWS).round().astype(int)], 1, MAXS - 2)
    return np.ascontiguousarray(vol[np.stack([c - 1, c, c + 1], 1)])           # [94, 3, 384, 384]


# ---------------------------------------------------------------- d96n builder (B2 kernel)
def native_picks(n):
    if n <= 1:
        return [0] * n
    lo, hi = int(n * 0.02), max(int(n * 0.98) - 1, int(n * 0.02))
    if hi - lo + 1 <= CAP:
        return list(range(lo, hi + 1))
    return [int(p) for p in np.linspace(lo, hi, CAP).round().astype(int)]


def build_d96n(split, study_id, rows, pool):
    slices, slot_of, used, errors = [], [], set(), 0
    for j, (plane, fluid) in enumerate(SLOTSN):
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
    while len(slices) < MIN_ROWS:
        slices.append(np.zeros((IMG, IMG), np.uint8)); slot_of.append(-1)
    return np.stack(slices), np.array(slot_of), errors


def build_both(job):
    split, study_id, rows = job
    with ThreadPoolExecutor(max_workers=8) as pool:
        vol, mask, e1 = build_d96(split, study_id, rows, pool)
        nvol, slot_of, e2 = build_d96n(split, study_id, rows, pool)
    return study_id, vol, mask, nvol, slot_of, e1 + e2


# ---------------------------------------------------------------- model
def ensure_timm():
    try:
        import timm
        if timm.__version__ == "1.0.29":
            return
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "-q", str(WHEEL)], check=True)


def build_model(device):
    import timm
    import torch.nn as nn
    from torchvision.models.video import swin3d_t

    class Raptor(nn.Module):                                   # Exp 1e (coatnet_d96_runner Raptor, no context)
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model("resnet50.tv_in1k", pretrained=False, num_classes=0)
            d = self.backbone.num_features
            self.norm = nn.LayerNorm(d)
            self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
            self.clsW = nn.Parameter(torch.zeros(12, d))
            self.clsb = nn.Parameter(torch.zeros(12))

    class SwinB(nn.Module):                                    # Model B2 (B2 kernel)
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
            for m in self.branch[0]:
                h = m(h)
            return h.flatten(0, 3)                                   # [T/2 * 10 * 10, 768]

    class JointNet(nn.Module):                                 # scripts/joint_r50_swin_runner.py JointNet, eval
        def __init__(self):
            super().__init__()
            self.R, self.S, D = Raptor(), SwinB(), 512
            self.pr = nn.Sequential(nn.Linear(self.R.backbone.num_features, D), nn.LayerNorm(D))
            self.ps = nn.Sequential(nn.Linear(self.S.clsW.shape[1], D), nn.LayerNorm(D))
            self.kind = nn.Parameter(torch.zeros(2, D))
            self.att = nn.Sequential(nn.Linear(D, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
            self.gate = nn.Parameter(torch.zeros(2, 12))
            self.clsW = nn.Parameter(torch.zeros(12, D)); self.clsb = nn.Parameter(torch.zeros(12))

        def forward(self, xr, series, use_amp):
            """xr [94, 3, 320, 320] one study; series [(slot, [1, 3, T, 320, 320])] -> fused, ResNet-head, Swin-head
            logits [12] and the fused attention mass on ResNet tokens [12]."""
            R, S = self.R, self.S
            with torch.autocast(xr.device.type, dtype=torch.float16, enabled=use_amp):
                f = R.backbone(xr)
            hr = R.norm(f.float())                                               # [K, 2048]
            ar = torch.softmax(R.att(hr), dim=0)
            lr_ = (torch.einsum("kn,kf->nf", ar, hr) * R.clsW).sum(-1) + R.clsb
            ur = self.pr(hr) + self.kind[0]
            parts, ls = [(0, ur)], S.clsb
            if series:
                toks = [S.encode(v, j) for j, v in series]
                ids = torch.cat([torch.full((len(t),), j, device=t.device) for t, (j, _) in zip(toks, series)])
                hs = S.norm(torch.cat(toks)) + S.series(ids)
                a_s = torch.softmax(S.att(hs), dim=0)
                ls = (torch.einsum("nk,nf->kf", a_s, hs) * S.clsW).sum(-1) + S.clsb
                parts.append((1, self.ps(hs) + self.kind[1]))
            u = torch.cat([q for _, q in parts])
            prior = torch.cat([(self.gate[k] - math.log(len(q))).expand(len(q), 12) for k, q in parts])
            a = torch.softmax(self.att(u) + prior, dim=0)                         # [K + N, 12]
            lf = (torch.einsum("nk,nf->kf", a, u) * self.clsW).sum(-1) + self.clsb
            return lf, lr_, ls, a[:len(ur)].sum(0)

    path = ART / "weights_swa16_20.pt"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != WEIGHTS_SHA:
        raise ValueError(f"Unexpected weights SHA-256: {got}")
    model = JointNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    model.R.backbone.to(memory_format=torch.channels_last)
    return model


def main():
    started = time.monotonic()
    ensure_timm()
    import torchvision
    from torchvision.models.video import Swin3D_T_Weights
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = build_model(device)
    kt = Swin3D_T_Weights.KINETICS400_V1.transforms()
    kmean = torch.tensor(list(kt.mean), device=device).view(1, 3, 1, 1, 1)
    kstd = torch.tensor(list(kt.std), device=device).view(1, 3, 1, 1, 1)
    imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    istd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    lo = (IMG - RES) // 2
    print(f"torch {torch.__version__} torchvision {torchvision.__version__} | device {device}", flush=True)

    def predict(vol, mask, nvol, slot_of):
        xr = torch.from_numpy(windows(vol, mask)).to(device).float().div_(255.0)
        xr = ((xr[..., lo:lo + RES, lo:lo + RES] - imean) / istd).contiguous(memory_format=torch.channels_last)
        x = torch.from_numpy(nvol).to(device).float().div_(255.0)
        series = []
        for j in range(4):
            rows = np.where(slot_of == j)[0]
            if len(rows):
                v = x[int(rows[0]):int(rows[-1]) + 1, lo:lo + RES, lo:lo + RES]
                series.append((j, (v[None, None].expand(1, 3, *v.shape) - kmean) / kstd))
        lf, lr_, ls, sh = model(xr, series, use_amp)
        out = [t.float().sigmoid().cpu().numpy() for t in (lf, lr_, ls)] + [sh.float().cpu().numpy()]
        if any(o.shape != (12,) or not np.isfinite(o).all() for o in out):
            raise ValueError("Invalid predictions")
        return out

    # ---- parity: 8 fold-0 studies from train_series vs the saved Modal (bf16) predictions
    ref = json.loads((ART / "parity_fold0.json").read_text())
    order = [ref["targets_in_order"].index(t) for t in TARGETS]
    train_rows = {}
    for row in read_csv(DATA / "train_series.csv"):
        train_rows.setdefault(row["StudyInstanceUID"], []).append(row)
    diffs = {"fused": [], "r50": [], "swin": []}
    with Pool(4) as procs, torch.inference_mode():
        jobs = [("train_series", r["study_id"], train_rows.get(r["study_id"], [])) for r in ref["predictions"]]
        for (s, vol, mask, nvol, slot_of, _), r in zip(procs.imap(build_both, jobs), ref["predictions"]):
            pf, pr_, ps_, _ = predict(vol, mask, nvol, slot_of)
            for k, p in (("fused", pf), ("r50", pr_), ("swin", ps_)):
                diffs[k].append(float(np.abs(p - np.array(r[k])[order]).max()))
            print(f"parity {s[-12:]}: max |dp| fused {diffs['fused'][-1]:.4f} r50 {diffs['r50'][-1]:.4f} "
                  f"swin {diffs['swin'][-1]:.4f}", flush=True)
    if max(diffs["fused"]) > PARITY_TOL:
        raise ValueError(f"Parity check failed: fused max |dp| {max(diffs['fused']):.4f} > {PARITY_TOL}")

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
    preds, heads = {}, {"r50_head": {}, "swin_head": {}, "r50_share": {}}
    stats = {"studies": len(ids), "empty_studies": 0, "no_swin_series": 0, "decode_errors": 0}
    with Pool(4) as procs, torch.inference_mode():
        for n, (s, vol, mask, nvol, slot_of, errors) in enumerate(
                procs.imap(build_both, [("test_series", s, by_study.get(s, [])) for s in ids]), 1):
            stats["decode_errors"] += errors
            if not mask.any() and not (slot_of >= 0).any():
                stats["empty_studies"] += 1
                preds[s] = fallback[s]
                continue
            stats["no_swin_series"] += int(not (slot_of >= 0).any())
            pf, pr_, ps_, sh = predict(vol, mask, nvol, slot_of)
            preds[s] = pf.tolist()
            heads["r50_head"][s], heads["swin_head"][s], heads["r50_share"][s] = pr_.tolist(), ps_.tolist(), sh.tolist()
            if n % 50 == 0 or n == len(ids):
                print(f"{n}/{len(ids)} studies; {time.monotonic() - started:.0f}s", flush=True)

    write(preds)
    written = read_csv(OUT)
    vals = np.array([[float(r[t]) for t in TARGETS] for r in written])
    if [r["StudyInstanceUID"] for r in written] != ids or not np.isfinite(vals).all() \
            or vals.min() < 0 or vals.max() > 1:
        raise ValueError("Saved submission is invalid")
    Path("/kaggle/working/branch_heads.json").write_text(json.dumps(heads, indent=1) + "\n")
    summary = {**stats, "device": str(device), "weights_sha256": WEIGHTS_SHA,
               "precision": "ResNet backbone fp16 autocast, Swin + heads fp32",
               "parity_max_abs_dp": {k: round(max(v), 5) for k, v in diffs.items()}, "parity_studies": len(diffs["fused"]),
               "model": "Exp J1 fused head: ResNet-50 (1e, d96) + Swin3D-T (B2, d96n), combined per-finding attention, label-v2, SWA 16-20, fold-0 training studies",
               "fold0_clear6": 0.9725, "fold0_10target": 0.9084, "seconds": round(time.monotonic() - started, 1)}
    Path("/kaggle/working/inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
