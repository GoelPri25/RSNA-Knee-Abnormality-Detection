"""r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells):
fused 0.9475 vs label-v2 r50 0.9425 (LB 0.937).

One ResNet-50 reads each study twice:
  global  the 94 eval windows over the valid range, 320 centre crop (label-v2 / E3 path) -> window attention head g;
  local   for each of the 10 cropped findings, 4 zoomed crops (192 px box, 128 px for medial meniscus / contusion,
          resampled to 384) run through the shared stem-layer3 (crop BN statistics) and the crops' own layer4 ->
          local head l, weighted by the window attention within the finding.
  fused   sigmoid((g + l) / 2) on the 10 cropped findings, sigmoid(g) on effusion / synovitis.
Crop placement is self-located (SL-0v4), no MS1: per finding the 4 windows with the largest global attention, on each the
gradient x activation centroid of that window's logit contribution over the 10 x 10 layer-4 cells.
Volume: E3 kernel's d96 builder, unchanged. fp16 autocast with an fp32 retry for non-finite logits; a study that
cannot be read keeps the sample_submission values.
"""
from __future__ import annotations

import copy
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
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IMG, CROP_MM, EVAL_RES, EVAL_WINDOWS = 384, 140.0, 320, 94
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(k for _, _, k in SLOTS)
BOX = {"ACL": 192, "MCL": 192, "Medial Meniscus": 128, "Lateral Meniscus": 192, "Medial OA": 192, "Lateral OA": 192,
       "PF OA": 192, "Baker's": 192, "Contusion": 128, "Fracture": 192}
CF = np.array([k for k, t in enumerate(TARGETS) if t in BOX])
BOXN = np.array([BOX.get(t, 0) for t in TARGETS])
CMASK = np.isin(np.arange(12), CF)
TOP, CROP_RES = 4, 384
WEIGHTS = {"dataset": "tumis001/rsna-surpass-f1-fold0", "file": "weights_swa21_25.pt",
           "sha256": "295b56ae275d8a640d54c1973b9cd2f23d992d26150df20ab9a88da0a0eedbd1"}
GLOBAL_ONLY = True   # SURPASS-F F1: fastMRI studies trained the global path only; submit the global head (no crop pass)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def competition_dir():
    return next(p for p in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                            Path("/kaggle/input/rsna-knee-abnormality-detection")) if p.exists())


def dataset_dir(ref):
    owner, slug = ref.split("/")
    return next(p for p in (Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input") / slug) if p.exists())


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------- d96 builder (E3 kernel, unchanged)
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
    study_id, rows, series_dir = job                     # series_dir: <competition>/test_series (train for checks)
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
            files = sorted((series_dir / study_id / row["SeriesInstanceUID"]).glob("*.dcm"))
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


def uniform_centres(mask):
    """94 centres over the whole valid range (coatnet_d96_runner.centres, eval), clipped to [1, MAXS - 2]."""
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, MAXS - 2))])
    return np.clip(cs[np.linspace(0, len(cs) - 1, EVAL_WINDOWS).round().astype(int)], 1, MAXS - 2)


# ---------------------------------------------------------------- model (r50_gl4_runner.RaptorGL4, arm l4copy)
class DualBN(nn.BatchNorm2d):
    """BatchNorm2d with a second set of running statistics for crop images (crop = True); affine shared."""

    def __init__(self, bn):
        super().__init__(bn.num_features, bn.eps, bn.momentum, bn.affine, bn.track_running_stats)
        self.register_buffer("crop_mean", torch.zeros(bn.num_features))
        self.register_buffer("crop_var", torch.ones(bn.num_features))
        self.register_buffer("crop_nbt", torch.zeros((), dtype=torch.long))
        self.crop = False

    def forward(self, x):
        if not self.crop:
            return super().forward(x)
        return F.batch_norm(x, self.crop_mean, self.crop_var, self.weight, self.bias, False, 0.0, self.eps)


def wrap_bns(module):
    out = []
    for name, ch in module.named_children():
        if isinstance(ch, nn.BatchNorm2d):
            d = DualBN(ch); setattr(module, name, d); out.append(d)
        else:
            out += wrap_bns(ch)
    return out


class V4L4Copy(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model("resnet50.tv_in1k", pretrained=False, num_classes=0)
        d = self.backbone.num_features
        self.norm = nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, d)); self.clsb = nn.Parameter(torch.zeros(12))
        self.bns = wrap_bns(self.backbone)
        assert len(self.bns) == 53
        self.layer4_c = copy.deepcopy(self.backbone.layer4)
        for m in self.layer4_c.modules():
            if isinstance(m, DualBN):
                m.crop = True
        self.norm_l = nn.LayerNorm(d)
        self.clsV = nn.Parameter(torch.zeros(12, d)); self.clsvb = nn.Parameter(torch.zeros(12))

    def crop_features(self, x):
        bb = self.backbone
        for m in self.bns:
            m.crop = True
        try:
            h = bb.maxpool(bb.act1(bb.bn1(bb.conv1(x))))
            h = self.layer4_c(bb.layer3(bb.layer2(bb.layer1(h))))
            return bb.forward_head(h)
        finally:
            for m in self.bns:
                m.crop = False


def load_model(path, device, sha256=None):
    if sha256 is not None and hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
        raise ValueError("Weights hash mismatch: v4 l4copy")
    model = V4L4Copy()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def windows(vol, cs, device):
    """[k] centre slices -> uint8 [k, 3, IMG, IMG] on device (slices c - 1, c, c + 1)."""
    return torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(device)


def normalise(x):
    return ((x - MEAN.to(x.device)) / STD.to(x.device)).contiguous(memory_format=torch.channels_last)


def locate(model, fm, a, cs):
    """selfloc_v4_gpu.locate on the global pass: fm [K, 2048, c, c] fp32, a [K, 12] -> (slices, rows, cols, weights)
    [10, TOP] for the cropped findings."""
    dev = fm.device
    W = model.clsW.float()[CF]
    ac = a[:, CF]
    top = ac.topk(TOP, dim=0).indices                                            # [TOP, 10]
    U = torch.unique(top); row = {int(k): j for j, k in enumerate(U.tolist())}
    with torch.enable_grad():
        t = fm[U].detach().clone().requires_grad_(True)
        contrib = ac[U] * (model.norm.float()(t.mean((2, 3))) @ W.T)          # [u, 10], a fixed
        c = t.shape[-1]; px = EVAL_RES / c; gi = torch.arange(c, device=dev, dtype=torch.float32)
        lo = (IMG - EVAL_RES) // 2
        S = np.zeros((len(CF), TOP), np.int64); R = S.copy(); Q = S.copy(); A = np.zeros((len(CF), TOP), np.float32)
        for j in range(len(CF)):
            gr, = torch.autograd.grad(contrib[:, j].sum(), t, retain_graph=j < len(CF) - 1)
            ks = top[:, j].tolist(); rows = [row[k] for k in ks]
            mp = (gr[rows] * t.detach()[rows]).sum(1).clamp_min(0)
            tot = mp.sum((1, 2)); ok = tot > 0; tot = tot.clamp_min(1e-12)
            cy = torch.where(ok, (mp.sum(2) * gi).sum(1) / tot, torch.full_like(tot, (c - 1) / 2))
            cx = torch.where(ok, (mp.sum(1) * gi).sum(1) / tot, torch.full_like(tot, (c - 1) / 2))
            S[j] = cs[ks]
            R[j] = np.round((lo + (cy + 0.5) * px).cpu().numpy()).astype(np.int64)
            Q[j] = np.round((lo + (cx + 0.5) * px).cpu().numpy()).astype(np.int64)
            A[j] = ac[top[:, j], j].detach().cpu().numpy()
    return S, R, Q, A


def predict_study(model, vol, mask, device, amp=True, table=None):
    """-> (fused, global, local) sigmoid [12] each; table (MS1-style slices, rows, cols, weights [10, TOP]) overrides
    self-location (test hook)."""
    cs = uniform_centres(mask)
    lo = (IMG - EVAL_RES) // 2
    xg = windows(vol, cs, device)[..., lo:lo + EVAL_RES, lo:lo + EVAL_RES].float().div_(255.0)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        fm = model.backbone.forward_features(normalise(xg))
    fm = fm.float()
    h = model.norm.float()(fm.mean((2, 3)))
    a = torch.softmax(model.att.float()(h), dim=0)                               # [K, 12]
    g = ((a.T @ h) * model.clsW.float()).sum(-1) + model.clsb.float()
    if GLOBAL_ONLY:
        return torch.stack([g, g, g]).sigmoid().cpu().numpy()
    S, R, Q, A = table if table is not None else locate(model, fm, a, cs)
    fid = np.repeat(CF, TOP)
    S, R, Q = S.ravel(), R.ravel(), Q.ravel()
    hb = BOXN[fid] // 2
    R, Q = np.clip(R, hb, IMG - hb), np.clip(Q, hb, IMG - hb)
    wt = (A / A.sum(1, keepdims=True)).ravel()
    xc = windows(vol, S, device).float().div_(255.0)
    s = torch.as_tensor(BOXN[fid] / IMG, dtype=torch.float32, device=device).view(-1, 1, 1)
    pxy = torch.stack([torch.as_tensor((Q + 0.5) / IMG * 2 - 1), torch.as_tensor((R + 0.5) / IMG * 2 - 1)], 1)
    th = torch.cat([s * torch.eye(2, device=device).expand(len(fid), 2, 2), pxy.float().to(device)[..., None]], 2)
    xc = F.grid_sample(xc, F.affine_grid(th, (len(fid), 3, CROP_RES, CROP_RES), align_corners=False), mode="bilinear",
                       padding_mode="zeros", align_corners=False)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        hc = model.crop_features(normalise(xc))
    hc = model.norm_l.float()(hc.float())
    fidt = torch.as_tensor(fid, device=device)
    lc = (hc * model.clsV.float()[fidt]).sum(-1) + model.clsvb.float()[fidt]
    l = torch.zeros(12, device=device).scatter_add(0, fidt, lc * torch.as_tensor(wt, device=device))
    cm = torch.as_tensor(CMASK, device=device)
    fused = torch.where(cm, 0.5 * (g + l), g)
    return torch.stack([fused, g, torch.where(cm, l, g)]).sigmoid().cpu().numpy()   # local: g where no crops


def ensure_timm():
    try:
        import timm
        if timm.__version__ == "1.0.29":
            return
    except ImportError:
        pass
    art = dataset_dir("tumis001/rsna-coatnet-d96-exp1b-fold0")                 # holds the timm 1.0.29 wheel
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "-q",
                    str(art / "timm-1.0.29-py3-none-any.whl")], check=True)


def main():
    started = time.monotonic()
    data, work = competition_dir(), Path("/kaggle/working")
    sample = read_csv(data / "sample_submission.csv")
    if not sample or list(sample[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("sample_submission.csv columns differ from the model target order")
    # EVAL VARIANT: the 883 training studies of tumis001/rsna-fast-eval/eval_ids.txt (gold-58 + fold 0), else unchanged
    ids = [s for s in (dataset_dir("tumis001/rsna-fast-eval") / "eval_ids.txt").read_text().split() if s]
    fallback = {s: [0.5] * 12 for s in ids}

    def write(preds):
        with (work / "submission.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["StudyInstanceUID", *TARGETS])
            for s in ids:
                w.writerow([s, *[f"{v:.8f}" for v in preds[s]]])
    write(fallback)                                          # valid file even if inference is interrupted

    ensure_timm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(dataset_dir(WEIGHTS["dataset"]) / WEIGHTS["file"], device, WEIGHTS["sha256"])
    print(f"Studies {len(ids)} | device {device} | v4 l4copy, " + ("global head only" if GLOBAL_ONLY else "self-located crops"), flush=True)
    by_study = {}
    for row in read_csv(data / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    preds, heads = {}, {}
    stats = {"studies": len(ids), "empty_studies": 0, "decode_errors": 0, "fp32_retries": 0}
    with Pool(4) as procs, torch.no_grad():
        for n, (sid, vol, mask, errors) in enumerate(
                procs.imap(build_study, [(s, by_study.get(s, []), data / "train_series") for s in ids]), 1):
            stats["decode_errors"] += errors
            if not mask.any():
                stats["empty_studies"] += 1
                continue
            p = predict_study(model, vol, mask, device)
            if not np.isfinite(p).all():
                stats["fp32_retries"] += 1
                p = predict_study(model, vol, mask, device, amp=False)
            if p.shape != (3, 12) or not np.isfinite(p).all():
                raise ValueError(f"Invalid prediction {sid}")
            preds[sid] = p[0].tolist(); heads[sid] = {"global": p[1].tolist(), "local": p[2].tolist()}
            if n % 50 == 0 or n == len(ids):
                print(f"v4 {n}/{len(ids)} studies; {time.monotonic() - started:.0f}s", flush=True)
    final = {s: preds.get(s, fallback[s]) for s in ids}
    write(final)
    written = read_csv(work / "submission.csv")
    vals = np.array([[float(r[t]) for t in TARGETS] for r in written])
    if [r["StudyInstanceUID"] for r in written] != ids or not np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
        raise ValueError("Saved submission is invalid")
    (work / "head_predictions.json").write_text(json.dumps(heads) + "\n")
    (work / "eval_predictions.json").write_text(json.dumps({"targets_in_order": TARGETS, "predictions": [
        {"study_id": s, "probabilities": final[s]} for s in ids if s in preds]}) + "\n")
    summary = {"model": "SURPASS-F F1: ra recipe (r50-GL v4 l4copy s2026 SWA 21-25, Raptor-only pure targets) + 9,196 fastMRI studies labelled by the Raptor teacher; global head only", "weights_sha256": WEIGHTS["sha256"],
               "stats": stats, "seconds": round(time.monotonic() - started, 1)}
    (work / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
