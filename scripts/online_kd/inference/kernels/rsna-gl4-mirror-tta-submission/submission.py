"""Anatomical-mirror TTA submission for one GL4 student: each test study is read twice, as is and anatomically mirrored
(the left / right mirror the model saw in training with probability 0.5: sagittal windows reverse their 3 slice
channels, coronal / axial windows flip width; the fused head also self-locates and crops on the mirrored view), and the
submission is the mean of the two probabilities, as the runner's val_swa_mirror.

Earlier evidence: Exp 1e (r50 d96) mirror TTA was +0.0009 on fold 0 and unchanged on the LB (0.936). Not yet measured
for the E4 recipe or ConvNeXt-S: run MODE "fold0" first and score the three JSONs with score_fold0.py. The mirror
doubles the inference time (efficiency track: ~0.00087 AUC per extra minute).

MODE "test" writes submission.csv (COMBINE "prob" = mean of probabilities, "logit" = mean of logits). MODE "fold0"
predicts the fold-0 training studies from DICOM and writes fold0_<name>_{plain,mirror,tta}.json (a commit run).
Before the test set the model re-predicts FID_N fold-0 studies (plain view, global head) against its run's saved
val_swa_global.json (max |dp| < FID_MAX).
Built by build_kernels.py (the GL4 core is inlined below); edit the source, not the built kernel.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- GL4 inference core (inlined by build_kernels.py)
"""Shared inference code for the r50-GL4 l4copy students (E4 family: ResNet-50; KD-E4CNXS: ConvNeXt-S 22k).

Volume: the E3 / E4 kernels' d96 builder, unchanged (5 slots, 96 slices over 2-98%, per-series 2/98 window, 140 mm centre
crop, INTER_AREA to 384). Model: r50_gl4_runner.RaptorGL4 (arm l4copy; BB-CMP for non-ResNet backbones: the crop branch
swaps in its own copy of the last stage). Global = 94 eval windows, 320 centre crop, window attention head; fused =
sigmoid((g + l) / 2) on the 10 cropped findings with self-located crops (SL-0v4), as the E4 submission kernel.

Anatomical mirror (the training augmentation, mirror_p 0.5 in every E4-family run, = coatnet_d96_runner.mirror):
sagittal windows reverse their 3 slice channels, coronal / axial windows flip width. Under mirror=True the whole study
is read mirrored: the global windows, the self-location on them, and the crops cut from the mirrored windows.
"""
import copy
import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IMG, CROP_MM, EVAL_WINDOWS = 384, 140.0, 94
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(k for _, _, k in SLOTS)
SAGITTAL = np.repeat([p == "Sagittal" for p, _, _ in SLOTS], [k for _, _, k in SLOTS])   # [96] slice -> sagittal slot
BOX = {"ACL": 192, "MCL": 192, "Medial Meniscus": 128, "Lateral Meniscus": 192, "Medial OA": 192, "Lateral OA": 192,
       "PF OA": 192, "Baker's": 192, "Contusion": 128, "Fracture": 192}
CF = np.array([k for k, t in enumerate(TARGETS) if t in BOX])
BOXN = np.array([BOX.get(t, 0) for t in TARGETS])
CMASK = np.isin(np.arange(12), CF)
TOP, CROP_RES, WIN_CHUNK = 4, 384, 24
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# r50_gl4_runner.BACKBONES (+ apply_cnxs_patch.py): timm arch, the crop branch's own copies, BatchNorm count
# rap2 = PUB-KD rap2-D (nartaa's CoAtNet-rmlp-2 384, teammate kernel pubkd_rap2d_global_submission): globals at 384, no
# centre crop, 25 BatchNorms (timm BatchNormAct2d), global head only in these kernels.
ARCHS = {"r50": {"timm": "resnet50.tv_in1k", "last": ("layer4",), "bns": 53, "eval_res": 320},
         "cnxs": {"timm": "convnext_small.fb_in22k_ft_in1k", "last": ("stages.3", "head.norm"), "bns": 0, "eval_res": 320},
         "r101": {"timm": "resnet101.tv_in1k", "last": ("layer4",), "bns": 104, "eval_res": 320},   # BB-CMP crop copy
         "bit101": {"timm": "resnetv2_101x1_bit.goog_in21k_ft_in1k", "last": ("stages.3", "norm"), "bns": 0,
                    "eval_res": 320, "norm": [0.5, 0.5]},
         "rap2": {"timm": "coatnet_rmlp_2_rw_384", "last": ("stages.3", "norm"), "bns": 25, "eval_res": 384,
                  "kw": {"img_size": IMG}, "global_only": True}}


def competition_dir():
    return next(p for p in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                            Path("/kaggle/input/rsna-knee-abnormality-detection")) if p.exists())


def dataset_dir(ref):
    owner, slug = ref.split("/")
    return next(p for p in (Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input") / slug) if p.exists())


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


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
        self.post = nn.Sequential(bn.drop, bn.act) if hasattr(bn, "act") else None
        self.crop = False

    def forward(self, x):
        if not self.crop:
            y = super().forward(x)
        else:
            y = F.batch_norm(x, self.crop_mean, self.crop_var, self.weight, self.bias, False, 0.0, self.eps)
        return y if self.post is None else self.post(y)


def wrap_bns(module):
    out = []
    for name, ch in module.named_children():
        if isinstance(ch, nn.BatchNorm2d):
            d = DualBN(ch); setattr(module, name, d); out.append(d)
        else:
            out += wrap_bns(ch)
    return out


def _get(mod, path):
    for p in path.split("."):
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return mod


def _set(mod, path, new):
    *a, last = path.split(".")
    setattr(_get(mod, ".".join(a)) if a else mod, last, new)


class GL4(nn.Module):
    """RaptorGL4 l4copy. crops=False builds the global path only (a global-only checkpoint, or head "global")."""

    def __init__(self, arch, crops=True):
        super().__init__()
        import timm
        spec = ARCHS[arch]
        self.arch, self.crops, self.last, self.eval_res = arch, crops, spec["last"], spec["eval_res"]
        mean, std = ([spec["norm"][0]] * 3, [spec["norm"][1]] * 3) if "norm" in spec else (MEAN.flatten().tolist(), STD.flatten().tolist())
        self.register_buffer("in_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.backbone = timm.create_model(spec["timm"], pretrained=False, num_classes=0, **spec.get("kw", {}))
        bb, d = self.backbone, self.backbone.num_features
        self.norm = nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, d)); self.clsb = nn.Parameter(torch.zeros(12))
        self.bns = wrap_bns(bb)
        assert len(self.bns) == spec["bns"], (arch, len(self.bns))
        if not crops:
            return
        if arch == "r50":                                    # crop-only layer4, own (crop) BN statistics
            self.layer4_c = copy.deepcopy(bb.layer4)
            for m in self.layer4_c.modules():
                if isinstance(m, DualBN):
                    m.crop = True
        else:                                                # BB-CMP: crop-only copy of the last stage, swapped in
            self.crop_parts = nn.ModuleDict({p.replace(".", "_"): copy.deepcopy(_get(bb, p)) for p in self.last})
            for m in self.crop_parts.modules():
                if isinstance(m, DualBN):
                    m.crop = True
        self.norm_l = nn.LayerNorm(d)
        self.clsV = nn.Parameter(torch.zeros(12, d)); self.clsvb = nn.Parameter(torch.zeros(12))

    def pool(self, fm):
        """Feature map -> the runner's window feature bb(x) (r50: average pool; ConvNeXt: pool + head.norm)."""
        return self.backbone.forward_head(fm)

    def crop_features(self, x):
        bb = self.backbone
        for m in self.bns:
            m.crop = True
        try:
            if self.arch == "r50":
                h = bb.maxpool(bb.act1(bb.bn1(bb.conv1(x))))
                return bb.forward_head(self.layer4_c(bb.layer3(bb.layer2(bb.layer1(h)))))
            orig = {p: _get(bb, p) for p in self.last}
            try:
                for p in self.last:
                    _set(bb, p, self.crop_parts[p.replace(".", "_")])
                return bb(x)
            finally:
                for p, mod in orig.items():
                    _set(bb, p, mod)
        finally:
            for m in self.bns:
                m.crop = False


CROP_KEYS = ("layer4_c.", "crop_parts.", "norm_l.", "clsV", "clsvb")


def load_model(spec, device):
    """spec: {"arch", "path", "sha256", "head"}. Head "global" (or a global-only checkpoint) builds the global path only;
    the crop-branch keys it drops are checked to be exactly the crop branch."""
    path = Path(spec["path"])
    if spec.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Weights hash mismatch: {spec['name']}")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    has_crops = any(k.startswith(CROP_KEYS) for k in sd)
    crops = has_crops and spec.get("head", "fused") != "global"
    if spec.get("head", "fused") != "global" and (not has_crops or ARCHS[spec["arch"]].get("global_only")):
        raise ValueError(f"{spec['name']}: head {spec.get('head')} needs the crop branch (global-only checkpoint or arch)")
    model = GL4(spec["arch"], crops=crops)
    if not crops:
        dropped = [k for k in sd if k.startswith(CROP_KEYS)]
        sd = {k: v for k, v in sd.items() if not k.startswith(CROP_KEYS)}
        if dropped:
            print(f"{spec['name']}: global head only, {len(dropped)} crop-branch tensors not loaded", flush=True)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=True)
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def windows(vol, cs, device, mirror=False):
    """[k] centre slices -> uint8 [k, 3, IMG, IMG] on device (slices c - 1, c, c + 1); mirror = anatomical mirror."""
    x = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(device)
    if mirror:
        sag = torch.as_tensor(SAGITTAL[cs], device=device).view(-1, 1, 1, 1)
        x = torch.where(sag, x.flip(1), x.flip(-1))
    return x


def normalise(x, model):
    """The checkpoint's input normalisation (ImageNet mean / std, or 0.5 / 0.5 for BiT)."""
    return ((x - model.in_mean) / model.in_std).contiguous(memory_format=torch.channels_last)


def locate(model, fm, a, cs):
    """selfloc_v4_gpu.locate on the global pass: fm [K, d, c, c] fp32, a [K, 12] -> (slices, rows, cols, weights)
    [10, TOP] for the cropped findings."""
    dev = fm.device
    W = model.clsW.float()[CF]
    ac = a[:, CF]
    top = ac.topk(TOP, dim=0).indices                                            # [TOP, 10]
    U = torch.unique(top); row = {int(k): j for j, k in enumerate(U.tolist())}
    with torch.enable_grad():
        t = fm[U].detach().clone().requires_grad_(True)
        contrib = ac[U] * (model.norm.float()(model.pool(t)) @ W.T)             # [u, 10], a fixed
        c = t.shape[-1]; px = model.eval_res / c; gi = torch.arange(c, device=dev, dtype=torch.float32)
        lo = (IMG - model.eval_res) // 2
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


def predict_study(model, vol, mask, device, mirror=False, amp=True):
    """-> sigmoid [3, 12]: fused, global, local (local = g where no crops; a global-only model returns g three times)."""
    cs = uniform_centres(mask)
    lo = (IMG - model.eval_res) // 2
    xg = windows(vol, cs, device, mirror)[..., lo:lo + model.eval_res, lo:lo + model.eval_res].float().div_(255.0)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        fm = torch.cat([model.backbone.forward_features(normalise(xg[i:i + WIN_CHUNK], model))
                        for i in range(0, len(xg), WIN_CHUNK)])
    fm = fm.float()
    h = model.norm.float()(model.pool(fm))
    a = torch.softmax(model.att.float()(h), dim=0)                               # [K, 12]
    g = ((a.T @ h) * model.clsW.float()).sum(-1) + model.clsb.float()
    if not model.crops:
        return torch.stack([g, g, g]).sigmoid().cpu().numpy()
    S, R, Q, A = locate(model, fm, a, cs)
    fid = np.repeat(CF, TOP)
    S, R, Q = S.ravel(), R.ravel(), Q.ravel()
    hb = BOXN[fid] // 2
    R, Q = np.clip(R, hb, IMG - hb), np.clip(Q, hb, IMG - hb)
    wt = (A / A.sum(1, keepdims=True)).ravel()
    xc = windows(vol, S, device, mirror).float().div_(255.0)
    s = torch.as_tensor(BOXN[fid] / IMG, dtype=torch.float32, device=device).view(-1, 1, 1)
    pxy = torch.stack([torch.as_tensor((Q + 0.5) / IMG * 2 - 1), torch.as_tensor((R + 0.5) / IMG * 2 - 1)], 1)
    th = torch.cat([s * torch.eye(2, device=device).expand(len(fid), 2, 2), pxy.float().to(device)[..., None]], 2)
    xc = F.grid_sample(xc, F.affine_grid(th, (len(fid), 3, CROP_RES, CROP_RES), align_corners=False), mode="bilinear",
                       padding_mode="zeros", align_corners=False)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        hc = torch.cat([model.crop_features(normalise(xc[i:i + WIN_CHUNK], model)) for i in range(0, len(xc), WIN_CHUNK)])
    hc = model.norm_l.float()(hc.float())
    fidt = torch.as_tensor(fid, device=device)
    lc = (hc * model.clsV.float()[fidt]).sum(-1) + model.clsvb.float()[fidt]
    l = torch.zeros(12, device=device).scatter_add(0, fidt, lc * torch.as_tensor(wt, device=device))
    cm = torch.as_tensor(CMASK, device=device)
    fused = torch.where(cm, 0.5 * (g + l), g)
    return torch.stack([fused, g, torch.where(cm, l, g)]).sigmoid().cpu().numpy()


HEAD_ROW = {"fused": 0, "global": 1, "local": 2}


def predict_safe(model, vol, mask, device, mirror, stats):
    """fp16 with an fp32 retry for non-finite logits; raises on a still-invalid prediction."""
    p = predict_study(model, vol, mask, device, mirror)
    if not np.isfinite(p).all():
        stats["fp32_retries"] += 1
        p = predict_study(model, vol, mask, device, mirror, amp=False)
    if p.shape != (3, 12) or not np.isfinite(p).all():
        raise ValueError("Invalid prediction")
    return p


def fidelity(models, refs, data, device, limit):
    """Re-predict fold-0 training studies from DICOM (plain view, global head) and compare with each run's saved
    val_swa_global.json values (refs: model name -> {study_id: [12]}). Returns name -> max |dp|."""
    from multiprocessing import Pool
    refs = {n: dict(sorted(r.items())[:limit]) for n, r in refs.items()}       # each model: its own first `limit` studies
    ids = sorted({s for r in refs.values() for s in r})
    if not ids:
        return {}
    tr = {}
    for row in read_csv(data / "train_series.csv"):
        if row["StudyInstanceUID"] in ids:
            tr.setdefault(row["StudyInstanceUID"], []).append(row)
    out = {n: 0.0 for n in refs}
    with Pool(4) as procs, torch.no_grad():
        for sid, vol, mask, _ in procs.imap(build_study, [(s, tr.get(s, []), data / "train_series") for s in ids]):
            for n, ref in refs.items():
                if sid in ref:
                    p = predict_study(models[n], vol, mask, device)[1]
                    out[n] = max(out[n], float(np.abs(p - np.array(ref[sid])).max()))
    return out


def run_models(models, specs, ids, by_study, series_dir, device, stats, started, tag=""):
    """Every model on every study (each volume built once). -> {name: {"plain": [n, 12], "mirror": [n, 12]}} of the
    spec's head; a model gets "mirror" only when spec["mirror"]; rows stay NaN for studies that could not be read."""
    from multiprocessing import Pool
    out = {s["name"]: {v: np.full((len(ids), 12), np.nan) for v in (("plain", "mirror") if s.get("mirror") else ("plain",))}
           for s in specs}
    with Pool(4) as procs, torch.no_grad():
        for n, (sid, vol, mask, errors) in enumerate(
                procs.imap(build_study, [(s, by_study.get(s, []), series_dir) for s in ids])):
            stats["decode_errors"] += errors
            if not mask.any():
                stats["empty_studies"] += 1
                continue
            for s in specs:
                for v, arr in out[s["name"]].items():
                    arr[n] = predict_safe(models[s["name"]], vol, mask, device, v == "mirror", stats)[HEAD_ROW[s["head"]]]
            if (n + 1) % 50 == 0 or n + 1 == len(ids):
                print(f"{tag}{n + 1}/{len(ids)} studies; {time.monotonic() - started:.0f}s", flush=True)
    return out


def pct_rank(P):
    """Per-target percentile rank over the studies of this run, (rank - 0.5) / n with tied values averaged; studies
    without a prediction (NaN) get the middle rank 0.5. Monotone per target, so each model's AUC is unchanged."""
    from scipy.stats import rankdata
    out = np.full(P.shape, 0.5)
    for k in range(P.shape[1]):
        ok = np.isfinite(P[:, k])
        if ok.any():
            out[ok, k] = (rankdata(P[ok, k]) - 0.5) / ok.sum()
    return out


def weighted(parts, weights):
    """parts: list of [n, 12]; weights: per part a scalar or 12 per-target values -> per-target weighted mean (NaN
    entries left out of their study's mean)."""
    W = np.stack([np.broadcast_to(np.asarray(w, float), (12,)) for w in weights])[:, None, :]       # [m, 1, 12]
    X = np.stack(parts); ok = np.isfinite(X)
    num = np.where(ok, X, 0) * W; den = ok * W
    return np.where(den.sum(0) > 0, num.sum(0) / np.maximum(den.sum(0), 1e-12), np.nan)


def val_json(path, ids, P, note):
    rows = [{"study_id": s, "probabilities": [round(float(v), 6) for v in P[i]]} for i, s in enumerate(ids)
            if np.isfinite(P[i]).all()]
    path.write_text(json.dumps({"targets_in_order": TARGETS, "note": note, "predictions": rows}) + "\n")


def write_submission(path, ids, preds):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["StudyInstanceUID", *TARGETS])
        for s in ids:
            w.writerow([s, *[f"{v:.8f}" for v in preds[s]]])


def check_submission(path, ids):
    written = read_csv(path)
    vals = np.array([[float(r[t]) for t in TARGETS] for r in written])
    if [r["StudyInstanceUID"] for r in written] != ids or not np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
        raise ValueError(f"Saved submission is invalid: {path}")
# ---------------------------------------------------------------- end of GL4 inference core

MODE = "test"                     # "test" | "fold0"
FOLD0_LIMIT = None                # fold0 mode: first N studies only (None = all 825)
COMBINE = "prob"                  # "prob" | "logit"
MODEL = {"name": "cnxs", "arch": "cnxs", "head": "global", "mirror": True,                     # LB 0.947 without mirror
         "dataset": "priyanjaliml/rsna-pubkd-e4cnxs-fold0", "file": "weights_swa21_25_global_only.pt",
         "sha256": "1c21356b312082a4ad2ccc7470244f74f7a06686c25450e33a8d56dca546d64d"}
# fused head instead: "head": "fused", "file": "weights_swa21_25.pt",
#   "sha256": "29af81b432de5b339aa016b965709bb06b0045d2db089a2fc23e764e9265e2f3"
FOLD0_IDS = ("priyanjaliml/rsna-pubkd-e4cnxs-fold0", "fold0_predictions/val_swa_global.json")
FID_N, FID_MAX = 8, 0.01
FID = {"cnxs": {"1.2.826.0.1.3680043.8.498.10021288557171866794624317186288618513": [0.197311, 0.286846, 0.318299, 0.347849, 0.35687, 0.418308, 0.538143, 0.337819, 0.37879, 0.302222, 0.197276, 0.317438], "1.2.826.0.1.3680043.8.498.10048842414095994814596043262132965018": [0.902006, 0.588711, 0.121634, 0.585738, 0.23919, 0.490343, 0.266071, 0.971246, 0.725292, 0.209695, 0.979926, 0.920202], "1.2.826.0.1.3680043.8.498.10066945398485042741116657288959323922": [0.359248, 0.437317, 0.243427, 0.201247, 0.339246, 0.274117, 0.555126, 0.844296, 0.756634, 0.411078, 0.40945, 0.298878], "1.2.826.0.1.3680043.8.498.10087350795575215080051569638065974347": [0.178665, 0.154349, 0.054987, 0.077342, 0.063447, 0.107326, 0.089564, 0.313416, 0.242882, 0.22475, 0.082076, 0.198823], "1.2.826.0.1.3680043.8.498.10090631811016939819371208464230332678": [0.135979, 0.203285, 0.096455, 0.077946, 0.062447, 0.083636, 0.087124, 0.111571, 0.19981, 0.193875, 0.086812, 0.252404], "1.2.826.0.1.3680043.8.498.10096063158753121251217031021181036863": [0.1953, 0.190043, 0.061849, 0.063077, 0.087447, 0.13497, 0.124603, 0.141692, 0.203249, 0.304751, 0.144662, 0.236913], "1.2.826.0.1.3680043.8.498.10129462959695121379466274688956171166": [0.453249, 0.457733, 0.981628, 0.535996, 0.980746, 0.900508, 0.966197, 0.526253, 0.535809, 0.608265, 0.472122, 0.361047], "1.2.826.0.1.3680043.8.498.10147180484860577398303524379319212586": [0.392392, 0.592485, 0.899314, 0.324227, 0.951996, 0.581759, 0.566436, 0.84942, 0.588578, 0.954995, 0.90226, 0.729936]}}                          # <<FID>> model name -> {fold-0 study: saved global probabilities}


def combine(plain, mirror):
    if COMBINE == "logit":
        z = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))  # noqa: E731
        return 1 / (1 + np.exp(-(z(plain) + z(mirror)) / 2))
    return (plain + mirror) / 2


def main():
    started = time.monotonic()
    data, work = competition_dir(), Path("/kaggle/working")
    sample = read_csv(data / "sample_submission.csv")
    if not sample or list(sample[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("sample_submission.csv columns differ from the model target order")
    test_ids = [r["StudyInstanceUID"] for r in sample]
    fallback = {r["StudyInstanceUID"]: [float(r[t]) for t in TARGETS] for r in sample}
    write_submission(work / "submission.csv", test_ids, fallback)        # valid file even if inference is interrupted

    ensure_timm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = {**MODEL, "mirror": True}
    models = {spec["name"]: load_model({**spec, "path": dataset_dir(spec["dataset"]) / spec["file"]}, device)}
    t_load = time.monotonic() - started
    print(f"device {device} | {spec['name']} {spec['head']} head + anatomical mirror ({COMBINE}) | ready {t_load:.1f}s",
          flush=True)
    stats = {"empty_studies": 0, "decode_errors": 0, "fp32_retries": 0}

    if MODE == "fold0":
        ref = json.loads((dataset_dir(FOLD0_IDS[0]) / FOLD0_IDS[1]).read_text())["predictions"]
        ids = [r["study_id"] for r in ref][:FOLD0_LIMIT]
        by_study, wanted = {}, set(ids)
        for row in read_csv(data / "train_series.csv"):
            if row["StudyInstanceUID"] in wanted:
                by_study.setdefault(row["StudyInstanceUID"], []).append(row)
        v = run_models(models, [spec], ids, by_study, data / "train_series", device, stats, started, "fold0 ")[spec["name"]]
        tag = f"{spec['name']}_{spec['head']}"
        val_json(work / f"fold0_{tag}_plain.json", ids, v["plain"], "plain view")
        val_json(work / f"fold0_{tag}_mirror.json", ids, v["mirror"], "anatomical mirror only")
        val_json(work / f"fold0_{tag}_tta.json", ids, combine(v["plain"], v["mirror"]), f"mirror TTA ({COMBINE})")
        summary = {"mode": "fold0", "studies": len(ids), "model": spec, "combine": COMBINE, "stats": stats,
                   "seconds": round(time.monotonic() - started, 1)}
        (work / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return

    t0 = time.monotonic()
    fid = fidelity(models, {n: r for n, r in FID.items() if n in models}, data, device, FID_N)
    print(f"FIDELITY max |dp| {fid} ({time.monotonic() - t0:.1f}s)", flush=True)
    if any(x >= FID_MAX for x in fid.values()):
        raise ValueError(f"fidelity gate failed: {fid}")
    t_test = time.monotonic()
    by_study = {}
    for row in read_csv(data / "test_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    v = run_models(models, [spec], test_ids, by_study, data / "test_series", device, stats, started, "test ")[spec["name"]]
    P = combine(v["plain"], v["mirror"])
    final = {s: (P[i].tolist() if np.isfinite(P[i]).all() else fallback[s]) for i, s in enumerate(test_ids)}
    write_submission(work / "submission.csv", test_ids, final)
    check_submission(work / "submission.csv", test_ids)
    summary = {"mode": "test", "model": spec, "combine": COMBINE, "fidelity_max_abs": fid, "stats": stats,
               "studies": len(test_ids), "seconds_model_load": round(t_load, 1),
               "seconds_test": round(time.monotonic() - t_test, 1), "seconds": round(time.monotonic() - started, 1)}
    (work / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
