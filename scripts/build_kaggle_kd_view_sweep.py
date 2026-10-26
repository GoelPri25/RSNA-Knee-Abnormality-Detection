"""Build the Kaggle kernel for KD-VIEW (ledger row KD-VIEW): which student training view keeps the teacher's answer and
the evidence? No training, no submission; free Kaggle T4 (both GPUs when present); weights sha256-checked.
Out: kaggle/submission_kernels/kd_view_sweep/ (or data/_session_tmp/kd_view_sweep_test with --test).

Fold 0 studies (Steven-confident labels are joined locally; no labels in the kernel). Per study and view: the Raptor 384
teacher's probabilities (all-data teacher: in-sample, judged relative to its own clean view) and E4's (out of fold).
Views = slice draw x augmentation:
  slices: u = the 94 uniform eval windows; rK = K stratified random windows (runner centres(mask, K, train=True));
          gK = K of the 94 drawn without replacement with p = 0.5 * teacher clean attention (mean over findings) + 0.5
          uniform (teacher-guided).
  augmentation: none (320 centre crop), mirror (always), fov (whole 384 field resampled to 320), geom (whole field +
          rotation 10 deg / scale 0.10 / shift 0.06), int (crop + gamma 0.25 / gain 0.10), full (the runner's: mirror 0.5 +
          whole field + geom + int), light (crop, no mirror, rotation 5 / scale 0.05 / shift 0.03, gamma 0.10, gain 0.05).
Plus the teacher-attention mass each draw covers (windows within 1 slice), the teacher's clean attention, and the
Grad-CAM baseline: the 224 Raptor sibling's map on the 384 teacher's top window (mapped onto the 384's 10 x 10 grid of
the 320 crop), to set against teacher-student 0.13 and student-student 0.47-0.58 (KD-GAP part 1).
Usage: python3 -I scripts/build_kaggle_kd_view_sweep.py <repo root> [--test]
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]); TEST = "--test" in sys.argv
OUT = ROOT / ("data/_session_tmp/kd_view_sweep_test" if TEST else "kaggle/submission_kernels/kd_view_sweep")
OUT.mkdir(parents=True, exist_ok=True)
emb = {"kd_sub.py": base64.b64encode((ROOT / "kaggle/submission_kernels/pubkd_kd_e4_submission/submission.py")
                                     .read_bytes()).decode()}
SHA = {"E4": "080cf63f8997a768d0b6290e71a597c53d7be02c30564df47eee951c20398a36",
       "384": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef",
       "224": "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"}
code = '''"""KD-VIEW sweep (no training, no submission). Built by scripts/build_kaggle_kd_view_sweep.py; kd_sub = the E4
submission script: DICOM builder, windows, r50 l4copy model."""
import base64
import hashlib
import json
import math
import sys
import threading
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EMB = %s
SHA = %s
N, BUDGET_S = %d, %d
W = Path("/kaggle/working"); CODE = W / "code"; CODE.mkdir(exist_ok=True)
for name, b in EMB.items():
    (CODE / name).write_bytes(base64.b64decode(b))
sys.path.insert(0, str(CODE))
import kd_sub as sub  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

RES, IMG = 320, sub.IMG
BOUNDS = np.cumsum([0, 26, 22, 18, 12, 18])
slot_of = lambda c: np.searchsorted(BOUNDS, c, side="right") - 1
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
Z = dict(mirror_p=0.0, crop=True, rot=0.0, scale=0.0, shift=0.0, gamma=0.0, gain=0.0)
AUGS = {"none": Z, "mirror": {**Z, "mirror_p": 1.0}, "fov": {**Z, "crop": False},
        "geom": {**Z, "crop": False, "rot": 10.0, "scale": 0.10, "shift": 0.06},
        "int": {**Z, "gamma": 0.25, "gain": 0.10},
        "full": dict(mirror_p=0.5, crop=False, rot=10.0, scale=0.10, shift=0.06, gamma=0.25, gain=0.10),
        "light": dict(mirror_p=0.0, crop=True, rot=5.0, scale=0.05, shift=0.03, gamma=0.10, gain=0.05)}
VIEWS = [("clean", "u", 94, "none"), ("r16", "r", 16, "none"), ("r32", "r", 32, "none"), ("r48", "r", 48, "none"),
         ("g16", "g", 16, "none"), ("g32", "g", 32, "none"),
         ("u_mirror", "u", 94, "mirror"), ("u_fov", "u", 94, "fov"), ("u_geom", "u", 94, "geom"),
         ("u_int", "u", 94, "int"), ("u_full", "u", 94, "full"), ("u_light", "u", 94, "light"),
         ("r16_full", "r", 16, "full"), ("r32_full", "r", 32, "full"), ("r48_full", "r", 48, "full"),
         ("r16_light", "r", 16, "light"), ("r32_light", "r", 32, "light"), ("r48_light", "r", 48, "light"),
         ("g16_light", "g", 16, "light"), ("g32_light", "g", 32, "light")]
COVER = ["r16", "r32", "r48", "g16", "g32"]


def find(owner, slug):
    return next(p for p in (Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input") / slug) if p.exists())


def checked(path, sha):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
        raise ValueError(f"weights hash mismatch: {path}")
    return str(path)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, d, n=12):
        super().__init__()
        self.backbone, self.norm = backbone, nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.0), nn.Linear(256, n))
        self.clsW, self.clsb = nn.Parameter(torch.zeros(n, d)), nn.Parameter(torch.zeros(n))


def load_teacher(device, which):
    import timm
    name = {"384": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt",
            "224": "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt"}[which]
    p = sorted(find("nartaa", "rsna-knee-publication-swa-weights-20261007").rglob(name))
    assert len(p) == 1, p
    ck = torch.load(checked(p[0], SHA[which]), map_location="cpu", weights_only=True)
    assert list(ck["lab"]) == list(sub.TARGETS), ck["lab"]
    bb = timm.create_model(f"coatnet_rmlp_2_rw_{which}", pretrained=False, num_classes=0,
                           **({"img_size": 320} if which == "384" else {}))
    m = RaptorClassifier(bb, bb.num_features); m.load_state_dict(ck["model"], strict=True)
    for q in m.parameters():
        q.requires_grad_(False)
    return m.eval().to(device)


class Wrap:
    def __init__(self, m, teacher):
        self.m, self.teacher = m, teacher

    def fmaps(self, x):
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
            fm = torch.cat([self.m.backbone.forward_features(c) for c in x.split(16)])
        return fm.float()

    def pool(self, fm):
        return self.m.backbone.forward_head(fm, pre_logits=True) if self.teacher else fm.mean((2, 3))

    def head(self, fm):
        h = self.m.norm(self.pool(fm))
        a = torch.softmax(self.m.att(h), 0)
        return a, ((a.T @ h) * self.m.clsW).sum(-1) + self.m.clsb

    def prob(self, x):
        return torch.sigmoid(self.head(self.fmaps(x))[1]).cpu().numpy()


def norm_x(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def view(vol, cs, device, mode, gen):
    """uint8 windows -> normalised [k, 3, 320, 320]; the clean / mirror / int crops are exact pixel slices."""
    A = AUGS[mode]; u = lambda: torch.rand(1, generator=gen).item() * 2 - 1
    x = sub.windows(vol, cs, device)
    if A["mirror_p"] > 0 and torch.rand(1, generator=gen).item() < A["mirror_p"]:
        sag = torch.as_tensor(slot_of(cs) <= 1, device=device)
        x = torch.where(sag.view(-1, 1, 1, 1), x.flip(1), x.flip(-1))
    x = x.float().div_(255.0)
    if A["crop"] and A["rot"] == 0 and A["scale"] == 0 and A["shift"] == 0:
        lo = (IMG - RES) // 2
        x = x[..., lo:lo + RES, lo:lo + RES]
    else:
        f = RES / IMG if A["crop"] else 1.0
        ang, sc = u() * math.radians(A["rot"]), 1 + u() * A["scale"]
        tx, ty = u() * A["shift"] * 2, u() * A["shift"] * 2
        c, s = math.cos(ang) / sc * f, math.sin(ang) / sc * f
        theta = torch.tensor([[c, -s, tx], [s, c, ty]], device=device, dtype=torch.float32).expand(len(cs), 2, 3).contiguous()
        x = F.grid_sample(x, F.affine_grid(theta, (len(cs), 3, RES, RES), align_corners=False), mode="bilinear",
                          padding_mode="zeros", align_corners=False)
    if A["gamma"] > 0:
        gamma, gain = math.exp(u() * A["gamma"]), 1 + u() * A["gain"]
        x = (x.clamp_min(1e-4).pow(gamma) * gain).clamp_(0, 1)
    return norm_x(x)


def view224(vol, cs, device):
    """The 224 sibling's own eval view (scripts/pubkd_raptor_teacher.py): 0.8 centre crop (307 px) -> 224."""
    x = sub.windows(vol, cs, device).float().div_(255.0)
    c = int(round(IMG * 0.8)); o = (IMG - c) // 2
    return norm_x(F.interpolate(x[..., o:o + c, o:o + c], size=(224, 224), mode="bilinear", align_corners=False))


def train_centres(mask, k, rng):
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, sub.MAXS - 2))])
    edges = np.linspace(0, len(cs), k + 1)
    idx = np.minimum((edges[:-1] + rng.random(k) * (edges[1:] - edges[:-1])).astype(int), len(cs) - 1)
    return np.clip(cs[idx], 1, sub.MAXS - 2)


def guided_centres(cs, att, k, rng):
    p = 0.5 * att.mean(1) / att.mean(1).sum() + 0.5 / len(cs)
    return cs[np.sort(rng.choice(len(cs), size=min(k, len(cs)), replace=False, p=p / p.sum()))]


def cam224_on(w, x, kt):
    """224 sibling's gradient x activation map for finding f on window kt[f], mapped to the 384's 10 x 10 grid."""
    out = np.zeros((12, 10, 10), np.float32)
    a, _ = w.head(w.fmaps(x))
    c = int(round(IMG * 0.8)); o = (IMG - c) // 2; lo = (IMG - RES) // 2; pl = o - lo; pr_ = RES - c - pl
    for k in sorted(set(int(v) for v in kt)):
        fs = [f for f in range(12) if int(kt[f]) == k]
        with torch.enable_grad():
            t = w.fmaps(x[k:k + 1]).detach().requires_grad_(True)
            hk = w.m.norm(w.pool(t))[0]
            for i, f in enumerate(fs):
                g, = torch.autograd.grad(a[k, f] * (hk @ w.m.clsW[f]), t, retain_graph=i < len(fs) - 1)
                cam = (g * t.detach()).sum(1).clamp_min(0)[None]
                cam = F.pad(F.interpolate(cam, size=(c, c), mode="bilinear", align_corners=False), (pl, pr_, pl, pr_))
                out[f] = F.avg_pool2d(cam, 32)[0, 0].cpu().numpy()
    return out


def worker(rank, device, ids, it, res, t0, log):
    torch.set_grad_enabled(False)
    try:
        T = Wrap(load_teacher(device, "384"), True); T2 = Wrap(load_teacher(device, "224"), True)
        p = checked(find("tumis001", "rsna-pubkd-kd-e4-fold0") / "weights_swa21_25.pt", SHA["E4"])
        S = Wrap(sub.load_model(p, device).to(memory_format=torch.contiguous_format), False)
        rng = np.random.default_rng(100 + rank); gen = torch.Generator().manual_seed(100 + rank)
        if True:
            for n, (sid, vol, mask, err) in enumerate(it):
                if time.monotonic() - t0 > BUDGET_S:
                    log[f"stopped_{rank}"] = n; break
                cs = sub.uniform_centres(mask)
                x = view(vol, cs, device, "none", gen)
                a, z = T.head(T.fmaps(x)); att = a.cpu().numpy(); kt = att.argmax(0)
                r = {"att": att.astype(np.float16), "top": kt, "err": err}
                for name, mode, k, aug in VIEWS:
                    if name == "clean":
                        c_ = cs; xv = x
                    else:
                        c_ = cs if mode == "u" else train_centres(mask, k, rng) if mode == "r" else guided_centres(cs, att, k, rng)
                        xv = view(vol, c_, device, aug, gen)
                    r[f"T_{name}"] = T.prob(xv) if name != "clean" else torch.sigmoid(z).cpu().numpy()
                    r[f"S_{name}"] = S.prob(xv)
                    if name in COVER:
                        r[f"cov_{name}"] = att[(np.abs(cs[:, None] - c_[None, :]) <= 1).any(1)].sum(0)
                x2 = view224(vol, cs, device)
                r["T224_clean"] = T2.prob(x2)
                r["cam224"] = cam224_on(T2, x2, kt).astype(np.float16)
                res[sid] = r
                if n %% 25 == 0:
                    print(f"gpu {rank}: {n + 1}/{len(ids)} {time.monotonic() - t0:.0f}s", flush=True)
    except Exception:
        log[f"error_{rank}"] = traceback.format_exc(); print(log[f"error_{rank}"], flush=True)


def main():
    t0 = time.monotonic()
    sub.ensure_timm()
    data = sub.competition_dir()
    by_study = {}
    for row in sub.read_csv(data / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    fold0 = json.loads((find("tumis001", "rsna-d96-fold0-ids") / "fold0_ids.json").read_text())[:N]
    ng = max(1, torch.cuda.device_count())
    procs = Pool(4)                                   # forked before any CUDA context and before the worker threads
    devs = [torch.device(f"cuda:{i}") for i in range(ng)] if torch.cuda.is_available() else [torch.device("cpu")]
    log = {"devices": [torch.cuda.get_device_name(i) for i in range(ng)] if torch.cuda.is_available() else ["cpu"],
           "weights_sha256": SHA, "views": VIEWS, "augs": AUGS}
    res = {}
    shards = [fold0[i::len(devs)] for i in range(len(devs))]
    its = [procs.imap(sub.build_study, [(s, by_study.get(s, []), data / "train_series") for s in sh]) for sh in shards]
    th = [threading.Thread(target=worker, args=(i, d, shards[i], its[i], res, t0, log)) for i, d in enumerate(devs)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    procs.terminate()
    ids = [s for s in fold0 if s in res]
    keys = list(res[ids[0]])
    np.savez_compressed(W / "kd_view_sweep.npz", ids=np.array(ids), **{k: np.array([res[s][k] for s in ids]) for k in keys})
    log["n_done"] = len(ids); log["seconds"] = round(time.monotonic() - t0, 1)
    log["s_per_study_per_gpu"] = round(log["seconds"] * len(devs) / max(1, len(ids)), 2)
    (W / "kd_view_sweep_log.json").write_text(json.dumps(log, indent=1, default=str))
    print(json.dumps({k: v for k, v in log.items() if k not in ("views", "augs")}, default=str), flush=True)


if __name__ == "__main__":
    main()
''' % (json.dumps(emb), json.dumps(SHA), 6 if TEST else 825, 600 if TEST else 3 * 3600)
(OUT / "run.py").write_text(code)
slug = "rsna-kd-view-sweep" + ("-test" if TEST else "")
meta = {"id": f"tumis001/{slug}", "title": slug, "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["tumis001/rsna-coatnet-d96-exp1b-fold0", "tumis001/rsna-d96-fold0-ids",
                            "tumis001/rsna-pubkd-kd-e4-fold0", "nartaa/rsna-knee-publication-swa-weights-20261007"],
        "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("built", OUT / "run.py", len(code), "bytes")
