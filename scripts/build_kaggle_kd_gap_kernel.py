"""Build the Kaggle kernel for KD-GAP (ledger row KD-GAP): why do the r50 students stay below the Raptor teacher?
No training, no submission; free Kaggle T4; every weights file sha256-checked. Out: kaggle/submission_kernels/kd_gap_xai/.

Part 1, fold 0 (825 studies, DICOMs decoded by the submission builder): the Raptor 384 teacher (plain view, as its
targets) and the r50 l4copy students v4 (no KD, LB 0.939), ra (Raptor targets, 0.943), B (full blend mix73, 0.946),
E4 (0.946), global head only. Per model: probabilities, the 94 x 12 window attention, series ablation (global logits
with each series slot dropped / alone), and gradient x activation maps (10 x 10, last feature stage) of each finding's
contribution on the TEACHER's top window for that finding (in-plane: same image, does the student look where the
teacher looks?) plus on the model's own top window. Panels (PNG) for 14 teacher-right / E4-wrong cases.
Part 2, 200 training studies (not fold 0; seed 2026): the teacher on the view its targets came from (94 windows,
320 centre crop, no augmentation) vs the views a student trains on: 16 stratified random windows (the runner's
centres(mask, 16, train=True)) without / with the training augmentation (mirror 0.5, rotation 10 deg, scale 0.10,
shift 0.06, gamma 0.25, gain 0.10, whole 384 field resampled; output 320 for the teacher), 94 windows with the
augmentation, and 94 windows on the whole field without augmentation. Plus the share of the teacher's own attention
mass that a 16-window draw covers (decisive slices missed?), and the students ra / E4 on the clean training view.
Usage: python3 -I scripts/build_kaggle_kd_gap_kernel.py <repo root>
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]); OUT = ROOT / "kaggle/submission_kernels/kd_gap_xai"; OUT.mkdir(parents=True, exist_ok=True)
emb = {"kd_sub.py": base64.b64encode((ROOT / "kaggle/submission_kernels/pubkd_kd_e4_submission/submission.py")
                                     .read_bytes()).decode()}
cases = json.loads((ROOT / "data/coatnet_d96_fold0/kd_gap_analysis/xai_cases.json").read_text())
SHA = {"v4": "2a570689d07a3a01caa96ca058076a7d695181848033a9e9dcd0f4b3a68d063b",
       "ra": "5107e5479e02c64e6bf239361b5fc577ea62fa6622e1e8c897c99fe39318bb3c",
       "B": "f04a1220efb7f8353c51389751c562bc0a6d3d5f5f1ef1691513556ea7cf4f2f",
       "E4": "080cf63f8997a768d0b6290e71a597c53d7be02c30564df47eee951c20398a36",
       "raptor384": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"}
DS = {"v4": "rsna-v4-l4copy-fold0", "ra": "rsna-pubkd-r-ra-fold0", "B": "rsna-pubkd-kd-b-fold0", "E4": "rsna-pubkd-kd-e4-fold0"}
code = '''"""KD-GAP xAI (no training, no submission). Built by scripts/build_kaggle_kd_gap_kernel.py; kd_sub = the E4 submission
script (kaggle/submission_kernels/pubkd_kd_e4_submission/submission.py): DICOM builder, windows, r50 l4copy model."""
import base64
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EMB = %s
SHA = %s
DS = %s
CASES = %s
W = Path("/kaggle/working"); CODE = W / "code"; CODE.mkdir(exist_ok=True)
for name, b in EMB.items():
    (CODE / name).write_bytes(base64.b64decode(b))
sys.path.insert(0, str(CODE))
import kd_sub as sub  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

N1, N2, DRAWS = int(os.environ.get("KD_N1", 825)), int(os.environ.get("KD_N2", 200)), 4
RES, IMG, K = 320, sub.IMG, sub.EVAL_WINDOWS
BOUNDS = np.cumsum([0, 26, 22, 18, 12, 18])
slot_of = lambda c: np.searchsorted(BOUNDS, c, side="right") - 1
AUG = {"mirror_p": 0.5, "rot_deg": 10.0, "scale": 0.10, "shift": 0.06, "gamma": 0.25, "gain": 0.10}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find(owner, slug):
    return next(p for p in (Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input") / slug) if p.exists())


def checked(path, sha):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
        raise ValueError(f"weights hash mismatch: {path}")
    return str(path)


class RaptorClassifier(nn.Module):
    """Their head (scripts/pubkd_raptor_teacher.py)."""

    def __init__(self, backbone, d, n=12):
        super().__init__()
        self.backbone, self.norm = backbone, nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.0), nn.Linear(256, n))
        self.clsW, self.clsb = nn.Parameter(torch.zeros(n, d)), nn.Parameter(torch.zeros(n))

    def forward(self, x):
        f = torch.cat([self.backbone(c) for c in x.split(16)])
        h = self.norm(f)
        a = torch.softmax(self.att(h).float(), 0)
        return (torch.einsum("kn,kf->nf", a.to(h.dtype), h) * self.clsW).sum(-1) + self.clsb


def load_teacher(device):
    import timm
    p = sorted(find("nartaa", "rsna-knee-publication-swa-weights-20261007").rglob(
        "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"))
    assert len(p) == 1, p
    ck = torch.load(checked(p[0], SHA["raptor384"]), map_location="cpu", weights_only=True)
    assert list(ck["lab"]) == list(sub.TARGETS), ck["lab"]
    bb = timm.create_model("coatnet_rmlp_2_rw_384", pretrained=False, num_classes=0, img_size=320)
    m = RaptorClassifier(bb, bb.num_features); m.load_state_dict(ck["model"], strict=True)
    for q in m.parameters():
        q.requires_grad_(False)
    return m.eval().to(device)


class Wrap:
    """Uniform access: feature maps -> pooled -> head, for the teacher (CoAtNet) and the r50 students."""

    def __init__(self, m, teacher):
        self.m, self.teacher = m, teacher

    def fmaps(self, x):                                  # x normalised [k, 3, R, R] -> fp32 [k, C, c, c]
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
            fm = torch.cat([self.m.backbone.forward_features(c) for c in x.split(16)])
        return fm.float()

    def pool(self, fm):
        if self.teacher:
            return self.m.backbone.forward_head(fm, pre_logits=True)
        return fm.mean((2, 3))

    def head(self, fm):
        h = self.m.norm(self.pool(fm))
        al = self.m.att(h)
        a = torch.softmax(al, 0)
        z = ((a.T @ h) * self.m.clsW).sum(-1) + self.m.clsb
        return h, al, a, z


def norm_x(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def view(vol, cs, device, aug=False, whole=False, gen=None):
    """uint8 windows -> normalised [k, 3, 320, 320]. Clean: 320 centre crop (eval / target view). aug / whole: the
    runner's prepare() geometry on the whole 384 field (one draw per study), output 320 for the teacher's fixed size."""
    x = sub.windows(vol, cs, device)
    if aug and torch.rand(1, generator=gen).item() < AUG["mirror_p"]:
        sag = torch.as_tensor(slot_of(cs) <= 1, device=device)
        x = torch.where(sag.view(-1, 1, 1, 1), x.flip(1), x.flip(-1))
    x = x.float().div_(255.0)
    if not (aug or whole):
        lo = (IMG - RES) // 2
        return norm_x(x[..., lo:lo + RES, lo:lo + RES])
    u = lambda: (torch.rand(1, generator=gen).item() * 2 - 1)
    ang = u() * math.radians(AUG["rot_deg"]) if aug else 0.0
    sc = 1 + u() * AUG["scale"] if aug else 1.0
    tx, ty = (u() * AUG["shift"] * 2, u() * AUG["shift"] * 2) if aug else (0.0, 0.0)
    c, s = math.cos(ang) / sc, math.sin(ang) / sc
    theta = torch.tensor([[c, -s, tx], [s, c, ty]], device=device, dtype=torch.float32).expand(len(cs), 2, 3).contiguous()
    grid = F.affine_grid(theta, (len(cs), 3, RES, RES), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if aug:
        gamma, gain = math.exp(u() * AUG["gamma"]), 1 + u() * AUG["gain"]
        x = (x.clamp_min(1e-4).pow(gamma) * gain).clamp_(0, 1)
    return norm_x(x)


def train_centres(mask, k, rng):
    """coatnet_d96_runner.centres(mask, k, train=True): one random centre per 1/k of the valid range."""
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, sub.MAXS - 2))])
    edges = np.linspace(0, len(cs), k + 1)
    idx = np.minimum((edges[:-1] + rng.random(k) * (edges[1:] - edges[:-1])).astype(int), len(cs) - 1)
    return np.clip(cs[idx], 1, sub.MAXS - 2)


def series_ablation(w, h, al, cs):
    sl = slot_of(cs); drop = np.full((5, 12), np.nan, np.float32); only = drop.copy()
    for j in range(5):
        for keep, out in ((sl != j, drop), (sl == j, only)):
            if keep.sum() == 0:
                continue
            kk = torch.as_tensor(keep, device=h.device)
            aa = torch.softmax(al[kk], 0)
            out[j] = (((aa.T @ h[kk]) * w.m.clsW).sum(-1) + w.m.clsb).detach().cpu().numpy()
    return drop, only


def cams(w, x, a, wins):
    """gradient x activation of finding f's contribution a[k, f] * (norm(pool(fm_k)) . clsW[f]) on window wins[f]."""
    out = np.zeros((12, 10, 10), np.float32)
    for k in sorted(set(int(v) for v in wins)):
        fs = [f for f in range(12) if int(wins[f]) == k]
        with torch.enable_grad():
            t = w.fmaps(x[k:k + 1]).detach().requires_grad_(True)
            hk = w.m.norm(w.pool(t))[0]
            for i, f in enumerate(fs):
                contrib = a[k, f].detach() * (hk @ w.m.clsW[f])
                g, = torch.autograd.grad(contrib, t, retain_graph=i < len(fs) - 1)
                cam = (g * t.detach()).sum(1)[0].clamp_min(0)
                if cam.shape[-1] != 10:
                    cam = F.adaptive_avg_pool2d(cam[None, None], 10)[0, 0]
                out[f] = cam.cpu().numpy()
    return out


def panel(vol, cs, f, k, cam_t, cam_s, pt, ps, path, title):
    from PIL import Image, ImageDraw
    lo = (IMG - RES) // 2
    img = vol[cs[k]][lo:lo + RES, lo:lo + RES].astype(np.float32)
    tiles = []
    for cam in (None, cam_t, cam_s):
        rgb = np.stack([img] * 3, -1)
        if cam is not None and cam.max() > 0:
            hm = np.kron(cam / cam.max(), np.ones((32, 32)))
            rgb[..., 0] = np.clip(rgb[..., 0] * 0.6 + 255 * hm * 0.8, 0, 255)
            rgb[..., 1] *= 1 - 0.5 * hm; rgb[..., 2] *= 1 - 0.5 * hm
        tiles.append(Image.fromarray(rgb.astype(np.uint8)))
    canvas = Image.new("RGB", (RES * 3, RES + 40), "black")
    for i, t in enumerate(tiles):
        canvas.paste(t, (i * RES, 40))
    d = ImageDraw.Draw(canvas)
    d.text((4, 2), title, fill="white")
    d.text((4, 20), f"window {k} (slice {cs[k]}, slot {slot_of(cs[k])}) | image | teacher CAM p={pt:.2f} | E4 CAM p={ps:.2f}",
           fill="white")
    canvas.save(path)


def main():
    t0 = time.monotonic()
    sub.ensure_timm()
    data = sub.competition_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"teacher": Wrap(load_teacher(device), True)}
    for tag in ("v4", "ra", "B", "E4"):
        p = checked(find("tumis001", DS[tag]) / "weights_swa21_25.pt", SHA[tag])
        models[tag] = Wrap(sub.load_model(p, device).to(memory_format=torch.contiguous_format), False)
    by_study = {}
    for row in sub.read_csv(data / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    fold0 = json.loads((find("tumis001", "rsna-d96-fold0-ids") / "fold0_ids.json").read_text())[:N1]
    rest = sorted(set(by_study) - set(json.loads((find("tumis001", "rsna-d96-fold0-ids") / "fold0_ids.json").read_text())))
    train = [rest[i] for i in np.random.default_rng(2026).choice(len(rest), N2, replace=False)]
    case_of = {s: f for f, lst in CASES.items() for s in lst}
    log = {"device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "weights_sha256": SHA}
    tags = list(models)
    P1 = {"ids": [], "cs": []}
    for t in tags:
        for k in ("prob", "att", "drop", "only", "cam_t", "cam_own", "top"):
            P1[f"{t}_{k}"] = []
    (W / "panels").mkdir(exist_ok=True)
    errors = 0
    with Pool(4) as procs:
        it = procs.imap(sub.build_study, [(s, by_study.get(s, []), data / "train_series") for s in fold0])
        for n, (sid, vol, mask, err) in enumerate(it):
            errors += err
            cs = sub.uniform_centres(mask)
            x = view(vol, cs, device)
            res = {}
            with torch.no_grad():
                for t, w in models.items():
                    fm = w.fmaps(x); h, al, a, z = w.head(fm)
                    res[t] = (h, al, a, z)
                    if n == 0 and t == "teacher":
                        z_ref = w.m(x).float()
                        log["teacher_head_check_max_abs_logit_diff"] = float((z_ref - z).abs().max())
                    del fm
            kt = res["teacher"][2].argmax(0).cpu().numpy()                      # teacher's top window per finding
            P1["ids"].append(sid); P1["cs"].append(cs)
            for t, w in models.items():
                h, al, a, z = res[t]
                own = a.argmax(0).cpu().numpy()
                d_, o_ = series_ablation(w, h, al, cs)
                P1[f"{t}_prob"].append(torch.sigmoid(z).cpu().numpy()); P1[f"{t}_att"].append(a.cpu().numpy().astype(np.float16))
                P1[f"{t}_drop"].append(d_); P1[f"{t}_only"].append(o_); P1[f"{t}_top"].append(own)
                P1[f"{t}_cam_t"].append(cams(w, x, a, kt).astype(np.float16))
                P1[f"{t}_cam_own"].append(cams(w, x, a, own).astype(np.float16))
            if sid in case_of:
                f = list(sub.TARGETS).index(case_of[sid])
                panel(vol, cs, f, int(kt[f]), P1["teacher_cam_t"][-1][f].astype(np.float32),
                      P1["E4_cam_t"][-1][f].astype(np.float32), P1["teacher_prob"][-1][f], P1["E4_prob"][-1][f],
                      W / "panels" / f"{case_of[sid].replace(' ', '_').replace(chr(39), '')}_{sid[-8:]}.png",
                      f"{case_of[sid]} {sid[-12:]}: teacher-right / E4-wrong case")
            if n %% 50 == 0:
                print(f"part 1: {n + 1}/{len(fold0)} {time.monotonic() - t0:.0f}s", flush=True)
    np.savez_compressed(W / "kd_gap_part1.npz", **{k: np.array(v) for k, v in P1.items()})
    log["part1_s"] = round(time.monotonic() - t0, 1); log["decode_errors_part1"] = errors

    rng = np.random.default_rng(7); gen = torch.Generator().manual_seed(7)
    tm = models["teacher"]
    P2 = {k: [] for k in ("ids", "clean", "sub16", "sub16aug", "all94aug", "whole", "cover16", "ra_clean", "E4_clean")}
    with Pool(4) as procs, torch.no_grad():
        it = procs.imap(sub.build_study, [(s, by_study.get(s, []), data / "train_series") for s in train])
        for n, (sid, vol, mask, err) in enumerate(it):
            cs = sub.uniform_centres(mask)
            x = view(vol, cs, device)
            _, _, a, z = tm.head(tm.fmaps(x))
            pr = lambda xx: torch.sigmoid(tm.head(tm.fmaps(xx))[3]).cpu().numpy()
            s16, s16a, cov = [], [], []
            for _ in range(DRAWS):
                c16 = train_centres(mask, 16, rng)
                s16.append(pr(view(vol, c16, device)))
                s16a.append(pr(view(vol, c16, device, aug=True, gen=gen)))
                near = (np.abs(cs[:, None] - c16[None, :]) <= 1).any(1)               # uniform windows within 1 slice
                cov.append(a.cpu().numpy()[near].sum(0))
            P2["ids"].append(sid); P2["clean"].append(torch.sigmoid(z).cpu().numpy())
            P2["sub16"].append(s16); P2["sub16aug"].append(s16a); P2["cover16"].append(cov)
            P2["all94aug"].append([pr(view(vol, cs, device, aug=True, gen=gen)) for _ in range(2)])
            P2["whole"].append(pr(view(vol, cs, device, whole=True)))
            for t in ("ra", "E4"):
                P2[f"{t}_clean"].append(torch.sigmoid(models[t].head(models[t].fmaps(x))[3]).cpu().numpy())
            if n %% 25 == 0:
                print(f"part 2: {n + 1}/{len(train)} {time.monotonic() - t0:.0f}s", flush=True)
    np.savez_compressed(W / "kd_gap_part2.npz", **{k: np.array(v) for k, v in P2.items()})
    log["seconds"] = round(time.monotonic() - t0, 1)
    (W / "kd_gap_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
''' % (json.dumps(emb), json.dumps(SHA), json.dumps(DS), json.dumps(cases))
(OUT / "run.py").write_text(code)
meta = {"id": "tumis001/rsna-kd-gap-xai", "title": "rsna-kd-gap-xai", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["tumis001/rsna-coatnet-d96-exp1b-fold0", "tumis001/rsna-d96-fold0-ids",
                            *[f"tumis001/{v}" for v in DS.values()], "nartaa/rsna-knee-publication-swa-weights-20261007"],
        "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("built", OUT / "run.py", len(code), "bytes")
