"""Build the Kaggle kernel for the PUB-KD-R xAI comparison on fold 0 (kaggle/submission_kernels/pubkd_r_ra_xai_fold0/):
the v4 l4copy XAI + ablations (scripts/l4copy_xai.py on the L4C-D core, as scripts/build_kaggle_xai_kernel.py) for
v4 l4copy s2026 (LB weights), PUB-KD-R ra (pure) and rb (mix73), plus the public Raptor 384 teacher's window attention
per slot on the same 94 windows (plain view, 320 centre crop, ImageNet normalisation, fp16; scripts/pubkd_raptor_teacher.py).
Full mode (scales 0.67 / 1.0 / 1.5, top-1, MS1's table), no case panels. Every weights file is sha256-checked.
Usage: python3 -I scripts/build_kaggle_xai_kdr_kernel.py <repo root>"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]); OUT = ROOT / "kaggle/submission_kernels/pubkd_r_ra_xai_fold0"; OUT.mkdir(parents=True, exist_ok=True)
emb = {name: base64.b64encode(p.read_bytes()).decode() for name, p in (
    ("l4copy_diag_core.py", ROOT / "kaggle/submission_kernels/v4_l4copy_diag_fold0/run.py"),
    ("l4copy_xai.py", ROOT / "scripts/l4copy_xai.py"))}
SHA = {"s2026": "2a570689d07a3a01caa96ca058076a7d695181848033a9e9dcd0f4b3a68d063b",
       "ra": "5107e5479e02c64e6bf239361b5fc577ea62fa6622e1e8c897c99fe39318bb3c",
       "rb": "fea57667950280cfd14a9cde5f4d8585e1ef2341cdd258d04c68fce7d6418a21",
       "raptor384": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"}
code = '''"""PUB-KD-R xAI, fold 0 (no submission). Built by scripts/build_kaggle_xai_kdr_kernel.py; the embedded files are the
repo's kaggle/submission_kernels/v4_l4copy_diag_fold0/run.py and scripts/l4copy_xai.py. Models: v4 l4copy s2026, PUB-KD-R
ra / rb, plus the Raptor 384 teacher's window attention per slot."""
import base64
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EMB = %s
SHA = %s
os.environ.update({"XAI_NOCASES": "1"})
W = Path("/kaggle/working"); CODE = W / "code"; CODE.mkdir(exist_ok=True)
for name, b in EMB.items():
    (CODE / name).write_bytes(base64.b64decode(b))
sys.path.insert(0, str(CODE))


def find(owner, slug):
    return next(p for p in (Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input") / slug) if p.exists())


def checked(path, sha):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
        raise ValueError(f"weights hash mismatch: {path}")
    return str(path)


import l4copy_diag_core as run  # noqa: E402
PATHS = {"s2026": run.ART / run.sub.WEIGHTS["file"],
         "ra": find("tumis001", "rsna-pubkd-r-ra-fold0") / "weights_swa21_25.pt",
         "rb": find("tumis001", "rsna-pubkd-r-rb-fold0") / "weights_swa21_25.pt"}
os.environ["XAI_MODELS"] = ",".join(f"{k}={checked(p, SHA[k])}" for k, p in PATHS.items())
import l4copy_xai as xai  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sub = run.sub


class RaptorClassifier(nn.Module):
    """Their head (scripts/pubkd_raptor_teacher.py); forward returns logits [12] and the window attention [K, 12]."""

    def __init__(self, backbone, d, n=12):
        super().__init__()
        self.backbone, self.norm = backbone, nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.0), nn.Linear(256, n))
        self.clsW, self.clsb = nn.Parameter(torch.zeros(n, d)), nn.Parameter(torch.zeros(n))

    def forward(self, x):
        f = torch.cat([self.backbone(c) for c in x.split(16)])
        h = self.norm(f)
        a = torch.softmax(self.att(h).float(), 0)
        return (torch.einsum("kn,kf->nf", a.to(h.dtype), h) * self.clsW).sum(-1) + self.clsb, a


def teacher(device):
    import timm
    p = sorted(find("nartaa", "rsna-knee-publication-swa-weights-20261007").rglob(
        "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"))
    assert len(p) == 1, p
    ck = torch.load(checked(p[0], SHA["raptor384"]), map_location="cpu", weights_only=True)
    assert list(ck["lab"]) == list(sub.TARGETS), ck["lab"]
    bb = timm.create_model("coatnet_rmlp_2_rw_384", pretrained=False, num_classes=0, img_size=320)
    m = RaptorClassifier(bb, bb.num_features); m.load_state_dict(ck["model"], strict=True)
    return m.eval().to(device)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def teacher_attention(tm, vol, device):
    cs = sub.uniform_centres(vol.reshape(len(vol), -1).sum(1) > 0)                  # their mask: non-empty slices
    lo = (run.IMG - 320) // 2
    x = sub.windows(vol, cs, device)[..., lo:lo + 320, lo:lo + 320].float().div_(255.0)
    x = (x - MEAN.to(device)) / STD.to(device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        z, a = tm(x)
    sl = run.slot_of(cs); a = a.float().cpu().numpy()
    mass = np.stack([a[sl == j].sum(0) for j in range(5)], 1)                       # [12, 5]
    ent = -(a * np.log(np.clip(a, 1e-12, 1))).sum(0) / np.log(len(cs))             # [12] normalised entropy
    return torch.sigmoid(z.float()).cpu().numpy(), mass, ent


def main():
    t0 = time.monotonic()
    sub.ensure_timm()
    data = sub.competition_dir()
    ids = json.loads((find("tumis001", "rsna-d96-fold0-ids") / "fold0_ids.json").read_text()); assert len(ids) == 825, len(ids)
    if os.environ.get("XAI_N"):
        ids = ids[:int(os.environ["XAI_N"])]
    tabm = run.ms1_table(); assert all(s in tabm for s in ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = xai.load_models(device)
    tm = teacher(device)
    by_study = {}
    for row in sub.read_csv(data / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    errors = [0]
    T = {"ids": [], "prob": [], "mass": [], "ent": []}

    def items(it):
        for sid, vol, mask, err in it:
            errors[0] += err
            p, m, e = teacher_attention(tm, vol, device)
            T["ids"].append(sid); T["prob"].append(p); T["mass"].append(m); T["ent"].append(e)
            yield sid, vol, mask

    with Pool(4) as procs:
        it = procs.imap(sub.build_study, [(s, by_study.get(s, []), data / "train_series") for s in ids])
        R, panels, checks, done = xai.loop(items(it), len(ids), models, tabm, {}, device, t0)
    np.savez_compressed(W / "xai_teacher_raptor384.npz", ids=np.array(T["ids"]), prob=np.array(T["prob"], np.float32),
                        mass=np.array(T["mass"], np.float32), ent=np.array(T["ent"], np.float32))
    xai.save(W, done, models, R, panels, checks, {"source": "kaggle DICOM (sub.build_study)", "decode_errors": errors[0],
             "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "teacher": "raptor384 plain",
             "weights_sha256": SHA, "seconds": round(time.monotonic() - t0, 1)})


if __name__ == "__main__":
    main()
''' % (json.dumps(emb, indent=0), json.dumps(SHA))
(OUT / "run.py").write_text(code)
meta = {"id": "tumis001/rsna-pubkd-r-xai-fold0", "title": "rsna-pubkd-r-xai-fold0", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["tumis001/rsna-coatnet-d96-exp1b-fold0", "tumis001/rsna-v4-l4copy-fold0",
                            "tumis001/rsna-d96-fold0-ids", "tumis001/rsna-pubkd-r-ra-fold0", "tumis001/rsna-pubkd-r-rb-fold0",
                            "nartaa/rsna-knee-publication-swa-weights-20261007"],
        "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("built", OUT / "run.py", len(code), "bytes")
