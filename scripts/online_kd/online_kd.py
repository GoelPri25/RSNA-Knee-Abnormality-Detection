"""Online distillation for the r50-GL4 l4copy student (E4 recipe): the Raptor teacher scores the student's own view.

E4's offline target for a study is one number per finding: the public 0.958 blend's rank order with Raptor values,
mixed 0.7 / 0.3 with label-v2 (pubkd_targets_fullorder_rapvalues_mix73.json). It ignores which 16 windows and which
augmentation the student sees in a step. Here the frozen Raptor teacher (nartaa CoAtNet-rmlp-2 384 + 224) runs every
step on the student's exact augmented windows. R6 and the reader cannot run on d96 windows, so the blend part stays
offline and is shifted by how much the Raptor's opinion moves between its clean view and this view:

    v      = 0.65 * z384(view) + 0.35 * z224(view)                    teacher logits on the student's 16 windows
    c      = 0.65 * logit(p384) + 0.35 * logit(p224)                  same, clean view (94 windows, mirror-averaged)
    d      = alpha * clamp(v - c, -D, D)
    y_view = sigmoid(logit(y_pure) + d)                               y_pure = pubkd_targets_fullorder_rapvalues.json
    y      = 0.7 * y_view + 0.3 * y_label-v2  where label-v2 weight > 0, else y_view    (E4's mix73 rule)

With d = 0 this is E4's target exactly (checked at start-up on the training studies).
Teacher views (kit README): the 384 model gets the 320 centre crop of the student's augmented 384 window; the 224 model
the 0.8 centre crop (307 px) resized to 224. Inputs are already ImageNet-normalised by prepare_gl, as the teacher expects.

Env:
  V4_ONLINE=1
  V4_ONLINE_CKPT384, V4_ONLINE_CKPT224    nartaa checkpoints (sha-checked); V4_ONLINE_T224=0 runs the 384 model only
  V4_ONLINE_CLEAN                         dir with raptor384_{0,1}.json and raptor224_{0,1}.json (clean-view predictions)
  V4_ONLINE_PURE                          pubkd_targets_fullorder_rapvalues.json
  V4_ONLINE_MIX (0.3), V4_ONLINE_CLIP (4.0), V4_ONLINE_ALPHA (1.0)
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SHA = {"384": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef",
       "224": "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"}
ARCH = {"384": "coatnet_rmlp_2_rw_384", "224": "coatnet_rmlp_2_rw_224"}
W384, W224 = 0.65, 0.35
IMG = 384


def config():
    return {"ckpt384": Path(os.environ["V4_ONLINE_CKPT384"]).name,
            "ckpt224": Path(os.environ["V4_ONLINE_CKPT224"]).name if os.environ.get("V4_ONLINE_T224", "1") == "1" else None,
            "pure": Path(os.environ["V4_ONLINE_PURE"]).name, "mix": float(os.environ.get("V4_ONLINE_MIX", "0.3")),
            "clip": float(os.environ.get("V4_ONLINE_CLIP", "4.0")), "alpha": float(os.environ.get("V4_ONLINE_ALPHA", "1.0")),
            "views": "384: 320 centre crop of the augmented 384 window; 224: 0.8 centre crop resized to 224",
            "rule": "sigmoid(logit(y_pure) + alpha*clamp(v - c)), then 0.7/0.3 label-v2 mix where label-v2 weight > 0"}


class RaptorClassifier(nn.Module):
    """nartaa's head (as scripts/pubkd_raptor_teacher.py): LayerNorm, per-finding attention, softmax over windows."""

    def __init__(self, backbone, d, n=12):
        super().__init__()
        self.backbone, self.norm = backbone, nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.0), nn.Linear(256, n))
        self.clsW, self.clsb = nn.Parameter(torch.zeros(n, d)), nn.Parameter(torch.zeros(n))

    def forward(self, x):                                         # x [K, 3, H, W] -> logits [12]
        f = torch.cat([self.backbone(c) for c in x.split(16)])
        h = self.norm(f)
        a = torch.softmax(self.att(h).float(), 0).to(h.dtype)
        return (torch.einsum("kn,kf->nf", a, h) * self.clsW).sum(-1) + self.clsb


def load_teacher(model, path, targets, dev, check_sha=True):
    import timm
    path = Path(path)
    if check_sha:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        assert h.hexdigest() == SHA[model], f"teacher {model} checkpoint hash differs"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert list(ck["lab"]) == list(targets), ck["lab"]
    kw = {"img_size": 320} if model == "384" else {}
    bb = timm.create_model(ARCH[model], pretrained=False, num_classes=0, **kw)
    net = RaptorClassifier(bb, bb.num_features)
    net.load_state_dict(ck["model"], strict=True)
    net.eval().requires_grad_(False)
    return net.to(dev).to(memory_format=torch.channels_last)


def teacher_inputs(x, model):
    """x: [K, 3, 384, 384] ImageNet-normalised windows -> the teacher's native view."""
    if model == "384":
        lo = (IMG - 320) // 2
        return x[..., lo:lo + 320, lo:lo + 320].contiguous(memory_format=torch.channels_last)
    c = int(round(IMG * 0.8)); o = (IMG - c) // 2
    return F.interpolate(x[..., o:o + c, o:o + c].float(), size=(224, 224), mode="bilinear",
                         align_corners=False).contiguous(memory_format=torch.channels_last)


def logit(p, eps=1e-5):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def clean_logits(clean_dir, ids, use224):
    """c per study [n, 12] from the stored clean-view teacher predictions (mirror-averaged probabilities)."""
    def load(model):
        pred = {}
        for f in sorted(Path(clean_dir).glob(f"raptor{model}_*.json")):
            d = json.loads(f.read_text()); assert d["sha256"] == SHA[model], f
            pred.update(d["pred"])
        missing = [s for s in ids if s not in pred]
        assert not missing, f"{len(missing)} training studies without clean raptor{model} predictions"
        return np.stack([logit(0.5 * (np.array(pred[s]["plain"]) + np.array(pred[s]["mirror"]))) for s in ids])
    c384 = load("384")
    return W384 * c384 + W224 * load("224") if use224 else c384


class OnlineTeacher:
    def __init__(self, ids, labels, tlabels, targets, dev):
        """ids: the training dataset's study order (batch[11] indexes it); labels: label-v2 {id: (y, w)};
        tlabels: E4's offline targets (mix73), used only for the start-up identity check."""
        self.cfg, self.dev, self.ids = config(), dev, list(ids)
        self.use224 = self.cfg["ckpt224"] is not None
        pure = json.loads(Path(os.environ["V4_ONLINE_PURE"]).read_text())
        assert pure["targets_in_order"] == list(targets)
        pmap = {r_["study_id"]: r_["probabilities"] for r_ in pure["labels"]}
        assert all(s in pmap for s in self.ids), "pure targets must cover every training study"
        y0 = np.array([pmap[s] for s in self.ids], np.float64)
        ylv = np.array([labels[s][0] for s in self.ids], np.float64)
        wlv = np.array([labels[s][1] for s in self.ids], np.float64)
        m = self.cfg["mix"]
        # identity check: with d = 0 the rule reproduces E4's offline target file
        mix = np.where(wlv > 0, (1 - m) * y0 + m * ylv, y0)
        e4 = np.array([tlabels[s][0] for s in self.ids], np.float64)
        err = float(np.abs(mix - e4).max())
        print(f"ONLINE_KD identity check (d = 0 vs offline E4 targets): max |diff| {err:.2e}", flush=True)
        assert err < 2e-4, "the mix rule does not reproduce the offline E4 targets"
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
        self.z0, self.ylv, self.wlv = t(logit(y0)), t(ylv), t(wlv)
        self.c = t(clean_logits(os.environ["V4_ONLINE_CLEAN"], self.ids, self.use224))
        self.t384 = load_teacher("384", os.environ["V4_ONLINE_CKPT384"], targets, dev)
        self.t224 = load_teacher("224", os.environ["V4_ONLINE_CKPT224"], targets, dev) if self.use224 else None
        self.stats = np.zeros(4)                                  # sum |d|, sum d^2, clipped cells, cells

    @torch.no_grad()
    def check_clean(self, meta, n=6):
        """Parity gate: the loaded 384 teacher on the clean plain view (pubkd_raptor_teacher.py's windows) must reproduce
        the stored plain predictions of the first n training studies (max |dp| <= 0.03, the teammate's parity gate)."""
        stored = {}
        for f in sorted(Path(os.environ["V4_ONLINE_CLEAN"]).glob("raptor384_*.json")):
            stored.update(json.loads(f.read_text())["pred"])
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        worst = 0.0
        for s in self.ids[:n]:
            m = meta[s]
            vol = np.asarray(np.load(m["file"], mmap_mode="r")[m["row"]:m["row"] + 96])
            mask = vol.reshape(96, -1).sum(1) > 0
            valid = np.where(mask)[0]
            lo, hi = (int(valid.min()), int(valid.max())) if len(valid) >= 3 else (0, 2)
            cs = np.array([c for c in range(lo + 1, hi)] or [max(1, min((lo + hi) // 2, 94))])
            cs = np.clip(cs[np.linspace(0, len(cs) - 1, 94).round().astype(int)], 1, 94)
            x = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(self.dev).float() / 255.0
            x = (x - mean) / std
            with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda"):
                p = torch.sigmoid(self.t384(teacher_inputs(x, "384")).float()).cpu().numpy()
            worst = max(worst, float(np.abs(p - np.array(stored[s]["plain"])).max()))
        print(f"ONLINE_KD teacher parity (384, clean plain view, {n} studies): max |dp| {worst:.4f}", flush=True)
        assert worst <= 0.03, "teacher does not reproduce its stored clean-view predictions"

    def reset_stats(self):
        self.stats[:] = 0

    def epoch_stats(self):
        n = max(1.0, self.stats[3])
        return {"online_mean_abs_d": round(float(self.stats[0] / n), 4), "online_rms_d": round(float(np.sqrt(self.stats[1] / n)), 4),
                "online_clipped": round(float(self.stats[2] / n), 5)}

    @torch.no_grad()
    def view_logits(self, xg, b):
        """xg [b*k, 3, 384, 384] (prepare_gl training output) -> teacher logits on these windows [b, 12]."""
        x = xg.view(b, -1, *xg.shape[1:])
        out = []
        with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            for i in range(b):
                z = self.t384(teacher_inputs(x[i], "384")).float()
                if self.use224:
                    z = W384 * z + W224 * self.t224(teacher_inputs(x[i], "224")).float()
                out.append(z)
        return torch.stack(out)

    @torch.no_grad()
    def target(self, xg, b, idx):
        idx = idx.to(self.dev).long()
        v = self.view_logits(xg, b)
        raw = v - self.c[idx]
        D = self.cfg["clip"]
        d = self.cfg["alpha"] * raw.clamp(-D, D)
        self.last_logit = self.z0[idx] + d                          # teacher part, before the label-v2 mix
        y = torch.sigmoid(self.last_logit)
        m = self.cfg["mix"]
        y = torch.where(self.wlv[idx] > 0, (1 - m) * y + m * self.ylv[idx], y)
        self.stats += [float(d.abs().sum()), float((d ** 2).sum()), float((raw.abs() > D).sum()), d.numel()]
        return y


class LogitTargets:
    """Offline logit matching (V4_LOGIT > 0 without V4_ONLINE): the teacher logit of each training study is
    logit(pure teacher target) from V4_ONLINE_PURE (pubkd_targets_fullorder_rapvalues.json = E4's teacher part, before
    the 0.3 label-v2 mix). The loss adds V4_LOGIT * MSE(student logit, teacher logit) to BCE on E4's mixed target."""

    def __init__(self, ids, targets, dev):
        pure = json.loads(Path(os.environ["V4_ONLINE_PURE"]).read_text())
        assert pure["targets_in_order"] == list(targets)
        pmap = {r_["study_id"]: r_["probabilities"] for r_ in pure["labels"]}
        assert all(s in pmap for s in ids), "pure targets must cover every training study"
        self.z = torch.as_tensor(logit(np.array([pmap[s] for s in ids], np.float64)), dtype=torch.float32, device=dev)
        print(f"LOGIT_KD teacher logits: {len(ids)} studies, mean {float(self.z.mean()):.3f}, sd {float(self.z.std()):.3f}", flush=True)

    def logits(self, idx):
        return self.z[idx.to(self.z.device).long()]


def logit_cfg():
    return {"weight": float(os.environ["V4_LOGIT"]), "teacher": "online shifted logit" if os.environ.get("V4_ONLINE") == "1"
            else f"logit({Path(os.environ['V4_ONLINE_PURE']).name})", "loss": "weight * mean w*(z_student - z_teacher)^2, global (12) and crop (10) heads"}


class RankBank:
    """Pairwise ranking loss with a memory bank (V4_RANK > 0): for each finding, every study in the micro-batch is paired with
    the other batch studies and with the last `size` studies seen (their logits stored detached). A pair counts when the two
    training targets differ by more than `margin`; the loss pushes the study with the higher target above the other:
    softplus(-(z_hi - z_lo)) (RankNet / AUC surrogate). Gradients flow through the current batch only. mask [12] limits
    findings (the crop head: the 10 cropped findings)."""

    def __init__(self, dev, size=512, margin=0.2, n=12):
        self.size, self.margin, self.dev = size, margin, dev
        self.s = torch.zeros(size, n, device=dev); self.y = torch.zeros(size, n, device=dev)
        self.ptr, self.filled = 0, 0
        self.stats = np.zeros(2)                                   # pairs, steps

    def loss(self, z, y, mask=None):
        z, y = z.float(), y.float()
        S = torch.cat([z, self.s[:self.filled]]); Y = torch.cat([y, self.y[:self.filled]])   # [b + m, 12]
        dz = z[:, None, :] - S[None]                                  # [b, b + m, 12]
        dy = y[:, None, :] - Y[None]
        valid = (dy.abs() > self.margin).float()
        if mask is not None:
            valid = valid * mask.view(1, 1, -1)
        l = F.softplus(-torch.sign(dy) * dz) * valid
        npair = valid.sum()
        out = l.sum() / npair.clamp_min(1)
        b = z.shape[0]; idx = (self.ptr + torch.arange(b, device=self.dev)) % self.size
        self.s[idx], self.y[idx] = z.detach(), y.detach()
        self.ptr = int((self.ptr + b) % self.size); self.filled = min(self.size, self.filled + b)
        self.stats += [float(npair), 1]
        return out


def rank_cfg():
    return {"weight": float(os.environ["V4_RANK"]), "margin": 0.2, "bank": 512,
            "loss": "BCE + weight * pairwise softplus(-(z_hi - z_lo)) on target-ordered pairs (batch + memory bank), global (12) and crop (10) heads"}
