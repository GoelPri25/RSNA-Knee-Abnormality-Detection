"""Exp 5a step 1: continued self-supervised pretraining (Barlow Twins) of a ResNet-50 on the d96 knee-MRI cache (H100).

Start: the Barlow Twins ImageNet ResNet-50 (facebookresearch/barlowtwins, MIT licence,
dl.fbaipublicfiles.com/barlowtwins/ljng/resnet50.pth; torchvision key names = timm resnet50 key names). Chosen from public
transfer benchmarks, not by our runs: Hosseinzadeh Taher et al. 2021 (arXiv 2108.05930, 7 medical tasks) name SwAV, Barlow
Twins, SeLa-v2 and DeepCluster-v2 as consistently beating supervised ImageNet; Barlow Twins is the only one of the four
under a permissive licence.
Data: training folds only (fold != 0) and no gold studies, so fold 0 stays an honest ruler; no labels. A sample is one of
the runner's 3-slice windows (c-1, c, c+1), c in the study's valid range (slot boundaries ignored, as in the runner).
Each dataset item is one study with WINDOWS_PER_STUDY random windows (amortises reads); one "epoch" = REPEATS passes over
the studies, about one pass over every window.
Two views per window, augmented on the GPU: random resized crop (area 0.25-1 of the 384 field, aspect 3/4-4/3) to 224,
anatomical mirror (sagittal: reverse the slice channels; others: flip width), rotation +-15 deg, gamma / gain jitter,
Gaussian blur (p 0.5), Gaussian noise (p 0.5).
Loss: Barlow Twins, projector 8192-8192-8192, lambda 0.0051 (paper defaults). The projector is new (not in the public
checkpoint), so the backbone is frozen for the first PROJECTOR_WARMUP steps; then AdamW, backbone 1e-4 / projector 1e-3,
wd 1e-4, linear warm-up then cosine to 0. bf16 autocast, loss in fp32.

Usage: SSL_MODE=bench (timing only) or SSL_MODE=train; D96_CACHE, D96_ASSETS, SSL_OUT, SSL_INIT (checkpoint path),
SSL_EPOCHS. Writes backbone_epochNN.pt (timm resnet50 state dict, num_classes=0), recovery.pt, history.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path(os.environ.get("D96_CACHE", "/root/vol/d96"))
ASSETS = Path(os.environ.get("D96_ASSETS", "/root/assets"))
OUT = Path(os.environ.get("SSL_OUT", "/root/vol/experiments/ssl_barlow_d96"))
INIT = Path(os.environ.get("SSL_INIT", "/root/bt_resnet50.pth"))
MODE = os.environ.get("SSL_MODE", "bench")
CFG = {"version": "ssl_barlow_d96_v1", "epochs": int(os.environ.get("SSL_EPOCHS", "10")), "shards": 4, "maxs": 96,
       "windows_per_study": 8, "studies_per_batch": 32, "repeats": 12, "crop": 224, "area": [0.25, 1.0],
       "aspect": [0.75, 1.3333], "rot_deg": 15.0, "gamma": 0.4, "gain": 0.2, "blur_p": 0.5, "blur_sigma": [0.1, 1.5],
       "noise_p": 0.5, "noise_std": 0.03, "proj": 8192, "lambd": 0.0051, "lr": 1e-4, "proj_lr": 1e-3, "wd": 1e-4,
       "projector_warmup": 500, "warmup_frac": 0.03, "seed": 2026, "workers": 10,
       "folds_sha": "2dd1044adcc54ca4d9bcb9c9fd3b910a6784bd2b99e792bbeaf84a42d417f06a"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SLOT_OF = np.repeat(np.arange(5), [26, 22, 18, 12, 18])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def studies():
    assert sha(ASSETS / "folds.csv") == CFG["folds_sha"]
    fold = {r["study_id"]: r["fold"] for r in csv.DictReader((ASSETS / "folds.csv").open())}
    meta = {}
    for i in range(CFG["shards"]):
        root = CACHE / f"shard_{i}"
        for line in (root / "manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["gold"] or fold.get(r["study_id"], "0") == "0":
                continue
            valid = np.where(np.array(r["mask"], bool))[0]
            lo, hi = (int(valid.min()), int(valid.max())) if len(valid) >= 3 else (0, 2)
            meta[r["study_id"]] = {"file": str(root / r["part"]), "row": r["row_start"], "lo": lo, "hi": hi}
    assert len(meta) == 3524, len(meta)
    return meta


class Windows(torch.utils.data.Dataset):
    def __init__(self, meta):
        self.ids, self.meta, self._mm = sorted(meta), meta, {}

    def __len__(self):
        return len(self.ids) * CFG["repeats"]

    def __getitem__(self, i):
        m = self.meta[self.ids[i % len(self.ids)]]
        if m["file"] not in self._mm:
            self._mm[m["file"]] = np.load(m["file"], mmap_mode="r")
        cs = np.arange(m["lo"] + 1, m["hi"]) if m["hi"] - m["lo"] >= 2 else np.array([1])
        c = np.sort(np.random.choice(cs, CFG["windows_per_study"], replace=len(cs) < CFG["windows_per_study"]))
        rows = np.stack([c - 1, c, c + 1], 1)
        uniq, inv = np.unique(rows, return_inverse=True)
        block = np.asarray(self._mm[m["file"]][m["row"] + uniq])               # only the rows needed
        win = block[inv.reshape(rows.shape)]                                    # [W, 3, H, W] uint8
        return torch.from_numpy(np.ascontiguousarray(win)), torch.from_numpy(np.isin(SLOT_OF[c], (0, 1)))


def view(x, sag):
    """x uint8 [N, 3, 384, 384] on the GPU -> one augmented, normalised view [N, 3, 224, 224]."""
    n, dev = x.shape[0], x.device
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev)
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where((flip & sag).view(-1, 1, 1, 1), x.flip(1), x)
    x = torch.where((flip & ~sag).view(-1, 1, 1, 1), x.flip(-1), x).float().div_(255.0)
    area, aspect = u(*CFG["area"]), torch.exp(u(math.log(CFG["aspect"][0]), math.log(CFG["aspect"][1])))
    w, h = (area * aspect).sqrt().clamp(max=1.0), (area / aspect).sqrt().clamp(max=1.0)
    tx, ty = (1 - w) * u(-1, 1), (1 - h) * u(-1, 1)
    ang = u(-1, 1) * math.radians(CFG["rot_deg"])
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([w * cos, -h * sin, tx], 1), torch.stack([w * sin, h * cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (n, 3, CFG["crop"], CFG["crop"]), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gamma = torch.exp(u(-CFG["gamma"], CFG["gamma"])).view(-1, 1, 1, 1)
    gain = (1 + u(-CFG["gain"], CFG["gain"])).view(-1, 1, 1, 1)
    x = (x.clamp_min(1e-4).pow(gamma) * gain).clamp_(0, 1)
    sigma = u(*CFG["blur_sigma"]) * (torch.rand(n, device=dev) < CFG["blur_p"])
    t = torch.arange(-4, 5, device=dev, dtype=torch.float32)
    k = torch.exp(-t[None] ** 2 / (2 * sigma.clamp_min(1e-3)[:, None] ** 2))
    k = torch.where((sigma > 0)[:, None], k, (t == 0).float()[None].expand_as(k))
    k = (k / k.sum(1, keepdim=True)).repeat_interleave(3, 0)                   # [N*3, 9]
    y = x.reshape(1, n * 3, CFG["crop"], CFG["crop"])
    y = F.conv2d(F.pad(y, (4, 4, 0, 0), mode="reflect"), k[:, None, None, :], groups=n * 3)
    y = F.conv2d(F.pad(y, (0, 0, 4, 4), mode="reflect"), k[:, None, :, None], groups=n * 3)
    x = y.view(n, 3, CFG["crop"], CFG["crop"])
    noise = CFG["noise_std"] * torch.rand(n, device=dev) * (torch.rand(n, device=dev) < CFG["noise_p"])
    x = (x + noise.view(-1, 1, 1, 1) * torch.randn_like(x)).clamp_(0, 1)
    x = (x - MEAN.to(dev)) / STD.to(dev)
    return x.contiguous(memory_format=torch.channels_last)


class Barlow(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model("resnet50", pretrained=False, num_classes=0)
        sd = torch.load(INIT, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        self.backbone.load_state_dict(sd, strict=True)
        p = CFG["proj"]
        self.projector = nn.Sequential(nn.Linear(2048, p, bias=False), nn.BatchNorm1d(p), nn.ReLU(inplace=True),
                                       nn.Linear(p, p, bias=False), nn.BatchNorm1d(p), nn.ReLU(inplace=True),
                                       nn.Linear(p, p, bias=False))
        self.bn = nn.BatchNorm1d(p, affine=False)

    def forward(self, v1, v2):
        f = self.backbone(torch.cat([v1, v2]))
        z = self.projector(f).float()
        z1, z2 = z.chunk(2)
        with torch.autocast("cuda", enabled=False):                             # cross-correlation in fp32
            c = self.bn(z1).T @ self.bn(z2) / z1.shape[0]
        on = (torch.diagonal(c) - 1).pow(2).sum()
        off = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
        fs = F.normalize(f.float(), dim=1).std(0).mean()                       # collapse check (~1/sqrt(2048) = 0.022 if healthy-ish)
        return on + CFG["lambd"] * off, on.detach(), off.detach(), fs.detach()


def loader(ds, seed):
    return torch.utils.data.DataLoader(ds, batch_size=CFG["studies_per_batch"], shuffle=True, drop_last=True,
                                       num_workers=CFG["workers"], pin_memory=True, persistent_workers=False,
                                       worker_init_fn=lambda w: np.random.seed((seed * 1000 + w) % 2**32))


def main():
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    meta = studies()
    ds = Windows(meta)
    print(f"SSL {MODE}: {len(meta)} training-fold studies, init {INIT} sha {sha(INIT)[:16]}", flush=True)
    model = Barlow().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": CFG["lr"]},
                             {"params": model.projector.parameters(), "lr": CFG["proj_lr"]}], weight_decay=CFG["wd"])
    steps_per_epoch = len(ds) // CFG["studies_per_batch"]
    total = steps_per_epoch * CFG["epochs"]; warm = max(1, int(CFG["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))

    def step(x, sag, global_step):
        x, sag = x.to(dev, non_blocking=True).flatten(0, 1), sag.to(dev).flatten()
        frozen = global_step < CFG["projector_warmup"]
        for p in model.backbone.parameters():
            p.requires_grad_(not frozen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, on, off, fs = model(view(x, sag), view(x, sag))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        return loss.item(), on.item(), off.item(), fs.item()

    if MODE == "bench":
        dl = loader(ds, CFG["seed"]); it = iter(dl)
        batches = [next(it) for _ in range(10)]                                 # worker start-up excluded
        t0 = time.monotonic(); batches += [next(it) for _ in range(30)]
        data_s = (time.monotonic() - t0) / 30                                   # wall time per batch, all workers
        model.train()
        for i in range(5):
            step(*batches[i], 10**9)
        torch.cuda.synchronize(); t0 = time.monotonic()
        for i in range(5, 40):
            r = step(*batches[i], 10**9)
        torch.cuda.synchronize(); gpu_s = (time.monotonic() - t0) / 35
        win = CFG["studies_per_batch"] * CFG["windows_per_study"]
        per_epoch = steps_per_epoch * max(data_s, gpu_s)
        print(f"BENCH windows/batch {win}; data loading {data_s:.3f}s/batch with {CFG['workers']} workers "
              f"({win / max(data_s, 1e-9):.0f} windows/s), GPU step {gpu_s:.3f}s "
              f"({win / gpu_s:.0f} windows/s); steps/epoch {steps_per_epoch}; est. {per_epoch / 60:.1f} min/epoch; "
              f"last loss {r[0]:.1f} on {r[1]:.1f} off {r[2]:.1f} feat-std {r[3]:.4f}; "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    start, history, gstep = 1, [], 0
    rec = OUT / "recovery.pt"
    if rec.is_file():
        r = torch.load(rec, map_location="cpu", weights_only=False)
        assert r["cfg"] == CFG, "recovery config differs"
        model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"]); sched.load_state_dict(r["sched"])
        start, history, gstep = r["epoch"] + 1, r["history"], r["gstep"]
        print(f"RESUMED after epoch {r['epoch']}", flush=True)
    for epoch in range(start, CFG["epochs"] + 1):
        model.train(); t0 = time.monotonic(); acc = np.zeros(4); n = 0
        for i, (x, sag) in enumerate(loader(ds, CFG["seed"] + epoch)):
            r = step(x, sag, gstep); gstep += 1; acc += r; n += 1
            if (i + 1) % 200 == 0:
                a = acc / n
                print(f"  epoch {epoch} step {i + 1}/{steps_per_epoch} loss {a[0]:.1f} on {a[1]:.1f} off {a[2]:.1f} "
                      f"feat-std {a[3]:.4f} {(time.monotonic() - t0) / (i + 1):.3f}s/step", flush=True)
        a = acc / max(n, 1)
        history.append({"epoch": epoch, "loss": a[0], "on": a[1], "off": a[2], "feat_std": a[3],
                        "seconds": round(time.monotonic() - t0, 1), "lr": [g["lr"] for g in opt.param_groups]})
        torch.save(model.backbone.state_dict(), OUT / f"backbone_epoch{epoch:02d}.pt")
        torch.save({"cfg": CFG, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "gstep": gstep, "history": history}, OUT / "recovery.tmp")
        os.replace(OUT / "recovery.tmp", rec)
        (OUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"EPOCH {epoch}: loss {a[0]:.1f} on {a[1]:.1f} off {a[2]:.1f} feat-std {a[3]:.4f} "
              f"({history[-1]['seconds']:.0f}s)", flush=True)
    torch.save(model.backbone.state_dict(), OUT / "backbone_final.tmp")
    os.replace(OUT / "backbone_final.tmp", OUT / "backbone_final.pt")
    print("SSL_DONE " + json.dumps(history[-1]) + f" backbone_final sha {sha(OUT / 'backbone_final.pt')[:16]}", flush=True)


if __name__ == "__main__":
    main()
