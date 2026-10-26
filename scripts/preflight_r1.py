"""MeluXina preflight for the L4C-R1 job (first step of the allocation; exit != 0 stops the job): offline pretrained
weights, label / fold / validation hashes, cache layout, MS1 slim table coverage (the MS1-table lanes), one bf16
forward + backward of the l4copy and l3copy arms, and the round-1 runner's self-location (self_table) on 2 real studies
on the GPU, with its random-crop and slice-jitter training items."""
import os
import time

import numpy as np
import torch

t0 = time.monotonic()
print(f"host {os.uname().nodename} torch {torch.__version__} devices {torch.cuda.device_count()} "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
assert torch.cuda.is_available()
os.environ["V4_ARM"] = "l4copy"; os.environ["R1_SJIT"] = "2"
import coatnet_d96_runner as r                            # D96_PHASE=r50_gl
import l4copy_r1_runner as R
m = R.m

import timm
w = timm.create_model("resnet50.tv_in1k", pretrained=True, num_classes=0)
print(f"offline weights ok ({sum(p.numel() for p in w.parameters())} params)", flush=True)
labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
assert r.sha(r.ASSETS / "folds.csv") == r.CFG["sha256"]["folds"]
meta = r.read_manifest()
files = {}
for s, v in meta.items():
    files.setdefault(v["file"], []).append(v["row"])
for f, rows in sorted(files.items()):
    a = np.load(f, mmap_mode="r")
    assert a.shape[1:] == (r.CFG["img"], r.CFG["img"]) and a.dtype == np.uint8, (f, a.shape, a.dtype)
    assert max(rows) + r.CFG["maxs"] <= a.shape[0], (f, max(rows), a.shape)
tab = m.v2.crop_table(meta)
assert set(labels) <= set(meta) and set(labels) <= set(tab), "labelled studies missing from cache / crop table"
print(f"cache ok: {len(meta)} studies in {len(files)} arrays; labels {len(labels)}, validation {len(ref)}", flush=True)
dev = torch.device("cuda")
for arm in ("l4copy", "l3copy"):
    m.ARM = arm
    model, opt, clip = m.build(dev, pretrained=True)
    model.train(); b = 2
    xg = torch.randn(b * 4, 3, 384, 384, device=dev).contiguous(memory_format=torch.channels_last)
    xc = torch.randn(b * 10, 3, 384, 384, device=dev).contiguous(memory_format=torch.channels_last)
    fid = torch.as_tensor(m.CF, device=dev).repeat(b, 1); wc = torch.ones(b, 10, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        g, l = model(xg, b, xc, fid, wc)
    (g.float().mean() + l.float().mean()).backward()
    opt.step()
    print(f"{arm} forward/backward ok, mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    if arm == "l4copy":
        ids = sorted(labels)[:2]; ts = time.monotonic()
        st = R.self_table(model, ids, meta, dev)
        for s in ids:
            S, Rr, Q, A = (x[m.CF] for x in st[s])
            assert S.shape == (10, 4) and (S >= 1).all() and (S <= r.CFG["maxs"] - 2).all() and np.isfinite(A).all()
            assert ((Rr >= 0) & (Rr < 384) & (Q >= 0) & (Q < 384)).all()
        ds = R.R1Studies(ids, meta, labels, r.CFG["train_windows"], True, None); _ = ds[0]
        ds.tab = st; _ = ds[1]
        p = m.predict(model, ids, meta, st, dev)
        assert np.isfinite(p).all()
        print(f"self_table ok ({(time.monotonic() - ts) / len(ids):.2f} s/study incl. warm-up), R1Studies ok, predict ok", flush=True)
    del model, opt; torch.cuda.empty_cache()
print(f"PREFLIGHT_OK {time.monotonic() - t0:.0f}s", flush=True)
