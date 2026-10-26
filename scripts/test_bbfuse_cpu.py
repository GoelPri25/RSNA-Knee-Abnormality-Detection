"""CPU tests for BB-CMP (V4_BB) and V4-FUSE (V4_IMP=fuse) in scripts/r50_gl4_runner.py, localhost/rsna-mlx-train:v1,
--network=none, pretrained=False. Cache as test_v4imp_cpu.py (shard_0 = fold-0 export, shard_1 = smoke cache).
Run once per setting: V4_BB=r50 V4_IMP=fuse (fusion tests + r50 key parity), then V4_BB in cnx / coat0 / effv2s / dino2s / dino3s.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r50_gl4_runner as L  # noqa: E402

r, v2, CFG = L.r, L.v2, L.CFG
CFG["shards"] = 2; CFG["workers"] = 0
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8"))); dev = torch.device("cpu")
meta = r.read_manifest(); labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
tab = {}
for cp in sorted(v2.CP.glob("cp_*.npz")):
    z = np.load(cp)
    for s, cen, at in zip(z["ids"], z["centres"], z["a_top"]):
        s = str(s)
        if s in meta:
            c = np.clip(r.centres(meta[s]["mask"], v2.K, False), 1, v2.MAXS - 2)
            tab[s] = (c[cen[..., 0].astype(int)], cen[..., 1].astype(int), cen[..., 2].astype(int), at.astype(np.float32))
tr = sorted(s for s in meta if fold.get(s) not in (None, "0") and s in tab and s in labels)[:2]
va = sorted(s for s in meta if fold.get(s) == "0" and s in tab)[:2]
ok = lambda msg: print("TEST", msg, "OK", flush=True)
CM = torch.from_numpy(L.CMASK.astype(np.float32))


def step(model, dset):
    batch = torch.utils.data.default_collate([dset[0], dset[1]])
    parts = L.train_step_loss(model, batch, dev, CM)
    assert parts.shape == (2,) and torch.isfinite(parts).all()
    (parts[0] + 0.5 * parts[1]).backward()
    gr = [p.grad for p in model.parameters() if p.grad is not None]
    assert gr and all(torch.isfinite(g).all() for g in gr)
    return parts


if L.IMP == "fuse":
    assert L.BB == "r50"
    L.PAIRS.update(L.pair_table())
    # 1. GLStudiesF: own outputs = v2.GLStudies (same RNG); partner = rows P[c] -1..+1 inside the partner series, zeros
    #    where unpaired; in evaluation every paired centre is kept
    for train in (False, True):
        for s in va + tr:
            np.random.seed(5); a = v2.GLStudies([s], meta, labels, 8, train, tab)[0]
            np.random.seed(5); dsf = L.GLStudiesF([s], meta, labels, 8, train, tab); f = dsf[0]
            assert len(f) == 15 and all(torch.equal(u, v) for u, v in zip(a, f[:11]))
            P, _, lo, hi = L.PAIRS[s]
            m = meta[s]; vol = np.load(m["file"], mmap_mode="r")[m["row"]:m["row"] + 96]
            for xp, okp, c in ((f[11], f[12], dsf._c), (f[13], f[14], dsf._S)):
                okp = okp.numpy()
                assert (P[c][okp] >= 0).all() and xp[~okp].sum() == 0
                if not train:
                    assert np.array_equal(okp, P[c] >= 0)
                for j in np.where(okp)[0][:4]:
                    q = P[c[j]]; rows = np.clip(q + np.arange(-1, 2), lo[q], hi[q])
                    assert lo[q] == lo[rows].min() and torch.equal(xp[j], torch.from_numpy(np.ascontiguousarray(vol[rows])))
    cov = np.mean([(L.PAIRS[s][0][np.array(meta[s]["mask"], bool)] >= 0).mean() for s in meta])
    print(f"paired valid rows {cov:.3f}")
    ok("GLStudiesF own outputs identical to v4, partner triplets inside the partner series")

    # 2. prep: own images = v2.prepare_gl with the same draws; partners get the identical transform
    ds = L.GLStudiesF(va, meta, labels, 8, True, tab); batch = torch.utils.data.default_collate([ds[0], ds[1]])
    for train in (True, False):
        torch.manual_seed(1); xg, xc, part = L.prep(batch, train, dev)
        torch.manual_seed(1); x3, c3, _ = v2.prepare_gl(*batch[:2], batch[5], batch[6], batch[8], batch[9], train, dev)
        assert torch.equal(xg, x3) and torch.equal(xc, c3)
        torch.manual_seed(1); xp3, cp3, _ = v2.prepare_gl(batch[11], batch[1], batch[13], batch[6], batch[8], batch[9], train, dev)
        assert torch.equal(part["xp"], xp3.flatten(0, 0)[part["ig"]]) and torch.equal(part["xcp"], cp3[part["ic"]])
        assert len(part["ig"]) == int(batch[12].sum()) and len(part["ic"]) == int(batch[14].sum())
    ok("prep: own = v4's prepare_gl, partners share the per-study augmentation")

    # 2b. BN refresh: partners share the BN batch with their own windows, as in training
    mb = L.RaptorGL4(128, pretrained=False); seen, enc = [], mb.encode
    mb.encode = lambda x, crop=False: seen.append((crop, x.shape[0])) or enc(x, crop)
    L.bn_refresh(mb, va, meta, labels, tab, dev)
    np.random.seed(5); dsr = L.make_ds(va, meta, labels, CFG["train_windows"], False, tab)
    br = torch.utils.data.default_collate([dsr[0], dsr[1]]); xg, xc, part = L.prep(br, False, dev, gres=CFG["train_res"])
    assert seen == [(False, len(xg) + len(part["xp"])), (True, len(xc) + len(part["xcp"]))] and len(part["xp"]) > 0, seen
    ok(f"bn_refresh encodes own + partner windows {seen}")

    # 3. init equivalence: v4 and v4-fuse from one seed: same state (plus fuse_*), no extra RNG, same outputs at init
    L.IMP = "none"; torch.manual_seed(7); m3 = L.RaptorGL4(CFG["train_res"], pretrained=False).eval(); r3 = torch.rand(3)
    L.IMP = "fuse"; torch.manual_seed(7); mf = L.RaptorGL4(CFG["train_res"], pretrained=False).eval(); rf = torch.rand(3)
    assert torch.equal(r3, rf), "fuse init consumed global RNG"
    s3, sf = m3.state_dict(), mf.state_dict()
    assert {k.split(".")[0] for k in set(sf) - set(s3)} == {"fuse_g", "fuse_c"} and all(torch.equal(s3[k], sf[k]) for k in s3)
    eds = L.GLStudiesF(va, meta, labels, 6, False, tab); eb = torch.utils.data.default_collate([eds[0]])
    with torch.no_grad():
        xg, xc, part = L.prep(eb, False, dev)
        gf, lf = mf(xg, 1, xc, eb[9], eb[10], part)
        L.IMP = "none"; g3, l3 = m3(xg, 1, xc, eb[9], eb[10]); L.IMP = "fuse"
    assert len(part["ig"]) > 0 and (gf - g3).abs().max() < 1e-5 and (lf - l3).abs().max() < 1e-5
    ok("init equivalence with partners present (fuse output layer zero), no extra RNG draws")

    # 4. the partner path learns at the head LR: fuse params in the head group, output layer gets gradient at step 1,
    #    and after one SGD step the partner changes the output (not so with partner features shuffled to zero pairs)
    CFG["train_res"] = 128; v2.GL["crop_res"] = 128
    torch.manual_seed(0); L.IMP = "fuse"
    model, opt, clip = L.build(dev, pretrained=False)
    fz = {id(p) for n, p in model.named_parameters() if n.startswith("fuse_")}
    assert fz <= {id(p) for p in opt.param_groups[1]["params"]} and opt.param_groups[1]["lr"] == CFG["head_lr"]
    model.train(); dsT = L.GLStudiesF(tr, meta, labels, 4, True, tab)
    parts = step(model, dsT)
    assert model.fuse_g.fc2.weight.grad.abs().sum() > 0 and model.fuse_c.fc2.weight.grad.abs().sum() > 0
    assert model.fuse_stats[1] > 0 and model.fuse_stats[0] == 0                   # delta exactly 0 at init
    with torch.no_grad():
        for mod in (model.fuse_g, model.fuse_c):
            mod.fc2.weight.normal_(0, 0.01)
    model.eval()
    with torch.no_grad():
        xg, xc, part = L.prep(eb, False, dev); g1, _ = model(xg, 1, xc, eb[9], eb[10], part)
        xg, xc, part0 = L.prep(eb, False, dev, nopartner=True); g0, _ = model(xg, 1, xc, eb[9], eb[10], part0)
    assert len(part0["ig"]) == 0 and (g1 - g0).abs().max() > 1e-4
    print(f"  fuse step: loss parts {parts.detach().numpy().round(4).tolist()}; partner effect {(g1 - g0).abs().max():.4f}")
    ok("fuse params at head LR, gradient at step 1, partner changes the output, no-partner path")

    # 5. SWA strict load; v4 weights load into the r50 path unchanged (state-dict keys identical to v4)
    sd = model.state_dict(); model.load_state_dict({k: v.clone() for k, v in sd.items()}, strict=True)
    L.IMP = "none"; m4 = L.RaptorGL4(128, pretrained=False)
    w = Path(os.environ.get("V4_WEIGHTS", "/v4w/weights_swa21_25.pt"))
    if w.is_file():
        m4.load_state_dict(torch.load(w, map_location="cpu"), strict=True); print(f"  loaded {w.name} strict")
    ok("SWA strict load, r50 keys = v4")
else:
    BB, spec = L.BB, L.BACKBONES[L.BB]
    print(f"{BB}: {CFG['arch']} norm {r.MEAN.flatten().tolist()} train/eval/crop res {CFG['train_res']}/{CFG['eval_res']}/"
          f"{v2.GL['crop_res']} lr {CFG['lr']} layer_decay {CFG.get('layer_decay')} drop_path {CFG.get('drop_path')}")
    res = {"dino2s": 126, "coat0": 128}.get(BB, 128)
    torch.manual_seed(0); model = L.RaptorGL4(res, pretrained=False).eval()
    # 1. structure: crop copies of the last stage, BN wrapped (BatchNormAct keeps its activation), counts
    assert set(model.crop_parts) == {p.replace(".", "_") for p in spec["last"]}
    assert len(model.bns) == spec["bns"], len(model.bns)
    x = torch.randn(3, 3, res, res).contiguous(memory_format=torch.channels_last)
    orig = {p: L._get(model.backbone, p) for p in spec["last"]}
    with torch.no_grad():
        fg, fc = model.encode(x, False), model.encode(x, True)
    assert all(L._get(model.backbone, p) is m for p, m in orig.items()), "crop pass did not restore the backbone"
    assert all(not m.crop for m in model.bns) and all(m.crop for m in model.cbns)
    assert (fg - fc).abs().max() < 1e-4, (fg - fc).abs().max()                   # identical copies, same statistics
    if spec["bns"]:
        acts = [m for m in model.bns if m.post is not None]
        assert acts, "BatchNormAct2d lost its activation"
        with torch.no_grad():
            ref = __import__("timm").create_model(CFG["arch"], pretrained=False, num_classes=0,
                                                  **({"img_size": res} if BB == "coat0" else {})).eval()
            ref.load_state_dict({k[len("backbone."):]: v for k, v in model.state_dict().items()
                                 if k.startswith("backbone.") and "crop_" not in k}, strict=True)
            assert (ref(x) - fg).abs().max() < 1e-4, "global path differs from the plain timm model"
    with torch.no_grad():
        for p_ in model.crop_parts.parameters():
            p_.add_(0.01 * torch.randn_like(p_))
        fg2, fc2 = model.encode(x, False), model.encode(x, True)
    assert torch.equal(fg, fg2) and (fc2 - fc).abs().max() > 1e-4
    ok("crop copy routes only the crop pass; global path = plain timm model; BN counts")

    # 2. optimiser groups: crop copies at their source's LR (layer decay for DINOv2), heads at head LR, complete
    CFG["train_res"] = res; v2.GL["crop_res"] = res                                    # CoAtNet: built at this size
    model, opt, clip = L.build(dev, pretrained=False)
    lr_of = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    assert len(lr_of) == len(list(model.parameters()))
    names = dict(model.named_parameters())
    for p in spec["last"]:
        src = dict(L._get(model.backbone, p).named_parameters()); cpy = dict(model.crop_parts[p.replace(".", "_")].named_parameters())
        assert set(src) == set(cpy) and all(lr_of[id(src[k])] == lr_of[id(cpy[k])] for k in src)
    print("  LR groups: " + ", ".join(f"{g['lr']:.2e} x {len(g['params'])}" for g in opt.param_groups))
    # 3. one training step at small resolution, then SWA strict load
    model.train(); parts = step(model, v2.GLStudies(tr, meta, labels, 4, True, tab))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.crop_parts.parameters())
    sd = model.state_dict(); model.load_state_dict({k: v.clone() for k, v in sd.items()}, strict=True)
    print(f"  {BB} step: loss parts {parts.detach().numpy().round(4).tolist()}; params "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (crop copy {sum(p.numel() for p in model.crop_parts.parameters()) / 1e6:.1f}M)")
    ok("optimiser groups, one step, SWA strict load")
print("ALL_TESTS_OK")
