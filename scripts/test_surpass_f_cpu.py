"""CPU tests for SURPASS-F in scripts/r50_gl4_runner.py (V4_EXT_CACHE, V4_NOISE, V4_EPOCHS, V4_STAGE / V4_INIT,
V4_EVAL_ONLY helpers) and scripts/build_fm_targets.py's output layout; localhost/rsna-mlx-train:v1, --network=none,
pretrained=False. Competition cache as test_pubkd_cpu.py (shard_0 = fold-0 export, shard_1 = smoke cache); extra cache =
a small fastMRI d96 build (V4_EXT_CACHE, 2 shards); targets = PUB-KD-R's file + the extra studies but the holdout.
Env: V4_ARM=l4copy V4_TARGETS V4_TARGETS_SHA V4_EXT_CACHE V4_EXT_SHARDS=2 V4_EXT_HOLDOUT V4_NOISE=1 V4_EPOCHS=20.
"""
import csv
import json
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
ok = lambda msg: print("TEST", msg, "OK", flush=True)

assert L.EXT and L.NOISE and L.STAGE == "joint" and L.BB == "r50" and L.ARM == "l4copy"
assert CFG["epochs"] == 20 and CFG["swa_epochs"] == [16, 17, 18, 19, 20] and CFG["drop_path"] == 0.2
assert CFG["aug"]["rot_deg"] == 15.0 and CFG["aug"]["gamma"] == 0.35 and CFG["arch"] == "resnet50.tv_in1k"
sf = L.G4["surpass_f"]
assert sf["noise"]["feature_dropout"] == 0.3 and sf["ext_loss"] == "global only" and sf["epochs"] == 20
assert CFG["version"].endswith("_sf_joint_ext_noise") and "_kd_" in CFG["version"]
ok("1 config: 20 epochs / SWA 16-20, noise (drop path 0.2, feature dropout 0.3, stronger aug), version tag")

meta = r.read_manifest(); labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
tl = r.load_labels(Path(L.TARGETS_FILE), "kd_targets")
fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
tab = {}
for c_ in sorted(v2.CP.glob("cp_*.npz")):
    z = np.load(c_)
    for s, cen, at in zip(z["ids"], z["centres"], z["a_top"]):
        s = str(s)
        if s in meta:
            c = np.clip(r.centres(meta[s]["mask"], v2.K, False), 1, v2.MAXS - 2)
            tab[s] = (c[cen[..., 0].astype(int)], cen[..., 1].astype(int), cen[..., 2].astype(int), at.astype(np.float32))
n_comp = len(meta)
ext_ids, hold_ids = L.ext_setup(meta, tab, tl)
assert ext_ids and hold_ids and not set(ext_ids) & set(hold_ids) and len(meta) > n_comp
for s in ext_ids + hold_ids:
    m = meta[s]
    assert Path(m["file"]).is_file() and m["mask"].shape == (96,) and m["mask"].sum() >= 3
    assert np.isfinite(m["pos"]).all() and m["source"].count("/") == 1
    assert all(a.shape == (12, v2.GL["top"]) for a in tab[s])
ok(f"2 extra cache: {len(ext_ids)} trained + {len(hold_ids)} holdout studies merged, placeholder crop rows, positions")

comp = sorted(s for s in meta if fold.get(s) not in (None, "0") and s in tab and s in tl and not s.startswith("fm_"))[:2]
ids = comp + ext_ids
ds = v2.GLStudies(ids, meta, tl, 4, True, tab)
smp = L.SourceBatches(len(comp), len(ext_ids), 2, 2026)
smp.set_epoch(1); b1 = [list(b) for b in smp]; e1 = list(smp.is_ext)
assert sorted(i for b in b1 for i in b) == list(range(len(ids)))
assert all(all((i >= len(comp)) == e for i in b) for b, e in zip(b1, e1))
smp.set_epoch(1); assert [list(b) for b in smp] == b1
smp.set_epoch(2); assert sorted(i for b in smp for i in b) == list(range(len(ids)))
dl = torch.utils.data.DataLoader(ds, batch_sampler=smp, num_workers=0)
assert len(dl) == len(smp)
ok("3 source batches: one source per batch, every study once, seeded per epoch, DataLoader batch_sampler")

model = L.RaptorGL4(CFG["train_res"], pretrained=False)
assert model.backbone.drop_path_rate == 0.2 if hasattr(model.backbone, "drop_path_rate") else True
CM = torch.from_numpy(L.CMASK.astype(np.float32))
model.train()
bx = torch.utils.data.default_collate([ds[len(comp)]])
item = ds[len(comp)]
assert np.allclose(item[2].numpy(), tl[ext_ids[0]][0]), "extra study's y = its teacher target"
parts = L.train_step_loss(model, bx, dev, CM, None, True)
assert torch.isfinite(parts).all() and float(parts[1]) == 0.0
(parts[0] + 0.5 * parts[1]).backward()
gr = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
assert any(n.startswith("backbone.layer4") for n in gr) and "clsW" in gr
assert not any(n.startswith(("layer4_c.", "clsV", "clsvb", "norm_l.")) for n in gr), "crop branch touched by an extra batch"
ok("4 extra batch: global loss only, local 0, gradients reach the shared backbone / global head, none the crop branch")

model.zero_grad(set_to_none=True)
bc = torch.utils.data.default_collate([ds[0]])
parts = L.train_step_loss(model, bc, dev, CM, None, False)
assert torch.isfinite(parts).all() and float(parts[1]) > 0
(parts[0] + 0.5 * parts[1]).backward()
gr = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
assert any(n.startswith("layer4_c.") for n in gr) and "clsV" in gr and "clsW" in gr
ok("5 competition batch: both losses, gradients reach both branches")

model.eval()
xg, xcr, _ = L.prep(bc, False, dev)
fid, wc = v2.to_dev(dev, bc[9], bc[10])
with torch.no_grad():
    g1, _ = model(xg, 1, xcr, fid, wc); g2 = model.forward_global(xg, 1)
assert torch.allclose(g1, g2, atol=1e-5), "forward_global != forward's global head in eval"
model.train()
with torch.no_grad():
    t1, t2 = model.forward_global(xg, 1), model.forward_global(xg, 1)
assert not torch.allclose(t1, t2), "noise inactive in training"
ok("6 forward_global = forward's global head (eval, deterministic); noise active in training")

ph = L.predict_global(model, hold_ids, meta, tab, dev)
assert ph.shape == (len(hold_ids), 12) and ((ph > 0) & (ph < 1)).all()
L.OUT.mkdir(parents=True, exist_ok=True)
r.save_predictions(L.OUT / "fm_holdout_global.json", hold_ids, ph, "test")
d = json.loads((L.OUT / "fm_holdout_global.json").read_text())
assert [p["study_id"] for p in d["predictions"]] == hold_ids
ok("7 holdout global predictions [n, 12] saved in the predictions layout")

pre = L.RaptorGL4(CFG["train_res"], pretrained=False)
with torch.no_grad():
    for p in pre.backbone.layer4.parameters():
        p.add_(0.01)
    for m in pre.backbone.layer4.modules():
        if isinstance(m, L.DualBN4):
            m.running_mean.add_(0.5)
torch.save(pre.state_dict(), L.OUT / "pre_swa.pt")
fine = L.RaptorGL4(CFG["train_res"], pretrained=False)
L.init_from(fine, L.OUT / "pre_swa.pt")
sd, ps = fine.state_dict(), pre.state_dict()
for k, v in pre.backbone.layer4.state_dict().items():
    if "crop_" not in k:
        assert torch.equal(fine.layer4_c.state_dict()[k], v), k
for c, g in zip(fine.layer4_c.modules(), fine.backbone.layer4.modules()):
    if isinstance(c, L.DualBN4):
        assert torch.equal(c.crop_mean, g.running_mean) and torch.equal(c.crop_var, g.running_var)
assert all(torch.equal(sd[k], ps[k]) for k in ps if not k.startswith("layer4_c."))
ok("8 fine-stage init: pre weights strict, crop layer 4 = pre-trained layer 4, crop statistics = its global ones")

m2 = L.RaptorGL4(CFG["train_res"], pretrained=False)
m2.load_state_dict(model.state_dict(), strict=True)
ok("9 state dict round trip strict (no new parameters or buffers: ra / v4 weights load into SURPASS-F models)")
print("ALL SURPASS-F CPU TESTS PASSED", flush=True)
