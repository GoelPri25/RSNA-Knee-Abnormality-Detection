"""CPU tests for KD-CV (V4_VIEW=consistent) in scripts/r50_gl4_runner.py / r50_gl_runner.prepare_gl, localhost/rsna-mlx-train:v1,
--network=none, pretrained=False. Cache as test_v4imp_cpu.py (shard_0 = fold-0 export, shard_1 = smoke cache).
Env as a lane + V4_VIEW=consistent (optionally V4_VIEW_MIRROR, V4_EPOCHS). Checks: config; the training global view with
the intensity draw switched off equals the eval view (320 centre crop) exactly and the crop-branch input equals its eval
input; with intensity on, the global view is a monotone pixel map of the eval view (no geometry); 48-window training
batches; one training step gives finite losses and gradients on both branches; BN refresh runs on the 320 crop.
With KD_CV_DEFAULT=1 (no V4_VIEW): the default view is unchanged (16 windows, 384 whole field, full augmentation).
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
ok = lambda msg: print("TEST", msg, "OK", flush=True)

if os.environ.get("KD_CV_DEFAULT") == "1":
    assert not L.VIEW and CFG["train_windows"] == 16 and CFG["train_res"] == 384 and not CFG.get("train_crop")
    assert CFG["aug"] == {"mirror_p": 0.5, "rot_deg": 10.0, "scale": 0.10, "shift": 0.06, "gamma": 0.25, "gain": 0.10}
    assert "kd_view" not in L.G4 and "_view" not in CFG["version"]
    print("ALL KD-CV DEFAULT-PATH CHECKS PASSED", flush=True); sys.exit(0)

mp = float(os.environ.get("V4_VIEW_MIRROR", "0"))
assert L.VIEW == "consistent" and L.ARM == "l4copy" and L.BB == "r50" and L.TARGETS_FILE
assert CFG["train_windows"] == 48 and CFG["train_res"] == CFG["eval_res"] == 320 and CFG["train_crop"]
assert CFG["aug"] == {"mirror_p": mp, "rot_deg": 0.0, "scale": 0.0, "shift": 0.0, "gamma": 0.25, "gain": 0.10}
assert CFG["studies_per_step"] == 2 and CFG["accum"] == 4, "E4's batch: 2 studies x 4"
assert CFG["version"].endswith("_view48" + ("m" if mp else "")) and L.G4["kd_view"]["train_windows"] == 48
if os.environ.get("V4_EPOCHS"):
    E = int(os.environ["V4_EPOCHS"]); assert CFG["epochs"] == E and CFG["swa_epochs"] == list(range(E - 4, E + 1))
ok(f"1 config: 48 windows, train_res = eval_res 320, crop, intensity-only aug, mirror {mp}, epochs {CFG['epochs']} "
   f"SWA {CFG['swa_epochs'][0]}-{CFG['swa_epochs'][-1]}, version {CFG['version']}")

meta = r.read_manifest()
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
ids = sorted(s for s in meta if fold.get(s) not in (None, "0") and s in tab and s in tl)[:2]
assert len(ids) == 2, ids
ds = L.make_ds(ids, meta, tl, CFG["train_windows"], True, tab)
b = torch.utils.data.default_collate([ds[0], ds[1]])
assert b[0].shape[:2] == (2, 48), b[0].shape
ok(f"2 training batch: 2 studies x 48 windows {tuple(b[0].shape)}")

# the same windows through the train path (intensity off) and the eval path (gres 320) must give the same images
x, sag, xc, sagc, cen, fid = b[0], b[1], b[5], b[6], b[8], b[9]
a0 = dict(CFG["aug"]); CFG["aug"] = {**a0, "mirror_p": 0.0, "gamma": 0.0, "gain": 0.0}
torch.manual_seed(0)
gt, ct, _ = v2.prepare_gl(x, sag, xc, sagc, cen, fid, True, dev)
ge, ce, _ = v2.prepare_gl(x, sag, xc, sagc, cen, fid, False, dev, gres=320)
assert gt.shape == ge.shape == (96, 3, 320, 320), gt.shape
assert torch.allclose(gt, ge, atol=1e-3), float((gt - ge).abs().max())     # float32 grid rounding: < 0.1 grey level
assert torch.allclose(ct, ce, atol=1e-3), float((ct - ce).abs().max())
ok(f"3 train view (intensity off) = eval view: global max |diff| {float((gt - ge).abs().max()):.1e}, "
   f"crop {float((ct - ce).abs().max()):.1e}")

CFG["aug"] = {**a0, "mirror_p": 0.0}
torch.manual_seed(1)
gi, _, _ = v2.prepare_gl(x, sag, xc, sagc, cen, fid, True, dev)
m_, s_ = r.MEAN.view(1, 3, 1, 1), r.STD.view(1, 3, 1, 1)
for w in (0, 50):                                            # one window of each study
    p_t, p_e = (gi[w] * s_[0] + m_[0]).flatten(), (gt[w] * s_[0] + m_[0]).flatten()   # same grid, intensity off
    keep = (p_e > 1e-3) & (p_t < 0.999)                       # clamps aside
    o = torch.argsort(p_e[keep]); d = torch.diff(p_t[keep][o])
    assert (d >= -1e-5).all(), f"not a monotone pixel map: geometry changed ({float(d.min()):.2e})"
CFG["aug"] = a0
ok("4 intensity on: training pixels = a monotone map of the eval pixels (no geometry)")

model = L.RaptorGL4(CFG["train_res"], pretrained=False)
with torch.no_grad():
    for bn in model.bns + model.cbns:                        # random init zeroes the last BN gamma of each block
        bn.weight.uniform_(0.5, 1.5)
CM = torch.from_numpy(L.CMASK.astype(np.float32))
model.train()
ds12 = L.make_ds(ids, meta, tl, 12, True, tab)                # host RAM: 2 x 48 windows fp32 backward does not fit on CPU
b12 = torch.utils.data.default_collate([ds12[0], ds12[1]])
parts = L.train_step_loss(model, b12, dev, CM, None, False)
assert torch.isfinite(parts).all() and float(parts[0]) > 0 and float(parts[1]) > 0
(parts[0] + 0.5 * parts[1]).backward()
gr = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
assert any(n.startswith("backbone.layer4") for n in gr) and any(n.startswith("layer4_c.") for n in gr)
assert "clsW" in gr and "clsV" in gr
ok(f"5 training step on the consistent view (12 windows on CPU): losses {[round(float(v), 4) for v in parts]}, grads on both branches")

L.bn_refresh(model, ids, meta, tl, tab, dev)
assert all(torch.isfinite(m.running_mean).all() for m in model.bns)
ok("6 BN refresh on the 320 crop with 48 eval-centre windows")
print("ALL KD-CV CPU TESTS PASSED", flush=True)
