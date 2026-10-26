"""CPU tests for PUB-KD in scripts/r50_gl4_runner.py (V4_BB=rap2 and V4_TARGETS), localhost/rsna-mlx-train:v1,
--network=none. Cache as test_bbfuse_cpu.py (shard_0 = fold-0 export, shard_1 = smoke cache). V4_RAPTOR_DIR = the public
nartaa checkpoints. Checks: rap2 loads the public weights exactly (backbone + global head), the crop branch / crop head
start as copies of them, BatchNorm count, teacher targets load by sha and feed the training dataset, one training step
(gradient checkpointing on) gives finite losses and gradients, SWA-style state round trip is strict.
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
RAP = L.BB == "rap2"                                     # V4_BB=rap2 (lanes C/D) or r50 (lanes A/B, PUB-KD-R)
assert L.BB in ("rap2", "r50") and L.ARM == "l4copy" and L.TARGETS_FILE
assert L.TARGETS_SHA in CFG["sha256"].values() and "_kd_" in CFG["version"]
if RAP:
    assert CFG["epochs"] == 6 and CFG["swa_epochs"] == [4, 5, 6] and CFG["lr"] == 5e-5 and CFG["head_lr"] == 5e-4
    assert CFG["studies_per_step"] == 1 and CFG["accum"] == 8 and CFG["grad_ckpt"] and CFG["eval_res"] == 384
    ok("1 config: fine-tune schedule, 1 x 8 studies, gradient checkpointing, eval 384")
else:
    assert CFG["epochs"] == 25 and CFG["arch"] == "resnet50.tv_in1k" and CFG["eval_res"] == 320
    ok("1 config: v4 l4copy r50 schedule unchanged, targets pinned in CFG")

model = L.RaptorGL4(CFG["train_res"], pretrained=RAP)
if not RAP:
    assert hasattr(model, "layer4_c") and len(model.bns) == 53
    ok("2 r50 l4copy model (random init on CPU: no ImageNet download offline)")
ck = None if not RAP else torch.load(Path(os.environ["V4_RAPTOR_DIR"]) / L.BACKBONES["rap2"]["init"], map_location="cpu", weights_only=True)["model"]
if RAP:
    sd = model.state_dict()
    for k, v in ck.items():
        assert torch.equal(sd[k], v), k
    assert len(model.bns) == 25 and model.backbone.stages[0].grad_checkpointing
    cp = model.crop_parts
    for k, v in model.backbone.stages[3].state_dict().items():
        assert torch.equal(cp["stages_3"].state_dict()[k], v), k
    for k, v in model.backbone.norm.state_dict().items():
        assert torch.equal(cp["norm"].state_dict()[k], v), k
    assert torch.equal(model.clsV, model.clsW) and torch.equal(model.clsvb, model.clsb)
    assert torch.equal(model.norm_l.weight, model.norm.weight) and torch.equal(model.norm_l.bias, model.norm.bias)
    ps = model.param_sets()
    print("PARAMS", {k: round(sum(p.numel() for p in v) / 1e6, 2) for k, v in ps.items()}, flush=True)
    ok("2 public weights loaded exactly; crop stage / norm / head are copies; 25 BatchNorms")

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
tr = sorted(s for s in meta if fold.get(s) not in (None, "0") and s in tab and s in tl)[:1]
assert tr, "no training study with crops and targets in the test cache"
ds = v2.GLStudies(tr, meta, tl, 4, True, tab)
item = ds[0]
assert np.allclose(item[2].numpy(), tl[tr[0]][0]) and np.allclose(item[3].numpy(), tl[tr[0]][1])
ok("3 teacher targets load by sha and are the training dataset's y / w")

model.train()
batch = torch.utils.data.default_collate([item])
CM = torch.from_numpy(L.CMASK.astype(np.float32))
parts = L.train_step_loss(model, batch, dev, CM)
assert parts.shape == (2,) and torch.isfinite(parts).all()
(parts[0] + 0.5 * parts[1]).backward()
gr = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
assert all(torch.isfinite(g).all() for g in gr.values())
assert any(n.startswith(("crop_parts.", "layer4_c.")) for n in gr)
assert any(n.startswith(("backbone.stages.3", "backbone.layer4")) for n in gr)
assert "clsV" in gr and "clsW" in gr
print("LOSS global/local", parts.detach().tolist(), flush=True)
ok("4 one training step (16 -> 4 windows, crops, grad checkpointing): finite losses, gradients reach both branches")

m2 = L.RaptorGL4(CFG["train_res"], pretrained=False)
m2.load_state_dict(model.state_dict(), strict=True)
ok("5 state dict round trip strict (SWA / submission load path)")
print("ALL PUBKD CPU TESTS PASSED", flush=True)
