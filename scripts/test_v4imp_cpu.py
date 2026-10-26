"""CPU tests for the V4-IMP lanes of scripts/r50_gl4_runner.py (docs/v4_improve_spec.md v2), localhost/rsna-mlx-train:v1,
--network=none. Cache: shard_0 = the fold-0 export, shard_1 = the smoke cache (training-fold studies).
Env as a lane (D96_PHASE=r50_gl, V4_ARM=l4copy, GL_CROPS, GL_OUT, D96_CACHE, D96_ASSETS); V4_IMP is switched here.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r50_gl4_runner as L  # noqa: E402

r, v2, CFG = L.r, L.v2, L.CFG
CFG["shards"] = 2; CFG["workers"] = 0
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8"))); dev = torch.device("cpu")
meta = r.read_manifest(); labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
tab = {}                                                 # v2.crop_table restricted to the local shards
for cp in sorted(v2.CP.glob("cp_*.npz")):
    z = np.load(cp)
    for s, cen, at in zip(z["ids"], z["centres"], z["a_top"]):
        s = str(s)
        if s in meta:
            c = np.clip(r.centres(meta[s]["mask"], v2.K, False), 1, v2.MAXS - 2)
            tab[s] = (c[cen[..., 0].astype(int)], cen[..., 1].astype(int), cen[..., 2].astype(int), at.astype(np.float32))
tr = sorted(s for s in meta if fold.get(s) not in (None, "0") and s in tab and s in labels)[:2]
va = sorted(s for s in meta if fold.get(s) == "0" and s in tab)[:2]
assert len(tr) == 2 and len(va) == 2, (len(tr), len(va))
ok = lambda msg: print("TEST", msg, "OK", flush=True)

# 1. pair table: complementary slot, |d| <= 1.5 mm, opposite fluid flag, axial unpaired; registration beats a random row
L.IMP = "mcpair"; L.G4.update({"pair": {"max_mm": 1.5, "max_deg": 5.0, "max_inplane_mm": 2.0, "fluid_differs": True},
                                "partner_dropout": 0.2})
L.PAIRS.update(L.pair_table())
bounds = np.cumsum([0] + CFG["slot_sizes"]); slot = np.repeat(np.arange(5), CFG["slot_sizes"])
comp = {0: 1, 1: 0, 2: 3, 3: 2}
rows = [__import__("json").loads(l) for i in range(2) for l in (r.CACHE / f"shard_{i}" / "manifest.jsonl").read_text().splitlines()]
cov, ncc_p, ncc_r = [], [], []
rng = np.random.default_rng(0)
for m in rows:
    P, fl = L.PAIRS[m["study_id"]]
    assert (P[bounds[4]:] == -1).all()
    for q in np.where(P >= 0)[0]:
        a, b = slot[q], slot[P[q]]
        assert comp[a] == b, (q, P[q])
        A, B = m["slots"][a], m["slots"][b]
        assert A["fluid"] != B["fluid"] and fl[q] == (A["fluid"] == "1")
        ia = np.array(A["geometry"][0]["iop"]); n = np.cross(ia[:3], ia[3:]); n /= np.linalg.norm(n)
        d = (np.array(A["geometry"][q - A["row_start"]]["ipp"]) - np.array(B["geometry"][P[q] - B["row_start"]]["ipp"])) @ n
        assert abs(d) <= 1.5 + 1e-6, d
    mk = np.array(m["mask"], bool); cov.append((P[mk] >= 0).mean())
    if m["study_id"] in meta and (P >= 0).any():
        f, row = meta[m["study_id"]]["file"], meta[m["study_id"]]["row"]
        vol = np.load(f, mmap_mode="r")[row:row + 96]
        g = lambda a: (lambda e: (e - e.mean()) / (e.std() + 1e-6))(np.hypot(*np.gradient(a.astype(np.float32).reshape(96, 4, 96, 4).mean((1, 3)))))
        for q in rng.choice(np.where(P >= 0)[0], min(3, (P >= 0).sum()), replace=False):
            b = slot[P[q]]; other = rng.integers(bounds[b], bounds[b] + m["slots"][b]["k"])
            ncc_p.append((g(vol[q]) * g(vol[P[q]])).mean()); ncc_r.append((g(vol[q]) * g(vol[other])).mean())
print(f"paired valid rows {np.mean(cov):.3f}; NCC paired {np.mean(ncc_p):.3f} vs random partner-slot row {np.mean(ncc_r):.3f}")
assert np.mean(ncc_p) > np.mean(ncc_r) + 0.05
ok("pair table")

# 2. GLStudies6: own channels = v2.GLStudies' (same RNG), partner rows through P, zero where unpaired
for train in (False, True):
    np.random.seed(5); a = v2.GLStudies(va[:1], meta, labels, 8, train, tab)[0]
    np.random.seed(5); b6 = L.GLStudies6(va[:1], meta, labels, 8, train, tab)[0]
    assert torch.equal(a[0], b6[0][:, :3]) and torch.equal(a[5], b6[5][:, :3])
    assert all(torch.equal(u, v) for u, v in zip(a[1:5] + a[6:], b6[1:5] + b6[6:11]))
    okg, okc = b6[11].numpy(), b6[12].numpy()
    assert (b6[0][:, 3:][~okg].sum() == 0) and (b6[5][:, 3:][~okc].sum() == 0)
ok("GLStudies6 own channels identical, unpaired partner rows zero")

# 3. mirror6 = r.mirror per half; prepare_gl6 zeroes unpaired channels exactly, flags follow the sagittal reversal
x = torch.randint(0, 255, (4, 6, 8, 8), dtype=torch.uint8); sg = torch.tensor([True, False, True, False])
m6 = L.mirror6(x, sg)
assert torch.equal(m6[:, :3], r.mirror(x[:, :3], sg)) and torch.equal(m6[:, 3:], r.mirror(x[:, 3:], sg))
ds = L.GLStudies6(va, meta, labels, 8, True, tab)
batch = torch.utils.data.default_collate([ds[0], ds[1]])
for train in (True, False):
    torch.manual_seed(1); xg, xc, _ = L.prepare_gl6(*batch[:2], batch[5], batch[6], batch[8], batch[9], batch[11], batch[12], train, dev)
    assert xg.shape[1] == 6 and xc.shape[1] == 6
    # recover the post-mirror flags the same way prepare_gl6 does, then check exact zeros
    torch.manual_seed(1); flip = torch.rand(2) < CFG["aug"]["mirror_p"] if train else torch.zeros(2, dtype=torch.bool)
    fg = flip.repeat_interleave(8); sag = batch[1].flatten(); okg = batch[11].flatten(0, 1)
    okg = torch.where((fg & sag).view(-1, 1), okg.flip(1), okg)
    zero = xg[:, 3:][~okg]
    assert zero.abs().max() == 0, zero.abs().max()
    # own-channel path = v2.prepare_gl with the same draws
    torch.manual_seed(1); x3, c3, _ = v2.prepare_gl(batch[0][:, :, :3], batch[1], batch[5][:, :, :3], batch[6], batch[8], batch[9], train, dev)
    assert (x3 - xg[:, :3]).abs().max() < 1e-5 and (c3 - xc[:, :3]).abs().max() < 1e-5
ok("mirror6, prepare_gl6 own channels = prepare_gl, unpaired partner channels exactly 0")

# 4. init equivalence + RNG parity: v4 and v4-mcpair from one seed give the same outputs (crop path included)
L.IMP = "none"; torch.manual_seed(7); m3 = L.RaptorGL4(CFG["train_res"], pretrained=False).eval(); r3 = torch.rand(3)
L.IMP = "mcpair"; torch.manual_seed(7); m6 = L.RaptorGL4(CFG["train_res"], pretrained=False).eval(); r6 = torch.rand(3)
assert torch.equal(r3, r6), "mcpair init consumed RNG"
s3, s6 = m3.state_dict(), m6.state_dict()
assert set(s6) - set(s3) == {"seq_bias"} and all(torch.equal(s3[k], s6[k]) for k in s3 if k != "backbone.conv1.weight")
assert torch.equal(s6["backbone.conv1.weight"][:, :3], s3["backbone.conv1.weight"]) and s6["backbone.conv1.weight"][:, 3:].abs().sum() == 0
eds = L.GLStudies6(va, meta, labels, 6, False, tab); eb = torch.utils.data.default_collate([eds[0]])
with torch.no_grad():
    xg, xc, typ = L.prep(eb, False, dev)
    g6, l6 = m6(xg, 1, xc, eb[9], eb[10], typ)
    g3, l3 = m3(xg[:, :3].contiguous(), 1, xc[:, :3].contiguous(), eb[9], eb[10])
assert (g6 - g3).abs().max() < 1e-4 and (l6 - l3).abs().max() < 1e-4, ((g6 - g3).abs().max(), (l6 - l3).abs().max())
ok("init equivalence (global + crop path), no extra RNG draws")

# 5. SWA load of 6-channel state dicts (strict)
avg = {k: (s6[k].float() + s6[k].float()) / 2 if s6[k].is_floating_point() else s6[k] for k in s6}
m6.load_state_dict(avg, strict=True)
ok("SWA average loads strict")

# 6. silent lanes: table, loss identities, Hill equilibrium
L.G4.update({"rule_states_sha256": "ce4473c71f3ea100db021805750c6bb9038bc7b0e56a143c81516edf2d29e77b",
             "silent_weight": 0.25, "hill_lambda": 1.5})
all_tr = sorted(s for s in labels if fold[s] != "0")
st = L.silent_table(all_tr, labels)
print(f"silent assumed-negative cells {st.mean():.4f} (training studies {len(all_tr)})")
assert 0.40 < st.mean() < 0.50
z = torch.randn(2, 12); y = torch.rand(2, 12); w = torch.ones(2, 12); mz = torch.zeros(2, 12, dtype=torch.bool)
for imp in ("silw", "hill"):
    L.IMP = imp
    assert torch.allclose(L.silent_loss(z, y, w, mz), v2.bce(z, y, w))
L.IMP = "silw"; mall = torch.ones(2, 12, dtype=torch.bool)
assert torch.allclose(L.silent_loss(z, y, w, mall), 0.25 * v2.bce(z, y, w))              # denominator = cell count
L.IMP = "hill"; t = torch.zeros(1, requires_grad=True); opt = torch.optim.SGD([t], lr=2.0)
for _ in range(3000):
    opt.zero_grad(); L.silent_loss(t.view(1, 1), torch.tensor([[0.06]]), torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool)).backward(); opt.step()
p = float(torch.sigmoid(t)); print(f"Hill equilibrium for y = 0.06: p = {p:.3f} (BCE: 0.06)")
assert 0.12 < p < 0.20
ok("silent table and losses")

# 7. one training step per lane: finite loss and gradients, logging shapes unchanged (2 parts)
CFG["train_res"] = 128; v2.GL["crop_res"] = 128
for imp in ("none", "silw", "hill", "mcpair"):
    L.IMP = imp; torch.manual_seed(0)
    model = L.RaptorGL4(128, pretrained=False).train()
    if imp == "mcpair":
        dset = L.GLStudies6(tr, meta, labels, 4, True, tab); sil = None
    elif imp in ("silw", "hill"):
        dset = L.GLStudiesI(tr, meta, labels, 4, True, tab); sil = torch.from_numpy(L.silent_table(tr, labels))
    else:
        dset = v2.GLStudies(tr, meta, labels, 4, True, tab); sil = None
    batch = torch.utils.data.default_collate([dset[0], dset[1]])
    if sil is not None:
        assert batch[11].tolist() == [0, 1]
    parts = L.train_step_loss(model, batch, dev, torch.from_numpy(L.CMASK.astype(np.float32)), sil)
    assert parts.shape == (2,) and torch.isfinite(parts).all()
    (parts[0] + 0.5 * parts[1]).backward()
    gr = [p.grad for p in model.parameters() if p.grad is not None]
    assert gr and all(torch.isfinite(g).all() for g in gr)
    if imp == "mcpair":
        assert model.backbone.conv1.weight.grad[:, 3:].abs().sum() > 0, "partner channels get no gradient"
        assert model.seq_bias.grad.abs().sum() > 0
    print(f"  {imp}: loss parts {parts.detach().numpy().round(4).tolist()}")
ok("one step per lane")
print("ALL_TESTS_OK")
