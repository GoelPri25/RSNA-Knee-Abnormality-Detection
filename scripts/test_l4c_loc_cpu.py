"""CPU tests for scripts/l4c_loc_runner.py (L4C-LOC spec v2 §4), in localhost/rsna-mlx-train:v1, --network=none.
Cache: shard_0 = fold-0 export, shard_1 = the smoke cache (training-fold studies). Then the end-to-end smoke is
`LOC_N=4 LOC_EPOCHS=2 python l4c_loc_runner.py` (main asserts frozen weights, writes equivalence vs predict_study).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l4c_loc_runner as L  # noqa: E402

run, sub, xai, CF, TOP = L.run, L.sub, L.xai, L.CF, L.TOP
dev = torch.device("cpu"); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
meta = xai.run_manifest(Path(os.environ["D96_CACHE"])); tabm = run.ms1_table()
import csv  # noqa: E402
fold = {r["study_id"]: r["fold"] for r in csv.DictReader(open(os.environ["LOC_FOLDS"]))}
tr = sorted(s for s in meta if fold.get(s) not in (None, "0") and s in tabm)[:4]
va = sorted(s for s in meta if fold.get(s) == "0" and s in tabm)[:3]
assert len(tr) == 4 and len(va) == 3, (len(tr), len(va))
model = sub.load_model(os.environ["LOC_WEIGHTS"], dev).to(memory_format=torch.contiguous_format)
loc = L.Locator(model)

# 8. eval mode / no dropout in the attention copy
assert not loc.att.training
h = torch.randn(5, 2048)
assert torch.equal(loc.att(h), loc.att(h)), "dropout active"

# 1. init equivalence: learned table at init = sub.locate's windows and normalised weights
for s in va:
    f, row, mask = meta[s]; vol = np.ascontiguousarray(np.load(f, mmap_mode="r")[row:row + sub.MAXS])
    with torch.no_grad():
        cs, fm, hh, a, g = L.global_feats(model, vol, mask, dev)
        S, R, Q, A, top = loc.table(hh, torch.as_tensor(run.slot_of(cs)), lambda U: fm[U], cs)
    S0, R0, Q0, A0 = sub.locate(model, fm, a, cs)
    assert np.array_equal(np.sort(S, 1), np.sort(S0, 1)), (s, S, S0)
    ws = {(j, int(x)): float(w) for j in range(len(CF)) for x, w in zip(S[j], A[j])}
    w0 = A0 / A0.sum(1, keepdims=True)
    assert max(abs(ws[(j, int(x))] - float(w)) for j in range(len(CF)) for x, w in zip(S0[j], w0[j])) < 1e-4
    # 2. MS1 mapping = l4copy_xai's tms1
    w, q, Rm, Qm = L.ms1_targets(tabm, s, cs); cen, at = tabm[s]
    assert np.array_equal(cs[w], cs[cen[..., 0]]) and abs(q.sum(1) - 1).max() < 1e-5
    # 9. point targets use the clipped centre
    hb = run.BOXN[CF][:, None] // 2
    assert (Rm >= hb).all() and (Rm <= L.IMG - hb).all()
print("TEST init-equivalence / mapping / clip OK")

# 6. cell target + soft-argmax round trip
for r_, q_ in ((64, 64), (200, 150), (320, 300), (192, 192)):
    t = torch.as_tensor(L.cell_target(r_, q_, 1.0), dtype=torch.float32).view(L.C, L.C); g = torch.arange(L.C, dtype=torch.float32)
    cy, cx = (t.sum(1) * g).sum(), (t.sum(0) * g).sum()
    rr, qq = L.LO + (cy + 0.5) * L.CELL, L.LO + (cx + 0.5) * L.CELL
    tol = 2 if 96 <= min(r_, q_) and max(r_, q_) <= 288 else 10   # truncated Gaussian: soft-argmax pulled inward at the border
    assert abs(rr - r_) < tol and abs(qq - q_) < tol, (r_, q_, float(rr), float(qq))
print("TEST cell target round trip OK (interior < 2 px; border bias <= 9 px = 3.3 mm, under L4C-D's 5 mm no-loss shift)")

# 3. loss decreases on 4 training studies; 4. v4 weights unchanged by locator steps
frozen = {k: v.clone() for k, v in model.state_dict().items()}
feats = L.extract(model, tr, meta, tabm, dev, True)
opt = torch.optim.AdamW(loc.parameters(), lr=1e-3)
first = last = None
for step in range(20):
    opt.zero_grad(); tot = 0
    for s in tr:
        lw, lp = L.losses(loc, feats[s], dev); tot = tot + lw + lp
    tot.backward(); opt.step()
    first = float(tot) if first is None else first; last = float(tot)
assert last < first, (first, last)
assert all(torch.equal(frozen[k], v) for k, v in model.state_dict().items())
print(f"TEST loss {first:.3f} -> {last:.3f}, v4 weights unchanged OK")

# 10. fold-0 exclusion
assert not any(fold[s] == "0" for s in feats)
print("TEST no fold-0 study in training OK")
print("ALL_TESTS_OK")
