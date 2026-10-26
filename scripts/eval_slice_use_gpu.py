"""Exp MS1-L1: does a model use the neighbouring slices packed into R / B? (inference only; Modal H100)

Each d96 window is stack slices [c - 1, c, c + 1] as RGB. MedSigLIP was pretrained on grey (R = G = B) MRI and MS1's patch
embedding got LR ~2.6e-6, so MS1 may read the neighbours as colour, not depth. Fold 0 (825 studies, 94 windows, no TTA)
is re-scored with the trained weights under three inputs, same centres:
  normal  [c - 1, c, c + 1]   (must reproduce the model's val_swa.json)
  grey    [c, c, c]           (no depth)
  wide    [c - 3, c, c + 3]   (neighbours further away, clamped inside the centre's series slot)
Run once per model: MS1 (D96_PHASE=sig_lv2) and the label-v2 r50 control (D96_PHASE=r50_pertarget). Scored on CPU by
scripts/audit_slice_use.py.

Env: D96_CACHE, D96_PHASE, SLICE_WEIGHTS, SLICE_OUT, SLICE_TAG.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import coatnet_d96_runner as r

assert not r.NATIVE_LAYOUT and "alloc" not in r.CFG and "shift" not in r.CFG
OUT = Path(os.environ["SLICE_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
TAG = os.environ["SLICE_TAG"]
LO = {s: int(np.where(r.SLOT_OF == s)[0].min()) for s in range(5)}
HI = {s: int(np.where(r.SLOT_OF == s)[0].max()) for s in range(5)}
MODE = {"m": "normal"}
_orig = r.Studies.__getitem__


def getitem(self, i):
    win, sag, y, w, ctx = _orig(self, i)
    if MODE["m"] == "normal":
        return win, sag, y, w, ctx
    if MODE["m"] == "grey":
        win = win[:, 1:2].expand(-1, 3, -1, -1).contiguous()
        return win, sag, y, w, ctx
    m = self.meta[self.ids[i]]                                   # wide: same centres as the runner's eval path
    length = r.CFG["maxs"]
    c = np.clip(r.centres(m["mask"], self.k, False, None), 1, length - 2)
    s = r.SLOT_OF[c]
    lo = np.array([LO[v] for v in s]); hi = np.array([HI[v] for v in s])
    vol = self._mm[m["file"]][m["row"]:m["row"] + length]
    idx = np.stack([np.clip(c - 3, lo, hi), c, np.clip(c + 3, lo, hi)], 1)
    new = torch.from_numpy(np.ascontiguousarray(vol[idx]))
    assert torch.equal(new[:, 1], win[:, 1]), "wide: centre slices differ from the runner's"
    return new, sag, y, w, ctx


r.Studies.__getitem__ = getitem


def main():
    dev = torch.device("cuda")
    res = r.CFG["eval_res"]
    m = r.build(res, pretrained=False)
    m.load_state_dict(torch.load(os.environ["SLICE_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    m = m.to(dev).eval().to(memory_format=torch.channels_last)
    ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    meta = r.read_manifest()
    val_ids = sorted(ref)
    assert len(val_ids) == 825
    log = {}
    for mode in ("normal", "grey", "wide"):
        MODE["m"] = mode
        t = time.time()
        with torch.no_grad():
            pv = r.predict(m, val_ids, meta, dev, res=res)
        r.save_predictions(OUT / f"{TAG}_{mode}.json", val_ids, pv, f"{TAG} SWA, slice-use test, input {mode}")
        v12, per = r.macro_auc(pv, val_ids, ref)
        log[mode] = {"val12": round(v12, 4), "per_target": per, "seconds": round(time.time() - t)}
        print("SLICE", TAG, mode, json.dumps(log[mode]), flush=True)
    (OUT / f"{TAG}_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("SLICE_DONE", TAG, flush=True)


if __name__ == "__main__":
    main()
