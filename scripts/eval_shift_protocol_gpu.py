"""Exp P1-shift-dx: does P1-shift's fold-0 loss come from the train / eval window mismatch? (inference only; Modal)

P1-shift trained on runs of 2-4 windows at stride 3 (about 31% of neighbour slots empty) and was scored on 1e's 94
centres (nearly every window has both neighbours). This re-scores the P1-shift SWA weights under training-like window
layouts, with 1e under the matching 16-window protocol as the control. Each variant is a full fold-0 prediction file
(825 studies, 320 centre crop, no TTA), scored on CPU by scripts/audit_shift_protocol.py.

  shift_full94   P1-shift, 94 centres, neighbours by slice distance (must reproduce run_r50_shift/val_swa.json)
  shift_zero94   P1-shift, 94 centres, every neighbour absent (shifted channels zero): is neighbour content used?
  shift_runs_c   P1-shift, one 16-window pass of centred runs (the BN-refresh layout)
  shift_runs_r8  P1-shift, mean of 8 passes of 16-window runs with random starts (the training layout)
  e1_full94      1e, 94 centres (must reproduce run_r50_pertarget/val_swa.json)
  e1_strat_r8    1e, mean of 8 passes of 16 stratified random windows (1e's training layout)

Env: D96_CACHE, D96_PHASE=r50_shift, SHIFT_WEIGHTS, E1_WEIGHTS, DX_OUT.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

import coatnet_d96_runner as r

assert os.environ["D96_PHASE"] == "r50_shift" and "shift" in r.CFG
OUT = Path(os.environ["DX_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
PASSES = 8


@torch.no_grad()
def run(model, ids, meta, dev, k, train, seed=None, zero=False):
    """One prediction pass [n, 12]; train=True draws the training layout (random) without augmentation."""
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    ds = r.Studies(ids, meta, {}, k, train)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=r.CFG["workers"], pin_memory=True,
                                     generator=torch.Generator().manual_seed(seed or 0))
    out, nb = [], []
    for x, sag, _, _, ctx in dl:
        if ctx.shape[-1] == 5:
            if zero:
                ctx[..., 3:5] = -1
            nb.append((ctx[0, :, 3:5] >= 0).sum(1).numpy())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(r.prepare(x, sag, False, r.CFG["eval_res"], dev), 1, ctx.to(dev)).float().sigmoid()
        out.append(p[0].cpu().numpy())
    stats = None
    if nb:
        c = np.bincount(np.concatenate(nb), minlength=3) / sum(len(v) for v in nb)
        stats = {"windows_with_0_1_2_neighbours": [round(float(v), 4) for v in c]}
    return np.stack(out), stats


def load(path):
    m = r.build(r.CFG["eval_res"], pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return m.to(torch.device("cuda")).eval().to(memory_format=torch.channels_last)


def main():
    dev = torch.device("cuda")
    ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    ids = sorted(ref)
    meta = r.read_manifest()
    assert len(ids) == 825
    log = {}

    def save(name, pred, note, stats=None):
        r.save_predictions(OUT / f"{name}.json", ids, pred, note)
        log[name] = {"val12": round(r.macro_auc(pred, ids, ref)[0], 4), **(stats or {})}
        print("DX", name, json.dumps(log[name]), flush=True)

    k16, k94 = r.CFG["train_windows"], r.CFG["eval_windows"]
    m = load(os.environ["SHIFT_WEIGHTS"])
    assert len(m.backbone.layer1[0].conv1._forward_pre_hooks) == 1
    p, st = run(m, ids, meta, dev, k94, False); save("shift_full94", p, "P1-shift SWA, 94 centres", st)
    p, st = run(m, ids, meta, dev, k94, False, zero=True); save("shift_zero94", p, "P1-shift SWA, 94, no neighbours", st)
    p, st = run(m, ids, meta, dev, k16, False); save("shift_runs_c", p, "P1-shift SWA, 16 centred runs", st)
    ps = [run(m, ids, meta, dev, k16, True, seed=100 + j) for j in range(PASSES)]
    save("shift_runs_r8", np.mean([q for q, _ in ps], 0), f"P1-shift SWA, mean of {PASSES} random 16-window runs",
         ps[0][1])
    del m

    shift = r.CFG.pop("shift")                         # 1e: plain Raptor, no hooks, 1e window sampling
    m = load(os.environ["E1_WEIGHTS"])
    assert not m.backbone.layer1[0].conv1._forward_pre_hooks
    p, _ = run(m, ids, meta, dev, k94, False); save("e1_full94", p, "1e SWA, 94 centres")
    ps = [run(m, ids, meta, dev, k16, True, seed=100 + j)[0] for j in range(PASSES)]
    save("e1_strat_r8", np.mean(ps, 0), f"1e SWA, mean of {PASSES} random 16-window stratified draws")
    r.CFG["shift"] = shift
    (OUT / "dx_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("DX_DONE", flush=True)


if __name__ == "__main__":
    main()
