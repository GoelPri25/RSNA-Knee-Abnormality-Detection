"""Exp r50-GL diagnostic (inference only, no training): why does the reader not use the crops?
Model: r50-GL SWA weights (GL_WEIGHTS) of the run GL_VERSION. Studies: all 825 fold-0 studies (held out) and 825
training-fold studies (fixed draw, seed 2026; they were trained on). Same evaluation inputs as the run (94 windows,
40 attention-weighted crops). Per study and finding it saves:
  - sigmoid of fused, global, local (columns 7-8 = global, as the run), fused without crop tokens;
  - the reader's per-finding attention mass on: all crop tokens, the finding's OWN 4 crops, the global windows;
  - the reader delta (fused logit - global logit) with and without crop tokens.
Question 1 (CPU scorer afterwards): does mean(global, local) beat global on the TRAINING studies as it does on fold 0?
Question 2: does each finding's reader attention land on its own crops (chance = 4 / 134 = 3%)?
Env: D96_CACHE, D96_PHASE=r50_gl, GL_VERSION, GL_CROPS, GL_OUT (dir of the run), GL_WEIGHTS, DIAG_OUT.
"""
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

import r50_gl_runner as g

r, CFG = g.r, g.CFG
OUT = Path(os.environ["DIAG_OUT"]); OUT.mkdir(parents=True, exist_ok=True)


def reader_parts(model, hg, ctx, hc, ctxc, fidc, pxc):
    """Reader forward with the per-token attention exposed (same arithmetic as Reader.forward)."""
    rd = model.reader
    t = rd.inp(hg) + rd.kind.weight[0] + rd.place(ctx); ng = t.shape[1]
    f = rd.f.to(pxc.device)
    xy = torch.cat([torch.sin(pxc[..., :1] * f), torch.cos(pxc[..., :1] * f),
                    torch.sin(pxc[..., 1:] * f), torch.cos(pxc[..., 1:] * f)], -1)
    tc = rd.inp(hc) + rd.kind.weight[1] + rd.place(ctxc) + rd.fid(fidc) + rd.xy(xy)
    z = rd.enc(torch.cat([t, tc.to(t.dtype)], 1))
    a = torch.softmax(rd.att(z).float(), dim=1)                                          # [1, n, 12]
    logit = (torch.einsum("bnk,bnf->bkf", a.to(z.dtype), z) * rd.clsW).sum(-1).float() + rd.clsb
    own = (fidc[0][:, None] == torch.arange(12, device=fidc.device)[None]).float()       # [kc, 12]
    return logit, a[0, ng:].sum(0), (a[0, ng:] * own).sum(0), a[0, :ng].sum(0)


@torch.no_grad()
def run(model, ids, meta, tab, dev):
    model.eval()
    dl = torch.utils.data.DataLoader(g.GLStudies(ids, meta, {}, g.K, False, tab), batch_size=1,
                                     num_workers=CFG["workers"], pin_memory=True)
    keys = ("fused", "global", "local", "nocrop", "mass_crop", "mass_own", "mass_glob", "delta", "delta_nocrop")
    o = {k: np.zeros((len(ids), 12), np.float32) for k in keys}
    for i, (x, sag, _, _, ctx, xc, sagc, ctxc, cen, fid, wc) in enumerate(dl):
        xg, xcr, pxc = g.prepare_gl(x, sag, xc, sagc, cen, fid, False, dev)
        ctx, ctxc, fid, wc = g.to_dev(dev, ctx, ctxc, fid, wc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (fz, gz, lz, nz), _ = model(xg, 1, ctx, xcr, ctxc, fid, pxc, wc, nocrop=True)
            hg = model.norm(model.encode(xg, False).view(1, xg.shape[0], -1))
            hc = model.norm_l(model.encode(xcr, True).view(1, xcr.shape[0], -1))
            dz, mc, mo, mg = reader_parts(model, hg, ctx, hc, ctxc, fid, pxc)
        for k, v in (("fused", fz.sigmoid()), ("global", gz.sigmoid()), ("local", lz.sigmoid()), ("nocrop", nz.sigmoid()),
                     ("mass_crop", mc), ("mass_own", mo), ("mass_glob", mg), ("delta", fz - gz), ("delta_nocrop", nz - gz)):
            o[k][i] = v.float().reshape(-1)[:12].cpu().numpy()
        if i == 0:                                                                       # parity: exposed reader = model's
            assert torch.allclose(dz.float(), (fz - gz).float(), atol=2e-2), (dz, fz - gz)
    o["local"][:, ~g.CMASK] = o["global"][:, ~g.CMASK]
    return o


def main():
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
    dev = torch.device("cuda")
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest(); tab = g.crop_table(meta)
    train_ids = sorted(s for s in labels if fold[s] != "0"); val_ids = sorted(s for s in labels if fold[s] == "0")
    tr = sorted(random.Random(CFG["seed"]).sample(train_ids, len(val_ids)))
    model = g.RaptorGL(CFG["train_res"], pretrained=False).to(dev).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(os.environ["GL_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    for split, ids in (("val", val_ids), ("train", tr)):
        o = run(model, ids, meta, tab, dev)
        np.savez(OUT / f"diag_{split}.npz", ids=np.array(ids), targets=np.array(r.TARGETS), **o)
        print(f"DIAG {split} {len(ids)} studies; mean own-crop mass {o['mass_own'][:, g.CF].mean():.3f}, "
              f"all-crop mass {o['mass_crop'].mean():.3f}", flush=True)
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
