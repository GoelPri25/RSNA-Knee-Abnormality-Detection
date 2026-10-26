"""Exp r50-GL XAI (inference only, no training): where does each head look, what does the reader use, and which input
changes move each head. Models: r50-GL v2 SWA (GL_WEIGHTS) and its parent, the label-v2 r50 (LV2_WEIGHTS, plain Raptor,
the best single). Same evaluation inputs as the run (94 windows at the 320 centre crop, 40 MS1-attention-weighted crops).
Fold 0 (825 studies), per study:
  predictions  fused, global, local, fused without crops (reproduction), parent;
               global head on the whole 384 px window instead of the 320 centre crop (GL and parent);
               local head with uniform / top-1 crop weights (instead of MS1's weights);
               local and fused with every crop centre moved 32 / 64 px (random direction, fixed seed);
               fused with the crop content of the previous study (same findings and positions, other knee);
               fused with each finding's pooling limited to the global windows + its own crops (routing, untrained).
  Grad-CAM     layer 4, per finding. Global heads (GL and parent) on all 94 windows: share of positive CAM mass on the
               finding's crop windows and inside its crop boxes, with the area shares (chance). Local head on each
               crop: share of mass in the outer ring (outer 2 of 12 cells, area 0.556) and centroid offset from the crop
               centre (px of the 384 image).
  reader       per finding |grad x input| of the reader delta and attention mass, split over global windows / the
               finding's own crops / other findings' crops.
Training studies (825, seed-2026 draw as diag_r50_gl_gpu.py): predictions (fused, global, local, without crops) and
the reader split; no CAM, no ablations.
Panels (PANEL_IDS json [[study, finding], ...], fold 0): images and CAMs for local rendering.
Env: D96_CACHE, D96_PHASE=r50_gl, GL_VERSION, GL_CROPS, GL_OUT, GL_WEIGHTS, LV2_WEIGHTS, XAI_OUT, PANEL_IDS.
"""
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import r50_gl_runner as g

r, CFG = g.r, g.CFG
OUT = Path(os.environ["XAI_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
K, IMG, CF = g.K, g.IMG, g.CF
SHIFTS, OFF = (32, 64), (IMG - CFG["eval_res"]) // 2                       # OFF: 320 centre crop in the 384 image
RING = 2                                                                  # outer cells of the 12 x 12 crop map
bf16 = lambda dev: torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")   # CPU test: float32


def maps(model, x, crop):
    """layer-4 maps; GL picks its global / crop BatchNorm statistics, the parent has one set"""
    for m in getattr(model, "bns", []):
        m.crop = crop
    f = model.backbone.forward_features(x)
    for m in getattr(model, "bns", []):
        m.crop = False
    return f


def head_global(model, P):
    """1e head on pooled window features P [k, 2048] (float32) -> logits [12]"""
    h = model.norm(P)[None]
    a = torch.softmax(model.att(h).float(), dim=1)
    return ((torch.einsum("bkn,bkf->bnf", a, h) * model.clsW).sum(-1) + model.clsb)[0]


def cam(logit, P, Fm):
    """Grad-CAM through average pooling: relu(sum_c dlogit/dP_c * F_c) [k, h, w]"""
    gP, = torch.autograd.grad(logit, P, retain_graph=True)
    return torch.relu(torch.einsum("kc,kchw->khw", gP, Fm))


def reader(model, hg, ctx, hc, ctxc, fidc, pxc, route=False):
    """Reader.forward with optional routing (finding f pools only global tokens + crops of f) -> delta [12], attention
    [n, 12], number of global tokens"""
    rd = model.reader
    t = rd.inp(hg) + rd.kind.weight[0] + rd.place(ctx); ng = t.shape[1]
    f = rd.f.to(pxc.device)
    xy = torch.cat([torch.sin(pxc[..., :1] * f), torch.cos(pxc[..., :1] * f),
                    torch.sin(pxc[..., 1:] * f), torch.cos(pxc[..., 1:] * f)], -1)
    tc = rd.inp(hc) + rd.kind.weight[1] + rd.place(ctxc) + rd.fid(fidc) + rd.xy(xy)
    z = rd.enc(torch.cat([t, tc], 1))
    s = rd.att(z).float()
    if route:
        own = fidc[0][:, None] == torch.arange(12, device=s.device)[None]
        keep = torch.cat([torch.ones(ng, 12, dtype=torch.bool, device=s.device), own], 0)
        s = s.masked_fill(~keep[None], float("-inf"))
    a = torch.softmax(s, dim=1)
    return ((torch.einsum("bnk,bnf->bkf", a, z) * rd.clsW).sum(-1) + rd.clsb)[0], a[0], ng


def groups(fidc):
    """own crops of each finding -> [12, kc] bool"""
    return fidc[0][None, :] == torch.arange(12, device=fidc.device)[:, None]


def split3(v, ng, own):
    """v [n, 12] non-negative per token and finding -> [12, 3] (global, own crops, other crops)"""
    vg, vc = v[:ng].sum(0), v[ng:]                                        # [12], [kc, 12]
    vo = (vc * own.T).sum(0)
    return torch.stack([vg, vo, vc.sum(0) - vo], 1)


def attribution(model, hg, ctx, hc, ctxc, fidc, pxc):
    """|grad x input| of each reader delta per token -> [12, 3] shares; attention mass [12, 3]"""
    hg, hc = hg.detach().requires_grad_(), hc.detach().requires_grad_()
    dz, a, ng = reader(model, hg, ctx, hc, ctxc, fidc, pxc)
    own = groups(fidc)
    att = torch.zeros(hg.shape[1] + hc.shape[1], 12, device=hg.device)
    for f in range(12):
        gg, gc = torch.autograd.grad(dz[f], (hg, hc), retain_graph=f < 11)
        att[:, f] = torch.cat([(gg * hg).sum(-1)[0], (gc * hc).sum(-1)[0]]).abs().detach()
    s = split3(att, ng, own)
    return (s / s.sum(1, keepdim=True).clamp_min(1e-12)), split3(a.detach(), ng, own), dz.detach()


def local_logits(model, hc, fidc):
    return (hc * model.clsV[fidc]).sum(-1) + model.clsvb[fidc]                      # [1, kc]


def pool_local(lc, fidc, wc):
    return torch.zeros(1, 12, device=lc.device).scatter_add(1, fidc, lc * wc)


def study_windows(meta, s):
    return np.clip(r.centres(meta[s]["mask"], K, False), 1, g.MAXS - 2)


def cam_global(model, xg, idx):
    """layer-4 maps -> logits [12] and positive CAM [len(idx), k, 320, 320] (bilinear to the input size)"""
    with torch.no_grad(), bf16(xg.device):
        Fm = maps(model, xg, False)
    Fm = Fm.float(); P = Fm.mean((2, 3)).requires_grad_()
    z = head_global(model, P)
    c = torch.stack([cam(z[f], P, Fm) for f in idx])
    return z.detach(), F.interpolate(c, size=xg.shape[-2:], mode="bilinear", align_corners=False), P.detach()


def box_masks(cen, fidc, win, sl, res):
    """per crop: (window index or -1, row / col slice of its box in the res-px centre-cropped window)"""
    out = []
    for q in range(len(sl)):
        w = np.where(win == sl[q])[0]
        h = g.BOXN[fidc[q]] // 2; R, Q = int(cen[q, 0]) - OFF, int(cen[q, 1]) - OFF
        out.append((int(w[0]) if len(w) else -1, slice(max(R - h, 0), min(R + h, res)),
                    slice(max(Q - h, 0), min(Q + h, res))))
    return out


def cam_shares(C, boxes, fidc, idx):
    """C [len(idx), k, H, W] -> [12, 5]: share on crop windows, share in boxes, box share within crop windows, the two
    area shares (chance); top-CAM window is a crop window (1/0) in column 5"""
    o = np.full((12, 6), np.nan, np.float32)
    k, H, W = C.shape[1:]
    for j, f in enumerate(idx):
        bx = [b for b, q in zip(boxes, fidc) if q == f and b[0] >= 0]
        if not bx:
            continue
        wins = sorted({b[0] for b in bx})
        m = torch.zeros(k, H, W, dtype=torch.bool, device=C.device)
        for w, rs, cs in bx:
            m[w, rs, cs] = True
        tot = C[j].sum().clamp_min(1e-12); cw = C[j][wins].sum()
        per_win = C[j].sum((1, 2))
        o[f] = [float(cw / tot), float((C[j] * m).sum() / tot), float((C[j][wins] * m[wins]).sum() / cw.clamp_min(1e-12)),
                len(wins) / k, float(m[wins].float().mean()), float(int(per_win.argmax()) in wins)]
    return o


def local_cam_stats(Cc, fidc, wc, n=12):
    """Cc [kc, 12, 12] -> per finding (MS1-weighted over its crops): outer-ring share, centroid offset in 384 px"""
    tot = Cc.sum((1, 2)).clamp_min(1e-12)
    inner = Cc[:, RING:n - RING, RING:n - RING].sum((1, 2))
    ring = 1 - inner / tot
    ii = (torch.arange(n, device=Cc.device, dtype=torch.float32) + 0.5) / n - 0.5
    cy, cx = (Cc.sum(2) * ii).sum(1) / tot, (Cc.sum(1) * ii).sum(1) / tot
    box = torch.as_tensor(g.BOXN, device=Cc.device, dtype=torch.float32)[fidc]
    off = torch.sqrt(cy ** 2 + cx ** 2) * box
    o = torch.zeros(12, 2, device=Cc.device)
    o.index_add_(0, fidc, torch.stack([ring, off], 1) * wc[:, None])
    o[~torch.as_tensor(g.CMASK, device=Cc.device)] = float("nan")
    return o.cpu().numpy()


def run(gl, par, ids, meta, tab, dev, full, panels):
    keys = ["fused", "global", "local", "nocrop"] + (["parent", "g384", "parent384", "local_uniform", "local_top1",
            "fused_route", "fused_shuffle"] + [f"{h}_shift{d}" for d in SHIFTS for h in ("local", "fused")] if full else [])
    o = {k: np.full((len(ids), 12), np.nan, np.float32) for k in keys}
    o.update({k: np.full((len(ids), 12, 3), np.nan, np.float32) for k in ("attr", "att")})
    if full:
        o.update({k: np.full((len(ids), 12, 6), np.nan, np.float32) for k in ("cam_gl", "cam_parent")})
        o["cam_local"] = np.full((len(ids), 12, 2), np.nan, np.float32)
        o["shift_px"] = np.zeros((len(ids), len(SHIFTS)), np.float32); o["missing_windows"] = np.zeros(len(ids), np.int32)
    pan, prev = [], None
    dl = torch.utils.data.DataLoader(g.GLStudies(ids, meta, {}, K, False, tab), batch_size=1,
                                     num_workers=CFG["workers"], pin_memory=True)
    sig = lambda z: torch.sigmoid(z.float()).reshape(-1)[:12].cpu().numpy()
    t0 = time.time()
    for i, (x, sag, _, _, ctx, xc, sagc, ctxc, cen, fid, wc) in enumerate(dl):
        s = ids[i]
        xg, xcr, pxc = g.prepare_gl(x, sag, xc, sagc, cen, fid, False, dev)
        ctx, ctxc, fidd, wcd = g.to_dev(dev, ctx, ctxc, fid, wc)
        with torch.no_grad(), bf16(dev):    # reproduction (run's forward)
            (fz, gz, lz, nz), _ = gl(xg, 1, ctx, xcr, ctxc, fidd, pxc, wcd, nocrop=True)
        for k, v in (("fused", fz), ("global", gz), ("local", lz), ("nocrop", nz)):
            o[k][i] = sig(v)
        o["local"][i, ~g.CMASK] = o["global"][i, ~g.CMASK]
        idx = list(range(12)) if full else []
        if full:
            zg, Cg, Pg = cam_global(gl, xg, idx)                              # GL global head, Grad-CAM
        else:
            with torch.no_grad(), bf16(dev):
                Pg = maps(gl, xg, False).float().mean((2, 3))
            zg = head_global(gl, Pg).detach()
        with torch.no_grad(), bf16(dev):
            Fc = maps(gl, xcr, True)
        Fc = Fc.float(); Pc = Fc.mean((2, 3)).requires_grad_()
        hc = gl.norm_l(Pc)[None]
        lc = local_logits(gl, hc, fidd)
        hg = gl.norm(Pg)[None]
        if i == 0:                                                            # parity: float32 heads = run's forward
            assert torch.allclose(zg.float(), gz.float()[0], atol=5e-2), (zg, gz)
            assert torch.allclose(pool_local(lc.detach(), fidd, wcd)[0, g.CMASK], lz.float()[0, g.CMASK], atol=5e-2)
        attr, att, dz = attribution(gl, hg, ctx, hc.detach(), ctxc, fidd, pxc)
        o["attr"][i], o["att"][i] = attr.cpu().numpy(), att.cpu().numpy()
        if i == 0:
            assert torch.allclose(dz, (fz - gz).float()[0], atol=5e-2), (dz, fz - gz)
        if not full:
            continue
        with torch.no_grad():
            lw = lc.detach()[0]
            n4 = torch.zeros(12, device=dev).index_add_(0, fidd[0], torch.ones_like(lw)).clamp_min(1)
            o["local_uniform"][i] = sig(torch.zeros(12, device=dev).index_add_(0, fidd[0], lw) / n4)
            top = torch.zeros(12, device=dev).index_reduce_(0, fidd[0], wcd[0], "amax", include_self=False)
            pick = wcd[0] == top[fidd[0]]
            o["local_top1"][i] = sig(torch.zeros(12, device=dev).index_add_(0, fidd[0][pick], lw[pick])
                                     / torch.zeros(12, device=dev).index_add_(0, fidd[0][pick], torch.ones_like(lw[pick])).clamp_min(1))
            for k in ("local_uniform", "local_top1"):
                o[k][i, ~g.CMASK] = o["global"][i, ~g.CMASK]
            o["fused_route"][i] = sig(zg + reader(gl, hg, ctx, hc.detach(), ctxc, fidd, pxc, route=True)[0])
            if prev is not None:
                o["fused_shuffle"][i] = sig(zg + reader(gl, hg, ctx, prev, ctxc, fidd, pxc)[0])
            prev = hc.detach()
        # Grad-CAM: local head per crop (own finding), parent on the global windows
        Cc = torch.relu(torch.einsum("kc,kchw->khw", torch.autograd.grad(lc.sum(), Pc)[0], Fc))
        o["cam_local"][i] = local_cam_stats(Cc, fidd[0], wcd[0])
        zp, Cp, _ = cam_global(par, xg, idx)
        o["parent"][i] = sig(zp)
        win = study_windows(meta, s); sl = tab[s][0][CF].ravel()
        boxes = box_masks(cen[0].numpy(), fid[0].numpy(), win, sl, xg.shape[-1])
        o["missing_windows"][i] = sum(b[0] < 0 for b in boxes)
        fn = fid[0].numpy()
        o["cam_gl"][i], o["cam_parent"][i] = cam_shares(Cg, boxes, fn, idx), cam_shares(Cp, boxes, fn, idx)
        for sid, tname in panels.get(s, []):                                  # panels for local rendering
            f = r.TARGETS.index(tname); q = [j for j in range(len(fn)) if fn[j] == f]
            qt = max(q, key=lambda j: float(wc[0, j]))
            wt = boxes[qt][0]; wtop = int(Cg[f].sum((1, 2)).argmax())
            pan.append({"study": s, "finding": tname, "box": [int(cen[0, qt, 0]), int(cen[0, qt, 1]), int(g.BOXN[f])],
                        "crop_window": wt, "top_window": wtop,
                        "img_crop_window": x[0, max(wt, 0), 1].numpy(), "img_top_window": x[0, wtop, 1].numpy(),
                        "cam_gl_crop_window": Cg[f, max(wt, 0)].cpu().numpy()[::8, ::8],
                        "cam_parent_crop_window": Cp[f, max(wt, 0)].cpu().numpy()[::8, ::8],
                        "cam_gl_top_window": Cg[f, wtop].cpu().numpy()[::8, ::8],
                        "crop": ((xcr[qt, 1].float() * r.STD.flatten()[1].to(dev) + r.MEAN.flatten()[1].to(dev)).clamp(0, 1)
                                 * 255).byte().cpu().numpy(),
                        "cam_crop": Cc[qt].detach().cpu().numpy(),
                        "pred": {k: float(o[k][i, f]) for k in ("global", "local", "fused", "parent")}})
        # global head on the whole 384 px window (GL and parent)
        xg384, _, _ = g.prepare_gl(x, sag, xc, sagc, cen, fid, False, dev, gres=IMG)
        with torch.no_grad(), bf16(dev):
            o["g384"][i] = sig(head_global(gl, maps(gl, xg384, False).float().mean((2, 3))))
            o["parent384"][i] = sig(head_global(par, maps(par, xg384, False).float().mean((2, 3))))
        # crops moved d px in a random direction (fixed seed per study), clipped to keep the box inside the image
        rng = np.random.default_rng(CFG["seed"] + i)
        for j, d in enumerate(SHIFTS):
            ang = rng.uniform(0, 2 * math.pi, size=cen.shape[1])
            h = g.BOXN[fid[0].numpy()] // 2
            c2 = cen[0].numpy() + d * np.stack([np.sin(ang), np.cos(ang)], 1)
            c2 = np.stack([np.clip(c2[:, 0], h, IMG - h), np.clip(c2[:, 1], h, IMG - h)], 1).round()
            o["shift_px"][i, j] = float(np.sqrt(((c2 - cen[0].numpy()) ** 2).sum(1)).mean())
            _, xs, ps = g.prepare_gl(x, sag, xc, sagc, torch.from_numpy(c2[None].astype(np.float32)), fid, False, dev)
            with torch.no_grad(), bf16(dev):
                hs = gl.norm_l(maps(gl, xs, True).float().mean((2, 3)))[None]
                o[f"local_shift{d}"][i] = sig(pool_local(local_logits(gl, hs, fidd), fidd, wcd))
                o[f"fused_shift{d}"][i] = sig(zg + reader(gl, hg, ctx, hs, ctxc, fidd, ps)[0])
            o[f"local_shift{d}"][i, ~g.CMASK] = o["global"][i, ~g.CMASK]
        if i % 100 == 0:
            print(f"  {i}/{len(ids)} {(time.time() - t0) / (i + 1):.2f} s/study", flush=True)
    return o, pan


def main():
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest(); tab = g.crop_table(meta)
    train_ids = sorted(s for s in labels if fold[s] != "0"); val_ids = sorted(s for s in labels if fold[s] == "0")
    tr = sorted(random.Random(CFG["seed"]).sample(train_ids, len(val_ids)))
    if os.environ.get("XAI_N"):                                               # CPU test only
        val_ids, tr = val_ids[:int(os.environ["XAI_N"])], tr[:int(os.environ["XAI_N"])]
    gl = g.RaptorGL(CFG["train_res"], pretrained=False)
    gl.load_state_dict(torch.load(os.environ["GL_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    par = r.Raptor(CFG["train_res"], pretrained=False)
    par.load_state_dict(torch.load(os.environ["LV2_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    for m in (gl, par):
        m.to(dev).eval().to(memory_format=torch.channels_last)
        for p in m.parameters():
            p.requires_grad_(False)
    panels = {}
    for s, t in json.loads(Path(os.environ["PANEL_IDS"]).read_text()) if os.environ.get("PANEL_IDS") else []:
        panels.setdefault(s, []).append((s, t))
    for split, ids, full in (("val", val_ids, True), ("train", tr, False)):
        t = time.time()
        o, pan = run(gl, par, ids, meta, tab, dev, full, panels if full else {})
        np.savez(OUT / f"xai_{split}.npz", ids=np.array(ids), targets=np.array(r.TARGETS), **o)
        if pan:
            np.savez_compressed(OUT / "panels.npz", panels=np.array(json.dumps(
                [{k: v for k, v in p.items() if not isinstance(v, np.ndarray)} for p in pan])),
                **{f"{k}_{j}": v for j, p in enumerate(pan) for k, v in p.items() if isinstance(v, np.ndarray)})
        print(f"XAI {split} {len(ids)} studies {time.time() - t:.0f}s; reader attribution global / own / other "
              f"{np.nanmean(o['attr'][:, CF], (0, 1)).round(3).tolist()}", flush=True)
        if full:
            print(f"XAI missing crop windows {int(o['missing_windows'].sum())}; mean shift px "
                  f"{o['shift_px'].mean(0).round(1).tolist()}", flush=True)
    print("XAI_DONE", flush=True)


if __name__ == "__main__":
    main()
