"""Exp r50-GL v3 XAI + ablations (inference only, no training): why did v3 fail and can any fix beat the best single?
Model: r50-GL v3 SWA (GL_WEIGHTS). Evaluation inputs as the run (94 windows at the 320 centre crop; 8 findings x 4
regions at MS1's points, Baker's mapped to sagittal). Fold 0 (825) with everything; 825 training studies (seed-2026
draw as the v2 XAI) with predictions, gate and pooled features only (in-sample vs held-out).
Per fold-0 study:
  reproduction  fused / global / local logits, gate beta (vs val_swa_*.json)
  features      pooled global z_g,f and local z_l,f of the 8 region findings (float16) -> offline cross-fitted gate,
                redundancy (CKA), in-sample vs held-out
  ablations     local with uniform / top-1 region weights; crops moved 16 / 32 / 48 px (random direction, seed per
                study; precision); local + fused with the previous study's crops (no study information); crops through
                the GLOBAL BatchNorm statistics (dual-BN check); Baker's local on MS1's original axial regions
  Grad-CAM      layer 4. Global head per region finding over the 94 windows: share on the region windows, inside the
                region boxes, box share within those windows, the area shares (chance), top window is a region window
                (as v2's XAI, comparable). Local head per region: outer-ring share (outer 2 of 12 cells, area 0.556),
                centroid offset (px of 384), weighted by the learned region attention. Window-attention entropy.
Panels (PANEL_IDS json [[study, finding], ...], region findings only): images and CAMs for local rendering.
Env: D96_CACHE, D96_PHASE=r50_gl, GL_CROPS, GL_POINTS4, GL_OUT, GL_WEIGHTS, XAI_OUT, PANEL_IDS, XAI_N (CPU test).
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

import r50_gl3_runner as v

g, r, CFG = v.g, v.r, v.CFG
OUT = Path(os.environ["XAI_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
K, IMG, MAXS, RF, NR = v.K, v.IMG, v.MAXS, v.RF, v.NR
SHIFTS, OFF, RING = (16, 32, 48), (IMG - CFG["eval_res"]) // 2, 2
KB = list(RF).index(r.TARGETS.index("Baker's"))
bf16 = lambda dev: torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


def maps(model, x, crop):
    for m in model.bns:
        m.crop = crop
    f = model.backbone.forward_features(x)
    for m in model.bns:
        m.crop = False
    return f


def heads(model, Pg, Pc):
    """pooled window features Pg [k, d], region features Pc [32, d] (float32) -> global logits [12], local logits [8],
    beta [8], z_g [8, d], z_l [8, d], region weights [8, 4], window attention [k, 12]"""
    hg = model.norm(Pg)
    a = torch.softmax(model.att(hg).float(), dim=0)
    zg = torch.einsum("kn,kf->nf", a, hg)
    gz = (zg * model.clsW).sum(-1) + model.clsb
    zl, ac = local(model, Pc)
    lr = (zl * model.clsV[RF]).sum(-1) + model.clsvb[RF]
    beta = torch.sigmoid(model.gate(torch.cat([zg[RF], zl], -1)).float()[torch.arange(len(RF)), RF])
    return gz, lr, beta, zg[RF], zl, ac, a


def local(model, Pc, weights=None):
    hc = model.norm_l(Pc).view(len(RF), NR, -1)
    ac = model.att_l(hc).float()[torch.arange(len(RF)), :, RF] if weights is None else None
    w = torch.softmax(ac, dim=1) if weights is None else weights
    return (w[..., None] * hc).sum(1), w


def logits_from(model, zg, zl, gz):
    lr = (zl * model.clsV[RF]).sum(-1) + model.clsvb[RF]
    beta = torch.sigmoid(model.gate(torch.cat([zg, zl], -1)).float()[torch.arange(len(RF)), RF])
    return lr, beta * gz[RF] + (1 - beta) * lr


def cam(logit, P, Fm):
    gP, = torch.autograd.grad(logit, P, retain_graph=True)
    return torch.relu(torch.einsum("kc,kchw->khw", gP, Fm))


def cam_shares(C, boxes):
    """C [k, H, W] one finding; boxes [(window or -1, rows, cols)] of its regions -> 6 stats (as v2's XAI)"""
    bx = [b for b in boxes if b[0] >= 0]
    if not bx:
        return np.full(6, np.nan, np.float32)
    k = C.shape[0]; wins = sorted({b[0] for b in bx})
    m = torch.zeros_like(C, dtype=torch.bool)
    for w, rs, cs in bx:
        m[w, rs, cs] = True
    tot = C.sum().clamp_min(1e-12); cw = C[wins].sum()
    return np.array([float(cw / tot), float((C * m).sum() / tot), float((C[wins] * m[wins]).sum() / cw.clamp_min(1e-12)),
                     len(wins) / k, float(m[wins].float().mean()), float(int(C.sum((1, 2)).argmax()) in wins)], np.float32)


def region_boxes(cen, sl, win, res):
    out = []
    for q in range(len(sl)):
        w = np.where(win == sl[q])[0]; f = RF[q // NR]
        h = g.BOXN[f] // 2; R_, Q_ = int(cen[q, 0]) - OFF, int(cen[q, 1]) - OFF
        out.append((int(w[0]) if len(w) else -1, slice(max(R_ - h, 0), min(R_ + h, res)), slice(max(Q_ - h, 0), min(Q_ + h, res))))
    return out


def local_cam_stats(Cc, ac, n=12):
    """Cc [32, n, n], ac [8, 4] -> [8, 2]: outer-ring share, centroid offset (px of 384), region-weighted"""
    tot = Cc.sum((1, 2)).clamp_min(1e-12)
    ring = 1 - Cc[:, RING:n - RING, RING:n - RING].sum((1, 2)) / tot
    ii = (torch.arange(n, device=Cc.device, dtype=torch.float32) + 0.5) / n - 0.5
    cy, cx = (Cc.sum(2) * ii).sum(1) / tot, (Cc.sum(1) * ii).sum(1) / tot
    box = torch.as_tensor(g.BOXN[np.repeat(RF, NR)], device=Cc.device, dtype=torch.float32)
    off = torch.sqrt(cy ** 2 + cx ** 2) * box
    w = ac.reshape(-1)
    return torch.stack([(ring * w).view(len(RF), NR).sum(1), (off * w).view(len(RF), NR).sum(1)], 1).detach().cpu().numpy()


def crops_at(x_dummy, sag, vol_crops, sagc, cen, fid, dev):
    _, xc, _ = g.prepare_gl(x_dummy, sag, vol_crops, sagc, cen, fid, False, dev)
    return xc


def axial_bakers(meta, s, tab_ms1):
    """MS1's original 4 Baker's regions (pre-sagittal mapping): crops [1, 4, 3, IMG, IMG], sagittal flags, centres"""
    m = meta[s]; vol = np.load(m["file"], mmap_mode="r")[m["row"]:m["row"] + MAXS]
    sl, rr, cc, _ = tab_ms1[s]; kb = r.TARGETS.index("Baker's")
    S, R_, Q_ = sl[kb].astype(int), rr[kb].astype(int), cc[kb].astype(int)
    h = g.BOXN[kb] // 2
    R_, Q_ = np.clip(R_, h, IMG - h), np.clip(Q_, h, IMG - h)
    xc = np.ascontiguousarray(vol[np.stack([S - 1, S, S + 1], 1)])
    sagc = np.isin(r.SLOT_OF[S], r.SAGITTAL_SLOTS)
    return (torch.from_numpy(xc)[None], torch.from_numpy(sagc)[None],
            torch.from_numpy(np.stack([R_, Q_], 1).astype(np.float32))[None], torch.full((1, NR), kb, dtype=torch.long))


def run(model, ids, meta, tab, tab_ms1, dev, full, panels):
    n, d = len(ids), model.backbone.num_features
    P = {k: np.full((n, 12), np.nan, np.float32) for k in ("fused", "global", "local")}
    E = {k: np.full((n, len(RF)), np.nan, np.float32) for k in ("beta",) + ((
        "local_uniform", "local_top1", "local_shuffle", "fused_shuffle", "local_gbn", "fused_gbn", "local_bakers_axial")
        + tuple(f"{h}_shift{s}" for s in SHIFTS for h in ("local", "fused")) if full else ())}
    feats = {k: np.zeros((n, len(RF), d), np.float16) for k in ("zg", "zl")}
    extra = {"ac": np.zeros((n, len(RF), NR), np.float32)}
    if full:
        extra.update({"cam_g": np.full((n, len(RF), 6), np.nan, np.float32), "cam_l": np.full((n, len(RF), 2), np.nan, np.float32),
                      "att_entropy": np.zeros((n, 12), np.float32), "missing_windows": np.zeros(n, np.int32)})
    pan, prev = [], None
    dl = torch.utils.data.DataLoader(v.V3Studies(ids, meta, {}, K, False, tab), batch_size=1,
                                     num_workers=CFG["workers"] if dev.type == "cuda" else 0, pin_memory=True)
    sig = lambda z: torch.sigmoid(z.float()).detach().cpu().numpy()
    t0 = time.time()
    for i, (x, sag, _, _, _, xc, sagc, cen, fid) in enumerate(dl):
        s = ids[i]
        xg, xcr, _ = g.prepare_gl(x, sag, xc, sagc, cen, fid, False, dev)
        with torch.no_grad(), bf16(dev):
            fz, gz0, lz0, bt0 = model(xg, 1, xcr)
        P["fused"][i], P["global"][i], P["local"][i] = sig(fz[0]), sig(gz0[0]), sig(lz0[0]); E["beta"][i] = bt0[0].float().cpu().numpy()
        with torch.no_grad(), bf16(dev):
            Fg, Fc = maps(model, xg, False), maps(model, xcr, True)
        Fg, Fc = Fg.float(), Fc.float()
        Pg, Pc = Fg.mean((2, 3)).requires_grad_(full), Fc.mean((2, 3)).requires_grad_(full)
        gz, lr, beta, zg, zl, ac, a = heads(model, Pg, Pc)
        if i == 0:
            assert torch.allclose(gz.detach(), gz0.float()[0], atol=5e-2), (gz, gz0)
            assert torch.allclose(lr.detach(), lz0.float()[0, RF], atol=5e-2), (lr, lz0[0, RF])
        feats["zg"][i], feats["zl"][i] = zg.detach().cpu().numpy(), zl.detach().cpu().numpy()
        extra["ac"][i] = ac.detach().cpu().numpy()
        if not full:
            continue
        extra["att_entropy"][i] = (-(a * a.clamp_min(1e-12).log()).sum(0)).detach().cpu().numpy()
        with torch.no_grad():
            gzd, Pcd = gz.detach(), Pc.detach(); zgd = zg.detach()
            zu, _ = local(model, Pcd, torch.full((len(RF), NR), 1 / NR, device=dev))
            E["local_uniform"][i] = sig(logits_from(model, zgd, zu, gzd)[0])
            top = torch.zeros_like(ac).scatter_(1, ac.argmax(1, keepdim=True), 1.0)
            zt, _ = local(model, Pcd, top)
            E["local_top1"][i] = sig(logits_from(model, zgd, zt, gzd)[0])
            if prev is not None:
                zs, _ = local(model, prev)
                l_, f_ = logits_from(model, zgd, zs, gzd); E["local_shuffle"][i], E["fused_shuffle"][i] = sig(l_), sig(f_)
            prev = Pcd
            with bf16(dev):
                Fb = maps(model, xcr, False)
            zb, _ = local(model, Fb.float().mean((2, 3)))
            l_, f_ = logits_from(model, zgd, zb, gzd); E["local_gbn"][i], E["fused_gbn"][i] = sig(l_), sig(f_)
            xa, sa, ca, fa = axial_bakers(meta, s, tab_ms1)
            xca = crops_at(xa[:, :1], torch.zeros(1, 1, dtype=torch.bool), xa, sa, ca, fa, dev)
            with bf16(dev):
                Pa = model.norm_l(maps(model, xca, True).float().mean((2, 3)))      # [4, d]
            wa = torch.softmax(model.att_l(Pa).float()[:, RF[KB]], 0)
            za = (wa[:, None] * Pa).sum(0)
            E["local_bakers_axial"][i, KB] = float(torch.sigmoid((za * model.clsV[RF[KB]]).sum() + model.clsvb[RF[KB]]))
            rng = np.random.default_rng(CFG["seed"] + i)
            for sh in SHIFTS:
                ang = rng.uniform(0, 2 * math.pi, size=cen.shape[1]); h = g.BOXN[fid[0].numpy()] // 2
                c2 = cen[0].numpy() + sh * np.stack([np.sin(ang), np.cos(ang)], 1)
                c2 = np.stack([np.clip(c2[:, 0], h, IMG - h), np.clip(c2[:, 1], h, IMG - h)], 1).round()
                _, xs, _ = g.prepare_gl(x, sag, xc, sagc, torch.from_numpy(c2[None].astype(np.float32)), fid, False, dev)
                with bf16(dev):
                    zs, _ = local(model, maps(model, xs, True).float().mean((2, 3)))
                l_, f_ = logits_from(model, zgd, zs, gzd); E[f"local_shift{sh}"][i], E[f"fused_shift{sh}"][i] = sig(l_), sig(f_)
        # Grad-CAM: global head per region finding, local head per region
        Cg = torch.stack([cam(gz[f], Pg, Fg) for f in RF])                      # [8, k, c, c]
        Cg = F.interpolate(Cg, size=xg.shape[-2:], mode="bilinear", align_corners=False)
        Cc = torch.relu(torch.einsum("kc,kchw->khw", torch.autograd.grad(
            ((model.norm_l(Pc) * model.clsV[np.repeat(RF, NR)]).sum(-1)).sum(), Pc)[0], Fc))
        extra["cam_l"][i] = local_cam_stats(Cc, ac.detach())
        win = np.clip(r.centres(meta[s]["mask"], K, False), 1, MAXS - 2)
        S = tab[s][0].ravel(); boxes = region_boxes(cen[0].numpy(), S, win, xg.shape[-1])
        extra["missing_windows"][i] = sum(b[0] < 0 for b in boxes)
        for j in range(len(RF)):
            extra["cam_g"][i, j] = cam_shares(Cg[j], boxes[j * NR:(j + 1) * NR])
        for _, tname in panels.get(s, []):
            j = list(RF).index(r.TARGETS.index(tname)); q = j * NR + int(ac[j].argmax())
            wt = boxes[q][0]; wtop = int(Cg[j].sum((1, 2)).argmax())
            pan.append({"study": s, "finding": tname, "box": [int(cen[0, q, 0]), int(cen[0, q, 1]), int(g.BOXN[RF[j]])],
                        "crop_window": wt, "top_window": wtop,
                        "img_crop_window": x[0, max(wt, 0), 1].numpy(), "img_top_window": x[0, wtop, 1].numpy(),
                        "cam_gl_crop_window": Cg[j, max(wt, 0)].detach().cpu().numpy()[::8, ::8],
                        "cam_gl_top_window": Cg[j, wtop].detach().cpu().numpy()[::8, ::8],
                        "crop": ((xcr[q, 1].float() * r.STD.flatten()[1].to(dev) + r.MEAN.flatten()[1].to(dev)).clamp(0, 1)
                                 * 255).byte().cpu().numpy(),
                        "cam_crop": Cc[q].detach().cpu().numpy(),
                        "pred": {"global": float(P["global"][i, RF[j]]), "local": float(P["local"][i, RF[j]]),
                                 "fused": float(P["fused"][i, RF[j]]), "beta": float(E["beta"][i, j])}})
        if i % 100 == 0:
            print(f"  {i}/{n} {(time.time() - t0) / (i + 1):.2f} s/study", flush=True)
    return {**P, **E, **feats, **extra}, pan


def main():
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest(); tab = v.region_table(meta); tab_ms1 = g.crop_table(meta)
    train_ids = sorted(s for s in labels if fold[s] != "0"); val_ids = sorted(s for s in labels if fold[s] == "0")
    tr = sorted(random.Random(CFG["seed"]).sample(train_ids, len(val_ids)))
    if os.environ.get("XAI_N"):
        val_ids, tr = val_ids[:int(os.environ["XAI_N"])], tr[:int(os.environ["XAI_N"])]
    model = v.RaptorGL3(CFG["train_res"], pretrained=False)
    model.load_state_dict(torch.load(os.environ["GL_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    model = model.to(dev).eval().to(memory_format=torch.channels_last)
    for p in model.parameters():
        p.requires_grad_(False)
    panels = {}
    for s, t in json.loads(Path(os.environ["PANEL_IDS"]).read_text()) if os.environ.get("PANEL_IDS") else []:
        if t in v.R8:
            panels.setdefault(s, []).append((s, t))
    for split, ids, full in (("val", val_ids, True), ("train", tr, False)):
        t = time.time()
        o, pan = run(model, ids, meta, tab, tab_ms1, dev, full, panels if full else {})
        np.savez(OUT / f"xai3_{split}.npz", ids=np.array(ids), targets=np.array(r.TARGETS), regions=np.array(v.R8), **o)
        if pan:
            np.savez_compressed(OUT / "panels3.npz", panels=np.array(json.dumps(
                [{k: v_ for k, v_ in p.items() if not isinstance(v_, np.ndarray)} for p in pan])),
                **{f"{k}_{j}": v_ for j, p in enumerate(pan) for k, v_ in p.items() if isinstance(v_, np.ndarray)})
        print(f"XAI3 {split} {len(ids)} studies {time.time() - t:.0f}s", flush=True)
    print("XAI3_DONE", flush=True)


if __name__ == "__main__":
    main()
