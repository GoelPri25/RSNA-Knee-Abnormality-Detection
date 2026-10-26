"""Exp SL-0 (self-location): can r50-GL v3 pick its own regions, so inference needs no MS1?
Inference only (Modal H100), fold 0 (825 studies), v3 SWA weights. v3 was trained on MS1's regions (crop table: MS1's
4 most-attended windows per finding, gradient x activation centroid; Baker's mapped into sagittal, points4).
Self-location, from v3's own global pass (the 94 eval windows, 320 centre crop, 1e's head):
  windows  per region finding n, the 4 windows with the largest a_kn (v3's per-finding window attention); Baker's only
           among sagittal windows (as trained: points4 maps it to sagittal);
  points   on each, the gradient x activation map of a_kn (LN(h_k) . W_n) (a held fixed, as MS1-CP / LOC-0) over the
           10 x 10 layer-4 cells (32 px each, inside the 320 crop) -> centroid of the positive part -> (row, col) px
           of the 384 image; centre if the map is empty.
Then v3's ordinary forward with those regions (fused / global / local), next to the same forward on MS1's regions
(reproduces val_swa_*.json: check). Agreement with MS1 per finding: share of own windows within +-1 slice of an MS1
region slice, top-1 same plane, point distance (mm, 140 mm field = 384 px) where the top-1 slices match.
Out: val_self_{fused,global,local}.json, val_ms1_*.json, val_{self,ms1}_gate_beta.npy, selfloc_regions.npz,
selfloc_log.json. Scorer: scripts/audit_selfloc.py.

Env: D96_CACHE, D96_PHASE=r50_gl, GL_CROPS, GL_POINTS4, GL_OUT (output dir), GL_WEIGHTS (v3 SWA), SL_N (CPU test).
"""
import csv
import json
import os
import time

import numpy as np
import torch

import r50_gl3_runner as v

r, g, CFG = v.r, v.g, v.CFG
IMG, MAXS, K, RF, NR = v.IMG, v.MAXS, v.K, v.RF, v.NR
OUT = v.OUT
RES = CFG["eval_res"]; LO = (IMG - RES) // 2
MM = 140.0 / IMG
KB = list(RF).index(r.TARGETS.index("Baker's"))


def locate(model, x, sag, wc, dev):
    """one study's eval windows [1, K, 3, IMG, IMG] -> slices [8, 4], rows [8, 4], cols [8, 4], top-1 plane slot of the
    unrestricted Baker's choice"""
    z = torch.zeros(1, 1, dtype=torch.long)
    xg, _, _ = g.prepare_gl(x, sag, x[:, :1], torch.zeros(1, 1, dtype=torch.bool), torch.full((1, 1, 2), IMG / 2), z,
                            False, dev)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        fm = model.backbone.forward_features(xg)                                 # [K, 2048, c, c]
    fm = fm.float()
    W = model.clsW.float()[RF]                                                   # [8, d]
    with torch.no_grad():
        hn = model.norm(fm.mean((2, 3)))
        a = torch.softmax(model.att(hn).float(), dim=0)[:, RF]                  # [K, 8]
    sagw = torch.as_tensor(np.isin(r.SLOT_OF[wc], r.SAGITTAL_SLOTS), device=dev)
    b_free = int(a[:, KB].argmax())
    a_sel = a.clone(); a_sel[~sagw, KB] = -1.0       # Baker's: sagittal windows first (other planes only if < 4 sagittal)
    top = a_sel.topk(NR, dim=0).indices                                          # [4, 8]
    U = torch.unique(top); row = {int(k): j for j, k in enumerate(U.tolist())}
    t = fm[U].detach().requires_grad_(True)
    contrib = a[U] * (model.norm(t.mean((2, 3))) @ W.T)                          # [u, 8], a fixed
    c = t.shape[-1]; px = RES / c; gi = torch.arange(c, device=dev, dtype=torch.float32)
    S = np.zeros((len(RF), NR), np.int64); R = np.zeros((len(RF), NR), np.float32); Q = R.copy()
    for j in range(len(RF)):
        gr, = torch.autograd.grad(contrib[:, j].sum(), t, retain_graph=j < len(RF) - 1)
        ks = top[:, j].tolist(); rows = [row[k] for k in ks]
        mp = (gr[rows] * t.detach()[rows]).sum(1).clamp_min(0)                  # [4, c, c]
        tot = mp.sum((1, 2)); ok = tot > 0; tot = tot.clamp_min(1e-12)
        cy = torch.where(ok, (mp.sum(2) * gi).sum(1) / tot, torch.full_like(tot, (c - 1) / 2))
        cx = torch.where(ok, (mp.sum(1) * gi).sum(1) / tot, torch.full_like(tot, (c - 1) / 2))
        S[j] = wc[ks]; R[j] = (LO + (cy + 0.5) * px).cpu().numpy(); Q[j] = (LO + (cx + 0.5) * px).cpu().numpy()
    return S, np.round(R).astype(np.int64), np.round(Q).astype(np.int64), int(r.SLOT_OF[wc[b_free]])


def agreement(ids, tab_ms1, tab_self):
    """per region finding: share of own windows within +-1 slice of an MS1 region slice, top-1 same plane (sagittal
    fat-sat / non-fat-sat count as one), top-1 point distance (mm) where an MS1 region slice is within +-1"""
    agree = {}
    for j, t in enumerate(v.R8):
        near, plane, dist = [], [], []
        for s in ids:
            Sm, Rm, Qm = (a_[j] for a_ in tab_ms1[s]); Ss, Rs, Qs = (a_[j] for a_ in tab_self[s])
            near.append(np.mean([np.abs(Sm - k).min() <= 1 for k in Ss]))
            plane.append(r.SLOT_OF[Ss[0]] == r.SLOT_OF[Sm[0]] or
                         (np.isin(r.SLOT_OF[[Ss[0], Sm[0]]], r.SAGITTAL_SLOTS).all()))
            m_ = np.abs(Sm - Ss[0]) <= 1
            if m_.any():
                q = int(np.argmax(m_)); dist.append(MM * np.hypot(Rs[0] - Rm[q], Qs[0] - Qm[q]))
        agree[t] = {"windows_within_1_slice": round(float(np.mean(near)), 3), "top1_same_plane": round(float(np.mean(plane)), 3),
                    "top1_point_mm_median": round(float(np.median(dist)), 1) if dist else None,
                    "top1_matched_share": round(len(dist) / len(ids), 3)}
    return agree


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest()
    ids = sorted(s for s in labels if fold[s] == "0")
    if os.environ.get("SL_N"):
        ids = ids[:int(os.environ["SL_N"])]
    else:
        assert len(ids) == 825
    tab_ms1 = v.region_table(meta)
    model = v.RaptorGL3(CFG["train_res"], pretrained=False)
    model.load_state_dict(torch.load(os.environ["GL_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    model = model.to(dev).eval().to(memory_format=torch.channels_last)
    for p in model.parameters():
        p.requires_grad_(False)
    dl = torch.utils.data.DataLoader(r.Studies(ids, meta, {}, K, False), batch_size=1,
                                     num_workers=CFG["workers"] if dev.type == "cuda" else 0, pin_memory=True)
    tab_self, bslot, t0 = {}, [], time.time()
    for i, (x, sag, _, _, _) in enumerate(dl):
        wc = np.clip(r.centres(meta[ids[i]]["mask"], K, False), 1, MAXS - 2)   # window -> centre slice (crop_table)
        S, R, Q, bs = locate(model, x, sag, wc, dev)
        tab_self[ids[i]] = (S, R, Q); bslot.append(bs)
        if i % 100 == 0:
            print(f"SL locate {i}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    t_loc = (time.time() - t0) / len(ids)
    res = {"studies": len(ids), "locate_s_per_study": round(t_loc, 4)}
    for nm, tab in (("ms1", tab_ms1), ("self", tab_self)):
        t1 = time.time(); p, beta = v.predict(model, ids, meta, tab, dev)
        res[f"predict_{nm}_s_per_study"] = round((time.time() - t1) / len(ids), 4)
        for j, h in enumerate(v.HEADS):
            r.save_predictions(OUT / f"val_{nm}_{h}.json", ids, p[j], f"r50-GL v3 SWA, {nm} regions, {h}")
        np.save(OUT / f"val_{nm}_gate_beta.npy", beta)
        if not os.environ.get("SL_N"):
            sc = v.score(p, beta, ids, r.load_labels(r.ASSETS / "validation_labels.json", "validation"), labels)
            res[nm] = {k: x_ for k, x_ in sc.items() if not k.startswith("lv2_per")}
    res["agreement_vs_ms1"] = agreement(ids, tab_ms1, tab_self)
    res["bakers_unrestricted_top1_slot_share"] = {int(k): round(float(np.mean(np.array(bslot) == k)), 3)
                                                  for k in sorted(set(bslot))}
    np.savez(OUT / "selfloc_regions.npz", ids=np.array(ids), slices=np.stack([tab_self[s][0] for s in ids]),
             rows=np.stack([tab_self[s][1] for s in ids]), cols=np.stack([tab_self[s][2] for s in ids]))
    res["seconds"] = round(time.time() - t0)
    (OUT / "selfloc_log.json").write_text(json.dumps(res, indent=1) + "\n")
    print("SL_DONE " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
