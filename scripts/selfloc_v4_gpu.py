"""Exp SL-0v4 (self-location for v4): can an r50-GL v4 SWA model place its own crops, so Kaggle inference needs no MS1?
Inference only (Modal H100), fold 0 (825 studies). v4 was trained and scored on MS1's crop table (MS1's 4 most-attended
eval windows per finding, gradient x activation centroid of MS1's own logit contribution, scripts/
extract_ms1_crop_features_gpu.py). Self-location replays exactly that procedure with v4's own GLOBAL pass:
  windows  per cropped finding n, the 4 eval windows with the largest a_kn (v4's global window attention, 1e's head);
           any plane, as MS1 (no sagittal restriction);
  points   on each, the gradient x activation map of a_kn (LN(pool(F_k)) . W_n) (a held fixed) over the 10 x 10
           layer-4 cells of the 320 eval crop (32 px each) -> centroid of the positive part -> (row, col) px of the
           384 cache image; image centre if the map is empty;
  weights  a_kn of the 4 windows, renormalised (MS1's a_top role).
Then v4's ordinary predict on those crops (fused / global / local), next to the same predict on MS1's table (must
reproduce the run's val_swa_*.json). Agreement with MS1 per finding: share of own windows within +-1 slice of an MS1
slice, top-1 point distance (mm, 140 mm field = 384 px) where the top-1 slices are within +-1.
Out (GL_OUT): val_{self,ms1}_{fused,global,local}.json, selfloc_regions.npz, selfloc_log.json.
Scorer: scripts/audit_selfloc_v4.py.

Env: D96_CACHE, D96_PHASE=r50_gl, V4_ARM, GL_CROPS (slim MS1 table), GL_OUT, GL_WEIGHTS (v4 SWA; empty = random init,
CPU test only), SL_N (first N fold-0 studies, CPU test).
"""
import csv
import json
import os
import time

import numpy as np
import torch

import r50_gl4_runner as m

r, v2, CFG = m.r, m.v2, m.CFG
IMG, MAXS, K, CF = v2.IMG, v2.MAXS, v2.K, m.CF
RES = CFG["eval_res"]; LO = (IMG - RES) // 2
TOP, MM = v2.GL["top"], 140.0 / IMG


def locate(model, x, sag, wc, dev):
    """one study's eval windows [1, K, 3, IMG, IMG] -> table entry (slice, row, col, weight), each [12, TOP]
    (only the cropped findings' rows are filled; the others are never read)."""
    z = torch.zeros(1, 1, dtype=torch.long)
    xg, _, _ = v2.prepare_gl(x, sag, x[:, :1], torch.zeros(1, 1, dtype=torch.bool), torch.full((1, 1, 2), IMG / 2), z,
                             False, dev)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        fm = model.backbone.forward_features(xg)                                 # [K, 2048, c, c], global BN stats
    fm = fm.float()
    W = model.clsW.float()[CF]                                                   # [10, d]
    with torch.no_grad():
        a = torch.softmax(model.att(model.norm(fm.mean((2, 3)))).float(), dim=0)[:, CF]   # [K, 10]
    top = a.topk(TOP, dim=0).indices                                             # [TOP, 10]
    U = torch.unique(top); row = {int(k): j for j, k in enumerate(U.tolist())}
    t = fm[U].detach().requires_grad_(True)
    contrib = a[U] * (model.norm(t.mean((2, 3))) @ W.T)                          # [u, 10], a fixed
    c = t.shape[-1]; px = RES / c; gi = torch.arange(c, device=dev, dtype=torch.float32)
    S = np.zeros((12, TOP), np.int64); R = np.full((12, TOP), IMG // 2, np.int64); Q = R.copy()
    A = np.ones((12, TOP), np.float32)
    for j, f in enumerate(CF):
        gr, = torch.autograd.grad(contrib[:, j].sum(), t, retain_graph=j < len(CF) - 1)
        ks = top[:, j].tolist(); rows = [row[k] for k in ks]
        mp = (gr[rows] * t.detach()[rows]).sum(1).clamp_min(0)                  # [TOP, c, c]
        tot = mp.sum((1, 2)); ok = tot > 0; tot = tot.clamp_min(1e-12)
        cy = torch.where(ok, (mp.sum(2) * gi).sum(1) / tot, torch.full_like(tot, (c - 1) / 2))
        cx = torch.where(ok, (mp.sum(1) * gi).sum(1) / tot, torch.full_like(tot, (c - 1) / 2))
        S[f] = wc[ks]
        R[f] = np.round((LO + (cy + 0.5) * px).cpu().numpy()).astype(np.int64)
        Q[f] = np.round((LO + (cx + 0.5) * px).cpu().numpy()).astype(np.int64)
        A[f] = a[top[:, j], j].detach().cpu().numpy()
    return S, R, Q, A


def agreement(ids, tab_ms1, tab_self):
    out = {}
    for f in CF:
        near, dist = [], []
        for s in ids:
            Sm, Rm, Qm = (x[f] for x in tab_ms1[s][:3]); Ss, Rs, Qs = (x[f] for x in tab_self[s][:3])
            near.append(np.mean([np.abs(Sm - k).min() <= 1 for k in Ss]))
            hit = np.abs(Sm - Ss[0]) <= 1
            if hit.any():
                q = int(np.argmax(hit)); dist.append(MM * np.hypot(Rs[0] - Rm[q], Qs[0] - Qm[q]))
        out[r.TARGETS[f]] = {"windows_within_1_slice": round(float(np.mean(near)), 3),
                             "top1_matched_share": round(len(dist) / len(ids), 3),
                             "top1_point_mm_median": round(float(np.median(dist)), 1) if dist else None}
    return out


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = m.OUT; out.mkdir(parents=True, exist_ok=True)
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest()
    tab_ms1 = v2.crop_table(meta)
    if os.environ.get("SL_N"):                           # CPU test: any labelled studies of the smoke cache
        ids = sorted(s for s in labels if s in meta and s in tab_ms1)[:int(os.environ["SL_N"])]
    else:
        ids = sorted(s for s in labels if fold[s] == "0"); assert len(ids) == 825
    model = m.build(dev, pretrained=False)[0]
    if os.environ.get("GL_WEIGHTS"):
        model.load_state_dict(torch.load(os.environ["GL_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    else:
        assert os.environ.get("SL_N"), "random init only for the CPU test"
    model.set_dense()
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dl = torch.utils.data.DataLoader(r.Studies(ids, meta, {}, K, False), batch_size=1,
                                     num_workers=CFG["workers"] if dev.type == "cuda" else 0, pin_memory=True)
    tab_self, t0 = {}, time.time()
    for i, (x, sag, _, _, _) in enumerate(dl):
        wc = np.clip(r.centres(meta[ids[i]]["mask"], K, False), 1, MAXS - 2)   # window -> centre slice (crop_table)
        tab_self[ids[i]] = locate(model, x, sag, wc, dev)
        if i % 100 == 0:
            print(f"SL locate {i}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    res = {"arm": m.ARM, "weights": os.environ.get("GL_WEIGHTS"), "studies": len(ids),
           "locate_s_per_study": round((time.time() - t0) / len(ids), 4)}
    for nm, tab in (("ms1", tab_ms1), ("self", tab_self)):
        t1 = time.time(); p = m.predict(model, ids, meta, tab, dev)
        res[f"predict_{nm}_s_per_study"] = round((time.time() - t1) / len(ids), 4)
        for j, h in enumerate(m.HEADS):
            r.save_predictions(out / f"val_{nm}_{h}.json", ids, p[j], f"r50-GL4 {m.ARM} SWA, {nm} crops, {h}")
        if not os.environ.get("SL_N"):
            res[nm] = {h: r.macro_auc(p[j], ids, labels)[0] for j, h in enumerate(m.HEADS)}
    res["agreement_vs_ms1"] = agreement(ids, tab_ms1, tab_self)
    np.savez(out / "selfloc_regions.npz", ids=np.array(ids), **{k: np.stack([tab_self[s][j] for s in ids])
                                                                 for j, k in enumerate(("slices", "rows", "cols", "a"))})
    res["seconds"] = round(time.time() - t0)
    (out / "selfloc_log.json").write_text(json.dumps(res, indent=1) + "\n")
    print("SL_DONE " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
