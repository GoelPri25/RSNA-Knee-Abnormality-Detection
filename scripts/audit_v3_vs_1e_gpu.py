"""Exp v3 vs Exp 1e: branch ablation and spatial-attention XAI on fold 0 (inference only; runs on Modal).

Ablations of the v3 SWA model (each a full fold-0 prediction file, scored on CPU by audit_v3_vs_1e.py):
  full          v3 as trained (must match run_r50_spatial/val_swa.json)
  base_only     U.p removed: the GAP path of the stride-16 backbone alone (what v3's backbone gives without selection)
  spatial_only  W.g removed: the per-finding spatial pool alone (window attention kept)
  uniform       alpha replaced by the uniform average: the spatial classifier U reads the plain mean (extra
                capacity without selection)
plus 1e SWA (must match run_r50_pertarget/val_swa.json).

XAI per study and finding (top window = the window each model's per-finding window attention weights most):
  window-attention mass per series slot, both models;
  v3 alpha entropy (normalised) and the alpha-weighted centroid in the top window;
  1e class-activation map in its top window: cell c of window k contributes W_n . LN-scaled (z_c - mu_k) to the
  finding's logit (exact for GAP -> LayerNorm -> linear: the mean of the per-cell terms equals the window term);
  agreement when both pick the same window: Spearman of v3 alpha vs 1e CAM (CAM upsampled 2x to 20 x 20) and the
  distance between their argmax cells in mm (one 20 x 20 cell = 140 mm / 384 px x 16 = 5.83 mm).
Maps for 10 positive fold-0 studies per finding (lowest SHA-256 of the study id) are saved for rendering.

Env: D96_CACHE, D96_ASSETS, V3_WEIGHTS, E1_WEIGHTS, AUDIT_OUT; D96_PHASE must be r50_spatial.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import coatnet_d96_runner as r

assert os.environ["D96_PHASE"] == "r50_spatial"
OUT = Path(os.environ["AUDIT_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
CELL_MM = 140.0 / 384 * 16


def rank(a):
    return np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(np.float64)


def spearman(a, b):
    ra, rb = rank(a.ravel()), rank(b.ravel())
    return float(np.corrcoef(ra, rb)[0, 1])


def centroid(m):
    h, w = m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    s = m.sum()
    return float((m * ys).sum() / s / (h - 1)), float((m * xs).sum() / s / (w - 1))


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    ids = sorted(ref)
    meta = r.read_manifest()
    v3 = r.build(320, pretrained=False)
    v3.load_state_dict(torch.load(os.environ["V3_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    v3 = v3.to(dev).eval().to(memory_format=torch.channels_last)
    sp = r.CFG.pop("spatial")                                    # 1e: plain Raptor at output stride 32
    e1 = r.Raptor(320, pretrained=False)
    r.CFG["spatial"] = sp
    e1.load_state_dict(torch.load(os.environ["E1_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    e1 = e1.to(dev).eval().to(memory_format=torch.channels_last)

    truth = {s: ref[s][0] >= 0.7 for s in ids}
    pick = {}
    for n, t in enumerate(r.TARGETS):
        pos = [s for s in ids if truth[s][n] and ref[s][1][n] > 0]
        pick[t] = sorted(pos, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:10]
    want = {s for v in pick.values() for s in v}

    preds = {k: [] for k in ("full", "base_only", "spatial_only", "uniform", "e1")}
    rows, maps = [], {}
    ds = r.Studies(ids, meta, {}, r.CFG["eval_windows"], False)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=8, pin_memory=True)
    for i, (x, sag, _, _, ctx) in enumerate(dl):
        s = ids[i]
        xi = r.prepare(x, sag, False, 320, dev)
        slot = ctx[0, :, 2].long().numpy()                        # [K]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # ---- v3 decomposed
            z = v3.backbone.forward_features(xi)                  # [K, d, 20, 20]
            h = v3.norm(z.mean((2, 3)))[None]                     # [1, K, d]
            a = torch.softmax(v3.att(h).float(), dim=1)           # [1, K, 12]
            base = (torch.einsum("bkn,bkf->bnf", a, h.float()) * v3.clsW.float()).sum(-1)
            zc = z.flatten(2).transpose(1, 2)                     # [K, C, d]
            al = torch.softmax(v3.score(v3.norm_s(zc)).float(), dim=1)   # [K, C, 12]
            p = v3.norm_p(torch.einsum("kct,kcf->ktf", al.to(zc.dtype), zc)).float()
            spat = (torch.einsum("bkt,bktf->btf", a, p[None]) * v3.clsU.float()).sum(-1)
            pu = v3.norm_p(zc.mean(1)).float()[:, None].expand(-1, 12, -1)
            unif = (torch.einsum("bkt,bktf->btf", a, pu[None]) * v3.clsU.float()).sum(-1)
            b = v3.clsb.float()
            for k_, v_ in (("full", base + spat + b), ("base_only", base + b), ("spatial_only", spat + b),
                           ("uniform", base + unif + b)):
                preds[k_].append(v_.sigmoid()[0].cpu().numpy())
            # ---- 1e with per-cell CAM
            z1 = e1.backbone.forward_features(xi).float()         # [K, d, 10, 10]
            m1 = z1.mean((2, 3))                                  # [K, d]
            h1 = e1.norm(m1)[None]
            a1 = torch.softmax(e1.att(h1).float(), dim=1)         # [1, K, 12]
            preds["e1"].append(((torch.einsum("bkn,bkf->bnf", a1, h1.float()) * e1.clsW.float()).sum(-1)
                                + e1.clsb.float()).sigmoid()[0].cpu().numpy())
            mu = m1.mean(1, keepdim=True); sd = (m1.var(1, unbiased=False, keepdim=True) + e1.norm.eps).sqrt()
            g1 = (e1.clsW.float() * e1.norm.weight.float())       # [12, d]
            cam = torch.einsum("nf,kfhw->knhw", g1, (z1 - mu[..., None, None]) / sd[..., None, None])  # [K, 12, 10, 10]
        A, A1 = a[0].cpu().numpy(), a1[0].cpu().numpy()         # [K, 12]
        AL = al.float().cpu().numpy().reshape(len(slot), 20, 20, 12)
        CAM = cam.float().cpu().numpy()
        for n, t in enumerate(r.TARGETS):
            k3, k1 = int(A[:, n].argmax()), int(A1[:, n].argmax())
            am = AL[k3, :, :, n]
            ent = float(-(am * np.log(np.clip(am, 1e-12, None))).sum() / np.log(400))
            row = {"study": s, "target": t, "y": float(ref[s][0][n]), "w": float(ref[s][1][n]),
                   "slot_mass_v3": [float(A[slot == q, n].sum()) for q in range(5)],
                   "slot_mass_e1": [float(A1[slot == q, n].sum()) for q in range(5)],
                   "top_v3": k3, "top_e1": k1, "top_v3_slot": int(slot[k3]), "top_e1_slot": int(slot[k1]),
                   "same_window": k3 == k1, "entropy_top": ent, "centroid_v3": centroid(am)}
            cm = CAM[k1, n]
            row["centroid_e1"] = centroid(np.clip(cm, 0, None) + 1e-9)
            if k3 == k1:
                up = np.kron(cm, np.ones((2, 2)))
                row["spearman"] = spearman(am, up)
                y3, x3 = np.unravel_index(am.argmax(), am.shape)
                y1, x1 = np.unravel_index(up.argmax(), up.shape)
                row["argmax_mm"] = float(np.hypot(y3 - y1, x3 - x1) * CELL_MM)
            rows.append(row)
            if s in pick[t]:
                img3 = x[0, k3, 1].numpy()[32:352, 32:352]         # middle slice of the window, 320 crop
                img1 = x[0, k1, 1].numpy()[32:352, 32:352]
                maps[f"{t}|{s}"] = {"img_v3": img3, "img_e1": img1, "alpha": am, "cam": cm,
                                    "slot_v3": int(slot[k3]), "slot_e1": int(slot[k1]),
                                    "p_v3": float(preds["full"][-1][n]), "p_e1": float(preds["e1"][-1][n])}
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(ids)}", flush=True)

    for k_, v_ in preds.items():
        r.CFG["version"] = f"v3_audit_{k_}"
        r.save_predictions(OUT / f"val_{k_}.json", ids, np.stack(v_), f"v3 vs 1e audit: {k_}")
    (OUT / "xai_rows.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    np.savez_compressed(OUT / "maps.npz", **{f"{key}|{f}": np.asarray(v) for key, d in maps.items()
                                             for f, v in d.items()})
    print(f"AUDIT_DONE {len(ids)} studies, {len(maps)} maps", flush=True)


if __name__ == "__main__":
    main()
