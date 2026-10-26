"""Score + render the v4 l4copy XAI / ablation outputs (scripts/l4copy_xai.py; Kaggle kernel rsna-v4-l4copy-xai-fold0).
Judge: label-v2 confident cells (weight > 0, p <= .3 or >= .7), per-finding AUC, 1,000 paired study bootstraps.
  A series ablation   global AUC with each series slot removed / alone, vs the full global head
  B attention         attention mass per finding per slot (all studies; positives vs negatives); MS1-free vs LB model
  C crop ablation     fused AUC (g + l) / 2 with self crops x 0.67 / 1.0 / 1.5, top-1 only, MS1's table; global-only
  D panels            PNG per case: top-1 global window + its gradient x activation map, 94-window attention strip
                      coloured by slot, self and MS1 top-1 crops + the crop branch's Grad-CAM, logits
Usage: python3 -I scripts/audit_l4copy_xai.py <data/coatnet_d96_fold0> <xai output dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

C, X = Path(sys.argv[1]), Path(sys.argv[2])
lv = json.load(open(C / "assets/soft_labels_final_per_target_latD2.json")); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
SLOT = ["SagF", "SagNF", "CorF", "CorNF", "Ax"]
CF = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]
SH = [t[:7] for t in T]
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


tags = sorted(p.stem.split("xai_arrays_")[1] for p in X.glob("xai_arrays_*.npz"))
Z = {t: np.load(X / f"xai_arrays_{t}.npz") for t in tags}
ids = list(Z[tags[0]]["ids"]); assert all(list(Z[t]["ids"]) == ids for t in tags)
Y = np.array([lab[s][0] for s in ids]); W = np.array([lab[s][1] for s in ids])
M = (W > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7
rng = np.random.default_rng(2026); B = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
def per(S, i=None):
    """per-finding AUC; S rows aligned with ids, or with ids[i] when i is given."""
    m, t = (M, TT) if i is None else (M[i], TT[i])
    if i is not None:
        S = S[i]
    return np.array([auc(S[m[:, k], k], t[m[:, k], k]) for k in range(12)])
sig = lambda z: 1 / (1 + np.exp(-z))


def fused(z, loc):
    if loc not in z.files:
        return None
    f = z["glob"].copy(); f[:, CF] = 0.5 * (z["glob"][:, CF] + z[loc][:, CF]); return f


def ci(a, b):
    d = np.array([np.nanmean(per(b, i)) - np.nanmean(per(a, i)) for i in B]); return d.mean(), *np.percentile(d, [2.5, 97.5]), (d > 0).mean()


pr(f"studies {len(ids)}; models {tags}")
ref = {"s2026": C / "l4copy_diag/modal_v1/val_global.json"}
for t in tags:
    pr(f"\n#################### model {t} ####################")
    z = Z[t]; g = z["glob"]; pg = per(g)
    if t in ref and len(ids) > 100:
        d = json.load(open(ref[t])); m = {x["study_id"]: np.array(x["probabilities"]) for x in d["predictions"]}
        old = np.array([m[s] for s in ids]); pr(f"check: global AUC this run {pg.mean():.4f} vs cache-path diag {per(old).mean():.4f}; "
                                                f"max |dp| {np.abs(sig(g) - old).max():.3f}, median {np.median(np.abs(sig(g) - old)):.4f}")
    fs = fused(z, "loc_self_100"); pf = per(fs)
    pr(f"global {pg.mean():.4f}  fused(self) {pf.mean():.4f}")
    pr("\n=== A series ablation: global AUC change when a slot is removed (drop) / used alone (only) ===")
    pr(f"{'':10s} " + " ".join(f"{s:>7s}" for s in SH))
    pr(f"{'global':10s} " + " ".join(f"{v:7.4f}" for v in pg))
    for j in range(5):
        dr = z["drop"][:, j]; ok = ~np.isnan(dr).any(1)
        dd = np.where(ok[:, None], dr, g)                 # studies without the slot: unchanged
        pr(f"drop {SLOT[j]:5s} " + " ".join(f"{v:+7.4f}" for v in per(dd) - pg))
    for j in range(5):
        on = z["only"][:, j]; ok = ~np.isnan(on).any(1)
        if ok.mean() < 0.9:
            pr(f"only {SLOT[j]:5s} (present in {100 * ok.mean():.0f}% of studies; scored on those)")
        ii = np.where(ok)[0]
        pr(f"only {SLOT[j]:5s} " + " ".join(f"{v:+7.4f}" for v in per(on, ii) - per(g, ii)))
    pr("\n=== B attention mass per slot (mean over studies; pos = confident positives) ===")
    ms = z["mass"]                                           # [n, 12, 5]
    for k in range(12):
        pos = M[:, k] & TT[:, k]; neg = M[:, k] & ~TT[:, k]
        pr(f"{T[k]:16s} all " + " ".join(f"{SLOT[j]} {100 * ms[:, k, j].mean():4.0f}%" for j in range(5)) +
           " | pos " + " ".join(f"{100 * ms[pos, k, j].mean():3.0f}" for j in range(5)) + " | neg " + " ".join(f"{100 * ms[neg, k, j].mean():3.0f}" for j in range(5)))
    pr("\n=== C crop ablation: fused AUC vs fused(self x1.0), and global-only ===")
    rows = [("glob only", g), ("self x0.67", fused(z, "loc_self_067")), ("self x1.5", fused(z, "loc_self_150")),
            ("self top1", fused(z, "loc_self_top1")), ("ms1 x1.0", fused(z, "loc_ms1_100")), ("self peak", fused(z, "loc_selfpeak_100"))]
    rows = [(nm, S) for nm, S in rows if S is not None]
    pr(f"{'fused self':10s} " + " ".join(f"{v:7.4f}" for v in pf) + f"  mean {pf.mean():.4f}")
    for nm, S in rows:
        p_ = per(S); d = ci(fs, S)
        pr(f"{nm:10s} " + " ".join(f"{v:+7.4f}" for v in p_ - pf) + f"  mean {p_.mean() - pf.mean():+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] P>0 {100 * d[3]:.0f}%")
    pr("local-only AUC: " + " | ".join(f"{nm} " + " ".join(f"{SH[k]} {v:.3f}" for k, v in zip(CF, per(z[key])[CF]))
                                       for nm, key in (("self", "loc_self_100"), ("x1.5", "loc_self_150"), ("ms1", "loc_ms1_100"))))

if "r1self" in tags and "s2026" in tags:
    pr("\n=== MS1-free (r1self) vs LB model (s2026): attention mass on coronal (CorF + CorNF) and axial, all studies ===")
    for k in range(12):
        a_, b_ = Z["s2026"]["mass"][:, k], Z["r1self"]["mass"][:, k]
        pr(f"{T[k]:16s} coronal {100 * a_[:, 2:4].sum(1).mean():4.0f}% -> {100 * b_[:, 2:4].sum(1).mean():4.0f}% | axial {100 * a_[:, 4].mean():4.0f}% -> {100 * b_[:, 4].mean():4.0f}%")

(X / "audit_xai.txt").write_text("\n".join(L) + "\n")

# ------------------------------------------------------------------ panels
COLS = [(230, 60, 60), (240, 160, 60), (60, 200, 90), (60, 160, 230), (190, 90, 230)]


def heat(img, cam, alpha=0.45):
    im = Image.fromarray(img).convert("RGB")
    c = cam / cam.max() if cam.max() > 0 else cam
    hm = Image.fromarray((c * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)
    v = np.asarray(hm, np.float32) / 255
    rgb = np.stack([np.clip(1.5 - np.abs(4 * v - 3), 0, 1), np.clip(1.5 - np.abs(4 * v - 2), 0, 1), np.clip(1.5 - np.abs(4 * v - 1), 0, 1)], -1)
    out = (1 - alpha * v[..., None]) * np.asarray(im, np.float32) + alpha * v[..., None] * 255 * rgb
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


for t in tags:
    P = np.load(X / f"xai_panels_{t}.npz"); keys = sorted({k.rsplit("|", 1)[0] for k in P.files})
    od = X / f"panels_{t}"; od.mkdir(exist_ok=True)
    for key in keys:
        f, kind, sid = key.split("|"); fi = T.index(f)
        s_full = next(s for s in ids if s.endswith(sid)); y = lab[s_full][0][fi]
        sheet = Image.new("RGB", (3 * 384, 384 + 70), (0, 0, 0)); d = ImageDraw.Draw(sheet)
        wimg = P[f"{key}|win_img"]; wi = P[f"{key}|win_info"]
        sheet.paste(heat(wimg, P[f"{key}|win_cam"]).resize((384, 384)), (0, 0))
        d.text((4, 4), f"global top-1 window: slice {int(wi[0])} {SLOT[int(wi[1])]} att {wi[2]:.2f}", fill=(255, 255, 0))
        for c, nm in enumerate(("self", "ms1"), 1):
            if f"{key}|{nm}_crop" in P.files:
                inf = P[f"{key}|{nm}_info"]
                sheet.paste(heat(P[f"{key}|{nm}_crop"], P[f"{key}|{nm}_cam"]).resize((384, 384)), (c * 384, 0))
                d.text((c * 384 + 4, 4), f"{nm} crop: slice {int(inf[2])} {SLOT[int(inf[5])]} box {int(inf[1])} logit {inf[0]:+.2f}", fill=(255, 255, 0))
        att, sl = P[f"{key}|att"], P[f"{key}|att_slot"]; n = len(att); w = 3 * 384 / n; mx = att.max()
        for i in range(n):
            hgt = int(40 * att[i] / mx) if mx > 0 else 0
            d.rectangle([i * w, 384 + 45 - hgt, (i + 1) * w - 1, 384 + 45], fill=COLS[int(sl[i])])
        pdx = P[f"{key}|pred"]
        d.text((4, 384 + 50), f"{f} {kind.upper()}  label {y:.2f}  logits: global {pdx[0]:+.2f}  local self {pdx[1]:+.2f}  local ms1 {pdx[2]:+.2f}   "
               f"attention: " + " ".join(f"{SLOT[j]}" for j in range(5)) + " (red/orange/green/blue/purple)", fill=(255, 255, 255))
        sheet.save(od / f"{f.replace(' ', '').replace(chr(39), '')}_{kind}_{sid}.png")
    print(t, len(keys), "panels ->", od)
