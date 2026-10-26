"""Exp r50-GL3-XAI scorer (CPU): v3 XAI / ablations vs v2's XAI, and the pre-registered ceiling estimates.
Judge: label-v2 confident cells (weight > 0, y <= 0.3 / >= 0.7), fold 0; paired study bootstrap (2,000).
Inputs: run_r50_gl3/xai/xai3_{val,train}.npz (scripts/xai_r50_gl3_gpu.py), run_r50_gl_v2/xai/xai_{val,train}.npz,
run_r50_gl3/val_swa_*.json, kaggle_fold0_labelv2/val_swa.json (label-v2 r50, the best single).
Ceilings (cross-fitted inside fold 0, 5 study folds, seed 2026; each held-out fold scored by a blend chosen on the
other 4): (a) v3 global + v3 local, per-finding constant beta (logit space, grid 0..1 by 0.05); (b) v3 per-study
logistic gate on [g, l, |g - l|] (L2 logistic regression per finding); (c) v3 global + v2 local, per-finding constant
beta = optimistic retrain projection (gate fixed + local at v2's level).
RULE (pre-registered): leave the global + local family if (c) < 0.9455 or (c) beats label-v2 r50 in < 90% of bootstraps.
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_r50_gl3_xai.py <project root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); C = R / "repository/data/coatnet_d96_fold0"
X3, X2 = C / "run_r50_gl3/xai", C / "run_r50_gl_v2/xai"
OUT = X3 / "audit_xai3.txt"; lines = []
say = lambda s="": (print(s, flush=True), lines.append(s))
lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text()); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
R8 = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "Baker's", "Contusion"]
RF = [T.index(t) for t in R8]; KB = R8.index("Baker's")
v3, v3t = np.load(X3 / "xai3_val.npz"), np.load(X3 / "xai3_train.npz")
v2, v2t = np.load(X2 / "xai_val.npz"), np.load(X2 / "xai_train.npz")
ids = [str(s) for s in v3["ids"]]; assert ids == [str(s) for s in v2["ids"]]
idt = [str(s) for s in v3t["ids"]]; assert idt == [str(s) for s in v2t["ids"]]


def load(p):
    d = json.loads(p.read_text()); assert d["targets_in_order"] == T
    m = {x["study_id"]: np.array(x["probabilities"]) for x in d["predictions"]}
    return np.array([m[s] for s in ids])


def YM(ii):
    Y = np.array([lab[s][0] for s in ii]); W = np.array([lab[s][1] for s in ii])
    return (W > 0) & ((Y <= .3) | (Y >= .7)), Y >= .7


M, TT = YM(ids); Mt, TTt = YM(idt)


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(x, m=M, tt=TT, idx=None):
    idx = np.arange(len(x)) if idx is None else idx
    return np.array([auc(x[idx, k][m[idx, k]], tt[idx, k][m[idx, k]]) for k in range(x.shape[1])])


lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / np.clip(1 - q, 1e-6, 1))
sg = lambda z: 1 / (1 + np.exp(-z))


def full12(base, reg, fill=None):
    """12-column predictions: base (global) with the 8 region columns replaced; NaN cells (shuffle: study 0) from fill"""
    o = base.copy(); o[:, RF] = reg if fill is None else np.where(np.isnan(reg), fill[:, RF], reg); return o


# ---- 0 reproduction
say("r50-GL3-XAI: v3 SWA vs v2 SWA, fold 0 (825) + 825 training studies; label-v2 confident cells")
for h in ("fused", "global", "local"):
    ref = load(C / f"run_r50_gl3/val_swa_{h}.json")
    say(f"  reproduction {h:6s}: max |dAUC| {np.nanmax(np.abs(per(v3[h]) - per(ref))):.4f}, max |dp| {np.abs(v3[h] - ref).max():.4f}")
lv2 = load(C / "kaggle_fold0_labelv2/val_swa.json")
G3, L3, F3, G2, L2, F2 = v3["global"], v3["local"], v3["fused"], v2["global"], v2["local"], v2["fused"]
HG3 = sg((lg(G3) + lg(L3)) / 2)

# ---- 1 AUC table
P = {"label-v2 r50": lv2, "v2 global": G2, "v2 local": L2, "v2 fused": F2, "v2 mean(g,l)": sg((lg(G2) + lg(L2)) / 2),
     "v3 global": G3, "v3 local": L3, "v3 fused": F3, "v3 mean(g,l)": HG3}
for k in ("local_uniform", "local_top1", "local_shuffle", "local_gbn", "local_shift16", "local_shift32", "local_shift48"):
    P[f"v3 {k}"] = full12(L3, v3[k], L3)
for k in ("fused_shuffle", "fused_gbn", "fused_shift16", "fused_shift32", "fused_shift48"):
    P[f"v3 {k}"] = full12(F3, v3[k], F3)
for k in ("local_top1", "local_shift32", "local_shift64", "fused_shuffle"):
    P[f"v2 {k}"] = v2[k]
ok = np.isfinite(v3["fused_shuffle"]).all(1)                                     # study 0 has no previous study
tab = pd.DataFrame({k: per(x) for k, x in P.items()}, index=T).T
tab["mean8"] = tab[R8].mean(1); tab["mean12"] = tab[T].mean(1)
say("\n1. AUC per finding (fold 0); shuffle rows: other study's crops (study 0 kept as unshuffled)")
say(tab[R8 + ["mean8", "mean12"]].round(4).to_string())
bak = per(full12(G3, np.where(np.isnan(v3["local_bakers_axial"]), L3[:, RF], v3["local_bakers_axial"])))[T.index("Baker's")]
say(f"  Baker's local: sagittal regions (as trained) {tab.at['v3 local', 'Baker' + chr(39) + 's']:.4f}, MS1 axial regions "
    f"(not trained on) {bak:.4f}; v2 local (axial, trained) {tab.at['v2 local', 'Baker' + chr(39) + 's']:.4f}")

# ---- 2 gate and in-sample bias
say("\n2. Gate: in-sample (825 training studies) vs held-out (fold 0)")
pt = {"global": v3t["global"], "local": v3t["local"], "fused": v3t["fused"], "mean(g,l)": sg((lg(v3t["global"]) + lg(v3t["local"])) / 2)}
pv = {"global": G3, "local": L3, "fused": F3, "mean(g,l)": HG3}
rows = []
for h in pt:
    a_t, a_v = per(pt[h], Mt, TTt)[RF].mean(), per(pv[h])[RF].mean()
    rows.append({"head (8 region findings)": f"v3 {h}", "training": a_t, "fold 0": a_v, "gap": a_t - a_v})
for h in ("global", "local", "fused"):
    a_t, a_v = per(v2t[h], Mt, TTt)[RF].mean(), per(v2[h])[RF].mean()
    rows.append({"head (8 region findings)": f"v2 {h}", "training": a_t, "fold 0": a_v, "gap": a_t - a_v})
say(pd.DataFrame(rows).set_index("head (8 region findings)").round(4).to_string())
bt_, bv_ = v3t["beta"], v3["beta"]
say("  gate beta (weight on global) mean train / fold 0 per finding: " + ", ".join(
    f"{t} {bt_[:, j].mean():.2f}/{bv_[:, j].mean():.2f}" for j, t in enumerate(R8)))
gl_t = per(v3t["local"], Mt, TTt)[RF] - per(v3t["global"], Mt, TTt)[RF]; gl_v = per(L3)[RF] - per(G3)[RF]
say("  local - global per finding, training / fold 0: " + ", ".join(f"{t} {a:+.3f}/{b:+.3f}" for t, a, b in zip(R8, gl_t, gl_v)))

# ---- 3 redundancy
say("\n3. Redundancy: corr(global logit, local logit) per finding, fold 0; linear CKA(z_g, z_l) v3")


def cka(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return float(np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


zg, zl = v3["zg"].astype(np.float32), v3["zl"].astype(np.float32)
rr = []
for j, t in enumerate(R8):
    k = RF[j]
    rr.append({"finding": t, "v3 corr": np.corrcoef(lg(G3[:, k]), lg(L3[:, k]))[0, 1], "v2 corr": np.corrcoef(lg(G2[:, k]), lg(L2[:, k]))[0, 1],
               "v3 corr(global, label-v2 r50)": np.corrcoef(lg(G3[:, k]), lg(lv2[:, k]))[0, 1], "v3 CKA(z_g, z_l)": cka(zg[:, j], zl[:, j])})
say(pd.DataFrame(rr).set_index("finding").round(3).to_string())

# ---- 4 Grad-CAM v3 vs v2
say("\n4. Grad-CAM, global head: share on region windows (chance), inside boxes, box share within those windows (area), "
    "top window is a region window; positives | negatives (confident cells)")
c3, c2 = v3["cam_g"], v2["cam_gl"]
for j, t in enumerate(R8):
    k = RF[j]; pos, neg = M[:, k] & TT[:, k], M[:, k] & ~TT[:, k]
    f = lambda c, s: np.nanmean(c[s], 0)
    for nm, c in (("v3", c3[:, j]), ("v2", c2[:, k])):
        p_, n_ = f(c, pos), f(c, neg)
        say(f"  {t:16s} {nm} pos: win {p_[0]:.2f} ({p_[3]:.2f}) box {p_[1]:.3f} in-win {p_[2]:.2f} ({p_[4]:.2f}) top {p_[5]:.2f} | "
            f"neg: win {n_[0]:.2f} box {n_[1]:.3f} in-win {n_[2]:.2f} top {n_[5]:.2f}")
say("  local head CAM: outer-ring share (area 0.556) / centroid offset px, v3 vs v2")
for j, t in enumerate(R8):
    a3, a2 = np.nanmean(v3["cam_l"][:, j], 0), np.nanmean(v2["cam_local"][:, RF[j]], 0)
    say(f"  {t:16s} v3 {a3[0]:.3f} / {a3[1]:5.1f}px   v2 {a2[0]:.3f} / {a2[1]:5.1f}px")
ent = v3["att_entropy"].mean(0)
say("  v3 window-attention entropy (log 94 = 4.54): " + ", ".join(f"{t} {ent[k]:.2f}" for t, k in zip(T, range(12))))
acm = v3["ac"].max(2).mean(0)
say("  v3 region attention, mean max weight over 4 regions (0.25 = uniform): " + ", ".join(f"{t} {a:.2f}" for t, a in zip(R8, acm)))

# ---- 5 ceilings, cross-fitted inside fold 0
say("\n5. Ceilings (cross-fitted in fold 0, 5 study folds): best achievable from these heads without fold-0 selection")
rng = np.random.default_rng(2026); fold = rng.permutation(len(ids)) % 5
GRID = np.round(np.arange(0, 1.0001, .05), 2)


def const_blend(gp, lp):
    """per region finding, beta chosen on 4 folds (mean AUC), applied to the 5th; returns 12-col predictions, betas"""
    out = gp.copy(); chosen = np.zeros((5, len(RF)))
    for f in range(5):
        tr, te = fold != f, fold == f
        for j, k in enumerate(RF):
            sc = [auc(sg(b * lg(gp[tr, k]) + (1 - b) * lg(lp[tr, k]))[M[tr, k]], TT[tr, k][M[tr, k]]) for b in GRID]
            b = GRID[int(np.nanargmax(sc))]; chosen[f, j] = b
            out[te, k] = sg(b * lg(gp[te, k]) + (1 - b) * lg(lp[te, k]))
    return out, chosen.mean(0)


def logreg(Xa, y, lam=1.0, it=50):
    w = np.zeros(Xa.shape[1])
    for _ in range(it):
        p = sg(Xa @ w); H = Xa.T @ (Xa * (p * (1 - p))[:, None]) + lam * np.eye(len(w)); w -= np.linalg.solve(H, Xa.T @ (p - y) + lam * w)
    return w


def stack_gate(gp, lp):
    out = gp.copy()
    for f in range(5):
        tr, te = fold != f, fold == f
        for k in RF:
            fe = lambda i: np.stack([lg(gp[i, k]), lg(lp[i, k]), np.abs(lg(gp[i, k]) - lg(lp[i, k])), np.ones(i.sum())], 1)
            m = M[tr, k]; w = logreg(fe(tr)[m], TT[tr, k][m].astype(float))
            out[te, k] = sg(fe(te) @ w)
    return out


CE = {}
CE["(a) v3 g + v3 l, constant beta"], ba = const_blend(G3, L3)
CE["(b) v3 per-study logistic gate"] = stack_gate(G3, L3)
CE["(c) v3 g + v2 l, constant beta"], bc = const_blend(G3, L2)
CE["    v2 g + v2 l, constant beta"], _ = const_blend(G2, L2)
for k, x in CE.items():
    say(f"  {k:34s} 12-target {np.nanmean(per(x)):.4f}  (mean8 {per(x)[RF].mean():.4f})")
say("  chosen beta (weight on global) (a): " + ", ".join(f"{t} {b:.2f}" for t, b in zip(R8, ba)))
say("  chosen beta (weight on global) (c): " + ", ".join(f"{t} {b:.2f}" for t, b in zip(R8, bc)))
B = 2000; rngb = np.random.default_rng(2026)
pairs = {"(c) - label-v2 r50": (CE["(c) v3 g + v2 l, constant beta"], lv2), "(a) - label-v2 r50": (CE["(a) v3 g + v3 l, constant beta"], lv2),
         "(b) - label-v2 r50": (CE["(b) v3 per-study logistic gate"], lv2), "v3 mean(g,l) - v3 fused": (HG3, F3),
         "v3 local_shuffle - v3 local (8)": (P["v3 local_shuffle"], L3), "v3 local_gbn - v3 local (8)": (P["v3 local_gbn"], L3)}
bt = {k: [] for k in pairs}
for _ in range(B):
    i = rngb.integers(0, len(ids), len(ids)); i = i[ok[i]]
    for k, (a, b) in pairs.items():
        cols = RF if "(8)" in k else list(range(12))
        bt[k].append(np.nanmean(per(a, idx=i)[cols]) - np.nanmean(per(b, idx=i)[cols]))
say(""); io = np.where(ok)[0]
for k, (a, b) in pairs.items():
    cols = RF if "(8)" in k else list(range(12)); lo, hi = np.percentile(bt[k], [2.5, 97.5])
    say(f"  {k:32s} {np.nanmean(per(a, idx=io)[cols]) - np.nanmean(per(b, idx=io)[cols]):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}], "
        f"{np.mean(np.array(bt[k]) > 0):.0%} > 0")
cval = np.nanmean(per(CE["(c) v3 g + v2 l, constant beta"])); share = np.mean(np.array(bt["(c) - label-v2 r50"]) > 0)
leave = cval < .9455 or share < .9
say(f"\nRULE: (c) {cval:.4f} vs 0.9455, beats label-v2 r50 in {share:.0%} of bootstraps -> "
    f"{'LEAVE the global + local family' if leave else 'a fix can clear the bar: next run = the fix the ablations point to'}")
OUT.write_text("\n".join(lines) + "\n")
