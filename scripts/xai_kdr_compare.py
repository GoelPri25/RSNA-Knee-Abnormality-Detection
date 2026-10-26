"""PUB-KD-R xAI comparison on fold 0: v4 l4copy s2026 vs the Raptor-target students ra (pure) / rb (mix73), plus the
Raptor 384 teacher's window attention (Kaggle kernel rsna-pubkd-r-xai-fold0, scripts/build_kaggle_xai_kdr_kernel.py).
Judge: Steven-v4 fold-0 confident cells (clear6 = moderate+ positives, 12t = all 12 findings; audit_clear_band_lb.py),
1,000 paired study bootstraps. Descriptive: nine past "where to look" interventions were null, so every difference here
is a hypothesis, not a recipe. Teacher numbers are in-sample (it trained on all data).
Usage: python3 -I scripts/xai_kdr_compare.py <project root> <xai dir> <out stem>   (writes <stem>.txt, <stem>.json and
<xai dir>/branch_preds/*.json for scripts/lb_predictor.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, X, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
g = {"__name__": "x"}; argv = sys.argv; sys.argv = [argv[0], str(R)]
exec((R / "repository/scripts/audit_clear_band_lb.py").read_text().split("run = lambda")[0], g); sys.argv = argv
ids, T, scores, auc, mask, truth, clear, CB = (g[k] for k in ("ids", "T", "scores", "auc", "mask", "truth", "clear", "CB"))
SLOT = ["SagF", "SagNF", "CorF", "CorNF", "Ax"]
BOUNDS = np.cumsum([0, 26, 22, 18, 12, 18])
slot_of = lambda s: np.searchsorted(BOUNDS, s, side="right") - 1
CF = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]
SH = [t[:7] for t in T]
TAGS = ["s2026", "ra", "rb"]
L, J = [], {}


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


Z = {}
for t in TAGS:
    z = np.load(X / f"xai_arrays_{t}.npz"); pos = {s: i for i, s in enumerate(z["ids"])}; o = np.array([pos[s] for s in ids])
    Z[t] = {k: z[k][o] for k in z.files if k != "ids"}
tz = np.load(X / "xai_teacher_raptor384.npz"); pos = {s: i for i, s in enumerate(tz["ids"])}; o = np.array([pos[s] for s in ids])
TCH = {k: tz[k][o] for k in ("prob", "mass", "ent")}
sig = lambda z: 1 / (1 + np.exp(-z))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
sp = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())                 # Spearman (no scipy on the host)
rng = np.random.default_rng(2026); BS = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]


def fused(z, loc="loc_self_100"):
    f = z["glob"].copy(); f[:, CF] = 0.5 * (z["glob"][:, CF] + z[loc][:, CF]); return f


def crop(z, loc="loc_self_100"):
    f = z["glob"].copy() * np.nan; f[:, CF] = z[loc][:, CF]; return f


def pf(S, idx=None, clear_only=False):
    """per-finding AUC on Steven-v4 confident cells (clear_only: positives restricted to report moderate+)."""
    idx = np.arange(len(ids)) if idx is None else idx; out = np.full(12, np.nan)
    for k in range(12):
        if np.isnan(S[:, k]).all():
            continue
        m = mask[idx, k] & ((~truth[idx, k]) | clear[idx, k]) if clear_only else mask[idx, k]
        sel = idx[m]; out[k] = auc(S[sel, k], truth[sel, k])
    return out


def c6(S, idx=None):
    return np.nanmean(pf(S, idx, True)[CB])


def ci(fn, a, b):
    d = np.array([fn(b, i) - fn(a, i) for i in BS]); return fn(b) - fn(a), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


m12 = lambda S, idx=None: np.nanmean(pf(S, idx))
m10 = lambda S, idx=None: np.nanmean(pf(S, idx)[CF])
pr(f"studies {len(ids)}; models {TAGS} + teacher raptor384 (plain view, in-sample); judge Steven-v4 confident cells; clear6 = {[T[k] for k in CB]}")
log = json.load(open(X / "xai_log.json")); pr(f"kernel: {log.get('device')} {log.get('seconds')} s, equivalence checks {log.get('equiv_self_fused_max_abs')}, decode errors {log.get('decode_errors')}")
for t, d in (("s2026", R / "repository/data/coatnet_d96_fold0/run_r50_gl4_mlx/v4_l4copy_s2026"),
             ("ra", R / "repository/data/coatnet_d96_fold0/pubkd/students_r/pubkd_kd_ra"),
             ("rb", R / "repository/data/coatnet_d96_fold0/pubkd/students_r/pubkd_kd_rb")):
    f = d / "val_swa_fused.json"
    if f.exists():
        ref = g["load"](f); pr(f"check {t}: this run fused clear6 {c6(sig(fused(Z[t]))):.4f} vs cache-path {c6(ref):.4f}; "
                               f"max |dp| {np.abs(sig(fused(Z[t])) - ref).max():.3f} median {np.median(np.abs(sig(fused(Z[t])) - ref)):.4f}")

# ---------------------------------------------------------------- 1 branches
pr("\n=== 1 branch AUC (Steven-v4): global / crop (self x1.0) / fused; crop on its 10 findings (crop clear6 = the 5 clear6 findings with a crop, no Effusion) ===")
pr(f"{'':14s} {'clear6':>7s} {'12t':>7s} {'10crop':>7s}  " + " ".join(f"{s:>7s}" for s in SH))
BR = {}
for t in TAGS:
    for nm, S in (("global", Z[t]["glob"]), ("crop", crop(Z[t])), ("fused", fused(Z[t])), ("fused ms1", fused(Z[t], "loc_ms1_100"))):
        BR[(t, nm)] = S; p = pf(S)
        pr(f"{t:5s} {nm:9s} {c6(S):7.4f} {np.nanmean(p):7.4f} {np.nanmean(p[CF]):7.4f}  " + " ".join(f"{v:7.4f}" for v in p))
pr(f"{'teacher':14s} {c6(TCH['prob']):7.4f} {m12(TCH['prob']):7.4f} {m10(TCH['prob']):7.4f}  " + " ".join(f"{v:7.4f}" for v in pf(TCH["prob"])))
pr("\nstudent - v4, same branch (bootstrap 95% CI, P>0):")
J["branch"] = {}
for t in ("ra", "rb"):
    for nm in ("global", "crop", "fused"):
        a, b = BR[("s2026", nm)], BR[(t, nm)]
        r6 = ci(c6, a, b)                                          # crop: clear6 findings that have a crop
        r12 = ci(m12 if nm != "crop" else m10, a, b)
        d = pf(b) - pf(a)
        pr(f"{t:3s} {nm:6s} clear6 {r6[0]:+.4f} [{r6[1]:+.4f}, {r6[2]:+.4f}] {100 * r6[3]:3.0f}% | {'12t' if nm != 'crop' else '10c'} "
           f"{r12[0]:+.4f} [{r12[1]:+.4f}, {r12[2]:+.4f}] {100 * r12[3]:3.0f}%  | " + " ".join(f"{v:+.3f}" if not np.isnan(v) else "    -" for v in d))
        J["branch"][f"{t}_{nm}"] = {"clear6": r6, "m": r12, "per_finding": d.tolist()}
pr("\ncrop's contribution (fused - global) per model, 10 crop findings, Steven 12t AUC:")
for t in TAGS:
    d = pf(BR[(t, "fused")]) - pf(BR[(t, "global")])
    r = ci(m10, BR[(t, "global")], BR[(t, "fused")])
    pr(f"{t:5s} mean {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}] | " + " ".join(f"{SH[k]} {d[k]:+.3f}" for k in CF))
pr("global~crop logit correlation (Spearman, all studies; lower = more complementary):")
for t in TAGS:
    rs = [sp(Z[t]["glob"][:, k], Z[t]["loc_self_100"][:, k]) for k in CF]
    pr(f"{t:5s} mean {np.mean(rs):.3f} | " + " ".join(f"{SH[k]} {r:.2f}" for k, r in zip(CF, rs)))
pr("fidelity to the teacher (Spearman with raptor384 prob, all studies) per branch:")
for t in TAGS:
    for nm in ("global", "crop", "fused"):
        S = BR[(t, nm)]; ks = CF if nm == "crop" else range(12)
        rs = [sp(S[:, k], TCH["prob"][:, k]) for k in ks]
        pr(f"{t:5s} {nm:6s} mean {np.mean(rs):.3f}")
# branch predictions for lb_predictor
bp = X / "branch_preds"; bp.mkdir(exist_ok=True)
for t in TAGS:
    for nm in ("global", "fused", "fused ms1"):
        p = sig(BR[(t, nm)]); fn = bp / f"{t}_{nm.replace(' ', '_')}.json"
        fn.write_text(json.dumps({"targets_in_order": T, "predictions": [{"study_id": s, "probabilities": p[i].tolist()} for i, s in enumerate(ids)]}))

# ---------------------------------------------------------------- 2 attention
pr("\n=== 2 window-attention mass per slot (% , all studies) and on confident positives (pos) ===")
pr("finding          model    " + " ".join(f"{s:>5s}" for s in SLOT) + "  | pos " + " ".join(f"{s:>5s}" for s in SLOT) + " | L1 to teacher (all)")
J["mass"] = {}
MS = {**{t: Z[t]["mass"] for t in TAGS}, "teacher": TCH["mass"]}
for k in range(12):
    p = mask[:, k] & truth[:, k]
    for t in TAGS + ["teacher"]:
        m = MS[t][:, k]; l1 = 0.5 * np.abs(m - MS["teacher"][:, k]).sum(1).mean()
        pr(f"{T[k][:16]:16s} {t:8s} " + " ".join(f"{100 * m[:, j].mean():5.1f}" for j in range(5)) + "  |     " +
           " ".join(f"{100 * m[p, j].mean():5.1f}" for j in range(5)) + (f" | {l1:.3f}" if t != "teacher" else ""))
        J["mass"][f"{T[k]}|{t}"] = m.mean(0).tolist()
pr("\nmean over findings of the per-study L1 distance (0..1) of slot mass to the teacher's:")
for t in TAGS:
    pr(f"{t:5s} " + f"{np.mean([0.5 * np.abs(MS[t][:, k] - MS['teacher'][:, k]).sum(1).mean() for k in range(12)]):.3f}  "
       f"(to v4: {np.mean([0.5 * np.abs(MS[t][:, k] - MS['s2026'][:, k]).sum(1).mean() for k in range(12)]):.3f})")

pr("\n=== 3 attention concentration: normalised window entropy (0 = one window, 1 = uniform over 94) / top-1 weight ===")
pr(f"{'':8s} {'mean':>6s}  " + " ".join(f"{s:>7s}" for s in SH))
ENT = {**{t: Z[t]["att_ent"] for t in TAGS}, "teacher": TCH["ent"]}
for t in TAGS + ["teacher"]:
    pr(f"{t:8s} {ENT[t].mean():6.3f}  " + " ".join(f"{v:7.3f}" for v in ENT[t].mean(0)))
for t in TAGS:
    pr(f"{t + ' top1':8s} {Z[t]['att_top'].mean():6.3f}  " + " ".join(f"{v:7.3f}" for v in Z[t]["att_top"].mean(0)))
pr("entropy on confident positives vs negatives (mean over findings):")
for t in TAGS + ["teacher"]:
    e = ENT[t]; pp = np.nanmean([e[mask[:, k] & truth[:, k], k].mean() for k in range(12)]); nn = np.nanmean([e[mask[:, k] & ~truth[:, k], k].mean() for k in range(12)])
    pr(f"{t:8s} pos {pp:.3f} neg {nn:.3f}")
J["entropy"] = {t: ENT[t].mean(0).tolist() for t in ENT}

# ---------------------------------------------------------------- 4 series ablation
pr("\n=== 4 series ablation (global head): Steven 12t AUC change when a slot is dropped / used alone (mean over findings) ===")
for t in TAGS:
    gl = Z[t]["glob"]; pg = pf(gl); row = []
    for j in range(5):
        dr = Z[t]["drop"][:, j]; ok = ~np.isnan(dr).any(1); dd = np.where(ok[:, None], dr, gl)
        row.append(np.nanmean(pf(dd) - pg))
    row2 = []
    for j in range(5):
        on = Z[t]["only"][:, j]; ok = np.where(~np.isnan(on).any(1))[0]
        row2.append(np.nanmean(pf(on, ok) - pf(gl, ok)))
    pr(f"{t:5s} drop " + " ".join(f"{SLOT[j]} {v:+.4f}" for j, v in enumerate(row)) + " | only " + " ".join(f"{SLOT[j]} {v:+.4f}" for j, v in enumerate(row2)))
pr("per finding, drop of the slot each model leans on most (largest AUC loss):")
for k in range(12):
    s = []
    for t in TAGS:
        gl = Z[t]["glob"]; pg = pf(gl)[k]; dl = []
        for j in range(5):
            dr = Z[t]["drop"][:, j]; ok = ~np.isnan(dr).any(1); dl.append(pf(np.where(ok[:, None], dr, gl))[k] - pg)
        j = int(np.argmin(dl)); s.append(f"{t} {SLOT[j]} {dl[j]:+.3f}")
    pr(f"{T[k][:16]:16s} " + " | ".join(s))

# ---------------------------------------------------------------- 5 crops
pr("\n=== 5 crops: placement (self top-1 vs MS1) and crop-variant AUC (fused, 10 findings, vs fused self x1.0) ===")
J["crops"] = {}
for t in TAGS:
    st, mt = Z[t]["self_tab"], Z[t]["ms1_tab"]                    # [n, 3, 10, TOP]: slice, row, col
    s0 = st[:, 0, :, 0]; same = slot_of(s0)[:, :, None] == slot_of(mt[:, 0])
    near = same & (np.abs(s0[:, :, None] - mt[:, 0]) <= 2)
    st, mt = st.astype(float), mt.astype(float)
    dpx = np.sqrt((st[:, 1, :, 0][:, :, None] - mt[:, 1]) ** 2 + (st[:, 2, :, 0][:, :, None] - mt[:, 2]) ** 2)
    dmin = np.where(near, dpx, np.inf).min(2)
    hit = np.isfinite(dmin)
    pr(f"{t:5s} self top-1 in an MS1 slot {100 * same.any(2).mean():4.1f}% | within 2 slices of an MS1 point {100 * hit.mean():4.1f}% | "
       f"median in-plane px to it {np.median(dmin[hit]):5.1f} | per finding within-2-slices %: " + " ".join(f"{SH[k]} {100 * hit[:, j].mean():3.0f}" for j, k in enumerate(CF)))
    J["crops"][t] = {"ms1_slot": float(same.any(2).mean()), "near": float(hit.mean()), "px": float(np.median(dmin[hit]))}
for t in ("ra", "rb"):
    a, b = Z["s2026"]["self_tab"], Z[t]["self_tab"]
    same = (a[:, 0, :, 0] == b[:, 0, :, 0]); dpx = np.sqrt(((a[:, 1:, :, 0] - b[:, 1:, :, 0]).astype(float) ** 2).sum(1))
    pr(f"{t} vs v4 self top-1: same slice {100 * same.mean():4.1f}%, same slot {100 * (slot_of(a[:, 0, :, 0]) == slot_of(b[:, 0, :, 0])).mean():4.1f}%, "
       f"median px (same slice) {np.median(dpx[same]):.1f}")
for t in TAGS:
    base = fused(Z[t]); pb = pf(base); out = []
    for nm, key in (("x0.67", "loc_self_067"), ("x1.5", "loc_self_150"), ("top1", "loc_self_top1"), ("ms1", "loc_ms1_100")):
        r = ci(m10, base, fused(Z[t], key)); out.append(f"{nm} {r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}]")
    pr(f"{t:5s} " + " | ".join(out))
pr("crop-only AUC with MS1 placements vs self (10 findings): " + " | ".join(
    f"{t} self {m10(crop(Z[t])):.4f} ms1 {m10(crop(Z[t], 'loc_ms1_100')):.4f}" for t in TAGS))

OUT.with_suffix(".txt").write_text("\n".join(L) + "\n")
OUT.with_suffix(".json").write_text(json.dumps(J, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)) + "\n")
print("wrote", OUT.with_suffix(".txt"))
