"""KD-GAP, GPU part scored on CPU: the Kaggle kernel tumis001/rsna-kd-gap-xai (scripts/build_kaggle_kd_gap_kernel.py).
X1 view consistency (200 training studies): how far the teacher's OWN output moves from its target view (94 windows,
   320 centre crop, no augmentation) to the views students train on; KL in nats per cell (the students' train-KL floor
   is 0.012 for ra) and Spearman; the teacher-attention mass a 16-window draw covers and whether low cover moves the
   prediction; students ra / E4 on the clean training view vs fold 0 (do they fit training studies any better?).
X2 window attention (fold 0): Jensen-Shannon divergence over the 94 windows, top-1 within +-1 window, top-5 overlap,
   series-slot mass, all studies and Steven-confident positives.
X3 series reliance (label-free): mean |logit change| when a slot is dropped; AUC change on Steven-confident cells
   (teacher in-sample: its AUCs are an upper reference only).
X4 in-plane (Grad-CAM, gradient x activation, 10 x 10 cells of 32 px): on the teacher's top window per finding,
   Spearman of the student's map with the teacher's, top cell within 1 cell, centroid distance (px).
X5 do the differences explain the misses? Spearman, over Steven-confident positives, between the attention JS /
   CAM disagreement and E4's rank shortfall vs the teacher.
Usage: LBP_OUT=data/lb_predictor_v2 python3 -I scripts/audit_kd_gap_xai.py <kernel output dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_predictor as L                                   # noqa: E402

X = Path(sys.argv[1]); OUT = X / "kd_gap_xai.txt"; lines = []
T = L.T; SH = [t[:8] for t in T]; SLOT = ["SagF", "SagNF", "CorF", "CorNF", "Ax"]
BOUNDS = np.cumsum([0, 26, 22, 18, 12, 18]); slot_of = lambda c: np.searchsorted(BOUNDS, c, side="right") - 1
MOD = ["teacher", "v4", "ra", "B", "E4"]


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)


sp1 = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())
H = lambda t: -(t * np.log(np.clip(t, 1e-7, 1)) + (1 - t) * np.log(np.clip(1 - t, 1e-7, 1)))
kl = lambda t, p: (-(t * np.log(np.clip(p, 1e-7, 1)) + (1 - t) * np.log(np.clip(1 - p, 1e-7, 1))) - H(t))
row = lambda name, v, f="7.3f": pr(f"{name:22s} " + " ".join(f"{x:{f}}" for x in v) + f"  | mean {np.nanmean(v):{f}}")
hdr = lambda name: pr(f"{name:22s} " + " ".join(f"{s:>7s}" for s in [t[:7] for t in T]))

# ------------------------------------------------------------------ X1
z2 = np.load(X / "kd_gap_part2.npz")
clean = z2["clean"]; n2 = len(clean)
pr(f"== X1 view consistency: teacher (Raptor 384) on its target view vs student training views, {n2} training studies")
hdr("KL nats/cell")
views = {"16 windows": z2["sub16"], "16 windows + aug": z2["sub16aug"], "94 windows + aug": z2["all94aug"],
         "94 whole field": z2["whole"][:, None]}
for nm, V in views.items():
    row(nm, [np.mean([kl(clean[:, j], V[:, d, j]).mean() for d in range(V.shape[1])]) for j in range(12)], "7.4f")
hdr("Spearman to target view")
for nm, V in views.items():
    row(nm, [np.mean([sp1(clean[:, j], V[:, d, j]) for d in range(V.shape[1])]) for j in range(12)])
row("16 w: mean of 4 draws", [sp1(clean[:, j], views["16 windows"][:, :, j].mean(1)) for j in range(12)])
cov = z2["cover16"]
row("cover: attn mass hit", [cov[:, :, j].mean() for j in range(12)])
row("cover < 0.3 share", [(cov[:, :, j] < .3).mean() for j in range(12)])
dev = np.abs(views["16 windows"] - clean[:, None])
row("rho(cover, -|dp|)", [sp1(cov[:, :, j].ravel(), -dev[:, :, j].ravel()) for j in range(12)])
pos = clean >= .5
pr(f"teacher-positive cells (p >= .5 on the target view): 16-window draw p drops by "
   f"{np.mean([(clean[:, None] - views['16 windows'])[:, :, j][pos[:, j]].mean() for j in range(12) if pos[:, j].any()]):+.3f} "
   f"on average; share of positives flipped below .5: "
   f"{np.mean((views['16 windows'] < .5)[np.repeat(pos[:, None], 4, 1)]):.3f}")
for t in ("ra", "E4"):
    S = z2[f"{t}_clean"]
    pr(f"{t} on the clean training view: Spearman to teacher 384 {np.nanmean([sp1(S[:, j], clean[:, j]) for j in range(12)]):.3f} "
       f"(training studies)")

# ------------------------------------------------------------------ X2
z = np.load(X / "kd_gap_part1.npz"); ids = list(z["ids"]); cs = z["cs"]
pos_ = {s: i for i, s in enumerate(L.IDS)}; o = np.array([pos_[s] for s in ids])
t0, m0 = L.RL["steven"]; t0, m0 = t0[o], m0[o]
P = {m: z[f"{m}_prob"] for m in MOD}
for m in MOD[1:]:
    pr(f"(fold 0 check) {m}: Spearman to teacher 384 {np.nanmean([sp1(P[m][:, j], P['teacher'][:, j]) for j in range(12)]):.3f}")
A = {m: z[f"{m}_att"].astype(np.float64) for m in MOD}
sl = slot_of(cs)


def js(p, q):
    mm = 0.5 * (p + q)
    k_ = lambda a, b: (a * np.log(np.clip(a, 1e-12, 1) / np.clip(b, 1e-12, 1))).sum(1)
    return 0.5 * k_(p, mm) + 0.5 * k_(q, mm)


pr("\n== X2 window attention vs the teacher (fold 0; all studies | Steven-confident positives)")
JS = {}
for m in MOD[1:]:
    JS[m] = np.stack([js(A[m][:, :, j], A["teacher"][:, :, j]) for j in range(12)], 1) / np.log(2)
    pr(f"-- {m}")
    hdr("")
    row("JS (0..1), all", JS[m].mean(0))
    row("JS, positives", [JS[m][m0[:, j] & t0[:, j], j].mean() for j in range(12)])
    tt, tm_ = A["teacher"].argmax(1), A[m].argmax(1)
    row("top-1 within 1 window", [(np.abs(tt[:, j] - tm_[:, j]) <= 1).mean() for j in range(12)])
    top5 = lambda a, j: np.argsort(-a[:, :, j], 1)[:, :5]
    row("top-5 overlap /5", [np.mean([len(set(a) & set(b)) / 5 for a, b in zip(top5(A['teacher'], j), top5(A[m], j))])
                             for j in range(12)])
    row("same slot as teacher top1", [(sl[np.arange(len(ids)), tt[:, j]] == sl[np.arange(len(ids)), tm_[:, j]]).mean()
                                      for j in range(12)])
pr("slot mass (%) of top-1 window, teacher vs E4, per finding:")
for j, t in enumerate(T):
    pr(f"  {t:17s} teacher " + " ".join(f"{SLOT[k]} {100 * (sl[np.arange(len(ids)), A['teacher'].argmax(1)[:, j]] == k).mean():4.0f}"
                                        for k in range(5))
       + " | E4 " + " ".join(f"{100 * (sl[np.arange(len(ids)), A['E4'].argmax(1)[:, j]] == k).mean():4.0f}" for k in range(5)))

# ------------------------------------------------------------------ X3
pr("\n== X3 series reliance: mean |logit change| when the slot is DROPPED (label-free)")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
for j, t in enumerate(T):
    s_ = f"{t:17s}"
    for m in MOD:
        full = lg(P[m][:, j]); d = z[f"{m}_drop"][:, :, j]
        rel = np.nanmean(np.abs(d - full[:, None]), 0)
        s_ += f" | {m[:3]:3s} " + " ".join(f"{v:4.2f}" for v in rel)
    pr(s_)
pr(f"(columns per model: {' '.join(SLOT)})")
pr("AUC change on Steven-confident cells when the slot is dropped, mean over findings (teacher in-sample):")
for m in MOD:
    d = z[f"{m}_drop"]; out = []
    for k in range(5):
        v = []
        for j in range(12):
            sel = m0[:, j] & ~np.isnan(d[:, k, j])
            if (sel & t0[:, j]).sum() >= 5 and (sel & ~t0[:, j]).sum() >= 5:
                v.append(L.auc(d[sel, k, j], t0[sel, j]) - L.auc(P[m][sel, j], t0[sel, j]))
        out.append(np.mean(v))
    pr(f"  {m:8s} " + " ".join(f"{SLOT[k]} {out[k]:+.4f}" for k in range(5)))

# ------------------------------------------------------------------ X4
pr("\n== X4 Grad-CAM on the teacher's top window (10 x 10 cells, 32 px each)")
CT = z["teacher_cam_t"].astype(np.float64)


def cam_stats(Cm):
    rho = np.full(Cm.shape[:2], np.nan); hit = rho.copy(); dist = rho.copy()
    g = np.stack(np.meshgrid(np.arange(10), np.arange(10), indexing="ij"), -1).reshape(100, 2)
    for i in range(Cm.shape[0]):
        for j in range(12):
            a, b = CT[i, j].ravel(), Cm[i, j].ravel()
            if a.sum() <= 0 or b.sum() <= 0:
                continue
            rho[i, j] = sp1(a, b)
            hit[i, j] = np.abs(g[a.argmax()] - g[b.argmax()]).max() <= 1
            ca, cb = (g * a[:, None]).sum(0) / a.sum(), (g * b[:, None]).sum(0) / b.sum()
            dist[i, j] = 32 * np.linalg.norm(ca - cb)
    return rho, hit, dist


CS = {}
for m in MOD[1:]:
    CS[m] = cam_stats(z[f"{m}_cam_t"].astype(np.float64))
    rho, hit, dist = CS[m]
    pr(f"-- {m}")
    hdr("")
    row("CAM Spearman, all", np.nanmean(rho, 0))
    row("CAM Spearman, pos", [np.nanmean(rho[m0[:, j] & t0[:, j], j]) for j in range(12)])
    row("peak within 1 cell", np.nanmean(hit, 0))
    row("centroid dist px", np.nanmean(dist, 0), "7.1f")

# ------------------------------------------------------------------ X5
pr("\n== X5 do attention / CAM differences explain E4's misses? Spearman over Steven-confident positives between the")
pr("   difference and E4's percentile shortfall vs the teacher (positive = bigger difference, bigger miss)")
pct = lambda X_: np.column_stack([pd.Series(X_[:, j]).rank(pct=True).to_numpy() for j in range(12)])
short = pct(P["teacher"]) - pct(P["E4"])
hdr("")
row("attention JS", [sp1(JS["E4"][m0[:, j] & t0[:, j], j], short[m0[:, j] & t0[:, j], j]) for j in range(12)])
row("1 - CAM Spearman", [sp1(1 - CS["E4"][0][m0[:, j] & t0[:, j], j], short[m0[:, j] & t0[:, j], j]) for j in range(12)])
row("E4 att mass on teacher top±1", [sp1(-np.array([A["E4"][i, max(0, k - 1):k + 2, j].sum() for i, k in
                                                    enumerate(A["teacher"].argmax(1)[:, j])])[m0[:, j] & t0[:, j]],
                                         short[m0[:, j] & t0[:, j], j]) for j in range(12)])
OUT.write_text("\n".join(lines) + "\n"); print(f"\nwrote {OUT}")
