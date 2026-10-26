"""KD-VIEW: score the Kaggle sweep (scripts/build_kaggle_kd_view_sweep.py) on fold 0.
Per view (slice draw x augmentation): the teacher's KL / Spearman to its own clean view (how far the training view moves
the target), and the AUC change vs each model's clean view on Steven-confident cells (does the view lose evidence?):
the teacher is all-data (in-sample; memorisation can exaggerate drops), E4 is out of fold (but was trained on the
'r16_full' view, so it is partly invariant to it). Macro AUC change over findings with a paired study bootstrap SE.
Coverage: teacher-attention mass covered by each draw. Grad-CAM baseline: 224 sibling vs 384 on the 384's top window,
next to teacher-student and student-student agreement from KD-GAP part 1.
Usage: LBP_OUT=data/lb_predictor_v2 python3 -I scripts/audit_kd_view_sweep.py <sweep output dir> <kd_gap_xai dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_predictor as L                                   # noqa: E402

X, G = Path(sys.argv[1]), Path(sys.argv[2]); OUT = X / "kd_view_sweep.txt"; lines = []
z = np.load(X / "kd_view_sweep.npz"); ids = list(z["ids"]); n = len(ids)
o = np.array([L.POS[s] for s in ids]); T = L.T


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)


sp1 = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())
H = lambda t: -(t * np.log(np.clip(t, 1e-7, 1)) + (1 - t) * np.log(np.clip(1 - t, 1e-7, 1)))
kl = lambda t, p: (-(t * np.log(np.clip(p, 1e-7, 1)) + (1 - t) * np.log(np.clip(1 - p, 1e-7, 1))) - H(t)).mean()
views = [k[2:] for k in z.files if k.startswith("T_") and k != "T_clean"]
rng = np.random.default_rng(0); BOOT = [rng.integers(0, n, n) for _ in range(200)]


def macro(p, ruler, idx=None):
    t, m = L.RL[ruler]; t, m = t[o], m[o]
    if idx is not None:
        p, t, m = p[idx], t[idx], m[idx]
    v = [L.auc(p[m[:, j], j], t[m[:, j], j]) for j in range(12) if t[m[:, j], j].sum() >= 3 and (~t[m[:, j], j]).sum() >= 3]
    return np.mean(v)


def dauc(m_, v, ruler):
    d = macro(z[f"{m_}_{v}"], ruler) - macro(z[f"{m_}_clean"], ruler)
    se = np.std([macro(z[f"{m_}_{v}"], ruler, b) - macro(z[f"{m_}_clean"], ruler, b) for b in BOOT])
    return d, se


pr(f"== KD-VIEW sweep, fold 0, {n} studies")
pr(f"clean macro AUC (Steven-confident): teacher {macro(z['T_clean'], 'steven'):.4f} (in-sample)  "
   f"E4 {macro(z['S_clean'], 'steven'):.4f}  224 sibling {macro(z['T224_clean'], 'steven'):.4f}")
pr(f"{'view':12s} {'T KL':>7s} {'T rho':>6s} | {'T dAUC st':>10s} {'±':>6s} | {'E4 dAUC st':>10s} {'±':>6s} | "
   f"{'E4 dAUC bothc':>13s}")
for v in views:
    tk = np.mean([kl(z["T_clean"][:, j], z[f"T_{v}"][:, j]) for j in range(12)])
    tr = np.mean([sp1(z["T_clean"][:, j], z[f"T_{v}"][:, j]) for j in range(12)])
    td, ts = dauc("T", v, "steven"); sd, ss = dauc("S", v, "steven"); bd, _ = dauc("S", v, "bothc")
    pr(f"{v:12s} {tk:7.4f} {tr:6.3f} | {td:+10.4f} {ts:6.4f} | {sd:+10.4f} {ss:6.4f} | {bd:+13.4f}")
pr("\nteacher KL to clean per finding:")
pr(f"{'view':12s} " + " ".join(f"{t[:7]:>7s}" for t in T))
for v in views:
    pr(f"{v:12s} " + " ".join(f"{kl(z['T_clean'][:, j], z[f'T_{v}'][:, j]):7.4f}" for j in range(12)))
pr("\nE4 AUC change vs E4 clean per finding (Steven-confident):")
t_, m_ = L.RL["steven"]; t_, m_ = t_[o], m_[o]
fa = lambda p, j: L.auc(p[m_[:, j], j], t_[m_[:, j], j]) if t_[m_[:, j], j].sum() >= 3 else np.nan
pr(f"{'view':12s} " + " ".join(f"{t[:7]:>7s}" for t in T))
for v in views:
    pr(f"{v:12s} " + " ".join(f"{fa(z[f'S_{v}'], j) - fa(z['S_clean'], j):+7.3f}" for j in range(12)))
pr("\nteacher attention mass covered by the draw (windows within 1 slice):")
pr(f"{'draw':12s} " + " ".join(f"{t[:7]:>7s}" for t in T) + "    mean  <0.3 share")
for c in [k for k in z.files if k.startswith("cov_")]:
    C = z[c]
    pr(f"{c[4:]:12s} " + " ".join(f"{C[:, j].mean():7.3f}" for j in range(12)) + f"  {C.mean():6.3f}  {(C < .3).mean():6.3f}")

pr("\n== Grad-CAM baseline on the 384 teacher's top window (Spearman of 10 x 10 maps)")
g = np.load(G / "kd_gap_part1.npz"); gi = {s: i for i, s in enumerate(g["ids"])}; gidx = np.array([gi[s] for s in ids])
same_top = (g["teacher_top"][gidx] == z["top"]).mean()
pr(f"384 top window identical to KD-GAP part 1: {same_top:.3f}")


def agree(A, B):
    r = [sp1(a.ravel(), b.ravel()) for Ai, Bi in zip(A, B) for a, b in zip(Ai, Bi) if a.sum() > 0 and b.sum() > 0]
    return np.mean(r)


C384 = g["teacher_cam_t"][gidx].astype(np.float64)
pr(f"224 sibling vs 384 (same family, different resolution): {agree(z['cam224'].astype(np.float64), C384):.3f}")
for m in ("v4", "ra", "E4"):
    pr(f"{m} vs 384: {agree(g[f'{m}_cam_t'][gidx].astype(np.float64), C384):.3f}")
pr(f"ra vs E4 (student-student): {agree(g['ra_cam_t'][gidx].astype(np.float64), g['E4_cam_t'][gidx].astype(np.float64)):.3f}")
pr(f"224 vs 384 prediction Spearman (fold 0): {np.mean([sp1(z['T224_clean'][:, j], z['T_clean'][:, j]) for j in range(12)]):.3f}")
OUT.write_text("\n".join(lines) + "\n"); print(f"\nwrote {OUT}")
