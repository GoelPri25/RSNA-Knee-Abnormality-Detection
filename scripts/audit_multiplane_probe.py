"""Exp MP-0 (ledger, pre-registered 2026-10-18): does a zoomed view of each finding's anatomy point in the coronal and
axial series (not only the series MS1 picked it on) add signal to the label-v2 r50's whole-study view? CPU probe on the
frozen features of scripts/extract_multiplane_features_gpu.py.

Labels: label-v2 per target, confident cells (weight > 0, y <= 0.3 / >= 0.7), training and scoring alike. Probes per
finding as scripts/audit_ms1_crop_probe.py: each view's feature standardised and reduced to 256 principal components on
the training folds (missing view -> 0 + a missing flag), L2 logistic, lambda in {1e-3, 1e-2, 1e-1} chosen on training
fold 1 held out, refit on all training folds -> fold-0 score.
  s_src    zoom on the source plane only (the plane MS1 picked; r50-GL's local view, frozen)
  s_all    zoom on all 3 planes (3 x 256 components)
  s_slice  the same 3 mapped windows unzoomed (320 centre crop): right slices, no zoom (control)
  A        the label-v2 r50's fold-0 logit (kaggle_fold0_labelv2/val_swa.json, out of fold)
  B_x      2-input logistic [z(A), z(s_x)] cross-fitted inside fold 0 (5 stratified splits x 20 repeats).
Gate findings G5 = Lateral Meniscus, Lateral OA, Baker's, MCL, Contusion (where lv2 + r50-GL local gained).
Q1 (zoom adds to the global view): mean over G5 of B_all - A >= +0.005, bootstrap 95% lower bound > 0.
Q2 (other planes add beyond the same-plane zoom): mean over G5 of B_all - B_src >= +0.003 with >= 90% of bootstraps
    > 0, or per finding >= +0.005 with lower bound > 0 on >= 2 of G5.
Decision: Q2 pass -> v3 regions in all 3 planes (for the findings that gain); Q2 fail -> same-plane regions only.
A frozen probe does not stop the trained design (the encoder never saw zoomed inputs here); it sets the region plan.
Bias: on training studies the parent's encoder is in-sample, so probe weights fit a slightly optimistic feature.

Usage (container): podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_multiplane_probe.py /proj [out.txt]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

R = Path(sys.argv[1]); D = R / "repository/data/coatnet_d96_fold0"; A = D / "assets"
C = D / "run_sig_lv2/crop_probe"; X = C / "multiplane"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else X / "audit_multiplane_probe.txt"
lines = []
say = lambda s="": (print(s, flush=True), lines.append(s))
parts = [np.load(p) for p in sorted(X.glob("mp_*.npz"))]
ids = np.concatenate([p["ids"] for p in parts]).astype(str)
assert len(ids) == 4349 and len(set(ids)) == 4349
pz = np.load(C / "multiplane_points.npz"); assert [str(s) for s in pz["ids"]] == list(ids)
lab = json.loads((A / "soft_labels_final_per_target_latD2.json").read_text()); T = lab["targets_in_order"]
BOXED = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Baker's", "Contusion", "Fracture"]
CF = [T.index(t) for t in BOXED]
SRC = pz["src_plane"][:, CF].astype(int)                                              # [n, 10]
fold = {r["study_id"]: r["fold"] for r in csv.DictReader((A / "folds.csv").open())}
lv = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lab["labels"]}
tr = np.array([fold[s] != "0" for s in ids]); va = ~tr; inner = np.array([fold[s] == "1" for s in ids])
Y = np.array([lv[s][0] for s in ids]); Wt = np.array([lv[s][1] for s in ids])
M = (Wt > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7
G5 = [T.index(t) for t in ("Lateral Meniscus", "Lateral OA", "Baker's", "MCL", "Contusion")]
LAMS, PCS, SPLITS, REPS, BOOT = [1e-3, 1e-2, 1e-1], 256, 5, 20, 2000
rng = np.random.default_rng(2026)


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def fit(X, y, lam):
    """L2 logistic regression (mean loss + lam/2 |w|^2), L-BFGS; returns (w, b)."""
    n, d = X.shape

    def f(p):
        w, b = p[:d], p[d]; z = X @ w + b
        g = (1 / (1 + np.exp(-z)) - y) / n
        return np.mean(np.logaddexp(0, z) - y * z) + lam / 2 * w @ w, np.append(X.T @ g + lam * w, g.sum())
    r = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return r.x[:d], r.x[d]


def pca(X):
    """missing rows -> training mean; standardise + PCA-PCS on the training folds (unit-variance components) + a
    missing flag column"""
    miss = np.isnan(X).any(1); X = X.copy()
    mu = np.nanmean(X[tr & ~miss], 0); X[miss] = mu
    sd = X[tr].std(0) + 1e-6; Z = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Z[tr], full_matrices=False); P = Z @ Vt[:PCS].T
    return np.concatenate([P / (P[tr].std(0) + 1e-6), miss[:, None].astype(float)], 1)


def probe(X, k):
    """train-fold probe for finding k -> scores for all rows"""
    m = M[:, k] & tr
    best = max(LAMS, key=lambda l: auc(X[m & inner] @ fit(X[m & ~inner], TT[m & ~inner, k], l)[0], TT[m & inner, k]))
    w, b = fit(X[m], TT[m, k], best)
    return X @ w + b, best


def crossfit(a, s, k):
    """fold-0 cross-fitted 2-input logistic [z(a), z(s)] -> out-of-split scores (mean over repeats) on fold-0 rows"""
    v = np.where(va)[0]; m = M[v, k]; t = TT[v, k]
    X = np.stack([a[v], s[v]], 1); X = (X - X[m].mean(0)) / (X[m].std(0) + 1e-9)
    out = np.zeros(len(v)); cnt = np.zeros(len(v))
    cells = np.where(m)[0]
    for _ in range(REPS):
        sp = np.zeros(len(v), int)
        for cls in (True, False):                                        # stratified split ids
            c = rng.permutation(cells[t[cells] == cls]); sp[c] = np.arange(len(c)) % SPLITS
        for j in range(SPLITS):
            fitm = m & (sp != j); test = m & (sp == j)
            w, b = fit(X[fitm], t[fitm], 1e-4)
            out[test] += X[test] @ w + b; cnt[test] += 1
    return np.where(cnt > 0, out / np.maximum(cnt, 1), np.nan)


ref = json.loads((D / "kaggle_fold0_labelv2/val_swa.json").read_text())
pv = {p["study_id"]: np.clip(np.array(p["probabilities"]), 1e-6, 1 - 1e-6) for p in ref["predictions"]}
LOGA = np.full((len(ids), 12), np.nan)
for i in np.where(va)[0]:
    LOGA[i] = np.log(pv[ids[i]] / (1 - pv[ids[i]]))
assert not np.isnan(LOGA[va]).any()
say(f"MP-0: {tr.sum()} training-fold studies, {va.sum()} fold 0; label-v2 confident cells; A = label-v2 r50 fold 0 "
    f"{np.nanmean([auc(LOGA[va, k][M[va, k]], TT[va, k][M[va, k]]) for k in range(12)]):.4f} (12-target)")
say("source plane share sag / cor / ax per finding: " + ", ".join(
    f"{t} {np.bincount(SRC[:, j][SRC[:, j] >= 0], minlength=3) / len(ids)}" for j, t in enumerate(BOXED)))

cat = lambda key, j: np.concatenate([p[key][:, j] for p in parts]).astype(np.float32)   # [n, 3(, 2048)]
v = np.where(va)[0]
rows, S = [], {}
for j, t in enumerate(BOXED):
    k = CF[j]
    fz, fs, lz = cat("feat_zoom", j), cat("feat_slice", j), cat("logit_zoom", j)
    src = np.where(SRC[:, j, None, None] == np.arange(3)[None, :, None], fz, 0).sum(1)  # [n, 2048] source plane
    src[SRC[:, j] < 0] = np.nan
    Xz = [pca(fz[:, q]) for q in range(3)]
    s_src, l1 = probe(pca(src), k); s_all, l2 = probe(np.concatenate(Xz, 1), k)
    s_sl, _ = probe(np.concatenate([pca(fs[:, q]) for q in range(3)], 1), k)
    s_pl = [probe(Xz[q], k)[0] for q in range(3)]
    a = LOGA[:, k]
    S[k] = {"A": a[v], "B_src": crossfit(a, s_src, k), "B_all": crossfit(a, s_all, k), "B_slice": crossfit(a, s_sl, k)}
    m, tt = M[v, k], TT[v, k]
    sc = lambda x: auc(x[m], tt[m])
    zs = np.nanmean(lz, 1)[v]; zs = np.where(np.isnan(zs), np.nanmean(zs), zs)
    rows.append({"finding": t, "n+": int((m & tt).sum()), "n-": int((m & ~tt).sum()), "A": sc(a[v]),
                 "B_src": sc(S[k]["B_src"]), "B_all": sc(S[k]["B_all"]), "B_slice": sc(S[k]["B_slice"]),
                 "B_all-A": sc(S[k]["B_all"]) - sc(a[v]), "B_all-B_src": sc(S[k]["B_all"]) - sc(S[k]["B_src"]),
                 "s_src": sc(s_src[v]), "s_all": sc(s_all[v]), "s_sag": sc(s_pl[0][v]), "s_cor": sc(s_pl[1][v]),
                 "s_ax": sc(s_pl[2][v]), "s_slice": sc(s_sl[v]), "zero-shot zoom": sc(zs), "lam src/all": f"{l1:g}/{l2:g}"})
    print("done", t, flush=True)
df = pd.DataFrame(rows).set_index("finding")
say("\nPer finding, fold 0 AUC (label-v2 confident cells). B_* = A + probe, cross-fitted in fold 0; s_* = probe alone")
say(df.round(4).to_string())

pairs = [("B_all", "A"), ("B_src", "A"), ("B_all", "B_src"), ("B_slice", "A"), ("B_all", "B_slice")]
boot = {p: np.full((BOOT, 12), np.nan) for p in pairs}
for b_ in range(BOOT):
    i = rng.integers(0, len(v), len(v))
    for k in CF:
        m = M[v[i], k]; tt = TT[v[i], k][m]
        if 0 < tt.sum() < len(tt):
            sc = {x: auc(S[k][x][i][m], tt) for x in ("A", "B_src", "B_all", "B_slice")}
            for p in pairs:
                boot[p][b_, k] = sc[p[0]] - sc[p[1]]
say("\nPaired study bootstrap (2,000): point [95% CI], share > 0")
res = {}
for p in pairs:
    pt = {k: df.at[T[k], p[0]] - df.at[T[k], p[1]] for k in CF}
    g = np.nanmean(boot[p][:, G5], 1); lo, hi = np.nanpercentile(g, [2.5, 97.5])
    a10 = np.nanmean(boot[p][:, CF], 1); lo10, hi10 = np.nanpercentile(a10, [2.5, 97.5])
    res[p] = (np.mean([pt[k] for k in G5]), lo, hi, float(np.mean(g > 0)), pt)
    say(f"  {p[0]} - {p[1]}: G5 {res[p][0]:+.4f} [{lo:+.4f}, {hi:+.4f}] {np.mean(g > 0):.0%}; all 10 "
        f"{np.mean(list(pt.values())):+.4f} [{lo10:+.4f}, {hi10:+.4f}]")
    say("    " + ", ".join(f"{T[k]} {pt[k]:+.4f} [{np.nanpercentile(boot[p][:, k], 2.5):+.4f}, "
                         f"{np.nanpercentile(boot[p][:, k], 97.5):+.4f}]" for k in CF))
q1 = res[("B_all", "A")]; q2 = res[("B_all", "B_src")]
q2f = [T[k] for k in G5 if q2[4][k] >= .005 and np.nanpercentile(boot[("B_all", "B_src")][:, k], 2.5) > 0]
say(f"\nQ1 zoom adds to the global view (B_all - A, G5 >= +0.005, lower bound > 0): "
    f"{'PASS' if q1[0] >= .005 and q1[1] > 0 else 'FAIL'}")
say(f"Q2 other planes add beyond the same-plane zoom (B_all - B_src, G5 >= +0.003 with >= 90% > 0, or >= 2 findings "
    f"+0.005 with lower bound > 0 [{', '.join(q2f) or 'none'}]): "
    f"{'PASS -> v3 regions in all 3 planes' if (q2[0] >= .003 and q2[3] >= .9) or len(q2f) >= 2 else 'FAIL -> same-plane regions only'}")
gain = [T[k] for k in CF if q2[4][k] >= .003]
say(f"Findings with B_all - B_src >= +0.003: {', '.join(gain) or 'none'}")
OUT.write_text("\n".join(lines) + "\n")
