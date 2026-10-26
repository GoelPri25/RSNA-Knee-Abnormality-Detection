"""Exp MS1-CP (ledger, pre-registered 2026-10-17): does a crop chosen by MS1's own token maps add signal to MS1's
whole-field view? CPU probe on the frozen features of scripts/extract_ms1_crop_features_gpu.py.

Labels: label-v2 per target (MS1's training labels), confident cells (weight > 0, y <= 0.3 / >= 0.7), training and
scoring alike. Crop probe per finding, as scripts/audit_ssl_probe.py: crop feature c_n standardised and reduced to 256
principal components on the training folds, L2 logistic, lambda in {1e-3, 1e-2, 1e-1} chosen on training fold 1 held
out, refit on all training folds -> fold-0 crop score s_n.
  A = MS1's fold-0 logit (frozen, out of fold).
  B = 2-input logistic [z(MS1 logit), z(s_n)] cross-fitted inside fold 0 (5 stratified splits x 20 repeats, the
      out-of-split scores averaged), so neither input nor weights see the scored study.
Gate (pre-registered): over ACL, Medial Meniscus, Lateral Meniscus, Medial OA, Contusion, Fracture (the tight-map
findings of LOC-0-MS1), mean AUC(B) - AUC(A) >= +0.005 and its paired study-bootstrap 95% lower bound > 0 -> build
MS1 global + local, crops for the findings with own gain >= +0.005; otherwise stop.
Descriptive: s_n alone; the zero-shot crop logit (c_n . W_n + b_n) in place of s_n; probe A' (global g_n) vs B'
(g_n + c_n) both trained on the training folds (biased against crops: g_n is in-sample for MS1 there).

Usage (container): podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_ms1_crop_probe.py /proj
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

R = Path(sys.argv[1]); D = R / "repository/data"; A = D / "coatnet_d96_fold0/assets"
C = Path(sys.argv[3]) if len(sys.argv) > 3 else D / "coatnet_d96_fold0/run_sig_lv2/crop_probe"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else C / "audit_crop_probe.txt"
lines = []
say = lambda s="": (print(s, flush=True), lines.append(s))
parts = [np.load(p) for p in sorted(C.glob("cp_*.npz"))]
ids = np.concatenate([p["ids"] for p in parts]); row = {s: i for i, s in enumerate(ids)}
cat = lambda k: np.concatenate([p[k] for p in parts])
LOG, CLOG, Gf, Cf = cat("logits"), cat("crop_logits"), cat("g").astype(np.float32), cat("c").astype(np.float32)
assert len(ids) == 4349 and len(row) == 4349
fold = {r["study_id"]: r["fold"] for r in csv.DictReader((A / "folds.csv").open())}
lab = json.loads((A / "soft_labels_final_per_target_latD2.json").read_text()); T = lab["targets_in_order"]
lv = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lab["labels"]}
tr = np.array([fold[s] != "0" for s in ids]); va = ~tr; inner = np.array([fold[s] == "1" for s in ids])
Y = np.array([lv[s][0] for s in ids]); Wt = np.array([lv[s][1] for s in ids])
M = (Wt > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7
GATE = [T.index(t) for t in ("ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Contusion", "Fracture")]
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
    """standardise + PCA-PCS on the training folds -> transformed all rows (unit-variance components)"""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Z = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Z[tr], full_matrices=False); P = Z @ Vt[:PCS].T
    return P / (P[tr].std(0) + 1e-6)


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


ref = json.loads((D / "coatnet_d96_fold0/run_sig_lv2/val_swa.json").read_text())
pv = {p["study_id"]: np.array(p["probabilities"] if "probabilities" in p else p["prediction"]) for p in ref["predictions"]}
vids = ids[va]; P0 = np.array([pv[s] for s in vids])
say(f"SANITY fold 0: pooled MS1 logits vs val_swa.json: rank corr per target min "
    f"{min(pd.Series(LOG[va, k]).corr(pd.Series(P0[:, k]), method='spearman') for k in range(12)):.4f}")
say(f"studies: {tr.sum()} training folds, {va.sum()} fold 0; label-v2 confident cells\n")

rows, dA, dB = [], {}, {}
for k, t in enumerate(T):
    Xc = pca(Cf[:, k]); s, lam = probe(Xc, k)
    Bs = crossfit(LOG[:, k], s, k); Bz = crossfit(LOG[:, k], CLOG[:, k], k)
    Xg = pca(Gf[:, k]); ga, _ = probe(Xg, k); gb, _ = probe(np.concatenate([Xg, Xc], 1), k)
    v = np.where(va)[0]; m = M[v, k]; tt = TT[v, k]
    a_ = LOG[v, k]
    rows.append({"finding": t, "n+": int((m & tt).sum()), "n-": int((m & ~tt).sum()), "lam": lam,
                 "A MS1": auc(a_[m], tt[m]), "B MS1+crop": auc(Bs[m], tt[m]), "B-A": auc(Bs[m], tt[m]) - auc(a_[m], tt[m]),
                 "crop probe alone": auc(s[v][m], tt[m]), "zero-shot crop": auc(CLOG[v, k][m], tt[m]),
                 "MS1+zero-shot": auc(Bz[m], tt[m]), "A' global probe": auc(ga[v][m], tt[m]), "B' global+crop probe": auc(gb[v][m], tt[m])})
    dA[k], dB[k] = a_, Bs
    print(f"done {t}", flush=True)
df = pd.DataFrame(rows).set_index("finding")
say("Per finding, fold 0 AUC (label-v2 confident cells). PRIMARY: A vs B.")
say(df.round(4).to_string())

v = np.where(va)[0]
boot = np.zeros((BOOT, 12))
for b_ in range(BOOT):
    i = rng.integers(0, len(v), len(v))
    for k in range(12):
        m = M[v[i], k]; tt = TT[v[i], k][m]
        boot[b_, k] = auc(dB[k][i][m], tt) - auc(dA[k][i][m], tt) if 0 < tt.sum() < len(tt) else np.nan
g_pt = df["B-A"].iloc[GATE].mean(); g_bt = np.nanmean(boot[:, GATE], 1)
lo, hi = np.nanpercentile(g_bt, [2.5, 97.5])
say(f"\nPer-finding B-A 95% CI: " + ", ".join(f"{T[k]} [{np.nanpercentile(boot[:, k], 2.5):+.4f}, {np.nanpercentile(boot[:, k], 97.5):+.4f}]" for k in range(12)))
say(f"GATE (6 tight-map findings): mean B-A {g_pt:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
win = [T[k] for k in GATE if df["B-A"].iloc[k] >= .005]
say("VERDICT: " + (f"PASS -> build MS1 global + local; crops for {', '.join(win)}" if g_pt >= .005 and lo > 0 else
                   "FAIL -> stop; the frozen crop view adds no reliable signal to MS1"))
OUT.write_text("\n".join(lines) + "\n")
