"""Extent audit (CPU, cross-fitted, 2026-10-13): does the number of windows showing a finding carry signal that 1e's softmax
attention pooling throws away?

Motivation: the host thresholds are extent-based ("moderate or large amount of fluid", ">= 1 cm area of high-grade cartilage
loss", "abnormal signal contacting the surface on at least two images", "moderate or large" Baker's cyst), and the
residual error is grading mild vs moderate+. RaptorClassifier pools windows with a per-finding softmax (weights sum to 1),
so a finding seen in 3 windows and one seen in 15 give the same pooled feature if the windows look alike: the head is
extent-invariant by construction.

Data: Exp 1e SWA exact CAM decomposition on the 825 fold-0 studies (Kaggle `tumis001/rsna-d96-r50-attention-side-audit`),
att [S, 94, 12] and camx [S, 94, 12, 12] = attention-weighted positive CAM mass per window. Window evidence
e = camx.sum(-1) / att (the window's positive evidence had it taken all the attention). Label-free extent features per
finding: n95 / n99 = windows with e above the 95th / 99th percentile of e pooled over all studies and windows, log sum e.

Test: per target, 5-fold cross-fitted logistic regression on [1e logit] vs [1e logit + extent features], scored on the
Steven-v4 confident cells (all-confident 10-target set, and the clear band as in audit_clear_band_lb.py). Controls: extent
features permuted across studies (must give ~0).

Pre-registered (before running): extent supports count-aware pooling if clear6 delta >= +0.003 with the 95% bootstrap
lower bound > 0 and the permuted control within +-0.001; gains should sit on extent-defined targets (effusion, Baker's,
OA x3, menisci), not ACL / fracture. Otherwise the pooling hypothesis is closed.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds python -I /proj/repository/scripts/audit_extent_pooling.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
clear = np.zeros((len(ids), 12), bool)
for i, s in enumerate(ids):
    if s in st.index:
        for k, t in enumerate(T):
            clear[i, k] = st.at[s, f"{key(t)}__status"] == "present" and st.at[s, f"{key(t)}__severity"] in (
                "moderate", "moderate_to_severe", "severe")
CB = [k for k in range(12) if (mask[:, k] & truth[:, k] & clear[:, k]).sum() >= 20]
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]

z = np.load(D / "attention_side_audit/kaggle_output/attention_side_fold0.npz")
assert list(z["ids"]) == ids or set(z["ids"]) == set(ids)
order = {s: i for i, s in enumerate(z["ids"])}; o = np.array([order[s] for s in ids])
att = z["att"][o].astype(np.float64); pos = z["camx"][o].astype(np.float64).sum(-1); logit = z["logits"][o].astype(np.float64)
ok = z["ok"][o]
e = np.where(att > 1e-6, pos / np.maximum(att, 1e-6), 0.0)                     # [S, 94, 12] window evidence
feats = {}
for k in range(12):
    q95, q99 = np.percentile(e[ok, :, k], [95, 99])                            # label-free thresholds
    feats[k] = np.stack([logit[:, k], (e[:, :, k] > q95).sum(1), (e[:, :, k] > q99).sum(1),
                         np.log1p(e[:, :, k].sum(1))], 1)


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_lr(X, t, lam=1.0, iters=50):
    """L2 logistic regression (Newton); returns weights incl. intercept."""
    A = np.hstack([np.ones((len(X), 1)), X]); b = np.zeros(A.shape[1]); L = lam * np.eye(A.shape[1]); L[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ b)); g = A.T @ (p - t) + L @ b
        H = A.T @ (A * (p * (1 - p))[:, None]) + L
        b -= np.linalg.solve(H, g)
    return b


def folds(t, n=5, seed=0):
    """Stratified fold ids."""
    r = np.random.default_rng(seed); f = np.empty(len(t), int)
    for c in (0, 1):
        i = np.where(t == c)[0]; r.shuffle(i); f[i] = np.arange(len(i)) % n
    return f


def oof(X, k, rng_perm=None):
    """Cross-fitted score for target k on its confident cells; returns full-length array (nan elsewhere)."""
    idx = np.where(mask[:, k] & ok)[0]; out = np.full(len(ids), np.nan)
    Xk = X[idx].copy()
    if rng_perm is not None:
        Xk[:, 1:] = Xk[rng_perm.permutation(len(idx)), 1:]
    Xk = (Xk - Xk.mean(0)) / (Xk.std(0) + 1e-9)
    f = folds(truth[idx, k])
    for j in range(5):
        tr, te = f != j, f == j
        b = fit_lr(Xk[tr], truth[idx[tr], k].astype(float))
        out[idx[te]] = b[0] + Xk[te] @ b[1:]
    return out


base = np.stack([oof(feats[k][:, :1], k) for k in range(12)], 1)
ext = np.stack([oof(feats[k], k) for k in range(12)], 1)
perm = np.stack([oof(feats[k], k, np.random.default_rng(1)) for k in range(12)], 1)


def per_target(S, idx, k, band):
    sel = idx[mask[idx, k] & ok[idx]]
    if band:
        sel = sel[~truth[sel, k] | clear[sel, k]]
    return auc(S[sel, k], truth[sel, k])


def summary(S, idx):
    return (np.mean([per_target(S, idx, k, True) for k in CB]), np.mean([per_target(S, idx, k, False) for k in TEN]))


allidx = np.arange(len(ids))
print(f"studies {ok.sum()} / {len(ids)}; clear-band targets {[T[k] for k in CB]}\n")
print(f"{'target':18s} {'base all':>9s} {'+extent':>8s} {'d all':>7s} {'base clr':>9s} {'+extent':>8s} {'d clr':>7s} {'perm d':>7s}"
      f"  {'mean n95 pos/neg':>17s}")
for k in range(12):
    b_a, e_a, p_a = (per_target(S, allidx, k, False) for S in (base, ext, perm))
    b_c, e_c = (per_target(S, allidx, k, True) for S in (base, ext))
    sel = mask[:, k] & ok
    print(f"{T[k]:18s} {b_a:9.4f} {e_a:8.4f} {e_a - b_a:+7.4f} {b_c:9.4f} {e_c:8.4f} {e_c - b_c:+7.4f} {p_a - b_a:+7.4f}"
          f"  {feats[k][sel & truth[:, k], 1].mean():8.1f}/{feats[k][sel & ~truth[:, k], 1].mean():.1f}")
sb, se, sp = summary(base, allidx), summary(ext, allidx), summary(perm, allidx)
rng = np.random.default_rng(0); bs = []
for _ in range(1000):
    i = rng.integers(0, len(ids), len(ids))
    (c0, m0), (c1, m1) = summary(base, i), summary(ext, i)
    bs.append((c1 - c0, m1 - m0))
bs = np.array(bs)
print(f"\nclear6:    base {sb[0]:.4f}  +extent {se[0]:.4f}  delta {se[0] - sb[0]:+.4f} "
      f"[{np.percentile(bs[:, 0], 2.5):+.4f}, {np.percentile(bs[:, 0], 97.5):+.4f}]  permuted control {sp[0] - sb[0]:+.4f}")
print(f"10-target: base {sb[1]:.4f}  +extent {se[1]:.4f}  delta {se[1] - sb[1]:+.4f} "
      f"[{np.percentile(bs[:, 1], 2.5):+.4f}, {np.percentile(bs[:, 1], 97.5):+.4f}]  permuted control {sp[1] - sb[1]:+.4f}")
# extent within positives: does n95 separate report-moderate+ from report-mild positives? (the grading question itself)
print("\nwithin ruler positives, n95 window count: report moderate+ vs mild/other (AUC; 0.5 = no extent difference)")
for k in range(12):
    sel = mask[:, k] & ok & truth[:, k]
    if clear[sel, k].sum() >= 10 and (~clear[sel, k]).sum() >= 10:
        print(f"  {T[k]:18s} n {sel.sum():4d}  extent AUC {auc(feats[k][sel, 1], clear[sel, k]):.3f}  "
              f"logit AUC {auc(feats[k][sel, 0], clear[sel, k]):.3f}")
