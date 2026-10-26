"""JF-0: do the other 11 findings' logits add signal to each finding (radiologist secondary signs / co-occurrence)?
v4 l4copy SWA fused fold-0 logits (s2026, s2027). Per target k, a cross-fitted L2 logistic regression inside fold 0
(5 folds x 4 repeats, fit on label-v2 confident cells of the training splits) on
  own      own logit only (monotone: reproduces raw AUC; the fair baseline for the fitting noise)
  joint    all 12 logits
  perm     own logit + the other 11 logits shuffled across studies (control: should be ~0)
Scored on clear6 (audit_clear_band_lb.py) and 12-target label-v2 confident; 1,000 paired study bootstraps of joint - own.
Pre-registered decision (2026-10-19, before running): joint - own clear6 >= +0.003 with >= 90% bootstraps > 0 on BOTH seeds
and perm - own within +-0.001 -> a joint-findings head is worth a training lane; otherwise co-occurrence is already used.
Usage (container): python -I audit_joint_findings.py <project root> <out dir> <v4 run root>
"""
import json
import sys
from pathlib import Path

import numpy as np

R, X, V = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); X.mkdir(parents=True, exist_ok=True)
g = {"__name__": "x"}; argv = sys.argv; sys.argv = [argv[0], str(R)]
exec((R / "repository/scripts/audit_clear_band_lb.py").read_text().split("run = lambda")[0], g); sys.argv = argv
ids, T, scores, auc, load = g["ids"], g["T"], g["scores"], g["auc"], g["load"]
lv = json.load(open(R / "repository/data/coatnet_d96_fold0/assets/soft_labels_final_per_target_latD2.json"))
lab = {r["study_id"]: np.array([.5 if p is None else p for p in r["probabilities"]]) for r in lv["labels"]}
Y = np.array([lab[s] for s in ids]); M = (Y <= .3) | (Y >= .7); TT = Y >= .7
n = len(ids); L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


def m12(S, idx=None):
    idx = np.arange(n) if idx is None else idx
    return np.mean([auc(S[idx[M[idx, k]], k], TT[idx[M[idx, k]], k]) for k in range(12)])


def fit(F, t, lam=1.0, iters=50):
    """L2 logistic regression (Newton; as audit_demographics.py), intercept unpenalised -> coefficients [1 + d]."""
    A = np.hstack([np.ones((len(F), 1)), F]); b = np.zeros(A.shape[1]); Lm = lam * np.eye(A.shape[1]); Lm[0, 0] = 0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-A @ b)); H = A.T @ (A * (q * (1 - q))[:, None]) + Lm
        b -= np.linalg.solve(H, A.T @ (q - t) + Lm @ b)
    return b


def crossfit(Z, mode, seed):
    out = np.zeros((n, 12)); rng = np.random.default_rng(seed)
    for rep in range(4):
        fold = rng.permutation(n) % 5
        for k in range(12):
            if mode == "own":
                F = Z[:, [k]]
            elif mode == "joint":
                F = Z
            else:
                F = Z.copy(); others = [j for j in range(12) if j != k]
                F[:, others] = F[rng.permutation(n)][:, others]
            F = (F - F.mean(0)) / F.std(0)
            for f in range(5):
                tr = (fold != f) & M[:, k]; te = fold == f
                b = fit(F[tr], TT[tr, k].astype(float))
                out[te, k] += (b[0] + F[te] @ b[1:]) / 4
    return out


rng = np.random.default_rng(2026); BS = [rng.integers(0, n, n) for _ in range(1000)]
res = {}
for seed in ("s2026", "s2027"):
    P = np.clip(load(V / f"v4_l4copy_{seed}" / "val_swa_fused.json"), 1e-6, 1 - 1e-6); Z = np.log(P / (1 - P))
    S = {m: crossfit(Z, m, 7) for m in ("own", "joint", "perm")}
    pr(f"\n## v4 l4copy {seed}: raw clear6 {scores(P)[0]:.4f} 12t {m12(P):.4f}")
    for m in ("own", "joint", "perm"):
        pr(f"  {m:5s} clear6 {scores(S[m])[0]:.4f} 12t {m12(S[m]):.4f}")
    dc = np.array([scores(S['joint'], b)[0] - scores(S['own'], b)[0] for b in BS])
    da = np.array([m12(S['joint'], b) - m12(S['own'], b) for b in BS])
    dp = scores(S["perm"])[0] - scores(S["own"])[0]
    res[seed] = (dc.mean(), (dc > 0).mean(), dp)
    pr(f"  joint - own clear6 {scores(S['joint'])[0] - scores(S['own'])[0]:+.4f} [{np.percentile(dc, 2.5):+.4f}, {np.percentile(dc, 97.5):+.4f}] "
       f"P>0 {100 * (dc > 0).mean():.0f}% | 12t {m12(S['joint']) - m12(S['own']):+.4f} [{np.percentile(da, 2.5):+.4f}, {np.percentile(da, 97.5):+.4f}] | perm - own clear6 {dp:+.4f}")
    per = []
    for k in range(12):
        sel = M[:, k]
        per.append(f"{T[k][:8]} {auc(S['joint'][sel, k], TT[sel, k]) - auc(S['own'][sel, k], TT[sel, k]):+.3f}")
    pr("  per-target 12t joint - own: " + " ".join(per))
    # which other findings carry weight (full-data fit, standardised coefficients), top 3 per target
    Fz = (Z - Z.mean(0)) / Z.std(0); tops = []
    for k in range(12):
        c = fit(Fz[M[:, k]], TT[M[:, k], k].astype(float))[1:]
        o = [j for j in np.argsort(-np.abs(c)) if j != k][:3]
        tops.append(f"{T[k][:8]}: " + ", ".join(f"{T[j][:8]} {c[j]:+.2f}" for j in o) + f" (own {c[k]:+.2f})")
    pr("  strongest other-finding coefficients:\n    " + "\n    ".join(tops))
ok = all(d >= 0.003 and p >= 0.90 and abs(dp) <= 0.001 for d, p, dp in res.values())
pr(f"\nDECISION: {'joint-findings head worth a lane' if ok else 'no lane: co-occurrence already used (or signal too small)'}")
(X / "audit_joint_findings.txt").write_text("\n".join(L) + "\n")
