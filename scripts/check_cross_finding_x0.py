"""Exp X0 (pre-registered in docs/experiment_ledger.md row X0): does a combiner that sees all 12 findings beat averaging?

Fold-0 logits of the member set (E3: lv2, 4net, 1b, cnx; P: lv2, B2). Per finding k, a residual on the members' mean
logit m_k (Delta = 0 is logit averaging): C-own (finding k's member logits), C-cross (+ all 12 mean logits), C-quad
(+ pairwise products of the 12 mean logits). L2 logistic regression fitted by Newton on label-v2 soft targets with their
weights; lambda by inner 4-fold CV (weighted BCE); outer 5-fold CV x 5 seeds inside fold 0, seed-averaged logits. Scored
with clear6 / 10-target on the Steven-v4 ruler; paired study bootstrap 2,000, seed 0.

Usage (CPU, single-threaded): python -I check_cross_finding_x0.py <project root>
"""
import sys
from pathlib import Path

src = (Path(sys.argv[1]) / "repository/scripts/screen_lean_ensembles.py").read_text()
exec(src.split("def table(pool)")[0])

LV2 = json.loads((D / "teammate_labelv2/soft_labels_final_per_target_latD2.json").read_text())
tg = LV2["targets_in_order"]; lab = {r["study_id"]: r for r in LV2["labels"]}
YS = np.array([[.5 if lab[s]["probabilities"][tg.index(t)] is None else lab[s]["probabilities"][tg.index(t)] for t in T] for s in ids])
WS = np.array([[lab[s]["weights"][tg.index(t)] for t in T] for s in ids], float)
LAMS = (0.1, 1, 10, 100, 1000)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))


def fit(X, off, y, w, lam):
    """Weighted L2 logistic regression with offset; intercept unpenalised. Returns (beta, b)."""
    beta, b = np.zeros(X.shape[1]), 0.0
    for _ in range(25):
        p = sig(off + X @ beta + b); r = w * (p - y); h = w * p * (1 - p) + 1e-9
        A = np.empty((X.shape[1] + 1,) * 2)
        A[:-1, :-1] = X.T @ (X * h[:, None]) + 2 * lam * np.eye(X.shape[1]); A[:-1, -1] = A[-1, :-1] = X.T @ h; A[-1, -1] = h.sum()
        g = np.concatenate([X.T @ r + 2 * lam * beta, [r.sum()]])
        step = np.linalg.solve(A, g); beta -= step[:-1]; b -= step[-1]
        if np.abs(step).max() < 1e-6:
            break
    return beta, b


def wbce(z, y, w):
    p = np.clip(sig(z), 1e-7, 1 - 1e-7)
    return -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / max(w.sum(), 1e-9)


def design(Lm, M, k, kind):
    own = Lm[:, :, k].T                                                      # [n, members]
    if kind == "own":
        return own
    x = np.concatenate([M, own], 1)
    if kind == "quad":
        iu = np.triu_indices(12, 1)
        x = np.concatenate([x, (M[:, :, None] * M[:, None, :])[:, iu[0], iu[1]]], 1)
    return x


def combine(members, kind, seeds=5):
    Lm = np.stack([lg(P[m]) for m in members])                               # [members, n, 12]
    M = Lm.mean(0); out = np.zeros_like(M); lam_used = []
    n = len(ids)
    for seed in range(seeds):
        fold = np.random.default_rng(seed).permutation(n) % 5
        for k in range(12):
            X = design(Lm, M, k, kind)
            for f in range(5):
                tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                Xs = (X - mu) / sd
                inner = np.random.default_rng(100 + seed).permutation(len(tr)) % 4
                cv = []
                for lam in LAMS:
                    loss = 0
                    for g in range(4):
                        a, v = tr[inner != g], tr[inner == g]
                        be, b0 = fit(Xs[a], M[a, k], YS[a, k], WS[a, k], lam)
                        loss += wbce(M[v, k] + Xs[v] @ be + b0, YS[v, k], WS[v, k])
                    cv.append(loss)
                lam = LAMS[int(np.argmin(cv))]; lam_used.append(lam)
                be, b0 = fit(Xs[tr], M[tr, k], YS[tr, k], WS[tr, k], lam)
                out[te, k] += (M[te, k] + Xs[te] @ be + b0) / seeds
    return M, out, lam_used


rng = np.random.default_rng(0); boots = [rng.choice(idx, len(idx)) for _ in range(2000)]
print("X0: logit-mean baseline vs combiners (out-of-fold inside fold 0); clear6 / 10-target; Δ with 95% bootstrap CI")
for name, members in (("E3", ("lv2", "4net", "1b", "cnx")), ("P", ("lv2", "B2"))):
    pm = np.mean([P[m] for m in members], 0)
    print(f"\n{name} = {'+'.join(members)}: prob mean {scores(pm, idx)[0]:.4f} / {scores(pm, idx)[1]:.4f}")
    base = None
    for kind in ("own", "cross", "quad"):
        M, S, lams = combine(members, kind)
        if base is None:
            base = scores(M, idx)[:2]
            print(f"  logit mean           {base[0]:.4f} / {base[1]:.4f}")
        c = scores(S, idx)[:2]
        d = np.array([np.subtract(scores(S, b)[:2], scores(M, b)[:2]) for b in boots])
        lo, hi = np.percentile(d, [2.5, 97.5], 0)
        per = " ".join(f"{T[k][:5]} {auc(S[mask[:, k], k], truth[mask[:, k], k]) - auc(M[mask[:, k], k], truth[mask[:, k], k]):+.3f}" for k in range(12))
        print(f"  C-{kind:5s}              {c[0]:.4f} / {c[1]:.4f}  Δ clear6 {c[0] - base[0]:+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}]  "
              f"Δ 10t {c[1] - base[1]:+.4f} [{lo[1]:+.4f},{hi[1]:+.4f}]  λ median {np.median(lams):g}")
        print(f"      per finding Δ AUC: {per}")
