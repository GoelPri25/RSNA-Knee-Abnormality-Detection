"""Exp F0 (CPU): does a learned fusion of 1e and B2 per-finding features beat averaging their predictions?

Inputs: fold-0 features from scripts/extract_fusion_features_gpu.py (data/coatnet_d96_fold0/fusion_f0/), both runs'
val_swa.json (parity), the Steven-v4 fold-0 ruler. Neither model saw fold 0 in training, so the features are clean.
Cells, clear6 and 10-target exactly as scripts/audit_phase1.py. Arms, per finding k, out-of-fold over 5-fold CV inside
fold 0 (5 repeats with different splits; repeat-averaged out-of-fold scores):
  A   mean of the two probabilities (E2-style averaging; nothing fitted)
  S   L2 logistic on [logit_1e, logit_B2]                      (learned reweighting control)
  F   L2 logistic on [logit_1e, logit_B2, PCA-32(z_1e,k), PCA-32(z_B2,k)]   (feature fusion; PCA fitted per training
      split on that finding's features; logits scaled x10 after standardising so the penalty mostly acts on features)
  Fr, Fs  as F with one branch's PCA features only (which branch carries the extra signal)
C picked from {0.001, 0.01, 0.1, 1} by inner 3-fold CV AUC on the training split. Fits use the ruler's confident cells.
Paired bootstrap (2,000 study resamples) for F - A and F - S. Decision rule: see ledger row F0.

Usage (CPU only): CUDA_VISIBLE_DEVICES="" <venv>/python -I fusion_diagnostic.py <project root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
FD = Path(sys.argv[2]) if len(sys.argv) > 2 else C / "fusion_f0"          # optional: dry-run directory
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]
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
lines = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def scores(S, idx):
    c6 = np.mean([auc(S[sel, k], truth[sel, k]) for k in CB
                  for sel in [idx[mask[idx, k] & (~truth[idx, k] | clear[idx, k])]]])
    m10 = np.mean([auc(S[idx[mask[idx, k]], k], truth[idx[mask[idx, k]], k]) for k in TEN])
    return c6, m10


def load_pred(p):
    d = json.loads(Path(p).read_text()); m = {r["study_id"]: r["probabilities"] for r in d["predictions"]}
    tg = d["targets_in_order"]; a = np.array([m[s] for s in ids], float)
    return a[:, [tg.index(t) for t in T]]


# ---------------------------------------------------------------- features + parity
feat = {}
for name, run in (("r50", "run_r50_pertarget"), ("swin", "run_swin3d_b2")):
    z = np.load(FD / f"{name}_fold0.npz", allow_pickle=False)
    order = {s: i for i, s in enumerate(z["ids"].tolist())}
    assert set(order) == set(ids), name
    ix = [order[s] for s in ids]
    Z, L = z["z"][ix], z["logit"][ix]
    p_saved = load_pred(C / run / "val_swa.json")
    p_now = 1 / (1 + np.exp(-L))
    diff = np.abs(p_now - p_saved).max(); corr = np.corrcoef(p_now.ravel(), p_saved.ravel())[0, 1]
    log(f"parity {name}: max |p - val_swa| {diff:.4f}, corr {corr:.5f}, z {Z.shape}")
    assert diff <= 0.01 and corr >= 0.999, f"{name} parity failed: features void"
    feat[name] = (Z, L)
(Zr, Lr), (Zs, Ls) = feat["r50"], feat["swin"]
sig = lambda x: 1 / (1 + np.exp(-x))
P = {"1e": sig(Lr), "B2": sig(Ls), "A": 0.5 * (sig(Lr) + sig(Ls))}

CS = [0.001, 0.01, 0.1, 1.0]


def design(k, tr, te, which):
    """Standardised design for finding k; PCA and scaling fitted on the training rows only."""
    parts_tr, parts_te = [np.c_[Lr[tr, k], Ls[tr, k]]], [np.c_[Lr[te, k], Ls[te, k]]]
    for nm, Z in (("r", Zr), ("s", Zs)):
        if nm in which:
            pca = PCA(32, random_state=0).fit(Z[tr, k])
            parts_tr.append(pca.transform(Z[tr, k])); parts_te.append(pca.transform(Z[te, k]))
    Xtr, Xte = np.hstack(parts_tr), np.hstack(parts_te)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Xtr[:, :2] *= 10; Xte[:, :2] *= 10                               # logits: lightly penalised
    return Xtr, Xte


def fit_predict(Xtr, ytr, Xte):
    best, bc = -1, CS[0]
    if len(CS) > 1:
        for c in CS:
            vals = []
            for itr, iva in StratifiedKFold(3, shuffle=True, random_state=0).split(Xtr, ytr):
                m = LogisticRegression(C=c, max_iter=5000).fit(Xtr[itr], ytr[itr])
                vals.append(auc(m.decision_function(Xtr[iva]), ytr[iva]))
            if np.mean(vals) > best:
                best, bc = np.mean(vals), c
    return LogisticRegression(C=bc, max_iter=5000).fit(Xtr, ytr).decision_function(Xte), bc


ARMS = {"S": "", "F": "rs", "Fr": "r", "Fs": "s"}
REPEATS = 5
oof = {a: np.zeros((REPEATS, len(ids), 12)) for a in ARMS}
chosen = {a: [] for a in ARMS}
for rep in range(REPEATS):
    for k in range(12):
        cells = np.where(mask[:, k])[0]
        yk = truth[cells, k].astype(int)
        for trc, tec in StratifiedKFold(5, shuffle=True, random_state=100 + rep).split(cells, yk):
            tr, te = cells[trc], cells[tec]
            for a, which in ARMS.items():
                Xtr, Xte = design(k, tr, te, which)
                s, c = fit_predict(Xtr, truth[tr, k].astype(int), Xte)
                oof[a][rep, te, k] = s; chosen[a].append(c)
    log(f"repeat {rep + 1}/{REPEATS} done")

al = np.arange(len(ids))
res = {}
for a in ("1e", "B2", "A"):
    res[a] = scores(P[a], al)
for a in ARMS:
    per_rep = [scores(oof[a][r], al) for r in range(REPEATS)]
    res[a] = tuple(np.mean(per_rep, 0)); res[a + "_rep_sd"] = tuple(np.std(per_rep, 0))
log("\nclear-band targets:", [T[k] for k in CB])
for a in ("1e", "B2", "A", "S", "F", "Fr", "Fs"):
    sd = res.get(a + "_rep_sd")
    log(f"  {a:3s} clear6 {res[a][0]:.4f}  10-target {res[a][1]:.4f}"
        + (f"   (repeat sd {sd[0]:.4f} / {sd[1]:.4f}; C chosen {dict(pd.Series(chosen[a]).value_counts())})" if sd else ""))

# paired bootstrap on repeat-averaged out-of-fold scores (rank-based AUC, so averaging decision values is fine)
Fm, Sm = oof["F"].mean(0), oof["S"].mean(0)
rng = np.random.default_rng(0)
dFA, dFS = [], []
for _ in range(2000):
    b = rng.integers(0, len(ids), len(ids))
    f, a_, s_ = scores(Fm, b), scores(P["A"], b), scores(Sm, b)
    dFA.append(np.subtract(f, a_)); dFS.append(np.subtract(f, s_))
dFA, dFS = np.array(dFA), np.array(dFS)
pt = {"F-A": np.subtract(scores(Fm, al), res["A"]), "F-S": np.subtract(scores(Fm, al), scores(Sm, al))}
for nm, arr in (("F-A", dFA), ("F-S", dFS)):
    lo, hi = np.percentile(arr, [2.5, 97.5], 0)
    log(f"  {nm}: clear6 {pt[nm][0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}]  10-target {pt[nm][1]:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]")
log("\nper target (all confident cells)      1e      B2       A       S       F")
for k, t in enumerate(T):
    c = mask[:, k]
    log(f"  {t:17s} " + " ".join(f"{auc(S_[c, k], truth[c, k]):7.4f}" for S_ in (P['1e'], P['B2'], P['A'], Sm, Fm)))

lo_FA = np.percentile(dFA, 2.5, 0)
go = bool(res["F"][1] - res["A"][1] >= 0.004 and lo_FA[1] > 0 and res["F"][0] >= res["A"][0] - 0.001
          and res["F"][1] - res["S"][1] >= 0.002)
log(f"\nDECISION: {'GO to a pre-registered joint two-branch model (Step 1)' if go else 'NO-GO: learned fusion does not beat averaging enough'}")
summary = {k: (list(map(float, v)) if isinstance(v, tuple) else v) for k, v in res.items()}
summary.update({"F_minus_A": list(map(float, pt["F-A"])), "F_minus_S": list(map(float, pt["F-S"])),
                "F_minus_A_ci_low": list(map(float, lo_FA)), "go": go, "clear_band": [T[k] for k in CB]})
(FD / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(FD / "summary.txt").write_text("\n".join(lines) + "\n")
