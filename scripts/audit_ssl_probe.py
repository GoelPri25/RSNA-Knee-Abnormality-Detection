"""Exp 5a diagnostic: frozen-feature linear probe of r50 backbones (Kaggle kernel kaggle/submission_kernels/d96_ssl_probe).

Features: per study, frozen r50 features mean-pooled over 32 windows (all planes; or the three plane means concatenated).
Probe: per target, L2 logistic regression on standardised features reduced to 256 principal components (PCA fitted
on the training folds; added after the full-dimension fit took ~7 min per setting), trained on the training folds' confident cells of
1e's per-target labels (>= 0.7 / <= 0.3, weight > 0); lambda per backbone and target chosen on fold 1 held out from
the other training folds, then refit on all training folds. Scored on fold 0 against the Steven-v4 ruler (confident
cells): 10-target macro (excl. lateral OA, Baker's, whose labels are the ruler) and the clear-band screen (2-band).
Studies: whatever the kernel processed (v2: cache shard 0 only, ~1,100 studies, after v1 with all 4 shards was
cancelled at start); probe scores on that subset are noisier than the fine-tuned fold-0 scores.
Reading: a diagnostic of feature quality only (frozen features, linear head); fine-tuning (5a) is the decision.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_ssl_probe.py /proj
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

R = Path(sys.argv[1]); D = R / "repository/data"; A = D / "coatnet_d96_fold0/assets"
K = D / "ssl_probe/kaggle_output"
NAMES = ["tv_in1k", "bt_imagenet", "ssl_e02", "ssl_e05", "ssl_e10"]
ids = json.loads((K / "study_ids.json").read_text()); row = {s: i for i, s in enumerate(ids)}
fold = {r["study_id"]: r["fold"] for r in csv.DictReader((A / "folds.csv").open())}
lab = json.loads((A / "soft_labels_final_per_target_any_model.json").read_text())
T = lab["targets_in_order"]
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
assert ref["targets_in_order"] == T
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]

tr = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"]))
      for r in lab["labels"] if fold[r["study_id"]] != "0" and r["study_id"] in row}
va = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"]))
      for r in ref["labels"] if r["study_id"] in row}
tr_ids, va_ids = sorted(tr), sorted(va)
Ytr = np.array([tr[s][0] for s in tr_ids]); Wtr = np.array([tr[s][1] for s in tr_ids])
Yva = np.array([va[s][0] for s in va_ids]); Wva = np.array([va[s][1] for s in va_ids])
Mtr = (Wtr > 0) & ((Ytr <= .3) | (Ytr >= .7)); Mva = (Wva > 0) & ((Yva <= .3) | (Yva >= .7))
Ttr, Tva = Ytr >= .7, Yva >= .7
inner = np.array([fold[s] == "1" for s in tr_ids])

st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
clear = np.array([[s in st.index and st.at[s, f"{key(t)}__status"] == "present" and
                   st.at[s, f"{key(t)}__severity"] in ("moderate", "moderate_to_severe", "severe") for t in T] for s in va_ids])
CB = [k for k in range(12) if (Mva[:, k] & Tva[:, k] & clear[:, k]).sum() >= 20]


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def fit(X, y, lam):
    """L2 logistic regression (mean loss + lam/2 |w|^2), L-BFGS; returns (w, b)."""
    n, d = X.shape

    def f(p):
        w, b = p[:d], p[d]; z = X @ w + b
        loss = np.mean(np.logaddexp(0, z) - y * z) + lam / 2 * w @ w
        g = (1 / (1 + np.exp(-z)) - y) / n
        return loss, np.append(X.T @ g + lam * w, g.sum())
    r = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return r.x[:d], r.x[d]


LAMS = [1e-3, 1e-2, 1e-1]
PCS = 256
rows = []
for name in NAMES:
    F = np.load(K / f"feats_{name}.npy").astype(np.float32)
    for view, X_all in (("all-mean", F[:, 0]), ("plane-means", F[:, 1:].reshape(len(F), -1))):
        Xtr, Xva = X_all[[row[s] for s in tr_ids]], X_all[[row[s] for s in va_ids]]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
        _, _, Vt = np.linalg.svd(Xtr, full_matrices=False)                     # PCA on training folds only
        Xtr, Xva = Xtr @ Vt[:PCS].T, Xva @ Vt[:PCS].T
        s_ = Xtr.std(0) + 1e-6; Xtr, Xva = Xtr / s_, Xva / s_
        P = np.zeros((len(va_ids), 12))
        for k in range(12):
            m = Mtr[:, k]
            best = max(LAMS, key=lambda l: auc(Xtr[m & inner] @ fit(Xtr[m & ~inner], Ttr[m & ~inner, k], l)[0],
                                               Ttr[m & inner, k]))
            w, b = fit(Xtr[m], Ttr[m, k], best)
            P[:, k] = Xva @ w + b
        per = np.array([auc(P[Mva[:, k], k], Tva[Mva[:, k], k]) for k in range(12)])
        cb = np.nanmean([auc(P[Mva[:, k] & (~Tva[:, k] | clear[:, k]), k], Tva[Mva[:, k] & (~Tva[:, k] | clear[:, k]), k]) for k in CB])
        rows.append({"backbone": name, "features": view, "10-target": per[TEN].mean(), "12-target": per.mean(),
                     "clear-band6": cb, **dict(zip(T, per.round(3)))})
        print(f"[{len(tr_ids)} train / {len(va_ids)} fold-0 studies] {name:12s} {view:12s} 10-target {per[TEN].mean():.4f}  clear6 {cb:.4f}", flush=True)
df = pd.DataFrame(rows)
print("\nFrozen linear probe on fold 0 (Steven-v4 confident cells). Fine-tuned 1e reference: 10-target 0.9073 / 0.9097.")
print(df.round(4).to_string(index=False))
