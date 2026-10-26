"""OrthoDiffusion go/no-go: frozen-feature linear probe of the public OrthoDiffusion backbones vs ImageNet r50.

Features from scripts/od_probe_features.py on Modal (all cache shards, gold excluded): per plane
(sagittal, coronal, axial) middle-block features of the plane's 3D U-Net, global mean (256) or 2x2 grid (1,024), for
2 in-plane orientations x 3 timesteps x 2 blocks; the three planes are concatenated (zeros + a presence flag if absent).
Probe exactly as scripts/audit_ssl_probe.py (PCA-256, L2 logistic, lambda on inner fold 1; 1e labels; scored on the
fold-0 Steven-v4 ruler, 10-target and clear-band6). The OrthoDiffusion setting (orientation, t, block, pooling) is
chosen on the inner fold-1 10-target AUC only, never on fold 0; all settings' fold-0 scores are printed for context.
Reference: tv_in1k (1e's ImageNet start) from the same run and studies; also tv_in1k + selected OrthoDiffusion.
Pre-registered gate (2026-10-12, set on the shard-0 probe: tv_in1k 0.759, 5a SSL epoch 2 0.814; all shards give
more training studies, so the gate is relative): continue to a fine-tune only if the selected OD setting beats
tv_in1k by >= 0.09 on the fold-0 10-target (the shard-0 0.85 bar), or tv_in1k + OD beats tv_in1k by >= 0.03.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_od_probe.py /proj [feature dir]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

R = Path(sys.argv[1]); D = R / "repository/data"; A = D / "coatnet_d96_fold0/assets"
K = O = Path(sys.argv[2]) if len(sys.argv) > 2 else D / "orthodiffusion_probe/modal_output/od_probe"  # OD + r50 from one run
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

od_ids = json.loads((O / "study_ids.json").read_text())
assert od_ids == ids, "OrthoDiffusion and r50 probes must cover the same studies"
OD = np.load(O / "od_feats.npy").astype(np.float32)          # [S, orient, plane, t, block, 1280]
PR = np.load(O / "od_present.npy").astype(np.float32)        # [S, 3]
TS, BLK = [50, 150, 250], ["mid_0", "mid_2"]


def evaluate(X_all):
    """Returns (inner fold-1 10-target AUC, fold-0 per-target AUC [12], fold-0 clear-band6)."""
    Xtr, Xva = X_all[[row[s] for s in tr_ids]], X_all[[row[s] for s in va_ids]]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    _, _, Vt = np.linalg.svd(Xtr, full_matrices=False)
    Xtr, Xva = Xtr @ Vt[:PCS].T, Xva @ Vt[:PCS].T
    s_ = Xtr.std(0) + 1e-6; Xtr, Xva = Xtr / s_, Xva / s_
    P = np.zeros((len(va_ids), 12)); inn = np.full(12, np.nan)
    for k in range(12):
        m = Mtr[:, k]
        sc = {l: auc(Xtr[m & inner] @ fit(Xtr[m & ~inner], Ttr[m & ~inner, k], l)[0], Ttr[m & inner, k]) for l in LAMS}
        best = max(LAMS, key=lambda l: sc[l]); inn[k] = sc[best]
        w, b = fit(Xtr[m], Ttr[m, k], best)
        P[:, k] = Xva @ w + b
    per = np.array([auc(P[Mva[:, k], k], Tva[Mva[:, k], k]) for k in range(12)])
    cb = np.nanmean([auc(P[Mva[:, k] & (~Tva[:, k] | clear[:, k]), k], Tva[Mva[:, k] & (~Tva[:, k] | clear[:, k]), k]) for k in CB])
    return np.nanmean(inn[TEN]), per, cb


def od_view(o, k, j, pool):
    f = OD[:, o, :, k, j, :256] if pool == "mean" else OD[:, o, :, k, j, 256:]
    return np.concatenate([f.reshape(len(OD), -1), PR], 1)


rows = []
def report(name, X):
    inn, per, cb = evaluate(X)
    rows.append({"features": name, "inner10": inn, "10-target": per[TEN].mean(), "clear-band6": cb, **dict(zip(T, per.round(3)))})
    print(f"{name:34s} inner {inn:.4f}  fold-0 10-target {per[TEN].mean():.4f}  clear6 {cb:.4f}", flush=True)


print(f"[{len(tr_ids)} train / {len(va_ids)} fold-0 studies]; OD planes present {PR.mean(0).round(3).tolist()}")
tv = np.load(K / "feats_tv_in1k.npy").astype(np.float32)[:, 0]
report("tv_in1k all-mean (reference)", tv)
for o in range(2):
    for k in range(3):
        for j in range(2):
            for pool in ("mean", "grid"):
                report(f"OD {'transposed' if o else 'as-cached'} t{TS[k]} {BLK[j]} {pool}", od_view(o, k, j, pool))
od_rows = [r for r in rows if r["features"].startswith("OD ")]
sel = max(od_rows, key=lambda r: r["inner10"])
print(f"\nSelected on inner fold 1: {sel['features']}  (fold-0 10-target {sel['10-target']:.4f}, clear6 {sel['clear-band6']:.4f})")
p = sel["features"].split()
X_sel = od_view(int(p[1] == "transposed"), TS.index(int(p[2][1:])), BLK.index(p[3]), p[4])
report("tv_in1k + selected OD", np.concatenate([tv, X_sel], 1))
df = pd.DataFrame(rows)
print("\nFrozen linear probe on fold 0 (Steven-v4 confident cells). Fine-tuned 1e reference: 10-target 0.9073 / 0.9097.")
print(df.round(4).to_string(index=False))
