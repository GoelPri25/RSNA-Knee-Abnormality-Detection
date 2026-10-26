"""Exp MG0 scorer (CPU, numpy only; ledger row MG0). Reads the Stage 0 extraction (part_*.npz, copied from the Modal
volume to data/medgemma_stage0/) and scores five readouts on fold 0 (Steven-v4 ruler, the usual clear6 / 10-target /
12-target) and gold-58 (development only):
  zs      MedGemma zero-shot yes-vs-no score (no training)
  h_last  linear probe on the final-layer hidden state at each finding's answer position (one probe per finding,
          on that finding's own question state)
  h_mid   the same on the middle layer
  vis     linear probe on MedGemma's per-slice image features, mean-pooled per series slot (5 x 2560)
  r50     the same probe on frozen ImageNet ResNet-50 slice features (5 x 2048): the same-protocol baseline
Probes: weighted ridge regression on label-v2 soft labels (training-fold studies only, label weights as sample weights),
features standardised on the training studies, ridge strength chosen by 5-fold CV inside the training studies
(confident-cell AUC). Dual form, so the 12,800-d slot features are cheap.
Pre-registered decision (ledger MG0): GO if a MedGemma probe (h_last, h_mid or vis) beats r50 by >= +0.02 on fold-0
10-target, or zs 10-target >= 0.80; STOP if every MedGemma probe <= r50 + 0.005 and zs < 0.70; otherwise report.

Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_medgemma_stage0.py <project root>
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1])
src = (R / "repository/scripts/audit_phase1.py").read_text()
sys.argv = [sys.argv[0], str(R)]
exec(src.split("def per_target")[0])                      # ids (fold 0), T, mask, truth, clear, auc, scores, TEN, C
D0 = Path(__import__("os").environ.get("MG0_DIR", str(R / "repository/data/medgemma_stage0")))
parts = sorted(D0.glob("part_*.npz"))
assert parts, f"no extraction under {D0}"
F = {}
for p in parts:
    z = np.load(p, allow_pickle=False)
    for k in z.files:
        F.setdefault(k, []).append(z[k])
F = {k: np.concatenate(v) for k, v in F.items()}
row = {s: i for i, s in enumerate(F["ids"].tolist())}
print(f"extracted studies {len(row)} from {len(parts)} parts")

lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text())
lt = lv["targets_in_order"]; order = [lt.index(t) for t in T]
LAB = {r["study_id"]: (np.array([.5 if v is None else v for v in r["probabilities"]])[order],
                       np.array(r["weights"])[order]) for r in lv["labels"]}
val_set = set(ids)
train = [s for s in row if s in LAB and s not in val_set]
missing = [s for s in ids if s not in row]
assert not missing, f"{len(missing)} fold-0 studies not extracted"
gl = [r for r in __import__("csv").DictReader((R / "reports_raw/train.csv").open(encoding="utf-8"))
      if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in gl if r["StudyInstanceUID"] in row]
gy = np.array([[int(r[t]) for t in T] for r in gl if r["StudyInstanceUID"] in row]).astype(bool)
TEST = ids + gids
print(f"probe training studies {len(train)}; fold-0 {len(ids)}; gold {len(gids)}")


def slot_pool(x):
    """[n, 96, d] -> [n, 5 x d]: mean over each slot's non-empty rows."""
    out, r = [], 0
    for k in (26, 22, 18, 12, 18):
        blk = x[:, r:r + k].astype(np.float32)
        n = (np.abs(blk).sum(-1) > 0).sum(1, keepdims=True)
        out.append(blk.sum(1) / np.maximum(n, 1))
        r += k
    return np.concatenate(out, 1)


def ridge_dual(Xtr, y, w, Xte, lam):
    """argmin sum_i w_i (y_i - b - x_i beta)^2 + lam |beta|^2; X already standardised. Returns test predictions."""
    yb = np.sum(w * y) / np.sum(w)
    K = Xtr @ Xtr.T
    a = np.linalg.solve(w[:, None] * K + lam * np.eye(len(y)), w * (y - yb))
    return Xte @ (Xtr.T @ a) + yb


def conf_auc(p, y, w):
    m = (w > 0) & ((y <= .3) | (y >= .7))
    t = y[m] >= .7
    return auc(p[m], t) if 0 < t.sum() < m.sum() else np.nan


def probe(feat_of, name):
    """feat_of(study list) -> [n, d] (or [n, 12, d] per finding). Fold-0 predictions [825, 12]."""
    Xtr_all, Xte_all = feat_of(train), feat_of(TEST)
    Y = np.stack([LAB[s][0] for s in train]); W = np.stack([LAB[s][1] for s in train])
    rng = np.random.default_rng(0); cv = rng.integers(0, 5, len(train))
    pred = np.zeros((len(TEST), 12)); lams = []
    for k in range(12):
        Xtr = Xtr_all[:, k] if Xtr_all.ndim == 3 else Xtr_all
        Xte = Xte_all[:, k] if Xte_all.ndim == 3 else Xte_all
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        y, w = Y[:, k], W[:, k]
        grid = [Xtr.shape[1] * f for f in (0.01, 0.1, 1, 10, 100)]
        cvs = []
        for lam in grid:
            pc = np.zeros(len(y))
            for f in range(5):
                tr, te = cv != f, cv == f
                pc[te] = ridge_dual(Xtr[tr], y[tr], w[tr], Xtr[te], lam)
            cvs.append(conf_auc(pc, y, w))
        lam = grid[int(np.nanargmax(cvs))]; lams.append(lam / Xtr.shape[1])
        pred[:, k] = ridge_dual(Xtr, y, w, Xte, lam)
    print(f"  {name}: ridge strength per finding (x d) {np.round(lams, 2).tolist()}")
    return pred


get = lambda key, f=lambda a: a.astype(np.float32): (lambda S: f(F[key][[row[s] for s in S]]))
P = {"zs": F["zs"][[row[s] for s in TEST]].astype(np.float64)}
for key in ("h_last", "h_mid"):
    P[key] = probe(get(key), key)
for key in ("vis", "r50"):
    P[key] = probe(get(key, slot_pool), key)

idx = np.arange(len(ids))
per = lambda S: [auc(S[mask[:, k], k], truth[mask[:, k], k]) for k in range(12)]
print("\nfold 0 (Steven-v4 ruler): clear6 / 10-target / 12-target; reference fine-tuned label-v2 r50 0.9739 / 0.9074")
res = {}
for name, S in P.items():
    S = S[:len(ids)]
    c6, t10 = scores(S, idx); m12 = float(np.mean(per(S))); res[name] = (c6, t10, m12)
    print(f"  {name:7s} {c6:.4f} / {t10:.4f} / {m12:.4f}   " + " ".join(f"{t[:6]} {v:.3f}" for t, v in zip(T, per(S))))

print(f"\ngold-{len(gids)} macro (development only; no LB ranking signal in Q-1017):")
for name, S in P.items():
    G = S[len(ids):]
    print(f"  {name:7s} {np.nanmean([auc(G[:, k], gy[:, k]) for k in range(12)]):.3f}")

b = res["r50"][1]; best = max(res[k][1] for k in ("h_last", "h_mid", "vis"))
verdict = ("GO" if best >= b + 0.02 or res["zs"][1] >= 0.80 else
           "STOP" if best <= b + 0.005 and res["zs"][1] < 0.70 else "IN BETWEEN (report, user decides)")
print(f"\nDECISION (pre-registered): best MedGemma probe 10-target {best:.4f} vs r50 {b:.4f} ({best - b:+.4f}); "
      f"zero-shot {res['zs'][1]:.4f} -> {verdict}")
