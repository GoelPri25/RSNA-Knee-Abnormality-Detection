"""Score Exp 1e (d96 r50 on the teammate's per-target labels) against the pre-registered rule.

Primary: SWA 21-25, no TTA, Steven-v4 fold-0 confident cells, 10-target macro (Lateral OA and Baker's excluded: their
training labels are the ruler itself) vs Exp 1c's 10-target macro. Submit standalone if delta >= -0.002.
Secondary: all-12 and per-target (LatOA/Baker's flagged as inflated), gold-58 per target, hexa + 1e.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_pertarget.py /proj
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
KEEP = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]


def load(p, order):
    d = json.loads(Path(p).read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
    tg = d.get("targets_in_order", d.get("targets", T)); a = np.array([m[s] for s in order], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(S, idx):
    return np.array([auc(S[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(len(T))])


rng = np.random.default_rng(0); n = len(ids); full = np.arange(n)
boots = [rng.integers(0, n, n) for _ in range(2000)]


def compare(A, B, keep):
    d = np.array([per(A, b)[keep].mean() - per(B, b)[keep].mean() for b in boots])
    return per(A, full)[keep].mean() - per(B, full)[keep].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


new = load(C / "run_r50_pertarget/val_swa.json", ids)
old = load(C / "run_r50/val_swa.json", ids)
cb = load(C / "run_continue/val_swa.json", ids)
pt = pd.DataFrame({"1c_r50_D2": per(old, full), "1e_r50_pertarget": per(new, full), "1b_coatnet_D2": per(cb, full)}, index=T)
pt["1e-1c"] = pt["1e_r50_pertarget"] - pt["1c_r50_D2"]
print("Fold-0 per target (Lateral OA / Baker's for 1e are trained on the ruler: inflated by construction):")
print(pt.round(4).to_string())
for name, keep in (("10-target (primary)", KEEP), ("12-target (secondary)", list(range(12)))):
    d, lo, hi, sh = compare(new, old, keep)
    print(f"\n{name}: 1e {per(new, full)[keep].mean():.4f} vs 1c {per(old, full)[keep].mean():.4f}, "
          f"delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
d, *_ = compare(new, old, KEEP)
print(f"\nPRE-REGISTERED (10-target delta >= -0.002): {'PASS -> submit standalone' if d >= -0.002 else 'FAIL'}")
rk = lambda a: pd.DataFrame(a).rank().to_numpy()
print("rank corr 1e vs 1c (mean over targets):",
      round(float(np.mean([np.corrcoef(rk(new)[:, k], rk(old)[:, k])[0, 1] for k in range(12)])), 3))

QUAD = [D / "r50_tta/kaggle_output/d2_ep15__none.json", D / "r50_tta/kaggle_output/hpo_v1_ep17__none.json",
        D / "highres336/results/oof_predictions.json", D / "efficientnet_v31_d2/results/oof_predictions.json"]
hexa = np.mean([load(p, ids) for p in QUAD] + [cb, old], 0)
for label, M in (("hexa + 1e (7 equal)", np.mean([load(p, ids) for p in QUAD] + [cb, old, new], 0)),
                 ("hexa with 1c swapped for 1e", np.mean([load(p, ids) for p in QUAD] + [cb, new], 0))):
    d, lo, hi, sh = compare(M, hexa, KEEP)
    print(f"{label} vs hexa, 10-target: delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")

rows = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in rows]; gy = np.array([[int(r[t]) for t in T] for r in rows]).astype(bool)
g = {}
for lab, run in (("1c_r50_D2", "run_r50"), ("1e_r50_pertarget", "run_r50_pertarget"), ("1b_coatnet_D2", "run_continue")):
    G = load(C / run / "gold_swa.json", gids)
    g[lab] = [auc(G[:, k], gy[:, k]) for k in range(12)]
g = pd.DataFrame(g, index=T); g.loc["MACRO"] = g.mean()
g["1e-1c"] = g["1e_r50_pertarget"] - g["1c_r50_D2"]
print("\nGold-58 per target (development evidence only):")
print(g.round(3).to_string())
