"""Score Exp 2c (Exp 1c + per-plane branches with auxiliary plane losses) against the pre-registered rule.

PASS if SWA 21-25 12-target delta vs Exp 1c (0.9074) >= +0.003 with >= 90% of 2,000 paired bootstraps better.
Secondary: per target, each plane's auxiliary head alone, mirror TTA, per epoch, gold-58, ensemble value.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_plane.py /proj
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


def compare(A, B):
    d = np.array([per(A, b).mean() - per(B, b).mean() for b in boots])
    return per(A, full).mean() - per(B, full).mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


X = C / "run_r50_plane"
new = load(X / "val_swa.json", ids); old = load(C / "run_r50/val_swa.json", ids)
d, lo, hi, sh = compare(new, old)
print(f"PRIMARY: 2c SWA {per(new, full).mean():.4f} vs 1c {per(old, full).mean():.4f}, delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
print(f"PRE-REGISTERED (delta >= +0.003 and >= 90% better): {'PASS' if d >= 0.003 and sh >= 0.9 else 'FAIL'}")
A, B = load(X / "val_swa_mirror.json", ids), load(C / "run_r50/val_swa_mirror.json", ids)
d, lo, hi, sh = compare(A, B)
print(f"secondary mirror TTA: 2c {per(A, full).mean():.4f} vs 1c {per(B, full).mean():.4f}, delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
h2 = {r["epoch"]: r["val_macro_auc"] for r in json.loads((X / "history.json").read_text())}
h1 = {r["epoch"]: r["val_macro_auc"] for r in json.loads((C / "run_r50/history.json").read_text())}
print("per epoch (2c / 1c): " + "; ".join(f"{e}: {h2[e]:.4f}/{h1[e]:.4f}" for e in sorted(h2) if e % 4 == 1 or e >= 20))
cols = {"1c": per(old, full), "2c": per(new, full)}
for nm in ("sagittal", "coronal", "axial"):
    cols[f"aux {nm}"] = per(load(X / f"val_swa_aux_{nm}.json", ids), full)
pt = pd.DataFrame(cols, index=T); pt["2c-1c"] = pt["2c"] - pt["1c"]
pt.loc["MACRO"] = pt.mean()
print("\nFold-0 per target (aux = that plane's auxiliary head alone):"); print(pt.round(4).to_string())

rows = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in rows]; gy = np.array([[int(r[t]) for t in T] for r in rows]).astype(bool)
g = {lab: [auc(load(C / run / "gold_swa.json", gids)[:, k], gy[:, k]) for k in range(12)]
     for lab, run in (("1c", "run_r50"), ("2c", "run_r50_plane"))}
g = pd.DataFrame(g, index=T); g.loc["MACRO"] = g.mean(); g["2c-1c"] = g["2c"] - g["1c"]
print("\nGold-58 per target (development evidence only; per-target noise ~ +-0.05):"); print(g.round(3).to_string())

QUAD = [D / "r50_tta/kaggle_output/d2_ep15__none.json", D / "r50_tta/kaggle_output/hpo_v1_ep17__none.json",
        D / "highres336/results/oof_predictions.json", D / "efficientnet_v31_d2/results/oof_predictions.json"]
q = [load(p, ids) for p in QUAD]; cb = load(C / "run_continue/val_swa.json", ids); e1 = load(C / "run_r50_pertarget/val_swa.json", ids)
hepta = np.mean(q + [cb, old, e1], 0)
for label, M in (("hepta + 2c (8 equal)", np.mean(q + [cb, old, e1, new], 0)),
                 ("hepta with 1c swapped for 2c", np.mean(q + [cb, new, e1], 0))):
    d, lo, hi, sh = compare(M, hepta)
    print(f"{label} vs hepta: delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
rk = lambda a: pd.DataFrame(a).rank().to_numpy()
print("rank corr 2c vs 1c (mean over targets):",
      round(float(np.mean([np.corrcoef(rk(new)[:, k], rk(old)[:, k])[0, 1] for k in range(12)])), 3))
