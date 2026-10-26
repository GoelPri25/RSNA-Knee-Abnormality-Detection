"""Score Exp 2a (Exp 1c + context between windows) against the pre-registered rule.

Primary: SWA 21-25, 320 crop, 94 windows, no TTA, Steven-v4 fold-0 confident cells, 12-target macro vs Exp 1c.
PASS if delta >= +0.003 with >= 90% of 2,000 paired bootstraps better.
Secondary: 6 x 16-window passes, mirror TTA, per epoch, per target, gold-58, as an ensemble member.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_ctx.py /proj
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
ALL = list(range(12))


def compare(A, B, keep=ALL):
    d = np.array([per(A, b)[keep].mean() - per(B, b)[keep].mean() for b in boots])
    return per(A, full)[keep].mean() - per(B, full)[keep].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


X = C / "run_r50_ctx"
new = load(X / "val_swa.json", ids); old = load(C / "run_r50/val_swa.json", ids)
d, lo, hi, sh = compare(new, old)
print(f"PRIMARY: 2a SWA {per(new, full).mean():.4f} vs 1c {per(old, full).mean():.4f}, delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
print(f"PRE-REGISTERED (delta >= +0.003, >= 90% better): {'PASS' if d >= 0.003 and sh >= 0.9 else 'FAIL'}")
for lab, f, base in (("6x16 passes", "val_swa_multi16.json", "run_r50/val_swa.json"),
                     ("mirror TTA", "val_swa_mirror.json", "run_r50/val_swa_mirror.json")):
    A, B = load(X / f, ids), load(C / base, ids)
    d, lo, hi, sh = compare(A, B)
    print(f"secondary {lab}: 2a {per(A, full).mean():.4f} vs 1c ({base.split('/')[-1]}) {per(B, full).mean():.4f}, "
          f"delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
h2 = {r["epoch"]: r["val_macro_auc"] for r in json.loads((X / "history.json").read_text())}
h1 = {r["epoch"]: r["val_macro_auc"] for r in json.loads((C / "run_r50/history.json").read_text())}
print("per epoch (2a / 1c): " + "; ".join(f"{e}: {h2[e]:.4f}/{h1[e]:.4f}" for e in sorted(h2) if e % 4 == 1 or e >= 20))
print("gate mean |g| at the end:", round(json.loads((X / "run_summary.json").read_text())["gate_abs_mean"], 5))
pt = pd.DataFrame({"1c": per(old, full), "2a": per(new, full)}, index=T); pt["2a-1c"] = pt["2a"] - pt["1c"]
print("\nFold-0 per target:"); print(pt.round(4).to_string())
rk = lambda a: pd.DataFrame(a).rank().to_numpy()
print("rank corr 2a vs 1c (mean over targets):",
      round(float(np.mean([np.corrcoef(rk(new)[:, k], rk(old)[:, k])[0, 1] for k in range(12)])), 3))

QUAD = [D / "r50_tta/kaggle_output/d2_ep15__none.json", D / "r50_tta/kaggle_output/hpo_v1_ep17__none.json",
        D / "highres336/results/oof_predictions.json", D / "efficientnet_v31_d2/results/oof_predictions.json"]
q = [load(p, ids) for p in QUAD]; cb = load(C / "run_continue/val_swa.json", ids); e1 = load(C / "run_r50_pertarget/val_swa.json", ids)
hepta = np.mean(q + [cb, old, e1], 0)
for label, M in (("hepta + 2a (8 equal)", np.mean(q + [cb, old, e1, new], 0)),
                 ("hepta with 1c swapped for 2a", np.mean(q + [cb, new, e1], 0))):
    d, lo, hi, sh = compare(M, hepta)
    print(f"{label} vs hepta: delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")

rows = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in rows]; gy = np.array([[int(r[t]) for t in T] for r in rows]).astype(bool)
g = {lab: [auc(load(C / run / "gold_swa.json", gids)[:, k], gy[:, k]) for k in range(12)]
     for lab, run in (("1c", "run_r50"), ("2a", "run_r50_ctx"))}
g = pd.DataFrame(g, index=T); g.loc["MACRO"] = g.mean(); g["2a-1c"] = g["2a"] - g["1c"]
print("\nGold-58 per target (development evidence only):"); print(g.round(3).to_string())
