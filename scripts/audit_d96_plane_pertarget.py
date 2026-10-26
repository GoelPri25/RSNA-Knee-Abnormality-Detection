"""Score Exp 2d (2c per-plane head + 1e per-target labels) against its pre-registered rule.

Submit standalone if fold-0 SWA 21-25 10-target (excluding Lateral OA, Baker's: trained on the ruler) delta vs Exp 1e
>= -0.002; PASS if public LB >= 0.939. Secondary: per target, plane heads alone, gold-58, value in L4.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_plane_pertarget.py /proj
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
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]


def load(p, order=ids):
    d = json.loads(Path(p).read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
    tg = d.get("targets_in_order", d.get("targets", T)); a = np.array([m[s] for s in order], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(S, idx):
    return np.array([auc(S[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(12)])


rng = np.random.default_rng(0); n = len(ids); full = np.arange(n)
boots = [rng.integers(0, n, n) for _ in range(2000)]


def compare(A, B, cols):
    d = np.array([per(A, b)[cols].mean() - per(B, b)[cols].mean() for b in boots])
    return per(A, full)[cols].mean() - per(B, full)[cols].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


X = C / "run_r50_plane_pertarget"
P = {"d2_224": load(D / "r50_tta/kaggle_output/d2_ep15__none.json"), "1b": load(C / "run_continue/val_swa.json"),
     "1c": load(C / "run_r50/val_swa.json"), "1e": load(C / "run_r50_pertarget/val_swa.json"),
     "2c": load(C / "run_r50_plane/val_swa.json"), "2d": load(X / "val_swa.json")}
d, lo, hi, sh = compare(P["2d"], P["1e"], TEN)
print(f"PRIMARY: 2d SWA 10-target {per(P['2d'], full)[TEN].mean():.4f} vs 1e {per(P['1e'], full)[TEN].mean():.4f}, "
      f"delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
print(f"PRE-REGISTERED submit gate (delta >= -0.002): {'SUBMIT' if d >= -0.002 else 'DO NOT SUBMIT'}")
for ref_name in ("2c", "1c"):
    d, lo, hi, sh = compare(P["2d"], P[ref_name], TEN)
    print(f"secondary 2d vs {ref_name} (10-target): {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
A, B = load(X / "val_swa_mirror.json"), load(C / "run_r50_pertarget/val_swa_mirror.json")
d, lo, hi, sh = compare(A, B, TEN); print(f"secondary mirror TTA vs 1e mirror: {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
h = {r["epoch"]: r["val_macro_auc"] for r in json.loads((X / "history.json").read_text())}
print("per epoch 12-target: " + "; ".join(f"{e}: {h[e]:.4f}" for e in sorted(h) if e % 4 == 1 or e >= 20))
cols = {k: per(P[k], full) for k in ("1c", "1e", "2c", "2d")}
for nm in ("sagittal", "coronal", "axial"):
    cols[f"aux {nm}"] = per(load(X / f"val_swa_aux_{nm}.json"), full)
pt = pd.DataFrame(cols, index=T); pt["2d-1e"] = pt["2d"] - pt["1e"]; pt["2d-2c"] = pt["2d"] - pt["2c"]
pt.loc["MACRO12"] = pt.iloc[:12].mean(); pt.loc["MACRO10"] = pt.iloc[TEN].mean()
print("\nFold-0 per target:"); print(pt.round(4).to_string())

rows = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in rows]; gy = np.array([[int(r[t]) for t in T] for r in rows]).astype(bool)
g = {lab: [auc(load(C / run / "gold_swa.json", gids)[:, k], gy[:, k]) for k in range(12)]
     for lab, run in (("1c", "run_r50"), ("1e", "run_r50_pertarget"), ("2c", "run_r50_plane"), ("2d", "run_r50_plane_pertarget"))}
g = pd.DataFrame(g, index=T); g.loc["MACRO"] = g.mean()
print("\nGold-58 per target (development evidence only; per-target noise ~ +-0.05):"); print(g.round(3).to_string())

ens = lambda ms: np.mean([P[m] for m in ms], 0)
L4 = ens(["d2_224", "1b", "2c", "1e"])
print(f"\nL4 (submitted, d2_224+1b+2c+1e) 10-target {per(L4, full)[TEN].mean():.4f}")
for label, ms in (("L4 with 1e -> 2d", ["d2_224", "1b", "2c", "2d"]), ("L4 with 2c -> 2d", ["d2_224", "1b", "2d", "1e"]),
                  ("L4 + 2d (5)", ["d2_224", "1b", "2c", "1e", "2d"])):
    d, lo, hi, sh = compare(ens(ms), L4, TEN)
    print(f"{label:18s} vs L4 (10-target): {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
rk = lambda a: pd.DataFrame(a).rank().to_numpy()
for o in ("1e", "2c"):
    print(f"rank corr 2d vs {o}:", round(float(np.mean([np.corrcoef(rk(P['2d'])[:, k], rk(P[o])[:, k])[0, 1] for k in range(12)])), 3))
