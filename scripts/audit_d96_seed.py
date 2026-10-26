"""Exp 1e-s2: seed re-run of Exp 1e (seed 2026 -> 2027, nothing else). Measurement of the noise floor.

Fold 0 (Steven-v4 confident cells): 12- and 10-target delta vs 1e, paired bootstrap; per-target |delta|;
per epoch; mirror TTA; gold-58; two-seed average vs 1e; L4 with 1e replaced by the two-seed average.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_seed.py /proj
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
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]; ALL = list(range(12))


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


S1, S2 = load(C / "run_r50_pertarget/val_swa.json"), load(C / "run_r50_pertarget_seed/val_swa.json")
for lab, cols in (("12-target", ALL), ("10-target", TEN)):
    d, lo, hi, sh = compare(S2, S1, cols)
    print(f"seed 2027 vs 2026 {lab}: {per(S2, full)[cols].mean():.4f} vs {per(S1, full)[cols].mean():.4f}, "
          f"delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
M1, M2 = load(C / "run_r50_pertarget/val_swa_mirror.json"), load(C / "run_r50_pertarget_seed/val_swa_mirror.json")
d, lo, hi, sh = compare(M2, M1, ALL); print(f"mirror TTA 12-target: delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
h1 = {r["epoch"]: r["val_macro_auc"] for r in json.loads((C / "run_r50_pertarget/history.json").read_text())}
h2 = {r["epoch"]: r["val_macro_auc"] for r in json.loads((C / "run_r50_pertarget_seed/history.json").read_text())}
print("per epoch 12-target (s2027 / s2026): " + "; ".join(f"{e}: {h2[e]:.4f}/{h1[e]:.4f}" for e in sorted(h2) if e % 4 == 1 or e >= 20))
avg = (S1 + S2) / 2
pt = pd.DataFrame({"1e s2026": per(S1, full), "1e s2027": per(S2, full), "two-seed avg": per(avg, full),
                   "1c (D2)": per(load(C / "run_r50/val_swa.json"), full)}, index=T)
pt["s2027-s2026"] = pt["1e s2027"] - pt["1e s2026"]
pt.loc["MACRO12"] = pt.iloc[:12].mean(); pt.loc["MACRO10"] = pt.iloc[TEN].mean()
print("\nPer target:"); print(pt.round(4).to_string())
print(f"mean |per-target seed delta| {pt['s2027-s2026'].iloc[:12].abs().mean():.4f}; max {pt['s2027-s2026'].iloc[:12].abs().max():.4f}")
d, lo, hi, sh = compare(avg, S1, TEN); print(f"\ntwo-seed average vs 1e (10-target): {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
rk = lambda a: pd.DataFrame(a).rank().to_numpy()
print("rank corr seed 2027 vs 2026:", round(float(np.mean([np.corrcoef(rk(S1)[:, k], rk(S2)[:, k])[0, 1] for k in range(12)])), 3))

P = {"d2_224": load(D / "r50_tta/kaggle_output/d2_ep15__none.json"), "1b": load(C / "run_continue/val_swa.json"),
     "2c": load(C / "run_r50_plane/val_swa.json")}
L4 = np.mean([P["d2_224"], P["1b"], P["2c"], S1], 0)
for label, M in (("L4, 1e -> s2027", np.mean([P["d2_224"], P["1b"], P["2c"], S2], 0)),
                 ("L4, 1e -> two-seed avg", np.mean([P["d2_224"], P["1b"], P["2c"], avg], 0)),
                 ("L4 + s2027 (5 equal)", np.mean([P["d2_224"], P["1b"], P["2c"], S1, S2], 0))):
    d, lo, hi, sh = compare(M, L4, TEN)
    print(f"{label:24s} vs L4 (10-target): {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")

rows = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in rows]; gy = np.array([[int(r[t]) for t in T] for r in rows]).astype(bool)
g = {lab: [auc(load(C / run / "gold_swa.json", gids)[:, k], gy[:, k]) for k in range(12)]
     for lab, run in (("s2026", "run_r50_pertarget"), ("s2027", "run_r50_pertarget_seed"))}
g = pd.DataFrame(g, index=T); g["delta"] = g["s2027"] - g["s2026"]; g.loc["MACRO"] = g.mean()
print("\nGold-58 (development only):"); print(g.round(3).to_string())
