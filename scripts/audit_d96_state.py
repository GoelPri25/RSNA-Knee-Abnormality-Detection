"""Score Exp 3b (Exp 1c, D2 re-valued by rule state: BORD -> 0.85 effusion/MCL/medial OA; BELOW -> 0.15
effusion/MCL/ACL/contusion) against the pre-registered rule.

Submit if fold-0 SWA delta vs 1c on the 7 UNCHANGED targets >= -0.002 AND gold-58 12-target macro >= 1c's (0.913).
PASS if public LB >= 0.936. Secondary: fold-0 12-target and per target, mirror TTA, per epoch, 3a comparison,
ensemble value.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_state.py /proj
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
CHANGED = ("Effusion", "MCL", "Medial OA", "ACL", "Contusion")
KEEP = [k for k, t in enumerate(T) if t not in CHANGED]; ALL = list(range(12))


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


def compare(A, B, keep=ALL):
    d = np.array([per(A, b)[keep].mean() - per(B, b)[keep].mean() for b in boots])
    return per(A, full)[keep].mean() - per(B, full)[keep].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


X = C / "run_r50_state"
new = load(X / "val_swa.json", ids); old = load(C / "run_r50/val_swa.json", ids); b3a = load(C / "run_r50_border/val_swa.json", ids)
d7, lo, hi, sh = compare(new, old, KEEP)
print(f"PRIMARY fold-0 (7 unchanged targets): 3b {per(new, full)[KEEP].mean():.4f} vs 1c {per(old, full)[KEEP].mean():.4f}, "
      f"delta {d7:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
d, lo, hi, sh = compare(new, old)
print(f"secondary fold-0 12-target: 3b {per(new, full).mean():.4f} vs 1c {per(old, full).mean():.4f}, delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
A, B = load(X / "val_swa_mirror.json", ids), load(C / "run_r50/val_swa_mirror.json", ids)
d, lo, hi, sh = compare(A, B)
print(f"secondary mirror TTA 12-target: 3b {per(A, full).mean():.4f} vs 1c {per(B, full).mean():.4f}, delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
h2 = {r["epoch"]: r["val_macro_auc"] for r in json.loads((X / "history.json").read_text())}
h1 = {r["epoch"]: r["val_macro_auc"] for r in json.loads((C / "run_r50/history.json").read_text())}
print("per epoch (3b / 1c): " + "; ".join(f"{e}: {h2[e]:.4f}/{h1[e]:.4f}" for e in sorted(h2) if e % 4 == 1 or e >= 20))
pt = pd.DataFrame({"1c": per(old, full), "3a": per(b3a, full), "3b": per(new, full)}, index=T); pt["3b-1c"] = pt["3b"] - pt["1c"]
pt["changed"] = [t in CHANGED for t in T]
print("\nFold-0 per target:"); print(pt.round(4).to_string())

rows = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in rows]; gy = np.array([[int(r[t]) for t in T] for r in rows]).astype(bool)
g = {lab: [auc(load(C / run / "gold_swa.json", gids)[:, k], gy[:, k]) for k in range(12)]
     for lab, run in (("1c", "run_r50"), ("3a", "run_r50_border"), ("1e", "run_r50_pertarget"), ("3b", "run_r50_state"))}
g = pd.DataFrame(g, index=T); g.loc["MACRO"] = g.mean(); g["3b-1c"] = g["3b"] - g["1c"]
print("\nGold-58 per target (development evidence; values were chosen with gold counts, so not independent):")
print(g.round(3).to_string())
gm_new, gm_old = g.at["MACRO", "3b"], g.at["MACRO", "1c"]
ok = d7 >= -0.002 and round(gm_new, 3) >= round(gm_old, 3)
print(f"\nPRE-REGISTERED gate: fold-0 7-target delta {d7:+.4f} (>= -0.002: {d7 >= -0.002}); gold macro {gm_new:.3f} vs 1c "
      f"{gm_old:.3f} (>=: {round(gm_new, 3) >= round(gm_old, 3)}) -> {'SUBMIT (PASS if LB >= 0.936)' if ok else 'DO NOT SUBMIT'}")

QUAD = [D / "r50_tta/kaggle_output/d2_ep15__none.json", D / "r50_tta/kaggle_output/hpo_v1_ep17__none.json",
        D / "highres336/results/oof_predictions.json", D / "efficientnet_v31_d2/results/oof_predictions.json"]
q = [load(p, ids) for p in QUAD]; cb = load(C / "run_continue/val_swa.json", ids); e1 = load(C / "run_r50_pertarget/val_swa.json", ids)
hepta = np.mean(q + [cb, old, e1], 0)
for label, M in (("hepta + 3b (8 equal)", np.mean(q + [cb, old, e1, new], 0)),
                 ("hepta with 1c swapped for 3b", np.mean(q + [cb, new, e1], 0))):
    d, lo, hi, sh = compare(M, hepta, KEEP)
    print(f"{label} vs hepta (7 unchanged targets): delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
