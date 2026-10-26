"""Phase 1 scorer (P1-r949, P1-r949b, later steps): pre-registered gate + per-target readout vs both 1e seeds.

Gate (ledger P1-r949 / P1-r949b): SWA 21-25, 320 crop, 94 windows, no TTA (val_swa.json); PASS if clear6 >= 0.9745
(1e seed mean 0.9705 + 0.004) AND 10-target >= 0.9053 (1e s2026 0.9073 - 0.002). Clear band and 10-target exactly as
scripts/audit_clear_band_lb.py. Secondary (no selection): per-target AUC vs 1e (rare findings flagged), paired bootstrap
of clear6 / 10-target vs 1e s2026, per-epoch val curve and train loss vs 1e.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds python -I /proj/repository/scripts/audit_phase1.py /proj run_r50_r949 [run_r50_r949_aug ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
RUNS = sys.argv[2:]
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]
RARE = {"Fracture", "MCL", "ACL", "Lateral Meniscus", "Contusion"}
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


def load(p):
    d = json.loads(Path(p).read_text()); m = {r["study_id"]: r["probabilities"] for r in d["predictions"]}
    tg = d["targets_in_order"]; a = np.array([m[s] for s in ids], float)
    return a[:, [tg.index(t) for t in T]]


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def scores(S, idx):
    c6 = np.mean([auc(S[sel, k], truth[sel, k]) for k in CB
                  for sel in [idx[mask[idx, k] & (~truth[idx, k] | clear[idx, k])]]])
    m10 = np.mean([auc(S[idx[mask[idx, k]], k], truth[idx[mask[idx, k]], k]) for k in TEN])
    return c6, m10


def per_target(S):
    return [auc(S[mask[:, k], k], truth[mask[:, k], k]) for k in range(12)]


al = np.arange(len(ids))
base = {"1e": load(C / "run_r50_pertarget/val_swa.json"), "1e_s27": load(C / "run_r50_pertarget_seed/val_swa.json")}
b0 = {n: scores(S, al) for n, S in base.items()}
print("clear-band targets:", [T[k] for k in CB])
for n, (c6, m10) in b0.items():
    print(f"  {n:8s} clear6 {c6:.4f}  10-target {m10:.4f}")
h1e = json.loads((C / "run_r50_pertarget/history.json").read_text())
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
for run in RUNS:
    p = C / run / "val_swa.json"
    if not p.is_file():
        print(f"\n{run}: no val_swa.json yet"); continue
    S = load(p); c6, m10 = scores(S, al)
    verdict = "PASS" if c6 >= 0.9745 and m10 >= 0.9053 else "FAIL"
    print(f"\n=== {run}: PRIMARY clear6 {c6:.4f} (gate 0.9745)  10-target {m10:.4f} (gate 0.9053)  -> {verdict}")
    for bn, B in base.items():
        d = np.array([np.subtract(scores(S, i), scores(B, i)) for i in boots[:500]])
        print(f"  vs {bn:7s} clear6 {c6 - b0[bn][0]:+.4f} [{np.percentile(d[:, 0], 2.5):+.4f}, {np.percentile(d[:, 0], 97.5):+.4f}] "
              f"{(d[:, 0] > 0).mean():.0%} better;  10-target {m10 - b0[bn][1]:+.4f} [{np.percentile(d[:, 1], 2.5):+.4f}, "
              f"{np.percentile(d[:, 1], 97.5):+.4f}]")
    pr, pe, ps = per_target(S), per_target(base["1e"]), per_target(base["1e_s27"])
    print(f"  {'target':17s} {'run':>7s} {'1e':>7s} {'1e_s27':>7s} {'d vs mean':>9s}")
    for k, t in enumerate(T):
        print(f"  {t + (' *' if t in RARE else ''):17s} {pr[k]:7.4f} {pe[k]:7.4f} {ps[k]:7.4f} {pr[k] - (pe[k] + ps[k]) / 2:+9.4f}")
    rare = [k for k, t in enumerate(T) if t in RARE]
    print(f"  rare-finding mean delta vs 1e seed mean: {np.mean([pr[k] - (pe[k] + ps[k]) / 2 for k in rare]):+.4f}; "
          f"others: {np.mean([pr[k] - (pe[k] + ps[k]) / 2 for k in range(12) if k not in rare]):+.4f}")
    pair, seeds = scores((S + base["1e"]) / 2, al), scores((base["1e"] + base["1e_s27"]) / 2, al)
    print(f"  partner check: mean(run, 1e) clear6 {pair[0]:.4f} 10-target {pair[1]:.4f} vs mean(1e, 1e_s27) "
          f"{seeds[0]:.4f} / {seeds[1]:.4f} (delta {pair[0] - seeds[0]:+.4f} / {pair[1] - seeds[1]:+.4f})")
    mt = C / run / "val_swa_mirror.json"
    if mt.is_file():
        print(f"  secondary mirror TTA: clear6 {scores(load(mt), al)[0]:.4f}  10-target {scores(load(mt), al)[1]:.4f}")
    hp = C / run / "history.json"
    if hp.is_file():
        h = json.loads(hp.read_text())
        print("  epoch  train_loss  val12 | 1e val12")
        for r, q in zip(h, h1e):
            v = r["val_macro_auc"]                     # None on epochs a run did not score (Model B)
            if v is not None and (r["epoch"] % 3 == 1 or r["epoch"] >= 21 or r["epoch"] in (5, 10, 15, 20)):
                print(f"  {r['epoch']:5d}  {r['train_loss']:.4f}  {v:.4f} | {q['val_macro_auc']:.4f}")
