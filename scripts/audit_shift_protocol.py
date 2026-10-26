"""Exp P1-shift-dx scorer (CPU): clear6 / 10-target for each window-layout variant, scored as scripts/audit_phase1.py.

Reads data/coatnet_d96_fold0/shift_dx/*.json (from scripts/eval_shift_protocol_gpu.py). Pre-registered readout
(ledger P1-shift-dx): protocol gain of P1-shift (shift_runs_r8 - shift_full94) minus that of 1e
(e1_strat_r8 - e1_full94), with a paired bootstrap CI.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds python -I /proj/repository/scripts/audit_shift_protocol.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"; X = C / "shift_dx"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]
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
    return np.array([c6, m10])


al = np.arange(len(ids))
saved = {"shift_full94": C / "run_r50_shift/val_swa.json", "e1_full94": C / "run_r50_pertarget/val_swa.json"}
P = {n: load(X / f"{n}.json") for n in ("shift_full94", "shift_zero94", "shift_runs_c", "shift_runs_r8",
                                         "e1_full94", "e1_strat_r8")}
print("sanity (re-run vs saved val_swa.json): max |dp|",
      {n: round(float(np.abs(P[n] - load(p)).max()), 5) for n, p in saved.items()})
log = json.loads((X / "dx_log.json").read_text())
print(f"\n{'variant':14s} {'clear6':>7s} {'10-tgt':>7s}  neighbours 0/1/2")
for n, S in P.items():
    c6, m10 = scores(S, al)
    print(f"{n:14s} {c6:7.4f} {m10:7.4f}  {log[n].get('windows_with_0_1_2_neighbours', '')}")
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]


def ci(f):
    d = np.array([f(i) for i in boots])
    return f(al), np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)


for name, f in [
    ("shift gain  runs_r8 - full94", lambda i: scores(P["shift_runs_r8"], i) - scores(P["shift_full94"], i)),
    ("1e gain     strat_r8 - full94", lambda i: scores(P["e1_strat_r8"], i) - scores(P["e1_full94"], i)),
    ("PRIMARY     shift gain - 1e gain", lambda i: (scores(P["shift_runs_r8"], i) - scores(P["shift_full94"], i))
     - (scores(P["e1_strat_r8"], i) - scores(P["e1_full94"], i))),
    ("shift_zero94 - shift_full94", lambda i: scores(P["shift_zero94"], i) - scores(P["shift_full94"], i)),
    ("shift_runs_r8 - e1_full94", lambda i: scores(P["shift_runs_r8"], i) - scores(P["e1_full94"], i)),
]:
    v, lo, hi = ci(f)
    print(f"{name:34s} clear6 {v[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}]  10-target {v[1]:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]")
