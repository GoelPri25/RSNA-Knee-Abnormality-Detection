"""Exp J1 scorer (pre-registered in docs/experiment_ledger.md row J1, before the run).

Inputs (fold 0, SWA 16-20, 320, no TTA): run_joint_j1/val_swa.json (fused head = primary), val_swa_r50.json and
val_swa_swin.json (the two co-trained branch heads), val_swa_share.npy (fused attention mass on ResNet tokens); the
standalone 1e (run_r50_pertarget) and B2 (run_swin3d_b2) val_swa.json. Cells, clear6 and 10-target exactly as
scripts/audit_phase1.py (Steven-v4 ruler). Paired bootstrap, 2,000 study resamples, seed 0.

Decision (fixed before the run):
  PRIMARY (does combined attention add anything over the same two branches averaged?):
    fused - mean(own branch heads) >= +0.003 on 10-target, bootstrap 95% CI lower bound > 0, and clear6 not lower
    by more than 0.001.
  SUBMISSION-WORTHY (does the one network beat what we already have?): fused 10-target >= A + 0.003, where
    A = mean(standalone 1e, B2) (F0: 0.9715 / 0.9112), and fused clear6 >= A clear6 - 0.001.
  LB CANDIDATE (the phase-1 single-model gate): clear6 >= 0.9745 AND 10-target >= 0.9085.
Diagnostics (no selection): each branch head vs its standalone model; per-target AUC; mean fused ResNet share per
finding; training curves from history.json.

Usage (CPU): python -I audit_joint_j1.py <project root> [run dir name, default run_joint_j1]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
RUN = C / (sys.argv[2] if len(sys.argv) > 2 else "run_joint_j1")
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
lines = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)


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


P = {"fused": load(RUN / "val_swa.json"), "r50_head": load(RUN / "val_swa_r50.json"),
     "swin_head": load(RUN / "val_swa_swin.json"),
     "1e": load(C / "run_r50_pertarget/val_swa.json"), "B2": load(C / "run_swin3d_b2/val_swa.json")}
P["mean_heads"] = 0.5 * (P["r50_head"] + P["swin_head"])
P["A"] = 0.5 * (P["1e"] + P["B2"])
al = np.arange(len(ids))
res = {k: scores(v, al) for k, v in P.items()}
log(f"run {RUN.name}; clear-band targets {[T[k] for k in CB]}")
for k in ("fused", "r50_head", "swin_head", "mean_heads", "1e", "B2", "A"):
    log(f"  {k:10s} clear6 {res[k][0]:.4f}  10-target {res[k][1]:.4f}")

rng = np.random.default_rng(0)
pairs = [("fused", "mean_heads"), ("fused", "A"), ("fused", "1e"), ("r50_head", "1e"), ("swin_head", "B2")]
boot = {p: [] for p in pairs}
for _ in range(2000):
    b = rng.integers(0, len(ids), len(ids))
    sc = {k: scores(P[k], b) for k in {x for p in pairs for x in p}}
    for p in pairs:
        boot[p].append(sc[p[0]] - sc[p[1]])
ci = {}
for p in pairs:
    arr = np.array(boot[p]); lo, hi = np.percentile(arr, [2.5, 97.5], 0); d = res[p[0]] - res[p[1]]
    ci[p] = (d, lo, hi)
    log(f"  {p[0]} - {p[1]}: clear6 {d[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}]  10-target {d[1]:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]")

log("\nper target (confident cells)   fused  r50_hd swin_hd  mean_hd     1e      B2       A   r50 share")
share = np.load(RUN / "val_swa_share.npy") if (RUN / "val_swa_share.npy").is_file() else None
for k, t in enumerate(T):
    c = mask[:, k]
    sh = f"{share[:, k].mean():9.3f}" if share is not None else ""
    log(f"  {t:17s} " + " ".join(f"{auc(P[m][c, k], truth[c, k]):7.4f}"
                                 for m in ("fused", "r50_head", "swin_head", "mean_heads", "1e", "B2", "A")) + sh)

hist = RUN / "history.json"
if hist.is_file():
    log("\nepoch  train(fused/r50/swin)        val fused  r50     swin    mean-br  val BCE")
    for h in json.loads(hist.read_text()):
        v = h.get("val_macro_auc")
        log(f"  {h['epoch']:2d}   {h['train_fused']:.4f}/{h['train_r50']:.4f}/{h['train_swin']:.4f}   "
            + ("-" if v is None else f"{v:.4f}   {h['val_r50']:.4f}  {h['val_swin']:.4f}  "
                                     f"{h['val_mean_of_branches']:.4f}  {h['val_bce']:.4f}"))

d, lo, _ = ci[("fused", "mean_heads")]
primary = bool(d[1] >= 0.003 and lo[1] > 0 and d[0] >= -0.001)
d_a = ci[("fused", "A")][0]
worthy = bool(d_a[1] >= 0.003 and d_a[0] >= -0.001)
lb = bool(res["fused"][0] >= 0.9745 and res["fused"][1] >= 0.9085)
log(f"\nPRIMARY (combined attention > own heads averaged): {'PASS' if primary else 'FAIL'}")
log(f"SUBMISSION-WORTHY (fused >= A + 0.003 on 10-target): {'PASS' if worthy else 'FAIL'}")
log(f"LB CANDIDATE (clear6 >= 0.9745 and 10-target >= 0.9085): {'PASS' if lb else 'FAIL'}")
summary = {"scores": {k: v.tolist() for k, v in res.items()},
           "deltas": {f"{a}-{b}": {"point": v[0].tolist(), "lo": v[1].tolist(), "hi": v[2].tolist()}
                      for (a, b), v in ci.items()},
           "primary": primary, "submission_worthy": worthy, "lb_candidate": lb}
(RUN / "audit_j1.json").write_text(json.dumps(summary, indent=2) + "\n")
(RUN / "audit_j1.txt").write_text("\n".join(lines) + "\n")
