"""Zero-training proxy for per-target plane routing: stitch saved fold-0 predictions by anatomy (pre-specified).

Routing idea: sagittal-read targets (medial / lateral / PF OA, Baker's) keep 1e's sagittal-dominated pool; coronal-
read targets (MCL, fracture, contusion) take a model whose pool gives coronal / axial more voice (4b native, or 2c
per-plane). Variants fixed before scoring (no per-target search on fold 0):
  S-4b    OA + Baker's + rest from 1e; MCL, fracture, contusion from 4b
  S-2c    same, coronal-read targets from 2c (D2 labels)
  S-4b+   OA + Baker's from 1e; every other target = mean(1e, 4b) probability
Controls: 1e, 4b, 2c alone; mean(1e, 4b) on all targets (is a gain routing or just averaging?); mean(1e, 1e-s2)
(a two-model average with no plane change).
Readouts vs 1e (paired bootstrap, 2,000): 10-target (excl. lateral OA, Baker's) and 12-target on Steven-v4 confident
cells; clear-band (report moderate+ positives vs confident negatives) on the six targets with >= 20 clear positives
(the 2-band screen; note the coronal-read targets are not in it), plus per target.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_routing_stitch.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; rid = {r["study_id"]: r for r in ref["labels"]}
ids = sorted(rid)
y = np.array([[.5 if p is None else p for p in rid[s]["probabilities"]] for s in ids])
w = np.array([rid[s]["weights"] for s in ids])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]
SHORT = ["ACL", "MCL", "MM", "LM", "MOA", "LOA", "PFOA", "Eff", "Syn", "Bak", "Cont", "Frac"]
SAG = [T.index(t) for t in ("Medial OA", "Lateral OA", "PF OA", "Baker's")]
COR = [T.index(t) for t in ("MCL", "Fracture", "Contusion")]


def load(run):
    d = json.loads((C / run / "val_swa.json").read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]}
    tg = d["targets_in_order"]
    return np.array([m[s] for s in ids], float)[:, [tg.index(t) for t in T]]


P = {k: load(v) for k, v in {"1e": "run_r50_pertarget", "1e-s2": "run_r50_pertarget_seed", "4b": "run_r50_native_pertarget",
                              "2c": "run_r50_plane"}.items()}


def stitch(base, other, cols):
    out = base.copy(); out[:, cols] = other[:, cols]; return out


rest = [k for k in range(12) if k not in SAG]
V = {"1e": P["1e"], "4b": P["4b"], "2c": P["2c"],
     "mean(1e,4b)": (P["1e"] + P["4b"]) / 2, "mean(1e,1e-s2)": (P["1e"] + P["1e-s2"]) / 2,
     "S-4b": stitch(P["1e"], P["4b"], COR), "S-2c": stitch(P["1e"], P["2c"], COR),
     "S-4b+": stitch(P["1e"], (P["1e"] + P["4b"]) / 2, rest)}

st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
clear = np.zeros((len(ids), 12), bool)
for i, s in enumerate(ids):
    if s in st.index:
        for k, t in enumerate(T):
            clear[i, k] = st.at[s, f"{key(t)}__status"] == "present" and \
                st.at[s, f"{key(t)}__severity"] in ("moderate", "moderate_to_severe", "severe")
cmask = mask & (~truth | clear)
CB = [k for k in range(12) if (cmask[:, k] & truth[:, k]).sum() >= 20]


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(S, m=mask, rows=None):
    rows = np.arange(len(ids)) if rows is None else rows
    return np.array([auc(S[rows][m[rows, k], k], truth[rows][m[rows, k], k]) for k in range(12)])


def boot(S, cols, m=mask, n=2000):
    rng = np.random.default_rng(0); out = []
    for _ in range(n):
        r = rng.integers(0, len(ids), len(ids))
        out.append(np.nanmean(per(S, m, r)[cols]) - np.nanmean(per(V["1e"], m, r)[cols]))
    out = np.array(out); return np.percentile(out, [2.5, 97.5]), (out > 0).mean()


print(f"fold 0: {len(ids)} studies; clear-band targets (>= 20 clear positives): {[SHORT[k] for k in CB]}")
print(f"{'variant':16s}{'10-tgt':>8s}{'d vs 1e':>9s}{'95% CI':>20s}{'%>':>6s}{'12-tgt':>8s}{'clear6':>8s}{'d':>8s}{'95% CI':>20s}{'%>':>6s}")
base10, base12, basec = np.nanmean(per(V["1e"])[TEN]), np.nanmean(per(V["1e"])), np.nanmean(per(V["1e"], cmask)[CB])
for name, S in V.items():
    a = per(S); c = per(S, cmask)
    a10, a12, ac = np.nanmean(a[TEN]), np.nanmean(a), np.nanmean(c[CB])
    if name == "1e":
        print(f"{name:16s}{a10:8.4f}{'':35s}{a12:8.4f}{ac:8.4f}"); continue
    (lo, hi), pb = boot(S, TEN)
    (clo, chi), cpb = boot(S, CB, cmask)
    print(f"{name:16s}{a10:8.4f}{a10 - base10:+9.4f}  [{lo:+.4f}, {hi:+.4f}]{pb:6.0%}{a12:8.4f}{ac:8.4f}{ac - basec:+8.4f}"
          f"  [{clo:+.4f}, {chi:+.4f}]{cpb:6.0%}")
print("\nPer target AUC (confident cells):")
print(f"  {'':16s}" + "".join(f"{s:>7s}" for s in SHORT))
for name, S in V.items():
    print(f"  {name:16s}" + "".join(f"{v:7.3f}" for v in per(S)))
print("\nPer target clear-band AUC (n clear positives in header):")
print(f"  {'':16s}" + "".join(f"{s + str((cmask[:, k] & truth[:, k]).sum()):>9s}" for k, s in enumerate(SHORT)))
for name, S in V.items():
    print(f"  {name:16s}" + "".join(f"{v:9.3f}" for v in per(S, cmask)))
