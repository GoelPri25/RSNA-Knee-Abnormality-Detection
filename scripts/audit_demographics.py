"""Demographics audit (CPU, cross-fitted, 2026-10-13): does patient age / sex (DICOM headers, Kaggle scan
`tumis001/rsna-dicom-demographics-scan`) add signal beyond the 1e image model?

Per target, 5-fold cross-fitted logistic regression on fold 0 (825 studies, 1e out-of-fold predictions): [1e logit] vs
[1e logit + age terms + sex (+ field strength)], scored against the Steven-v4 confident cells (10-target, clear band).
Control: demographics permuted across studies. Studies with missing age get the training-fold median plus a missing flag.

Pre-registered (before running): demographics are worth an end-to-end run if clear6 delta >= +0.004 with the 95%
bootstrap lower bound > 0, permuted control within +-0.001, AND age is present for >= 90% of test studies; expected
targets: OA x3, medial meniscus, Baker's (positive), ACL / contusion (younger).

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds python -I /proj/repository/scripts/audit_demographics.py /proj
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]]); mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
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
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]

H = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(D / "dicom_demographics/study_header_tags.csv"))}


def age(v):
    m = re.fullmatch(r"(\d{1,3})([YMWD]?)", v or "")
    if not m:
        return np.nan
    n = float(m.group(1)); return {"Y": n, "M": n / 12, "W": n / 52, "D": n / 365, "": n}[m.group(2)]


for split in ("train", "test"):
    rs = [r for r in H.values() if r["split"] == split]
    a = np.array([age(r["PatientAge"]) for r in rs])
    print(f"{split}: {len(rs)} studies; age present {np.isfinite(a).mean():.3f} (median {np.nanmedian(a) if np.isfinite(a).any() else float('nan'):.0f}); "
          f"sex present {np.mean([r['PatientSex'] in ('M', 'F') for r in rs]):.3f}; "
          f"field strength present {np.mean([bool(r['MagneticFieldStrength']) for r in rs]):.3f}")

A = np.array([age(H.get(s, {}).get("PatientAge", "")) for s in ids])
SEX = np.array([{"M": 1.0, "F": -1.0}.get(H.get(s, {}).get("PatientSex", ""), 0.0) for s in ids])
FS = np.array([float(H.get(s, {}).get("MagneticFieldStrength") or "nan") for s in ids])
FS = np.where(FS > 100, FS / 10000, FS)                                          # gauss -> tesla if mis-coded
print(f"fold 0: age present {np.isfinite(A).mean():.3f}, sex present {(SEX != 0).mean():.3f}, field strength {np.isfinite(FS).mean():.3f}")
d = json.loads((D / "coatnet_d96_fold0/run_r50_pertarget/val_swa.json").read_text())
m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
tg = d.get("targets_in_order", d.get("targets", T))
p = np.array([m[s] for s in ids], float)[:, [list(tg).index(t) for t in T]]
lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def fit(X, t, lam=1.0, iters=60):
    Am = np.hstack([np.ones((len(X), 1)), X]); b = np.zeros(Am.shape[1]); L = lam * np.eye(Am.shape[1]); L[0, 0] = 0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-Am @ b)); Hs = Am.T @ (Am * (q * (1 - q))[:, None]) + L
        b -= np.linalg.solve(Hs, Am.T @ (q - t) + L @ b)
    return b


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def demo(idx_tr, idx, perm=None):
    a = A.copy(); med = np.nanmedian(a[idx_tr]) if np.isfinite(a[idx_tr]).any() else 50.0
    miss = ~np.isfinite(a); a = np.where(miss, med, a); z = (a - 50) / 15
    f = np.where(np.isfinite(FS), FS, 0.0)
    X = np.stack([z, z ** 2, np.maximum(z, 0), SEX, miss.astype(float), (f >= 2.5).astype(float)], 1)
    if perm is not None:
        X = X[perm]
    return X[idx]


rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(ids)); permi = rng.permutation(len(ids))
base, ext, ctl = (np.full((len(ids), 12), np.nan) for _ in range(3))
for k in range(12):
    sel = np.where(mask[:, k])[0]
    for j in range(5):
        tr, te = sel[fold[sel] != j], sel[fold[sel] == j]
        for out, pm in ((ext, None), (ctl, permi)):
            Xtr = np.hstack([lg[tr, k:k + 1], demo(tr, tr, pm)]); Xte = np.hstack([lg[te, k:k + 1], demo(tr, te, pm)])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            b = fit((Xtr - mu) / sd, truth[tr, k].astype(float)); out[te, k] = b[0] + ((Xte - mu) / sd) @ b[1:]
        base[sel, k] = lg[sel, k]


def per(S, k, band, idx):
    s = idx[mask[idx, k]]
    if band:
        s = s[~truth[s, k] | clear[s, k]]
    return auc(S[s, k], truth[s, k])


def summ(S, idx):
    return np.mean([per(S, k, True, idx) for k in CB]), np.mean([per(S, k, False, idx) for k in TEN])


al = np.arange(len(ids))
print(f"\n{'target':16s} {'base':>7s} {'+demo':>7s} {'delta':>7s} {'perm':>7s}  clear-band delta")
for k, t in enumerate(T):
    b0, b1, b2 = per(base, k, False, al), per(ext, k, False, al), per(ctl, k, False, al)
    print(f"{t:16s} {b0:7.4f} {b1:7.4f} {b1 - b0:+7.4f} {b2 - b0:+7.4f}  {per(ext, k, True, al) - per(base, k, True, al):+.4f}")
bs = []
for _ in range(1000):
    i = rng.integers(0, len(ids), len(ids)); (c0, m0), (c1, m1) = summ(base, i), summ(ext, i); bs.append((c1 - c0, m1 - m0))
bs = np.array(bs); s0, s1, s2 = summ(base, al), summ(ext, al), summ(ctl, al)
print(f"\nclear6    {s0[0]:.4f} -> {s1[0]:.4f}  {s1[0] - s0[0]:+.4f} [{np.percentile(bs[:, 0], 2.5):+.4f}, {np.percentile(bs[:, 0], 97.5):+.4f}]  perm {s2[0] - s0[0]:+.4f}")
print(f"10-target {s0[1]:.4f} -> {s1[1]:.4f}  {s1[1] - s0[1]:+.4f} [{np.percentile(bs[:, 1], 2.5):+.4f}, {np.percentile(bs[:, 1], 97.5):+.4f}]  perm {s2[1] - s0[1]:+.4f}")
print("\nlabel rate by age band (fold 0 confident cells):")
bands = [(0, 30), (30, 45), (45, 60), (60, 120)]
for k, t in enumerate(T):
    print(f"  {t:16s} " + "  ".join(f"{lo}-{hi}: {truth[mask[:, k] & (A >= lo) & (A < hi), k].mean():.2f}" for lo, hi in bands))
