"""Quota prior for L4C-LOC's zero-training baseline Q (docs/l4c_loc_spec.md): from MS1's crop table on the training folds
(all folds except 0; fold 0 never read), per cropped finding
  quota   how many of the TOP=4 crops go to each series slot (SagF, SagNF, CorF, CorNF, Ax) = MS1's slot mix x 4, rounded by
          largest remainder
  median  per (finding, slot) the median MS1 crop centre (row, col) in the 384 px frame, after predict_study's clip
Out: JSON (quota [10][5], median [10][5][2], mix [10][5], n_studies).
Usage: python3 -I scripts/build_l4c_quota_prior.py <repo root> <out.json>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
A = ROOT / "data/kaggle_datasets/rsna-v4-l4copy-fold0"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]
BOX = {"ACL": 192, "MCL": 192, "Medial Meniscus": 128, "Lateral Meniscus": 192, "Medial OA": 192, "Lateral OA": 192,
       "PF OA": 192, "Baker's": 192, "Contusion": 128, "Fracture": 192}
CF = [k for k, t in enumerate(TARGETS) if t in BOX]; IMG, TOP = 384, 4
BOUNDS = np.cumsum([0, 26, 22, 18, 12, 18])
fold = {r["study_id"]: r["fold"] for r in csv.DictReader(open(ROOT / "data/synovitis_weight_candidate_seed2026/assets/folds.csv"))}
masks = json.load(open(ROOT / "data/coatnet_d96_fold0/l4copy_diag/d96_masks.json"))


def centres(m):                                      # v4_submission.uniform_centres
    v = np.where(np.array([c == "1" for c in m]))[0]
    if len(v) < 3:
        v = np.arange(3)
    lo, hi = int(v.min()), int(v.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, 94))])
    return np.clip(cs, 1, 94)


slots = [[] for _ in CF]; pts = [[[] for _ in range(5)] for _ in CF]; n = 0
for p in sorted(A.glob("cp_*.npz")):
    z = np.load(p)
    for s, cen in zip(z["ids"], z["centres"]):
        s = str(s)
        if fold.get(s, "0") == "0" or s not in masks:
            continue
        n += 1; cs = centres(masks[s])
        for j, f in enumerate(CF):
            hb = BOX[TARGETS[f]] // 2
            for w, r, q in cen[f]:
                sl = int(np.searchsorted(BOUNDS, cs[w], side="right") - 1)
                slots[j].append(sl); pts[j][sl].append((np.clip(r, hb, IMG - hb), np.clip(q, hb, IMG - hb)))
mix = np.array([np.bincount(slots[j], minlength=5) / len(slots[j]) for j in range(len(CF))])
quota = np.floor(mix * TOP).astype(int)
for j in range(len(CF)):
    rem = mix[j] * TOP - quota[j]
    for k in np.argsort(-rem)[:TOP - quota[j].sum()]:
        quota[j, k] += 1
med = np.array([[np.median(pts[j][k], 0) if pts[j][k] else [IMG / 2, IMG / 2] for k in range(5)] for j in range(len(CF))])
assert (quota.sum(1) == TOP).all()
json.dump({"quota": quota.tolist(), "median": np.round(med).astype(int).tolist(), "mix": np.round(mix, 4).tolist(),
           "n_studies": n, "slots": ["SagF", "SagNF", "CorF", "CorNF", "Ax"], "findings": [TARGETS[f] for f in CF]},
          open(OUT, "w"), indent=0)
print("studies", n)
for j, f in enumerate(CF):
    print(f"{TARGETS[f]:16s} mix " + " ".join(f"{100 * x:3.0f}" for x in mix[j]) + "  quota " + str(quota[j].tolist())
          + "  median(r,c) " + " ".join(f"{tuple(med[j, k].astype(int))}" for k in range(5) if quota[j, k]))
