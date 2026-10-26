"""Before/after scoring: name=predictions.json ... -> per-study |delta| vs the first file, gold-58 and fold-0 macro AUC.
Gold-58 = official-label rows of reports_raw/train.csv; fold 0 = Steven-v4 ruler, confident cells (<=0.3 / >=0.7, weight>0)."""
import csv
import json
import sys

import numpy as np

R = "./"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    p, n = s[y], s[~y]
    if not len(p) or not len(n):
        return np.nan
    return ((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n))


def load(f):
    d = json.load(open(f))
    tg = d.get("targets_in_order") or d["targets"]
    pr = {x["study_id"]: x["probabilities"] for x in d["predictions"]}
    return {s: [v[tg.index(t)] for t in T] for s, v in pr.items()}


gold = {r["StudyInstanceUID"]: [int(r[t]) for t in T] for r in csv.DictReader(open(R + "reports_raw/train.csv", encoding="utf-8"))
        if all(r[t] in ("0", "1") for t in T)}
ref = json.load(open(R + "repository/data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json"))
rt = ref["targets_in_order"]
f0 = {r["study_id"]: ([r["probabilities"][rt.index(t)] for t in T], [r["weights"][rt.index(t)] for t in T]) for r in ref["labels"]}

runs = {}
for spec in sys.argv[1:]:
    name, f = spec.split("=", 1)
    runs[name] = load(f)
names = list(runs)
base = runs[names[0]]
for n in names[1:]:
    com = sorted(set(base) & set(runs[n]))
    d = np.abs(np.array([base[s] for s in com]) - np.array([runs[n][s] for s in com]))
    print(f"{n} vs {names[0]}: n={len(com)} max|d| {d.max():.5f} mean|d| {d.mean():.6f} "
          f"p99|d| {np.percentile(d, 99):.5f}; studies with any |d|>0.01: {(d.max(1) > 0.01).sum()}")


def score(P, ids, lab):
    out = []
    for j in range(len(T)):
        s, y = [], []
        for i in ids:
            r = lab(i, j)
            if r is not None:
                s.append(P[i][j]); y.append(r)
        out.append(auc(s, y))
    return np.array(out)


def gold_lab(i, j):
    return gold[i][j]


def f0_lab(i, j):
    p, w = f0[i][0][j], f0[i][1][j]
    if p is None or w <= 0 or 0.3 < p < 0.7:
        return None
    return p >= 0.7


common_gold = sorted(set(gold).intersection(*[set(r) for r in runs.values()]))
common_f0 = sorted(set(f0).intersection(*[set(r) for r in runs.values()]))
print(f"\ngold-58: {len(common_gold)} studies in every run | fold 0: {len(common_f0)} studies in every run")
print(f"{'run':14s} {'gold macro':>10s} {'fold0 macro':>11s} | gold per target")
for n, P in runs.items():
    g = score(P, common_gold, gold_lab) if common_gold else np.full(12, np.nan)
    f = score(P, common_f0, f0_lab) if common_f0 else np.full(12, np.nan)
    print(f"{n:14s} {np.nanmean(g):10.4f} {np.nanmean(f):11.4f} | " + " ".join(f"{t[:6]} {v:.3f}" for t, v in zip(T, g)))
