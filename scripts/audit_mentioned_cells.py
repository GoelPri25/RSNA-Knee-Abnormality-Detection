"""Fold-0 Steven-v4 AUC restricted to cells the report actually mentions.

Steven-v4 gives silent cells (finding not mentioned) a confident-negative value, so a model that finds an
unreported finding is penalised on the standard ruler. This readout keeps only cells where pilkwang's
report verdict is YES or NO (not UNK), using the same confident-cell rule (<= 0.3 / >= 0.7).
Gold-74 (JEV) is not used: its 16 extra studies are report-verified and 13 sit in our training folds.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_mentioned_cells.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
pk = pd.read_csv(D / "public_labels/pilkwang_rsna-knee-llm-labels/report_labels_v2.csv").set_index("StudyInstanceUID")
mentioned = np.array([[pk.loc[s, t + "__verdict"] != "UNK" if s in pk.index else True for t in T] for s in ids])
conf = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
VIEWS = {"all_confident": conf, "mentioned_only": conf & mentioned}

MODELS = {
    "d2_224": "r50_tta/kaggle_output/d2_ep15__none.json",
    "d2_224_tta9": "r50_tta/kaggle_output/d2_ep15__tta9.json",
    "hpo_v1": "r50_tta/kaggle_output/hpo_v1_ep17__none.json",
    "r50_336": "highres336/results/oof_predictions.json",
    "effnet": "efficientnet_v31_d2/results/oof_predictions.json",
    "coatnet_exp1": "coatnet_d96_fold0/run/val_swa.json",
    "coatnet_exp1b": "coatnet_d96_fold0/run_continue/val_swa.json",
}


def load(p):
    d = json.loads((D / p).read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
    tg = d.get("targets_in_order", d.get("targets", T))
    return np.array([m[s] for s in ids], float)[:, [list(tg).index(t) for t in T]]


def auc(s, yy):
    r = pd.Series(s).rank().to_numpy(); n1 = yy.sum(); n0 = len(yy) - n1
    return np.nan if n1 == 0 or n0 == 0 else (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per(S, idx, m):
    return np.array([auc(S[idx, k][m[idx, k]], truth[idx, k][m[idx, k]]) for k in range(len(T))])


P = {k: load(v) for k, v in MODELS.items()}
P["quad"] = np.mean([P[k] for k in ("d2_224", "hpo_v1", "r50_336", "effnet")], 0)
full = np.arange(len(ids)); rng = np.random.default_rng(0)
boots = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]

print("Scored cells per target (all confident -> mentioned only):")
print(" ".join(f"{t[:8]}:{conf[:, k].sum()}->{(conf & mentioned)[:, k].sum()}" for k, t in enumerate(T)))
rows, pt = [], {}
for name, S in P.items():
    row = {"model": name}
    for v, m in VIEWS.items():
        p = per(S, full, m); row[v] = np.nanmean(p); pt[(name, v)] = p
    rows.append(row)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\nPer target, mentioned only:")
print(pd.DataFrame({n: pt[(n, "mentioned_only")] for n in P}, index=T).round(4).to_string())
for a, b in (("coatnet_exp1b", "d2_224"), ("coatnet_exp1", "d2_224")):
    for v, m in VIEWS.items():
        d = np.array([np.nanmean(per(P[a], i, m)) - np.nanmean(per(P[b], i, m)) for i in boots])
        print(f"{a} vs {b} [{v}]: {np.nanmean(pt[(a, v)]) - np.nanmean(pt[(b, v)]):+.4f} "
              f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}], {(d > 0).mean():.1%} better")
