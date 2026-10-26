"""Score Exp 1d (evaluation crop 320 / 352 / 384, inference only) against the pre-registered rule.

Primary: per model, 384 vs the same model's 320 on Steven-v4 fold-0 confident cells. PASS if delta >= +0.002 with
>= 90% of 2,000 paired bootstraps better, for both models. Per target and 352 are secondary.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_eval_crop.py /proj
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
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7


def load(p):
    m = {r["study_id"]: r["probabilities"] for r in json.loads(p.read_text())["predictions"]}
    return np.array([m[s] for s in ids], float)


def auc(s, yy):
    r = pd.Series(s).rank().to_numpy(); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per(S, idx):
    return np.array([auc(S[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(len(T))])


rng = np.random.default_rng(0); n = len(ids); full = np.arange(n)
boots = [rng.integers(0, n, n) for _ in range(2000)]
verdicts, pt = [], {}
for model in ("coatnet_1b", "r50_1c"):
    S = {res: load(D / f"coatnet_d96_fold0/eval_crop/{model}_val_{res}.json") for res in (320, 352, 384)}
    for res in S:
        pt[f"{model}_{res}"] = per(S[res], full)
    for res in (352, 384):
        d = np.array([per(S[res], b).mean() - per(S[320], b).mean() for b in boots])
        delta = pt[f"{model}_{res}"].mean() - pt[f"{model}_320"].mean()
        print(f"{model} {res} vs 320: macro {pt[f'{model}_{res}'].mean():.4f} vs {pt[f'{model}_320'].mean():.4f}, "
              f"delta {delta:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}], {(d > 0).mean():.1%} better")
        if res == 384:
            verdicts.append(delta >= 0.002 and (d > 0).mean() >= 0.90)
print("\nPer target:")
print(pd.DataFrame(pt, index=T).round(4).to_string())
print(f"\nPRE-REGISTERED (384 vs 320, both models, delta >= +0.002, >= 90% better): {'PASS' if all(verdicts) else 'FAIL'}")
