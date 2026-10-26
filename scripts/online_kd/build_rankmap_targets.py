"""E4 variant 2 targets: keep the teacher's rank order, give it label-v2's value distribution (quantile mapping).

E4's teacher part (`pubkd_targets_fullorder_rapvalues.json`) = the public 0.958 blend's rank order with Raptor's values,
which are compressed (most findings 0.4-0.5, fracture negatives ~0.45). Here, per finding, the k-th lowest teacher value
is replaced by the k-th lowest label-v2 value over the same studies (label-v2 quantile at the study's teacher rank), so
the ORDER is the teacher's and the VALUES follow label-v2's prevalence. Then E4's mix73 rule: 0.7 mapped + 0.3 label-v2
where label-v2 weight > 0. Studies without label-v2 (gold) take the label-v2 quantile only. Weights 1, as E4.
Usage: python build_rankmap_targets.py PURE.json LABELV2.json MIX73_E4.json OUT.json
"""
import hashlib
import json
import sys

import numpy as np

pure_p, lv2_p, e4_p, out_p = sys.argv[1:5]
pure = json.load(open(pure_p)); lv2 = json.load(open(lv2_p)); e4 = json.load(open(e4_p))
T = pure["targets_in_order"]; assert lv2["targets_in_order"] == T == e4["targets_in_order"]
ids = [r["study_id"] for r in pure["labels"]]
P = np.array([r["probabilities"] for r in pure["labels"]], np.float64)
L = {r["study_id"]: (np.array([np.nan if v is None else v for v in r["probabilities"]], np.float64),
                     np.array(r["weights"], np.float64)) for r in lv2["labels"]}
E4 = {r["study_id"]: np.array(r["probabilities"], np.float64) for r in e4["labels"]}
lab_ids = [s for s in ids if s in L]
Y = np.array([L[s][0] for s in lab_ids]); W = np.array([L[s][1] for s in lab_ids])

# identity check: E4's mix73 = 0.7 pure + 0.3 label-v2 where weight > 0
mix_e4 = np.array([np.where(L[s][1] > 0, 0.7 * P[ids.index(s)] + 0.3 * L[s][0], P[ids.index(s)]) for s in lab_ids[:200]])
assert np.abs(mix_e4 - np.array([E4[s] for s in lab_ids[:200]])).max() < 2e-4, "mix rule does not reproduce E4"

M = np.zeros_like(P)
for k in range(len(T)):
    # label-v2 value distribution for this finding, smoothed (Gaussian, sd 0.02, fixed seed) so that its quantile function is
    # strictly increasing: tied label-v2 values (e.g. hundreds of 0.02s) would otherwise flatten the teacher order inside them
    y = np.sort(np.clip(np.nan_to_num(Y[:, k], nan=0.5) + np.random.default_rng(k).normal(0, 0.02, len(Y)), 0.002, 0.998))
    rank = (np.argsort(np.argsort(P[:, k], kind="stable"), kind="stable") + 0.5) / len(ids)   # teacher percentile
    M[:, k] = np.quantile(y, rank, method="linear")
rows, out_rows = [], []
for i, s in enumerate(ids):
    m = M[i]
    if s in L:
        yl, wl = L[s]
        m = np.where(wl > 0, 0.7 * m + 0.3 * np.nan_to_num(yl, nan=0.5), m)
    out_rows.append({"study_id": s, "probabilities": [round(float(v), 6) for v in m], "weights": [1.0] * len(T)})
json.dump({"targets_in_order": T, "source": "E4 variant 2: teacher (full-blend) rank order with label-v2 values (quantile map), "
           "mix73 with label-v2", "labels": out_rows}, open(out_p, "w"))
sha = hashlib.sha256(open(out_p, "rb").read()).hexdigest()
print(f"wrote {out_p} ({len(out_rows)} studies) sha256 {sha}")
# report: order kept, scale changed
from scipy.stats import spearmanr
for k, t in enumerate(T):
    rho = spearmanr(P[:, k], M[:, k])[0]
    print(f"  {t:17s} order rho {rho:.4f} | teacher mean {P[:, k].mean():.3f} -> mapped {M[:, k].mean():.3f} | "
          f"label-v2 mean {np.nanmean(Y[:, k]):.3f} | E4 target mean {np.mean([E4[s][k] for s in ids]):.3f} -> new {np.mean([r['probabilities'][k] for r in out_rows]):.3f}")
