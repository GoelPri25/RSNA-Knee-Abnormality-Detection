"""Is 1c limited by its training labels? On fold 0 (never trained on), treat each training-label set's own soft
probability as a 'predictor' of the Steven-v4 ruler and compare its AUC with 1c's, per target. Also split 1c's
pairwise ranking errors by whether D2 agreed with the ruler on that cell.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds python -I /proj/repository/scripts/audit_label_ceiling.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7


def lab(p):
    d = json.loads(Path(p).read_text()); m = {r["study_id"]: r for r in d["labels"]}
    P = np.array([[.5 if v is None else v for v in m[s]["probabilities"]] for s in ids])
    W = np.array([m[s]["weights"] for s in ids]); return P, W


def pred(p):
    m = {r["study_id"]: r["probabilities"] for r in json.loads(Path(p).read_text())["predictions"]}
    return np.array([m[s] for s in ids])


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


d2, d2w = lab(C / "assets/soft_labels_hpo_v3_1_d2_structured.json")
tm, _ = lab(C / "assets/soft_labels_final_per_target_any_model.json")
m1c = pred(C / "run_r50/val_swa.json"); m1e = pred(C / "run_r50_pertarget/val_swa.json")
rows = {}
for k, t in enumerate(T):
    mk = mask[:, k]; tt = truth[mk, k]
    conf = (d2[:, k] <= .3) | (d2[:, k] >= .7)
    agree = conf & ((d2[:, k] >= .7) == truth[:, k])
    # pairwise errors of 1c: (pos, neg) pairs ranked wrong, attributed to the pair's cells' D2 status
    pos = np.where(mk & truth[:, k])[0]; neg = np.where(mk & ~truth[:, k])[0]
    wrong = m1c[pos, k][:, None] <= m1c[neg, k][None, :]
    bad = ~agree[pos][:, None] | ~agree[neg][None, :]
    rows[t] = {"n_pos": len(pos), "D2_label": auc(d2[mk, k], tt), "team_label": auc(tm[mk, k], tt),
               "1c": auc(m1c[mk, k], tt), "1e": auc(m1e[mk, k], tt),
               "D2_agree%": agree[mk].mean() * 100, "1c_AUC_on_D2agree": auc(m1c[mk & agree, k], truth[mk & agree, k]),
               "err_share_D2_not_agree%": (wrong & bad).sum() / max(wrong.sum(), 1) * 100,
               "pair_share_D2_not_agree%": bad.mean() * 100}
df = pd.DataFrame(rows).T
df.loc["MACRO"] = df.mean()
pd.set_option("display.width", 200)
print(df.round(3).to_string())
