"""Image model vs report readers on the 58 host gold studies, per target (evaluation only).

Question: per target, is the score limited by the training labels or by the images? Compares gold AUC of saved image
models (predictions from gold_image_model_predictions.py) with the v3.1 report-reader blend and rules on the same
studies. Gold is used for scoring only; nothing here feeds a label build or a model choice.

Readouts per target:
  - AUC of each model, a rank-average of the three ResNet50 models, the report blend, and the v3.1 rules;
  - paired bootstrap of (primary model - report blend), primary = r50_hpo_v31_d2_fold0 (the 0.925 public run);
  - recovery: gold positives the reports miss (blend < 0.5) ranked against all gold negatives, and gold negatives the
    reports over-call (blend >= 0.5) ranked against all gold positives, by the primary model (pooled within-target AUC);
  - complementarity: AUC of rank-average(primary model, report blend). Report text is not available at test time
    (public test.csv has StudyInstanceUID only); this only shows whether report evidence holds information the
    image model has not learned.

Container:
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_gold_image_model.py /proj /out
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(R / "label_rules/v3_1"))
import build_hpo_v3_1 as b  # noqa: E402

AUD = R / "repository/data/image_detail_audit_2026-10-07"
PRED = OUT / "gold_image_model_predictions.json"
NEGP, PRIMARY = 0.04, "r50_hpo_v31_d2_fold0"
R50 = ["r50_hpo_v31_d2_fold0", "r50_hpo_blend_fold0", "r50_hpo_v31_trainval"]

P = json.loads(PRED.read_text())
T = P["targets_in_order"]; ids = P["gold_study_ids"]
tr = pd.read_csv(R / "reports_raw/train.csv")
g = tr[tr[T].notna().all(1)].set_index("StudyInstanceUID")
assert sorted(g.index) == sorted(ids)
Y = g.loc[ids, T].astype(int)
M = {k: pd.DataFrame(v["gold_probabilities"], index=ids, columns=T) for k, v in P["models"].items()}

# report blend exactly as audit_gold_report_ceiling.py (v3.1 rules + LLM readers, no HPO-v1, all-silent -> NEG)
rules = pd.read_csv(AUD / "gold_rule_states_v31.csv", index_col=0).loc[ids]
Rd, _ = b.readers(R, AUD / "gold_rule_states_v31.csv", pd.Index(ids))
RR = dict(Rd); RR["rules"] = rules[T].replace({"SIL": np.nan})
votes = sum(RR[k][T].apply(lambda c: c.map(b.PROB)).fillna(0) for k in RR)
n = sum(RR[k][T].notna().astype(int) for k in RR)
BL = (votes / n.replace(0, np.nan)).fillna(NEGP).loc[ids]
RU = rules[T].apply(lambda c: c.map(lambda s: b.PROB.get(s, NEGP) if isinstance(s, str) else NEGP)).astype(float)


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


pct = lambda D: D.rank(pct=True)
ENS = sum(pct(M[k]) for k in R50) / len(R50)
COMB = (pct(M[PRIMARY]) + pct(BL)) / 2
cols = {"gold_pos": Y.sum(), "report_blend": {t: auc(BL[t], Y[t]) for t in T}, "rules_v31": {t: auc(RU[t], Y[t]) for t in T}}
for k in M:
    cols[k] = {t: auc(M[k][t], Y[t]) for t in T}
cols["ens_r50x3_rank"] = {t: auc(ENS[t], Y[t]) for t in T}
cols["primary_plus_report_rank"] = {t: auc(COMB[t], Y[t]) for t in T}
A = pd.DataFrame(cols).loc[T]
fold0 = pd.read_csv(AUD / "fold0_auc_single_and_ensemble.csv", index_col=0)
A["primary_fold0_vs_stevenv4"] = fold0.loc[T, "resnet_hpo_v31_d2"]
A.loc["MACRO"] = A.mean(); A.at["MACRO", "gold_pos"] = np.nan
A.round(4).to_csv(OUT / "gold_auc_image_vs_report.csv", index_label="target")

rng = np.random.default_rng(0); B = 2000
idx = [rng.integers(0, 58, 58) for _ in range(B)]
boot = []
for t in T + ["MACRO"]:
    d, c = [], []
    for i in idx:
        tt = T if t == "MACRO" else [t]
        a = np.nanmean([auc(M[PRIMARY][u].iloc[i], Y[u].iloc[i]) for u in tt])
        r = np.nanmean([auc(BL[u].iloc[i], Y[u].iloc[i]) for u in tt])
        cb = np.nanmean([auc(COMB[u].iloc[i], Y[u].iloc[i]) for u in tt])
        d.append(a - r); c.append(cb - a)
    d, c = np.array(d), np.array(c); d, c = d[np.isfinite(d)], c[np.isfinite(c)]
    boot.append({"target": t, "model_minus_report": round(float(A.at[t, PRIMARY] - A.at[t, "report_blend"]), 4),
                 "mmr_ci_low": round(float(np.percentile(d, 2.5)), 4), "mmr_ci_high": round(float(np.percentile(d, 97.5)), 4),
                 "combined_minus_model": round(float(A.at[t, "primary_plus_report_rank"] - A.at[t, PRIMARY]), 4),
                 "cmm_ci_low": round(float(np.percentile(c, 2.5)), 4), "cmm_ci_high": round(float(np.percentile(c, 97.5)), 4)})
BT = pd.DataFrame(boot); BT.to_csv(OUT / "gold_model_vs_report_bootstrap.csv", index=False)

rec = []
for t in T:
    s, y, bl = M[PRIMARY][t], Y[t].astype(bool), BL[t]
    miss, over = y & (bl < 0.5), ~y & (bl >= 0.5)
    rec.append({"target": t, "gold_pos": int(y.sum()), "report_missed_pos": int(miss.sum()), "gold_neg": int((~y).sum()),
                "report_overcalled_neg": int(over.sum()),
                # within-target AUC of the subset vs the opposite class (per pair: P(score_pos > score_neg))
                "missed_pos_vs_all_neg_pairs_won": float(sum((s[miss].to_numpy()[:, None] > s[~y].to_numpy()[None]).ravel())),
                "missed_pos_pairs": int(miss.sum() * (~y).sum()),
                "all_pos_vs_overcalled_neg_pairs_won": float(sum((s[y].to_numpy()[:, None] > s[over].to_numpy()[None]).ravel())),
                "overcalled_neg_pairs": int(y.sum() * over.sum()),
                "report_all_pos_vs_overcalled_neg_auc": auc(np.r_[bl[y], bl[over]], np.r_[np.ones(y.sum()), np.zeros(over.sum())])})
RC = pd.DataFrame(rec)
RC["model_auc_missed_pos_vs_neg"] = RC.missed_pos_vs_all_neg_pairs_won / RC.missed_pos_pairs.replace(0, np.nan)
RC["model_auc_pos_vs_overcalled_neg"] = RC.all_pos_vs_overcalled_neg_pairs_won / RC.overcalled_neg_pairs.replace(0, np.nan)
pool = {"model_auc_missed_pos_vs_neg_pooled": RC.missed_pos_vs_all_neg_pairs_won.sum() / RC.missed_pos_pairs.sum(),
        "model_auc_pos_vs_overcalled_neg_pooled": RC.all_pos_vs_overcalled_neg_pairs_won.sum() / RC.overcalled_neg_pairs.sum()}
RC.round(4).to_csv(OUT / "gold_model_recovery_of_report_errors.csv", index=False)

fid = {k: v.get("fidelity_32_fold0") for k, v in P["models"].items()}
meta = {"primary": PRIMARY, "r50_ensemble": R50, "bootstrap": {"draws": B, "seed": 0}, "pooled_recovery": pool,
        "fidelity_32_fold0": fid, "checkpoints": {k: {x: v[x] for x in ("checkpoint", "sha256", "epoch", "config_version")}
                                                   for k, v in P["models"].items()},
        "predictions_sha256": hashlib.sha256(PRED.read_bytes()).hexdigest()}
(OUT / "gold_image_model_meta.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
pd.set_option("display.width", 250)
print(A.round(3).to_string()); print(BT.to_string(index=False))
print(RC[["target", "gold_pos", "report_missed_pos", "model_auc_missed_pos_vs_neg", "gold_neg", "report_overcalled_neg",
          "report_all_pos_vs_overcalled_neg_auc", "model_auc_pos_vs_overcalled_neg"]].round(3).to_string(index=False))
print(json.dumps({"pooled": pool, "fidelity": fid}, default=float, indent=1))
