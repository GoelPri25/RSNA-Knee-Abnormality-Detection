"""How the public CC0 report-label sets relate to our HPO-v3.1-D2 labels and to the Steven-v4 fold-0 ruler.

Reads only label tables (data/public_labels/<owner_dataset>/, downloaded from Kaggle, CC0) plus our assets.
No gold labels, no fold-0 labels are changed. Outputs (data/public_labels/analysis/):
  agreement_spearman.csv   per target, mean Spearman of every source pair over all studies
  vs_ruler_fold0.csv       each source scored as a predictor of Steven-v4 confident cells on fold 0
  vs_hpo_train.csv         each source scored against HPO-D2 confident cells on the training folds (1-5)

Usage (container):
  podman run --rm --network=none -v "$P":/proj localhost/rsna-label-ds \
    python -I /proj/repository/scripts/public_label_agreement.py /proj/repository
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); B = R / "data/public_labels"; A = R / "data/coatnet_d96_fold0/assets"
OUT = B / "analysis"; OUT.mkdir(exist_ok=True)
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
# family: which labeller lineage a source belongs to (steven_v4 = steven_v2 + lixin; jev_hybrid = steven_v4/pilkwang)
SOURCES = {
    "steven_v2": ("stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv", "steven"),
    "steven_v4": ("stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv", "steven"),
    "lixin": ("lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv", "steven"),
    "jev_hybrid": ("riadmohamed42_jev-knee-labels/HYBRID_labels_scores.csv", "steven"),
    "pilkwang": ("pilkwang_rsna-knee-llm-labels/report_labels_v2.csv", "pilkwang"),
    "jev_final2": ("riadmohamed42_jev-knee-labels/FINAL2_labels_scores.csv", "jev"),
    "yunus": ("yunusgmsoy_rsna-knee-llm-report-labels/yunus_llm_labels.csv", "yunus"),
    "dread": ("dreaddevelopment_rsna-knee-labels/labels_llm_soft.csv", "dread"),
    "nartaa_v1_blend": ("nartaa_rsna-knee-hpo-assets/labels_v1_blend.parquet", "nartaa"),
}


def load_json_labels(p):
    d = json.loads(p.read_text()); tg = d["targets_in_order"]; idx = [tg.index(t) for t in T]
    y = {r["study_id"]: [np.nan if r["probabilities"][i] is None else r["probabilities"][i] for i in idx]
         for r in d["labels"]}
    w = {r["study_id"]: [r["weights"][i] for i in idx] for r in d["labels"]}
    return pd.DataFrame.from_dict(y, orient="index", columns=T), pd.DataFrame.from_dict(w, orient="index", columns=T)


hpo, hpo_w = load_json_labels(A / "soft_labels_hpo_v3_1_d2_structured.json")
ruler, ruler_w = load_json_labels(A / "validation_labels.json")
fold = {r["study_id"]: r["fold"] for r in csv.DictReader((A / "folds.csv").open())}
src = {"hpo_d2": hpo}
for name, (f, _) in SOURCES.items():
    d = pd.read_parquet(B / f) if f.endswith(".parquet") else pd.read_csv(B / f)
    src[name] = d.set_index("StudyInstanceUID")[T].astype(float)
FAMILY = {"hpo_d2": "hpo", **{k: v[1] for k, v in SOURCES.items()}}


def auc(s, y):
    r = pd.Series(s).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def score(ref, ref_w, ids):
    rows = []
    for name, d in src.items():
        common = [s for s in ids if s in d.index and s in ref.index]
        row = {"source": name, "family": FAMILY[name], "n": len(common)}
        for t in T:
            y, w, p = ref.loc[common, t].to_numpy(), ref_w.loc[common, t].to_numpy(), d.loc[common, t].to_numpy()
            m = (w > 0) & ((y <= .3) | (y >= .7)) & np.isfinite(p) & np.isfinite(y)
            row[t] = auc(p[m], y[m] >= .7)
        row["macro"] = np.nanmean([row[t] for t in T])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro", ascending=False)


val_ids = sorted(ruler.index); train_ids = sorted(s for s in hpo.index if fold[s] != "0")
assert len(val_ids) == 825 and len(train_ids) == 3524
v = score(ruler, ruler_w, val_ids); v.to_csv(OUT / "vs_ruler_fold0.csv", index=False)
h = score(hpo, hpo_w, train_ids); h.to_csv(OUT / "vs_hpo_train.csv", index=False)

names = list(src); common = sorted(set.intersection(*[set(d.index) for d in src.values()]))
rho = pd.DataFrame(0.0, index=names, columns=names)
for t in T:
    rho += pd.DataFrame({n: src[n].loc[common, t] for n in names}).corr("spearman") / len(T)
rho.round(3).to_csv(OUT / "agreement_spearman.csv")

pd.set_option("display.width", 250)
print("Each source as a predictor of the Steven-v4 ruler, fold 0 (825):")
print(v[["source", "family", "n", "macro"] + T].round(3).to_string(index=False))
print("\nEach source vs HPO-D2 confident cells, training folds (3524):")
print(h[["source", "family", "n", "macro"]].round(3).to_string(index=False))
print(f"\nMean Spearman over 12 targets, {len(common)} studies in all sources:")
print(rho.round(2).to_string())
