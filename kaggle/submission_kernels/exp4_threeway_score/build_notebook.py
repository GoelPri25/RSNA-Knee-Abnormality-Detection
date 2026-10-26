from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


cells = [
    md("""# Experiment 4: three-way report-label comparison

Score the saved Confidence-V2, Flight v4, and Steven v4 checkpoints on the same 29 development-gold studies. The primary metric is 12-target macro ROC AUC. Thresholded metrics at 0.5 are diagnostics; the competition metric is AUC.

This notebook reads official targets only for development IDs. It never materializes or scores evaluation-study labels. The training runs, architecture, seed, and feature cache were matched; the label source varied.
"""),
    code(r'''from pathlib import Path
import csv, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/exp4_threeway_score")
WORK.mkdir(parents=True, exist_ok=True)
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
DATASETS = {
    "confidence_v2_control": "rsna-exp4-report-label-comparison",
    "flight_v4": "rsna-exp4-alternate-label-runs-v1",
    "steven_v4": "rsna-exp4-steven-label-run-v1",
}

def exactly_one(paths, what):
    paths = list(paths)
    if len(paths) != 1: raise FileNotFoundError(f"Expected one {what}, found {paths}")
    return paths[0]

split_path = exactly_one((p for p in INPUT.rglob("report_gold_split.csv") if p.is_file()), "gold split")
split = pd.read_csv(split_path, dtype={"StudyInstanceUID": str})
dev_ids = set(split.loc[split["split"].eq("development"), "StudyInstanceUID"].astype(str))
assert len(dev_ids) == 29

# Read target values only for development IDs. Never materialize evaluation labels.
train_csv = INPUT / "competitions/rsna-knee-abnormality-detection/train.csv"
if not train_csv.is_file(): raise FileNotFoundError(train_csv)
gold = {}
with train_csv.open(newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        uid = str(row["StudyInstanceUID"])
        if uid in dev_ids: gold[uid] = np.asarray([int(row[t]) for t in TARGETS], dtype=np.int8)
assert set(gold) == dev_ids and len(gold) == 29
print("Development gold:", len(gold), "| evaluation target rows read: 0")

predictions = {}
for arm, slug in DATASETS.items():
    root = INPUT / "datasets/tumis001" / slug
    hits = list(root.rglob("development_predictions.csv")) if root.is_dir() else []
    if len(hits) != 1: raise FileNotFoundError(f"{arm}: expected one prediction CSV under {root}, found {hits}")
    df = pd.read_csv(hits[0], dtype={"StudyInstanceUID": str})
    if set(["StudyInstanceUID", *TARGETS]) - set(df.columns): raise ValueError(f"Wrong columns for {arm}")
    if df.StudyInstanceUID.duplicated().any() or set(df.StudyInstanceUID) != dev_ids:
        raise ValueError(f"{arm} predictions are not exactly the 29 development studies")
    df = df.set_index("StudyInstanceUID").loc[sorted(dev_ids)]
    p = df[TARGETS].to_numpy(dtype=np.float64)
    if p.shape != (29, 12) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"Invalid probabilities for {arm}: {p.shape}")
    predictions[arm] = p
    print(f"{arm}: loaded {p.shape} probabilities from {hits[0]}")

ids = sorted(dev_ids)
y = np.stack([gold[u] for u in ids]).astype(np.int8)
assert y.shape == (29, 12)
'''),
    code(r'''def target_aucs(y, p):
    return np.asarray([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])])

summary = {"primary_metric": "development macro ROC AUC", "development_studies": 29,
           "evaluation_studies_scored": 0, "targets": TARGETS,
           "threshold_diagnostic": "probability >= 0.5; not threshold-tuned"}
per_target = []
for arm, p in predictions.items():
    aucs = target_aucs(y, p)
    binary = (p >= 0.5).astype(np.int8)
    summary[arm] = {
        "macro_auc": float(aucs.mean()),
        "per_target_auc": {t: float(v) for t, v in zip(TARGETS, aucs)},
        "macro_accuracy_at_0_5": float(accuracy_score(y, binary)),
        "macro_precision_at_0_5": float(precision_score(y, binary, average="macro", zero_division=0)),
        "macro_recall_at_0_5": float(recall_score(y, binary, average="macro", zero_division=0)),
        "macro_f1_at_0_5": float(f1_score(y, binary, average="macro", zero_division=0)),
    }
    for t, v in zip(TARGETS, aucs): per_target.append({"arm": arm, "target": t, "auc": float(v)})

baseline = predictions["confidence_v2_control"]
comparisons = {}
rng = np.random.default_rng(2026)
for arm in ("flight_v4", "steven_v4"):
    challenger = predictions[arm]
    delta = float(target_aucs(y, challenger).mean() - target_aucs(y, baseline).mean())
    draws = []
    for _ in range(5000):
        ix = rng.integers(0, len(y), size=len(y)); ys = y[ix]
        valid = [j for j in range(12) if 0 < ys[:, j].sum() < len(ys)]
        if valid:
            draws.append(float(np.mean([roc_auc_score(ys[:, j], challenger[ix, j]) -
                                        roc_auc_score(ys[:, j], baseline[ix, j]) for j in valid])))
    ci = np.quantile(draws, [0.05, 0.95]).tolist() if draws else [None, None]
    comparisons[arm] = {"macro_auc_delta_vs_control": delta,
                        "paired_study_bootstrap_90pct_interval": ci,
                        "valid_replicates": len(draws)}

scores = pd.DataFrame([{"arm": arm, "macro_auc": summary[arm]["macro_auc"],
                        "macro_accuracy_at_0_5": summary[arm]["macro_accuracy_at_0_5"],
                        "macro_precision_at_0_5": summary[arm]["macro_precision_at_0_5"],
                        "macro_recall_at_0_5": summary[arm]["macro_recall_at_0_5"],
                        "macro_f1_at_0_5": summary[arm]["macro_f1_at_0_5"]}
                       for arm in predictions]).sort_values("macro_auc", ascending=False)
summary["paired_bootstrap_vs_control"] = comparisons
summary["candidate_selected_by_dev_macro_auc"] = str(scores.iloc[0].arm)
scores.to_csv(WORK / "scores.csv", index=False)
pd.DataFrame(per_target).pivot(index="target", columns="arm", values="auc").to_csv(WORK / "per_target_auc.csv")
(WORK / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("\nThree-way development comparison (AUC is primary):")
display(scores.round(4))
print("Paired uncertainty vs Confidence-V2:")
print(json.dumps(comparisons, indent=2))
print("Selected point-estimate candidate:", summary["candidate_selected_by_dev_macro_auc"])
print("Saved:", WORK)
'''),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"},
                   "kaggle": {"accelerator": "none", "dataSources": [], "dockerImageVersionId": 28755,
                              "isInternetEnabled": False, "language": "python", "sourceType": "notebook", "isGpuEnabled": False}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "score_three_label_sources.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
metadata = {"id": "tumis001/rsna-exp4-threeway-score", "title": "RSNA Experiment 4 Three-Way Label Source Score",
            "code_file": "score_three_label_sources.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["tumis001/rsna-exp4-report-label-comparison",
                                "tumis001/rsna-exp4-alternate-label-runs-v1",
                                "tumis001/rsna-exp4-steven-label-run-v1"],
            "kernel_sources": ["tumis001/report-label-audit"],
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Wrote scoring notebook under", OUT)
