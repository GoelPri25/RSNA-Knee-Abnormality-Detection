from __future__ import annotations

import json
from pathlib import Path


OUT = Path(__file__).parent
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(True),
    }


cells = [
    md("""# RSNA report-label source audit (CPU-only)

This notebook compares available report-derived label tables with the **29-study development gold split**. It does not read DICOMs or report text, does not train a model, and does not calculate any metric on the locked 29-study evaluation split. It reports source coverage because some extractors abstain or omit targets.

The internal reference is the already-saved conservative-v2 report extraction on development gold, passed through the same confidence/severity mapping used by Confidence V2. It is a reference for our report-to-label policy, not a claim that those 29 rows were training labels. The locked-evaluation extraction file is intentionally never opened.

Candidate sources include public report-label datasets. The teammate's blended-soft-label dataset is not attached because Kaggle currently returns HTTP 403 for its file manifest; it must be made accessible before it can be included. Sources with exact gold copies on the development split are flagged as contaminated and must not be treated as independent evidence.
"""),
    code(r'''from pathlib import Path
import json, re, unicodedata
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
INPUT = Path("/kaggle/input")

def competition_root():
    candidates = [p for p in [INPUT / "competitions" / "rsna-knee-abnormality-detection", INPUT / "rsna-knee-abnormality-detection"] if (p / "train.csv").is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one competition input; found {candidates}")
    return candidates[0]

def locate_input_file(basename, root_hint=None):
    roots = [Path(root_hint)] if root_hint and Path(root_hint).exists() else [p for p in INPUT.iterdir() if p.name not in {"competitions"}]
    found = []
    for root in roots:
        for p in root.rglob(basename):
            if p.is_file(): found.append(p)
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {basename}; found {found}")
    return found[0]

DATA_ROOT = competition_root()
train = pd.read_csv(DATA_ROOT / "train.csv", dtype={"StudyInstanceUID": str})
split_path = locate_input_file("report_gold_split.csv", "/kaggle/input/kernels/tumis001/report-label-audit")
split = pd.read_csv(split_path, dtype={"StudyInstanceUID": str})
dev_ids = set(split.loc[split["split"].eq("development"), "StudyInstanceUID"].astype(str))
eval_ids = set(split.loc[split["split"].eq("evaluation"), "StudyInstanceUID"].astype(str))
assert len(dev_ids) == 29 and len(eval_ids) == 29 and dev_ids.isdisjoint(eval_ids)

# Select development rows by UID before accessing target columns. No evaluation labels are scored or summarized.
dev_gold = train.loc[train["StudyInstanceUID"].astype(str).isin(dev_ids), ["StudyInstanceUID", *TARGETS]].copy()
assert len(dev_gold) == 29 and dev_gold[TARGETS].notna().all().all()
dev_gold["StudyInstanceUID"] = dev_gold["StudyInstanceUID"].astype(str)
print("CPU-only audit | development rows:", len(dev_gold), "| locked evaluation rows excluded:", len(eval_ids))
print("Data source:", DATA_ROOT)
print("No report text or DICOM paths are read.")'''),
    code(r'''def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", s)

target_norm = {norm(t): t for t in TARGETS}
target_norm.update({"bakerscyst": "Baker's", "poplitealcyst": "Baker's", "bonebruise": "Contusion", "acltear": "ACL", "mcltear": "MCL", "medialmeniscustear": "Medial Meniscus", "lateralmeniscustear": "Lateral Meniscus", "effusion": "Effusion", "joint_effusion": "Effusion", "synovitis": "Synovitis", "fracture": "Fracture"})
UID_NAMES = {"studyinstanceuid", "studyuid", "studyid", "uid", "id"}
VALUE_NAMES = {"probability", "prob", "score", "confidence", "label", "value", "prediction", "pred", "softlabel", "softtarget"}

def target_for_column(col):
    n = norm(col)
    for prefix in ("probability", "prob", "score", "confidence", "label", "softlabel", "softtarget", "prediction", "pred"):
        if n.startswith(prefix): n = n[len(prefix):]
        if n.endswith(prefix): n = n[:-len(prefix)]
    return target_norm.get(n)

def value_to_float(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float, np.number)):
        x = float(v)
        return x if 0.0 <= x <= 1.0 else np.nan
    s = str(v).strip().lower()
    mapped = {"positive": 1., "present": 1., "yes": 1., "true": 1., "negative": 0., "absent": 0., "no": 0., "false": 0., "uncertain": .5, "unknown": np.nan, "not_mentioned": np.nan, "not mentioned": np.nan}
    if s in mapped: return mapped[s]
    try:
        x = float(s)
        return x if 0.0 <= x <= 1.0 else np.nan
    except Exception:
        return np.nan

def normalize_candidate_frame(frame):
    cols = list(frame.columns)
    uid_col = next((c for c in cols if norm(c) in UID_NAMES), None)
    if uid_col is None: return None
    wide = {target_for_column(c): c for c in cols if target_for_column(c)}
    if len(wide) >= 8:
        out = pd.DataFrame({"StudyInstanceUID": frame[uid_col].astype(str)})
        for target, col in wide.items(): out[target] = frame[col].map(value_to_float)
        return out
    target_col = next((c for c in cols if norm(c) in {"target", "finding", "abnormality", "condition"}), None)
    value_col = next((c for c in cols if norm(c) in VALUE_NAMES), None)
    if target_col and value_col:
        tmp = frame[[uid_col, target_col, value_col]].copy()
        tmp["target"] = tmp[target_col].map(lambda x: target_norm.get(norm(x)))
        tmp["value"] = tmp[value_col].map(value_to_float)
        tmp = tmp.dropna(subset=["target"])
        if tmp.empty: return None
        out = tmp.pivot_table(index=uid_col, columns="target", values="value", aggfunc="first").reset_index()
        return out.rename(columns={uid_col: "StudyInstanceUID"})
    return None

def json_records(obj):
    """Extract row-like dicts from common JSON list/dict layouts without assuming the label schema."""
    rows = []
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj): rows.extend(obj)
        else:
            for x in obj: rows.extend(json_records(x))
    elif isinstance(obj, dict):
        if any(norm(k) in UID_NAMES for k in obj): rows.append(obj)
        for k, v in obj.items():
            if isinstance(v, (list, dict)) and norm(k) not in UID_NAMES: rows.extend(json_records(v))
    return rows

def read_candidate(path):
    try:
        if path.suffix.lower() == ".csv": return normalize_candidate_frame(pd.read_csv(path, low_memory=False))
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            rows = json_records(obj)
            if rows: return normalize_candidate_frame(pd.DataFrame(rows))
    except Exception as e:
        print("Could not parse", path.name, type(e).__name__)
    return None

def candidate_files():
    # Crawl only attached label-dataset roots; skip competition DICOM tree and notebook runtime folders.
    excluded = {"competitions"}
    for root in INPUT.iterdir():
        if root.name in excluded or root.name == "kernels": continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".csv", ".json"} and p.name not in {"train.csv", "test.csv", "sample_submission.csv", "train_series.csv", "test_series.csv", "gold74_eval_set.csv"}:
                yield p

sources = []
for path in candidate_files():
    table = read_candidate(path)
    if table is None or "StudyInstanceUID" not in table or sum(t in table.columns for t in TARGETS) < 8: continue
    table = table.drop_duplicates("StudyInstanceUID", keep="last")
    sources.append((str(path), table))
print("Candidate label tables parsed:", len(sources))
for name, table in sources: print("-", name, table.shape, "targets:", sum(t in table.columns for t in TARGETS))'''),
    code(r'''# Add our saved conservative-v2 development extraction as a reference. The evaluation JSONL is never opened.
v2_path = locate_input_file("development_report_extractions_v2.jsonl", "/kaggle/input/kernels/tumis001/report-label-audit")
PRESENT = {"severe": .97, "moderate_to_severe": .95, "moderate": .90, "mild_to_moderate": .82, "mild": .70, "unspecified": .88}
internal_rows = []
for line in v2_path.read_text(encoding="utf-8").splitlines():
    rec = json.loads(line)
    if rec.get("error") or rec.get("study_uid") not in dev_ids or not isinstance(rec.get("predictions"), dict): continue
    row = {"StudyInstanceUID": str(rec["study_uid"])}
    for target in TARGETS:
        pred = rec["predictions"].get(target, {})
        status = str(pred.get("status", "")).lower()
        severity = str(pred.get("severity", "unspecified")).lower().replace(" ", "_")
        if status == "absent": row[target] = .05
        elif status == "present": row[target] = PRESENT.get(severity, .88)
        else: row[target] = np.nan  # uncertain / unmentioned remain masked, matching our policy
    internal_rows.append(row)
internal_v2 = pd.DataFrame(internal_rows)
if len(internal_v2): sources.insert(0, ("INTERNAL: conservative-v2 dev extraction + Confidence-V2 severity mapping", internal_v2))
else: print("Internal reference unavailable: saved dev extraction file had no usable records.")

def score_source(name, predictions):
    available_targets = [t for t in TARGETS if t in predictions.columns]
    if not available_targets: return []
    gold_part = dev_gold[["StudyInstanceUID", *available_targets]].rename(columns={t: t + "__gold" for t in available_targets})
    pred_part = predictions[["StudyInstanceUID", *available_targets]].rename(columns={t: t + "__pred" for t in available_targets})
    merged = gold_part.merge(pred_part, on="StudyInstanceUID", how="inner")
    rows = []
    for target in available_targets:
        y = pd.to_numeric(merged[target + "__gold"], errors="coerce")
        p = pd.to_numeric(merged[target + "__pred"], errors="coerce")
        ok = y.notna() & p.notna() & p.between(0, 1)
        y, p = y[ok].astype(int), p[ok].astype(float)
        if len(y) == 0: continue
        auc = roc_auc_score(y, p) if y.nunique() == 2 else np.nan
        hard = (p >= .5).astype(int)
        rows.append({"source": name, "target": target, "n": len(y), "coverage": len(y) / len(dev_gold), "auc": auc,
                     "brier": brier_score_loss(y, p), "accuracy": accuracy_score(y, hard),
                     "balanced_accuracy": balanced_accuracy_score(y, hard), "precision": precision_score(y, hard, zero_division=0),
                     "recall": recall_score(y, hard, zero_division=0), "f1": f1_score(y, hard, zero_division=0),
                     "exact_binary_copy": bool(np.all(np.isclose(p.to_numpy(), y.to_numpy(), atol=0, rtol=0)))})
    return rows

target_rows = []
for name, table in sources: target_rows.extend(score_source(name, table))
per_target = pd.DataFrame(target_rows)
summary = (per_target.groupby("source", as_index=False)
          .agg(macro_auc=("auc", "mean"), macro_brier=("brier", "mean"), macro_accuracy=("accuracy", "mean"),
               macro_balanced_accuracy=("balanced_accuracy", "mean"), macro_precision=("precision", "mean"),
               macro_recall=("recall", "mean"), macro_f1=("f1", "mean"), mean_coverage=("coverage", "mean"),
               targets_scored=("target", "nunique"), all_targets_exact_copy=("exact_binary_copy", "all")))
summary["leakage_warning"] = np.where(summary["all_targets_exact_copy"], "EXACT COPY ON DEVELOPMENT — EXCLUDE", "")
summary = summary.sort_values(["macro_auc", "macro_brier"], ascending=[False, True])
internal_name = "INTERNAL: conservative-v2 dev extraction + Confidence-V2 severity mapping"
internal_table = next((table for name, table in sources if name == internal_name), None)
common_target_rows, common_summary_rows = [], []
if internal_table is not None:
    for name, table in sources:
        if name == internal_name: continue
        shared_targets = [t for t in TARGETS if t in table.columns and t in internal_table.columns]
        per_target_deltas = []
        for target in shared_targets:
            trio = dev_gold[["StudyInstanceUID", target]].rename(columns={target: "gold"})
            trio = trio.merge(table[["StudyInstanceUID", target]].rename(columns={target: "candidate"}), on="StudyInstanceUID", how="inner")
            trio = trio.merge(internal_table[["StudyInstanceUID", target]].rename(columns={target: "internal"}), on="StudyInstanceUID", how="inner")
            for col in ("gold", "candidate", "internal"): trio[col] = pd.to_numeric(trio[col], errors="coerce")
            trio = trio.dropna(subset=["gold", "candidate", "internal"])
            trio = trio[trio["candidate"].between(0, 1) & trio["internal"].between(0, 1)]
            if len(trio) == 0 or trio["gold"].nunique() < 2: continue
            ca = roc_auc_score(trio["gold"], trio["candidate"])
            ia = roc_auc_score(trio["gold"], trio["internal"])
            common_target_rows.append({"source": name, "target": target, "n_common": len(trio), "coverage_common": len(trio) / len(dev_gold), "candidate_auc": ca, "internal_auc_same_rows": ia, "delta_auc": ca - ia,
                                       "candidate_brier": brier_score_loss(trio["gold"], trio["candidate"]), "internal_brier_same_rows": brier_score_loss(trio["gold"], trio["internal"])})
            per_target_deltas.append(ca - ia)
        if per_target_deltas:
            common_summary_rows.append({"source": name, "macro_auc_on_internal_common_rows": float(np.mean([r["candidate_auc"] for r in common_target_rows if r["source"] == name])),
                                        "internal_macro_auc_same_rows": float(np.mean([r["internal_auc_same_rows"] for r in common_target_rows if r["source"] == name])),
                                        "paired_macro_auc_delta": float(np.mean(per_target_deltas)), "targets_with_both_classes": len(per_target_deltas),
                                        "mean_common_coverage": float(np.mean([r["coverage_common"] for r in common_target_rows if r["source"] == name]))})
common_summary = pd.DataFrame(common_summary_rows).sort_values("paired_macro_auc_delta", ascending=False) if common_summary_rows else pd.DataFrame()
print("Development-only source comparison (29 studies; 12 targets; do not over-interpret small-n differences):")
display(summary.round(3))
print("\nFair comparison against our internal reference on the exact same per-target studies (masked internal decisions excluded):")
display(common_summary.round(3))
print("\nPer-target details:")
display(per_target.sort_values(["source", "target"]).round(3))
OUT = Path("/kaggle/working")
summary.to_csv(OUT / "report_label_source_summary.csv", index=False)
per_target.to_csv(OUT / "report_label_source_per_target.csv", index=False)
pd.DataFrame(common_summary_rows).to_csv(OUT / "report_label_source_common_summary.csv", index=False)
pd.DataFrame(common_target_rows).to_csv(OUT / "report_label_source_common_per_target.csv", index=False)
print("Saved CSV summaries under /kaggle/working. Evaluation split remains unscored.")'''),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13"},
        "kaggle": {"accelerator": "none", "dataSources": [], "dockerImageVersionId": 28755,
                   "isInternetEnabled": False, "language": "python", "sourceType": "notebook", "isGpuEnabled": False},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(OUT / "report_label_source_audit.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))

metadata = {
    "id": "tumis001/rsna-report-label-source-audit",
    "title": "RSNA Report Label Source Audit",
    "code_file": "report_label_source_audit.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["cpu", "label-audit", "validation"],
    "dataset_sources": [
        "flight0234/rsna-knee-hybrid-report-labels",
        "stevenleehans/rsna-knee-llm-report-labels",
        "pilkwang/rsna-knee-llm-labels",
        "lixin73/rsna-knee-llm-report-labels-sol56",
        "yunusgmsoy/rsna-knee-llm-report-labels",
        "yehezkielhaganta/psuedo-labeling-knee",
        "vmohitrao/rsna-knee-report-labels",
        "riadmohamed42/jev-knee-labels",
        "nartaa/rsna-knee-hpo-assets",
        "saisahanas/rsna-knee-radiologist-report-labels",
        "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels",
    ],
    "kernel_sources": ["tumis001/report-label-audit"],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Wrote", OUT / "report_label_source_audit.ipynb")
