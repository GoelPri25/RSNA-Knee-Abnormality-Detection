"""Prepare fixed report-only evidence and validation inputs without GPU work."""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsna_knee.auc_margin import build_evidence
sys.path.insert(0, str(ROOT / "scripts"))
import csv
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def prepare_steven_validation(steven_csv: Path, folds_path: Path, gold_split_path: Path,
                              output_path: Path) -> tuple[int, int]:
    import numpy as np

    with gold_split_path.open(newline="", encoding="utf-8") as stream:
        gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    with folds_path.open(newline="", encoding="utf-8") as stream:
        fold_map = {row["study_id"]: int(row["fold"]) for row in csv.DictReader(stream)}
    validation_ids = {study_id for study_id, fold in fold_map.items() if fold == 0}
    if len(gold_ids) != 58 or len(validation_ids) != 825 or gold_ids & validation_ids:
        raise RuntimeError("Fold/gold split dimensions or separation check failed")

    records = []
    seen: set[str] = set()
    with steven_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"StudyInstanceUID", *TARGETS}.issubset(reader.fieldnames or []):
            raise ValueError("Steven v4 label file has an unexpected target schema")
        for row in reader:
            study_id = row["StudyInstanceUID"]
            if study_id not in validation_ids:
                continue
            if study_id in seen:
                raise ValueError(f"Duplicate Steven v4 validation row: {study_id}")
            seen.add(study_id)
            probabilities, weights = [], []
            for target in TARGETS:
                try:
                    value = float(row[target])
                except (TypeError, ValueError):
                    value = float("nan")
                valid = bool(np.isfinite(value) and 0.0 <= value <= 1.0)
                probabilities.append(value if valid else None)
                weights.append(float(valid))
            records.append({"study_id": study_id, "probabilities": probabilities, "weights": weights})
    if seen != validation_ids:
        raise RuntimeError(f"Steven v4 validation coverage mismatch: {len(validation_ids - seen)} missing")
    output_path.write_text(json.dumps({
        "version": "steven_v4_fold0_validation_only",
        "targets_in_order": TARGETS,
        "labels": records,
    }, allow_nan=False) + "\n", encoding="utf-8")
    return len(gold_ids), len(validation_ids)

def main():
    output = ROOT / "data/hpo_v1_resnet50_auc_margin_fold0/inputs"
    output.mkdir(parents=True, exist_ok=True)
    labels = json.loads((ROOT / "data/label_datasets/hpo_v1_blend/soft_labels_hpo_v1_blend.json").read_text())
    targets = labels["targets_in_order"]
    ids = [r["study_id"] for r in labels["labels"]]
    assets = ROOT / "data/external/nartaa_hpo_assets"
    frames = [pd.read_parquet(assets / name).set_index("StudyInstanceUID").loc[ids] for name in
              ("labels_v0raw.parquet", "labels_v1_gemlow.parquet", "train_labels_v0.parquet", "train_labels_v1.parquet")]
    columns = ["m_" + t.replace(" ", "_").replace("'", "") for t in targets]
    evidence = build_evidence(frames[0][targets].values, frames[1][targets].values,
                              frames[2][columns].values, frames[3][columns].values)
    source = ROOT / "kaggle/experiment_kernels/hpo_v1_blend_resnet50_warmstart/source/skanee"
    gold = set(pd.read_csv(source / "configs/gold_split_v1.csv")["StudyInstanceUID"])
    assert len(ids) == 4349 and not gold.intersection(ids)
    (output / "evidence.json").write_text(json.dumps({"targets": targets, "evidence": dict(zip(ids, evidence.tolist())),
        "note": "Source masks indicate availability, not independently verified textual evidence."}) + "\n")
    prepare_steven_validation(ROOT / "data/labels/steven_v4/llm_labels_v4_blend.csv",
        source / "configs/report_folds_hpo_add281_trainonly.csv", source / "configs/gold_split_v1.csv",
        output / "validation.json")
    print({t: {"positive": int((evidence[:, j] == 1).sum()), "negative": int((evidence[:, j] == -1).sum())}
           for j, t in enumerate(targets)})

if __name__ == "__main__":
    main()
