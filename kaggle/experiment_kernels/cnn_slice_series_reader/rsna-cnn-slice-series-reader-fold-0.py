"""Fold-0 Kaggle run for the slice-preserving, series-aware CNN reader."""

from __future__ import annotations

import csv
import json
import math
import runpy
import sys
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working/cnn_slice_series_reader_fold0")
INPUT = Path("/kaggle/input")
SOURCE_DATASET = INPUT / "datasets/tumis001/rsna-cnn-slice-series-reader-source-v1"
LABELS_CSV = (INPUT / "datasets/stevenleehans/rsna-knee-llm-report-labels"
              / "llm_labels_v4_blend.csv")
CACHE_DIRS = [
    INPUT / f"notebooks/tumis001/rsna-cnn-top-k-support-cache-shard-{shard}/slice_cache"
    for shard in range(3)
]


def main() -> None:
    print("Starting fold-0 slice-series CNN; resolving fixed Kaggle inputs.", flush=True)
    archive_path = SOURCE_DATASET / "skanee.zip"
    if archive_path.is_file():
        extracted = Path("/kaggle/working/cnn_slice_series_source")
        extracted.mkdir(parents=True, exist_ok=True)
        root = extracted.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                destination = (extracted / member.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise ValueError(f"Unsafe source archive path: {member.filename}")
            archive.extractall(extracted)
        source = extracted / "skanee"
    else:
        # Kaggle may expose uploaded directories at the root or retain one or
        # two archive folder levels. Probe only these known paths; never walk
        # the large /kaggle/input tree.
        source_candidates = [SOURCE_DATASET, SOURCE_DATASET / "skanee",
                             SOURCE_DATASET / "skanee/skanee"]
        source = next((path for path in source_candidates
                       if (path / "src/images/dataset.py").is_file()), None)
        if source is None:
            raise FileNotFoundError(
                f"Source files not found at known paths: {source_candidates}")
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"Steven v4 labels not found at {LABELS_CSV}")
    missing_caches = [str(path) for path in CACHE_DIRS
                      if not (path / "cache_summary.json").is_file()]
    if missing_caches:
        raise FileNotFoundError(f"Expected all three slice-cache shards; missing: {missing_caches}")
    labels_csv = LABELS_CSV

    sys.path.insert(0, str(source))
    from src.reports.schema import TARGETS

    with (source / "configs/gold_split_v1.csv").open(newline="", encoding="utf-8-sig") as stream:
        gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    if len(gold_ids) != 58:
        raise ValueError(f"Expected 58 locked gold studies, found {len(gold_ids)}")

    labels, seen = [], set()
    with labels_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = {"StudyInstanceUID", *TARGETS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Steven v4 CSV is missing columns: {sorted(missing)}")
        for row in reader:
            study_id = row["StudyInstanceUID"]
            if study_id in gold_ids:
                continue
            if study_id in seen:
                raise ValueError(f"Duplicate Steven-v4 study: {study_id}")
            seen.add(study_id)
            probabilities, weights = [], []
            for target in TARGETS:
                try:
                    value = float(row[target])
                except (TypeError, ValueError):
                    value = float("nan")
                valid = math.isfinite(value) and 0 <= value <= 1
                probabilities.append(value if valid else None)
                weights.append(float(valid))
            labels.append({"study_id": study_id, "probabilities": probabilities,
                           "weights": weights})

    ROOT.mkdir(parents=True, exist_ok=True)
    labels_path = ROOT / "soft_labels_public_v4.json"
    labels_path.write_text(json.dumps({"version": "steven_v4_blend",
                                       "targets_in_order": list(TARGETS),
                                       "labels": labels}, allow_nan=False) + "\n",
                           encoding="utf-8")
    caches = [str(path) for path in CACHE_DIRS]
    folds = source / "configs/report_folds_v5.csv"
    config = source / "configs/cnn_slice_series_reader_v1.json"
    output = ROOT / "run"
    print(f"Architecture source: {source}", flush=True)
    print(f"Label source: Steven v4 ({len(seen)} report-only studies; {len(gold_ids)} gold studies excluded)", flush=True)
    print(f"Cache shards: {caches}", flush=True)
    print("Single run: fold 0, 24 stratified slice tokens/study/epoch; no other model or fold will run.", flush=True)
    command = [str(source / "scripts/train_cnn_slice_series.py"),
               "--cache", *caches,
               "--soft-labels", str(labels_path),
               "--folds", str(folds), "--config", str(config),
               "--output-dir", str(output), "--validation-fold", "0",
               "--train-slices-per-study", "24"]
    print("Command: " + " ".join(command), flush=True)
    sys.argv = command
    runpy.run_path(command[0], run_name="__main__")
    summary = json.loads((output / "run_summary.json").read_text())
    (ROOT / "experiment_result.json").write_text(json.dumps({
        "experiment": "cnn_slice_series_reader_v1", "validation_fold": 0,
        "gold_locked": True, "label_source": "Steven v4",
        "best_epoch": summary["best"]["epoch"],
        "report_validation_macro_auc": summary["best"]["macro_auc"],
        "report_validation_bce": summary["best"]["metrics"]["validation_bce"],
        "epochs_completed": summary["epochs_run"]}, indent=2) + "\n")
    print("EXPERIMENT RESULT:", (ROOT / "experiment_result.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
