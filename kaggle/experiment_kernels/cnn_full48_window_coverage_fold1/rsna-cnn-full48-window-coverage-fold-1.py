"""Paired fold-1 test: 24 randomly sampled versus all 48 cached windows/study."""

from __future__ import annotations

import csv
import json
import math
import runpy
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path("/kaggle/working/cnn_full48_window_coverage_fold1")


def find_one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name}, found {matches}")
    return matches[0]


def main() -> None:
    print("Starting the single 48-window CNN run; no 24-window control will run.", flush=True)
    print("Locating source, labels, caches, and competition metadata...", flush=True)
    archive_matches = list(Path("/kaggle/input").rglob("skanee.zip"))
    if len(archive_matches) == 1:
        source = Path("/kaggle/working/cnn_window_coverage_source")
        source.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_matches[0]) as archive:
            root = source.resolve()
            for member in archive.infolist():
                destination = (source / member.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise ValueError(f"Unsafe source archive path: {member.filename}")
            archive.extractall(source)
    else:
        source = find_one("scripts/train_cnn_baseline.py").parents[1]
    if not (source / "scripts/train_cnn_baseline.py").is_file():
        raise FileNotFoundError(f"Patched training script is missing from {source}")
    writable_source = Path("/kaggle/working/cnn_full48_source")
    if source.resolve() != writable_source.resolve():
        if writable_source.exists():
            shutil.rmtree(writable_source)
        shutil.copytree(source, writable_source)
        source = writable_source
    instrument_training_logs(source / "scripts/train_cnn_baseline.py")
    print(f"Writable source prepared: {source}", flush=True)
    labels_csv = find_one("llm_labels_v4_blend.csv")
    data_root = next((path for path in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"))
        if (path / "train.csv").is_file()), None)
    if data_root is None:
        raise FileNotFoundError("Competition train.csv was not attached")

    sys.path.insert(0, str(source))
    from src.reports.schema import TARGETS

    with (data_root / "train.csv").open(newline="", encoding="utf-8-sig") as stream:
        official_rows = list(csv.DictReader(stream))
    gold_ids = {row["StudyInstanceUID"] for row in official_rows
                if all(row.get(target, "") in {"0", "1"} for target in TARGETS)}
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
    print("Scanning attached slice-cache shards...", flush=True)
    caches = sorted({str(path.parent) for path in Path("/kaggle/input").rglob("cache_summary.json")})
    if len(caches) != 3:
        raise FileNotFoundError(f"Expected 3 slice-cache shards, found {caches}")
    folds = source / "configs/report_folds_v5.csv"
    if not folds.is_file():
        raise FileNotFoundError(f"Fold configuration is missing: {folds}")
    print(f"Source: {source}\nLabels: {labels_path}\nCaches: {caches}", flush=True)
    print(f"Gold lock: {len(gold_ids)} excluded; fold 1 report validation only.", flush=True)

    print(f"Found {len(caches)} cache shards; running fold 1 with all 48 windows per study.", flush=True)
    output = ROOT / "full48"
    config = source / "configs" / "cnn_window_coverage_full48.json"
    command = [
        str(source / "scripts/train_cnn_baseline.py"),
        "--cache", *caches,
        "--soft-labels", str(labels_path),
        "--folds", str(folds),
        "--config", str(config),
        "--output-dir", str(output),
        "--validation-fold", "1",
        "--train-slices-per-study", "48",
        "--gradient-accumulation-steps", "2",
    ]
    print("Starting full48: 48 windows/study; accumulation=2", flush=True)
    print("Command: " + " ".join(command), flush=True)
    sys.argv = command
    runpy.run_path(command[0], run_name="__main__")

    path = output / "run_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run summary was not written: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "experiment": "cnn_full48_window_coverage_fold1",
        "gold_locked": True,
        "validation_fold": 1,
        "train_slices_per_study": 48,
        "gradient_accumulation_steps": 2,
        "selected_epoch": summary["best"]["epoch"],
        "report_validation_macro_auc": summary["best"]["macro_auc"],
        "report_validation_bce": summary["best"]["metrics"]["validation_bce"],
        "epochs_completed": summary["epochs_run"],
    }
    (ROOT / "full48_summary.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf-8")
    print("\nFull48 result:", json.dumps(result, indent=2), flush=True)


def instrument_training_logs(script_path: Path) -> None:
    """Add visible setup and minibatch progress without changing training math."""
    text = script_path.read_text(encoding="utf-8")
    manifest_line = "    manifest = load_manifest(args.cache, config[\"cache\"][\"exclude_order_sources\"])\n"
    if "Loading cache manifest (may take a few minutes)..." not in text:
        if text.count(manifest_line) != 1:
            raise RuntimeError("Could not safely add cache-manifest progress logging")
        text = text.replace(
            manifest_line,
            "    print(\"Loading cache manifest (may take a few minutes)...\", flush=True)\n"
            + manifest_line
            + "    print(f\"Cache manifest ready: {len(manifest)} studies\", flush=True)\n",
        )
    batch_tail = "            total_loss += float(loss.detach())\n            batches += 1\n"
    if "training progress:" not in text:
        if text.count(batch_tail) != 1:
            raise RuntimeError("Could not safely add minibatch progress logging")
        text = text.replace(
            batch_tail,
            batch_tail
            + "            if (batch_index + 1) % 100 == 0 or batch_index + 1 == batch_count:\n"
            + "                print(f\"epoch {epoch + 1} training progress: batch \"\n"
            + "                      f\"{batch_index + 1}/{batch_count} mean_loss={total_loss / batches:.4f}\",\n"
            + "                      flush=True)\n",
        )
    script_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
