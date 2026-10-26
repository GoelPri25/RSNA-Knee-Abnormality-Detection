#!/usr/bin/env python3
"""Convert the public HPO blended labels to our CNN reader's JSON format.

The report-only partition is checked against our fixed report fold and the
official train table before writing. Gold labels are never copied into output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="labels_v1_blend.parquet")
    parser.add_argument("--folds", type=Path, required=True, help="report_folds_v5.csv")
    parser.add_argument("--gold-split", type=Path, required=True,
                        help="gold_split_v1.csv; only IDs/split names are read")
    parser.add_argument("--output", type=Path, required=True, help="dataset staging directory")
    args = parser.parse_args()

    frame = pd.read_parquet(args.source)
    expected = ["StudyInstanceUID", *TARGETS]
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(f"Source labels lack columns: {sorted(missing)}")
    frame = frame[expected].copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("Source labels contain duplicate StudyInstanceUID values")
    probabilities = frame[list(TARGETS)].to_numpy(dtype="float64")
    if not (pd.notna(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("All 12 blended probabilities must be finite and in [0, 1]")

    fold_ids = set(pd.read_csv(args.folds, dtype={"study_id": str})["study_id"])
    gold_split = pd.read_csv(args.gold_split, dtype={"StudyInstanceUID": str})
    if not {"StudyInstanceUID", "split"}.issubset(gold_split.columns):
        raise ValueError("Gold split file must have StudyInstanceUID and split columns")
    gold_ids = set(gold_split["StudyInstanceUID"].astype(str))
    if len(gold_ids) != 58:
        raise ValueError(f"Expected exactly 58 locked gold studies, found {len(gold_ids)}")
    source_ids = set(frame["StudyInstanceUID"])
    if source_ids & gold_ids:
        raise ValueError(f"Source contains {len(source_ids & gold_ids)} gold study IDs")
    if not fold_ids.issubset(source_ids):
        raise ValueError(
            f"Some fixed-fold studies are missing labels: {len(fold_ids - source_ids)}"
        )

    items = []
    for row in frame.itertuples(index=False, name=None):
        study_id, *values = row
        items.append({
            "study_id": study_id,
            "probabilities": [float(value) for value in values],
            # This blend assigns a probability to every finding, including
            # findings the report did not mention; all cells are supervised.
            "weights": [1.0] * len(TARGETS),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    trainable = args.output / "soft_labels_hpo_v1_blend.json"
    trainable.write_text(json.dumps({
        "version": "hpo_v1_blend_gemlow_steven_v4",
        "targets_in_order": list(TARGETS),
        "labels": items,
    }, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        "# RSNA report labels: HPO v1 blend\n\n"
        "Trainable soft-label export for our support-aware ResNet-34 reader.\n\n"
        "- Source: `nartaa/rsna-knee-hpo-assets`, `labels_v1_blend.parquet` (CC0-1.0).\n"
        "- 4,349 report-only studies; all 58 official-label gold studies are excluded.\n"
        "- The blend averages Gemini-low and Steven-v4 probabilities where Steven-v4\n"
        "  addressed a finding, and uses Gemini-low where Steven-v4 used its 0.25\n"
        "  not-mentioned marker. Every target cell receives weight 1.0.\n"
        "- `soft_labels_hpo_v1_blend.json` follows the `read_soft_labels()` schema in\n"
        "  our CNN source: `study_id`, ordered `probabilities`, and `weights`.\n"
        "- For controlled comparison, train on this file and validate on the same\n"
        "  held-out report fold with Steven-v4 labels. Keep gold evaluation locked\n"
        "  until the checkpoint has been selected.\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": "tumis001/rsna-knee-hpo-v1-blended-report-labels",
        "source_dataset": "nartaa/rsna-knee-hpo-assets",
        "source_file": str(args.source.name),
        "source_sha256": sha256(args.source),
        "trainable_file": trainable.name,
        "trainable_sha256": sha256(trainable),
        "row_count": len(items),
        "gold_studies_included": 0,
        "targets_in_order": list(TARGETS),
        "weights": "1.0 for every source-provided probability",
        "format": "CNN reader soft-label JSON v1",
        "license": "CC0-1.0 (source dataset)",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(items),
                      "gold_ids_excluded": len(gold_ids),
                      "trainable_sha256": manifest["trainable_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
