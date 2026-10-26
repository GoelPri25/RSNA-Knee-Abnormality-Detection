"""Extend the HPO fold-0 training split with the 281 previously unassigned rows.

The original fold-0 validation IDs are preserved exactly. Added studies are
assigned to training only after checking report-group hashes against validation
and verifying that the gold lock remains intact.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--baseline-folds", type=Path, required=True)
    parser.add_argument("--report-groups", type=Path, required=True)
    parser.add_argument("--gold-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    label_payload = json.loads(args.labels.read_text(encoding="utf-8"))
    label_ids = {row["study_id"] for row in label_payload["labels"]}
    baseline_rows = read_csv(args.baseline_folds)
    baseline = {row["study_id"]: row for row in baseline_rows}
    groups = {
        row["StudyInstanceUID"]: row
        for row in read_csv(args.report_groups)
    }
    gold_ids = {row["StudyInstanceUID"] for row in read_csv(args.gold_split)}

    extra_ids = label_ids - set(baseline)
    validation_ids = {
        study_id for study_id, row in baseline.items() if int(row["fold"]) == 0
    }
    if (len(label_ids), len(baseline), len(extra_ids), len(validation_ids)) != (
        4349, 4068, 281, 825
    ):
        raise RuntimeError(
            "Unexpected source split dimensions: "
            f"labels={len(label_ids)}, baseline={len(baseline)}, "
            f"extra={len(extra_ids)}, validation={len(validation_ids)}"
        )
    if extra_ids & gold_ids:
        raise RuntimeError("One or more added studies are in the locked gold set")
    if not label_ids.issubset(groups):
        raise RuntimeError(
            f"Report-group coverage missing for {len(label_ids - set(groups))} labeled studies"
        )

    validation_hashes = {groups[study_id]["report_group_sha256"] for study_id in validation_ids}
    overlapping = {
        study_id for study_id in extra_ids
        if groups[study_id]["report_group_sha256"] in validation_hashes
    }
    if overlapping:
        raise RuntimeError(
            f"Added studies share exact normalized reports with validation: {len(overlapping)}"
        )

    group_sizes = Counter(groups[study_id]["report_group_sha256"] for study_id in label_ids)
    output_rows = []
    for study_id in sorted(label_ids):
        if study_id in baseline:
            row = baseline[study_id]
            fold = int(row["fold"])
            group_hash = row["report_group_sha256"]
        else:
            fold = 5  # build_studies treats every nonzero fold as training.
            group_hash = groups[study_id]["report_group_sha256"]
        output_rows.append({
            "study_id": study_id,
            "fold": fold,
            "report_group_sha256": group_hash,
            "report_group_size": group_sizes[group_hash],
        })

    train_ids = {row["study_id"] for row in output_rows if int(row["fold"]) != 0}
    heldout_ids = {row["study_id"] for row in output_rows if int(row["fold"]) == 0}
    if heldout_ids != validation_ids:
        raise RuntimeError("Expanded split changed the original fold-0 validation IDs")
    if (len(train_ids), len(heldout_ids)) != (3524, 825):
        raise RuntimeError(f"Unexpected expanded split: train={len(train_ids)}, val={len(heldout_ids)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["study_id", "fold", "report_group_sha256", "report_group_size"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(json.dumps({
        "hpo_label_rows": len(label_ids),
        "original_training_studies": 3243,
        "added_training_studies": len(extra_ids),
        "final_training_studies": len(train_ids),
        "validation_studies_unchanged": len(heldout_ids),
        "locked_gold_studies": len(gold_ids),
        "exact_report_group_overlap_with_validation": len(overlapping),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
