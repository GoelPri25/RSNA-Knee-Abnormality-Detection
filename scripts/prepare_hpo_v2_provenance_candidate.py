#!/usr/bin/env python3
"""Build a conservative, provenance-aware candidate from the HPO v1 labels.

For each report-only study and target:
  * both sources addressed it: preserve the established HPO v1 blend;
  * exactly one source addressed it: use that source's probability;
  * neither source addressed it: preserve the established HPO v1 value.

This is an experimental label-construction ablation, not an adjudication of
medical truth. The source mask columns are used for evidence availability;
numeric sentinel values are never used as a substitute for those masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "data/external/nartaa_hpo_assets"
DEFAULT_OUTPUT = REPO_ROOT / "data/label_datasets/hpo_v2_provenance_candidate"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def mask_suffix(target: str) -> str:
    return target.replace(" ", "_").replace("'", "")


def load_aligned(path: Path, name: str, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    require_columns(frame, ["StudyInstanceUID", *columns], name)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{name} has duplicate StudyInstanceUID values")
    return frame.set_index("StudyInstanceUID")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    asset_dir = args.asset_dir.resolve()
    output_dir = args.output.resolve()

    paths = {
        "hpo_blend": asset_dir / "labels_v1_blend.parquet",
        "steven": asset_dir / "labels_v0raw.parquet",
        "gemini": asset_dir / "labels_v1_gemlow.parquet",
        "steven_masks": asset_dir / "train_labels_v0.parquet",
        "gemini_masks": asset_dir / "train_labels_v1.parquet",
    }
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Required source files not found: {absent}")

    y_cols = list(TARGETS)
    steven_mask_cols = [f"m_{mask_suffix(target)}" for target in TARGETS]
    gemini_mask_cols = [f"m_{mask_suffix(target)}" for target in TARGETS]
    blend = load_aligned(paths["hpo_blend"], "HPO v1 blend", y_cols)
    steven = load_aligned(paths["steven"], "Steven v4 probabilities", y_cols)
    gemini = load_aligned(paths["gemini"], "Gemini-low probabilities", y_cols)
    steven_evidence = load_aligned(
        paths["steven_masks"], "Steven evidence masks", [*steven_mask_cols, "is_gold"]
    )
    gemini_evidence = load_aligned(
        paths["gemini_masks"], "Gemini evidence masks", gemini_mask_cols
    )

    report_ids = blend.index
    for name, frame in (("Steven probabilities", steven), ("Gemini probabilities", gemini)):
        if set(report_ids) != set(frame.index):
            raise ValueError(f"Study ID set differs between HPO blend and {name}")
    steven = steven.loc[report_ids]
    gemini = gemini.loc[report_ids]
    for name, frame in (("Steven masks", steven_evidence), ("Gemini masks", gemini_evidence)):
        if not report_ids.isin(frame.index).all():
            raise ValueError(f"{name} are missing report-only study IDs")
    gold_count = int(steven_evidence["is_gold"].sum())
    if gold_count != 58:
        raise ValueError(f"Expected 58 gold rows in the source mask table; found {gold_count}")
    steven_evidence = steven_evidence.loc[report_ids]
    gemini_evidence = gemini_evidence.loc[report_ids]
    if steven_evidence["is_gold"].astype(bool).any():
        raise ValueError("Gold studies are present in the report-only HPO label set")

    blend_values = blend[y_cols].to_numpy(dtype=np.float64)
    steven_values = steven[y_cols].to_numpy(dtype=np.float64)
    gemini_values = gemini[y_cols].to_numpy(dtype=np.float64)
    steven_mask = steven_evidence[steven_mask_cols].to_numpy(dtype=np.float64)
    gemini_mask = gemini_evidence[gemini_mask_cols].to_numpy(dtype=np.float64)

    for name, values in (("HPO v1 blend", blend_values), ("Steven v4", steven_values),
                         ("Gemini-low", gemini_values)):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{name} values must be finite and in [0, 1]")
    for name, values in (("Steven evidence masks", steven_mask), ("Gemini evidence masks", gemini_mask)):
        if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
            raise ValueError(f"{name} must contain only finite 0/1 values")

    steven_addressed = steven_mask == 1.0
    gemini_addressed = gemini_mask == 1.0
    both = steven_addressed & gemini_addressed
    steven_only = steven_addressed & ~gemini_addressed
    gemini_only = ~steven_addressed & gemini_addressed
    neither = ~steven_addressed & ~gemini_addressed

    candidate = blend_values.copy()
    # A source mask is the provenance authority, but a masked-in Steven value
    # of exactly 0.25 is internally contradictory: the published v4 format
    # uses 0.25 as its not-mentioned sentinel. Do not turn that sentinel into
    # supervision. Keep the established HPO v1 value for this exceptional cell
    # and let the audit report it explicitly.
    steven_only_usable = steven_only & (steven_values != 0.25)
    candidate[steven_only_usable] = steven_values[steven_only_usable]
    candidate[gemini_only] = gemini_values[gemini_only]

    # Exact policy checks: only single-source evidence cells may change.
    changed = ~np.isclose(candidate, blend_values, atol=1e-12, rtol=0.0)
    if (changed & ~(steven_only | gemini_only)).any():
        raise AssertionError("Candidate changed a both-addressed or neither-addressed cell")
    if not np.array_equal(candidate[both], blend_values[both]):
        raise AssertionError("Both-addressed cells must retain HPO v1 blend values")
    if not np.array_equal(candidate[neither], blend_values[neither]):
        raise AssertionError("Neither-addressed cells must retain HPO v1 values")
    if not np.allclose(candidate[steven_only_usable], steven_values[steven_only_usable], atol=0, rtol=0):
        raise AssertionError("Usable Steven-only cells must use Steven probabilities")
    if not np.array_equal(candidate[steven_only & ~steven_only_usable],
                          blend_values[steven_only & ~steven_only_usable]):
        raise AssertionError("Contradictory Steven sentinel cells must retain HPO v1 values")
    if not np.allclose(candidate[gemini_only], gemini_values[gemini_only], atol=0, rtol=0):
        raise AssertionError("Gemini-only cells must use Gemini probabilities")

    output_dir.mkdir(parents=True, exist_ok=True)
    items = [
        {
            "study_id": study_id,
            "probabilities": [float(value) for value in candidate[row_index]],
            # Preserve HPO v1's supervision density and training loss weighting.
            "weights": [1.0] * len(TARGETS),
        }
        for row_index, study_id in enumerate(report_ids)
    ]
    trainable = output_dir / "soft_labels_hpo_v2_provenance_candidate.json"
    trainable.write_text(json.dumps({
        "version": "hpo_v2_provenance_candidate",
        "targets_in_order": list(TARGETS),
        "labels": items,
    }, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")

    state_rows = []
    changed_rows = []
    state_arrays = {
        "both_addressed": both,
        "steven_only": steven_only,
        "gemini_only": gemini_only,
        "neither_addressed": neither,
    }
    for target_index, target in enumerate(TARGETS):
        for state, mask in state_arrays.items():
            state_rows.append({
                "target": target,
                "evidence_state": state,
                "cells": int(mask[:, target_index].sum()),
                "changed_cells": int((mask & changed)[:, target_index].sum()),
            })
        rows = np.flatnonzero(changed[:, target_index])
        for row_index in rows:
            changed_rows.append({
                "StudyInstanceUID": report_ids[row_index],
                "target": target,
                "evidence_state": "steven_only" if steven_only[row_index, target_index] else "gemini_only",
                "hpo_v1": float(blend_values[row_index, target_index]),
                "candidate": float(candidate[row_index, target_index]),
                "steven_probability": float(steven_values[row_index, target_index]),
                "gemini_probability": float(gemini_values[row_index, target_index]),
            })

    pd.DataFrame(state_rows).to_csv(output_dir / "evidence_state_summary.csv", index=False)
    pd.DataFrame(changed_rows, columns=[
        "StudyInstanceUID", "target", "evidence_state", "hpo_v1", "candidate",
        "steven_probability", "gemini_probability",
    ]).to_csv(output_dir / "changed_cells.csv", index=False)

    source_hashes = {name: sha256(path) for name, path in paths.items()}
    manifest = {
        "dataset": "hpo_v2_provenance_candidate",
        "kaggle_dataset_id": "tumis001/rsna-knee-hpo-v2-provenance-candidate",
        "status": "experimental_candidate_not_medical_adjudication",
        "policy": {
            "both_addressed": "preserve HPO v1 blend",
            "steven_only": "use Steven v4 probability",
            "gemini_only": "use Gemini-low probability",
            "neither_addressed": "preserve HPO v1 value",
        },
        "training_weights": "unchanged: 1.0 for all 12 targets",
        "row_count": len(items),
        "target_count": len(TARGETS),
        "gold_studies_included": 0,
        "changed_cells": int(changed.sum()),
        "changed_studies": int(np.any(changed, axis=1).sum()),
        "evidence_state_cells": {name: int(mask.sum()) for name, mask in state_arrays.items()},
        "source_files": {name: path.name for name, path in paths.items()},
        "source_sha256": source_hashes,
        "trainable_file": trainable.name,
        "trainable_sha256": sha256(trainable),
        "targets_in_order": list(TARGETS),
        "format": "CNN reader soft-label JSON v1",
        "license": "CC0-1.0 (source dataset license)",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# HPO v2 provenance candidate\n\n"
        "This is a controlled label-construction candidate, not a claim of medically corrected truth.\n\n"
        "## Rule\n\n"
        "The source-specific `m_*` masks indicate whether each labeling source addressed a target. "
        "They take precedence over numeric values such as Steven's 0.25 not-mentioned marker.\n\n"
        "- Both sources addressed the target: retain the HPO v1 blended value.\n"
        "- Exactly one source addressed the target: use that source's probability alone, unless its value conflicts with its not-mentioned sentinel.\n"
        "- Neither source addressed the target: retain the HPO v1 value, including Gemini's existing prior where that is what HPO v1 supplies.\n"
        "- If Steven's mask says addressed but its value is exactly 0.25 (the v4 not-mentioned sentinel), retain HPO v1 for that cell because the source evidence is contradictory.\n"
        "- Keep all target weights at 1.0, matching the HPO v1 training format.\n\n"
        "The experiment changes only source fusion in single-source cells with a usable source probability. It does not infer that silence means a negative, discard silent-cell priors, or use model predictions as labels. In one Steven-only Synovitis cell, the evidence mask conflicts with Steven's 0.25 not-mentioned sentinel, so the established HPO v1 probability is retained. `changed_cells.csv` lists every changed probability; `evidence_state_summary.csv` reports counts.\n\n"
        "To test, train the HPO v2 candidate using the HPO v1 +281 ResNet-50 recipe, fixed split, initialization, and stopping controls. Compare it against the existing HPO v1 ResNet-50 checkpoint on the same report-validation fold and submit only if the validation comparison justifies it. Validation/submission utility does not establish label truth; independent image review would still be needed for medical adjudication.\n",
        encoding="utf-8",
    )
    dataset_metadata = {
        "title": "RSNA Knee HPO v2 Provenance Candidate Labels",
        "id": "tumis001/rsna-knee-hpo-v2-provenance-candidate",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (output_dir / "datasets-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "output": str(output_dir),
        "rows": len(items),
        "changed_cells": int(changed.sum()),
        "changed_studies": int(np.any(changed, axis=1).sum()),
        "evidence_state_cells": manifest["evidence_state_cells"],
        "trainable_sha256": manifest["trainable_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
