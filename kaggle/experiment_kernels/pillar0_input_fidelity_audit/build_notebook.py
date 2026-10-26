#!/usr/bin/env python3
"""Build the no-training Kaggle audit of DICOM support and physical scale."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "pillar0_input_fidelity_audit.ipynb"
METADATA = ROOT / "kernel-metadata.json"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# RSNA Knee input-fidelity audit (no training)

This CPU-only audit measures DICOM pixel support, physical field of view, slice
spacing, and a conservative intensity-foreground extent proxy. It joins those
measurements to **report-only** present/absent target states to see whether
input coverage differs descriptively by target and report state. It does not
load a model, use official gold labels to choose cases, train, or export reports
or study/series identifiers.

The report-label kernel is attached as an input. Case selection is deterministic
and targets at least two present and two absent report-derived examples per
finding where available, capped at 36 unique studies. Interpret the tables as a
small audit sample, not a clinical association test. The foreground extent is
only a pixel-intensity proxy; it is not a knee segmentation.
"""
    ),
    code(
        '''from pathlib import Path
import json
import hashlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 20260925
MAX_STUDIES = 36
MIN_PER_TARGET_STATUS = 2
REPORT_STATUSES = ("present", "absent")
PLANES = ("Axial", "Coronal", "Sagittal")

data_candidates = [
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
]
DATA = next((path for path in data_candidates if path.is_dir()), None)
if DATA is None:
    raise FileNotFoundError("Attach the RSNA Knee competition dataset.")

label_candidates = list(Path("/kaggle/input").rglob("report_structured_labels.csv"))
if len(label_candidates) != 1:
    raise FileNotFoundError(
        "Expected exactly one attached report_structured_labels.csv; "
        f"found {len(label_candidates)}. Attach the report-label kernel output."
    )
LABELS_PATH = label_candidates[0]
SERIES_ROOT = DATA / "train_series"

train = pd.read_csv(DATA / "train.csv", low_memory=False)
series_table = pd.read_csv(DATA / "train_series.csv", low_memory=False)
report_labels = pd.read_csv(LABELS_PATH, low_memory=False)
TARGETS = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
if len(TARGETS) != 12:
    raise ValueError(f"Expected 12 target columns, found {TARGETS}")

for frame in (train, series_table, report_labels):
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
series_table["SeriesInstanceUID"] = series_table["SeriesInstanceUID"].astype(str)
report_only_ids = set(
    train.loc[train[TARGETS].isna().all(axis=1), "StudyInstanceUID"]
)
report_labels = report_labels[
    report_labels["StudyInstanceUID"].isin(report_only_ids)
].copy()
report_labels = report_labels.drop_duplicates("StudyInstanceUID", keep="last")

prefix_by_target = {
    "ACL": "acl", "MCL": "mcl",
    "Medial Meniscus": "medial_meniscus",
    "Lateral Meniscus": "lateral_meniscus",
    "Medial OA": "medial_oa", "Lateral OA": "lateral_oa",
    "PF OA": "pf_oa", "Effusion": "effusion",
    "Synovitis": "synovitis", "Baker's": "bakers",
    "Contusion": "contusion", "Fracture": "fracture",
}

for target, prefix in prefix_by_target.items():
    mask_col = f"{prefix}__label_mask"
    status_col = f"{prefix}__status"
    if mask_col not in report_labels or status_col not in report_labels:
        raise KeyError(f"Missing report label columns for {target}")
    report_labels[mask_col] = (
        report_labels[mask_col].astype(str).str.lower().isin(("true", "1"))
    )
    report_labels[status_col] = report_labels[status_col].astype(str).str.lower()

print("Competition data:", DATA)
print("Report-label input:", LABELS_PATH)
print("Report-only studies available:", len(report_only_ids))
print("Studies with extracted statuses:", len(report_labels))
print("No GPU/model/training is used by this notebook.")
'''
    ),
    code(
        '''# Deterministically select cases to cover each target's explicit present/absent states.
long_rows = []
for row in report_labels.itertuples(index=False):
    values = row._asdict()
    study_id = str(values["StudyInstanceUID"])
    for target, prefix in prefix_by_target.items():
        status = values.get(f"{prefix}__status", "")
        active = bool(values.get(f"{prefix}__label_mask", False))
        if active and status in REPORT_STATUSES:
            long_rows.append({"StudyInstanceUID": study_id, "target": target, "status": status})
label_long = pd.DataFrame(long_rows).drop_duplicates(
    ["StudyInstanceUID", "target", "status"]
)

status_counts = (
    label_long.groupby(["target", "status"]).size()
    .unstack(fill_value=0).reindex(columns=REPORT_STATUSES, fill_value=0)
    .reindex(TARGETS)
)
display(status_counts)

study_labels = {
    study_id: set(zip(group["target"], group["status"]))
    for study_id, group in label_long.groupby("StudyInstanceUID")
}
all_pairs = {(target, status) for target in TARGETS for status in REPORT_STATUSES}
selected = []
covered = {pair: 0 for pair in all_pairs}
plane_coverage = {plane: 0 for plane in PLANES}
study_planes = {
    study_id: set(group["Anatomical_Plane"].dropna().astype(str)) & set(PLANES)
    for study_id, group in series_table.groupby("StudyInstanceUID")
}
candidate_ids = sorted(
    study_labels,
    key=lambda uid: hashlib.sha256(f"{SEED}:{uid}".encode()).hexdigest(),
)

while len(selected) < MAX_STUDIES:
    best_id, best_score = None, 0
    for study_id in candidate_ids:
        if study_id in selected:
            continue
        # Strongly prefer filling the minimum per-target present/absent quotas.
        # Once those are filled, keep sampling to the cap while favoring rare
        # target/status strata and underrepresented planes.
        score = sum(
            (3.0 if covered[pair] < MIN_PER_TARGET_STATUS else 1.0)
            / (1 + covered[pair])
            for pair in study_labels[study_id]
            if pair in all_pairs
        )
        score += 0.25 * sum(
            1.0 / (1 + plane_coverage[plane])
            for plane in study_planes.get(study_id, set())
        )
        if score > best_score:
            best_id, best_score = study_id, score
    if best_id is None:
        break
    selected.append(best_id)
    for pair in study_labels[best_id]:
        if pair in covered:
            covered[pair] += 1
    for plane in study_planes.get(best_id, set()):
        plane_coverage[plane] += 1

coverage_rows = []
for target in TARGETS:
    row = {"target": target}
    for status in REPORT_STATUSES:
        row[status] = sum((target, status) in study_labels[sid] for sid in selected)
    coverage_rows.append(row)
coverage = pd.DataFrame(coverage_rows)
print(f"Selected {len(selected)} unique report-only studies.")
display(coverage)
'''
    ),
    code(
        '''# Decode sampled series and measure native support and physical geometry.
try:
    import pydicom
except ImportError as error:
    raise RuntimeError(
        "pydicom is not available in this Kaggle image. Stop here; do not run "
        "pip install from the notebook because Kaggle internet is unreliable. "
        "Attach a prebuilt, versioned pydicom wheel input instead."
    ) from error

def otsu_threshold(values, bins=128):
    """Histogram Otsu threshold used only for a rough foreground proxy."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 32 or float(values.max()) <= float(values.min()):
        return float(values.min()) if values.size else 0.0
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_left = np.cumsum(hist).astype(np.float64)
    weight_right = values.size - weight_left
    mean_left = np.cumsum(hist * centers)
    total_mean = mean_left[-1]
    valid = (weight_left > 0) & (weight_right > 0)
    between = np.zeros_like(centers)
    between[valid] = (
        (total_mean * weight_left[valid] - mean_left[valid] * values.size) ** 2
        / (weight_left[valid] * weight_right[valid])
    )
    return float(centers[int(np.argmax(between))])

def padding_support(ds, pixels):
    """Validity from finite pixels and DICOM padding tags; zero remains valid."""
    valid = np.isfinite(pixels)
    padding = getattr(ds, "PixelPaddingValue", None)
    limit = getattr(ds, "PixelPaddingRangeLimit", None)
    if padding is not None:
        try:
            p0 = float(padding)
            if limit is None:
                valid &= pixels != p0
            else:
                p1 = float(limit)
                valid &= ~((pixels >= min(p0, p1)) & (pixels <= max(p0, p1)))
        except (TypeError, ValueError):
            pass
    return valid

def series_measurements(study_id, series_row):
    folder = SERIES_ROOT / study_id / str(series_row.SeriesInstanceUID)
    paths = sorted(folder.glob("*.dcm"))
    if not paths:
        return [], {"decode_errors": 1, "missing_series": 1}

    headers = []
    for path in paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
            normal = np.cross(iop[:3], iop[3:])
            normal /= max(np.linalg.norm(normal), 1e-12)
            spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)
            headers.append((float(np.dot(ipp, normal)), path, ds, spacing, normal))
        except Exception:
            continue

    if not headers:
        return [], {"decode_errors": len(paths), "missing_series": 0}
    headers.sort(key=lambda item: item[0])
    coordinates = np.asarray([item[0] for item in headers], dtype=np.float64)
    gaps = np.diff(coordinates)
    gaps = gaps[np.isfinite(gaps) & (gaps > 1e-3)]
    sample_metrics = []
    decode_errors = 0

    for coordinate, path, header, spacing, normal in headers:
        try:
            ds = pydicom.dcmread(path)
            pixels = np.asarray(ds.pixel_array, dtype=np.float32)
            valid = padding_support(ds, pixels)
            valid_values = pixels[valid]
            if valid_values.size == 0:
                decode_errors += 1
                continue
            rows, cols = pixels.shape[-2:]
            row_mm, col_mm = float(spacing[0]), float(spacing[1])
            all_finite = pixels[np.isfinite(pixels)]
            threshold = otsu_threshold(valid_values)
            foreground = valid & (pixels > threshold)
            foreground_count = int(foreground.sum())
            if foreground_count:
                rr, cc = np.where(foreground)
                bbox_h_mm = float((rr.max() - rr.min() + 1) * row_mm)
                bbox_w_mm = float((cc.max() - cc.min() + 1) * col_mm)
                bbox_area_fraction = float(
                    (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)
                    / max(rows * cols, 1)
                )
            else:
                bbox_h_mm = bbox_w_mm = bbox_area_fraction = np.nan
            square_side = max(rows, cols)
            square_padding_fraction = 1.0 - (min(rows, cols) / square_side) ** 2
            sample_metrics.append({
                "study_id": study_id,
                "series_id": str(series_row.SeriesInstanceUID),
                "plane": str(series_row.Anatomical_Plane),
                "fluid_sensitive": int(bool(series_row.Fluid_Sensitive)),
                "slice_coordinate_mm": float(coordinate),
                "rows": int(rows), "columns": int(cols),
                "pixel_spacing_row_mm": row_mm,
                "pixel_spacing_column_mm": col_mm,
                "physical_fov_row_mm": rows * row_mm,
                "physical_fov_column_mm": cols * col_mm,
                "dicom_valid_support_fraction": float(valid.mean()),
                "native_nonfinite_fraction": float((~np.isfinite(pixels)).mean()),
                "valid_zero_fraction": float((valid & (pixels == 0)).sum() / max(valid.sum(), 1)),
                "dicom_padding_fraction": float(1 - valid.mean()),
                "square_pad_fraction_estimate": float(square_padding_fraction),
                "valid_p01": float(np.percentile(valid_values, 1)),
                "valid_p50": float(np.percentile(valid_values, 50)),
                "valid_p99": float(np.percentile(valid_values, 99)),
                "all_finite_p01": float(np.percentile(all_finite, 1)),
                "all_finite_p99": float(np.percentile(all_finite, 99)),
                "foreground_proxy_fraction": float(foreground_count / max(valid.sum(), 1)),
                "foreground_bbox_area_fraction": bbox_area_fraction,
                "foreground_bbox_height_mm": bbox_h_mm,
                "foreground_bbox_width_mm": bbox_w_mm,
            })
        except Exception:
            decode_errors += 1

    series_info = {
        "decode_errors": decode_errors,
        "missing_series": 0,
        "slices_found": len(paths),
        "slices_with_geometry": len(headers),
        "median_inter_slice_gap_mm": float(np.median(gaps)) if gaps.size else np.nan,
        "p90_inter_slice_gap_mm": float(np.percentile(gaps, 90)) if gaps.size else np.nan,
        "physical_span_mm": float(coordinates.max() - coordinates.min()) if len(coordinates) > 1 else 0.0,
    }
    return sample_metrics, series_info

selected_series = series_table[
    series_table["StudyInstanceUID"].isin(selected)
    & series_table["Anatomical_Plane"].isin(PLANES)
].copy()
slice_rows = []
series_summaries = []
for study_number, study_id in enumerate(selected, start=1):
    study_rows = selected_series[selected_series["StudyInstanceUID"] == study_id]
    for series_row in study_rows.itertuples(index=False):
        metrics, info = series_measurements(study_id, series_row)
        slice_rows.extend(metrics)
        series_summaries.append({
            "study_id": study_id,
            "series_id": str(series_row.SeriesInstanceUID),
            "plane": str(series_row.Anatomical_Plane),
            "fluid_sensitive": int(bool(series_row.Fluid_Sensitive)),
            **info,
        })
    if study_number % 4 == 0 or study_number == len(selected):
        print(f"Audited {study_number}/{len(selected)} selected studies; "
              f"decoded {len(slice_rows):,} slices so far.", flush=True)

slice_metrics = pd.DataFrame(slice_rows)
series_metrics = pd.DataFrame(series_summaries)
if slice_metrics.empty:
    raise RuntimeError("No DICOM pixels were decoded; inspect errors above.")
print("pydicom:", pydicom.__version__)
print("Audited series:", len(series_metrics), "decoded slices:", len(slice_metrics))
print("DICOM decode errors:", int(series_metrics["decode_errors"].sum()))
'''
    ),
    code(
        '''# Summarize input measurements by plane, sequence, target, and report state.
study_plane = (
    slice_metrics.groupby(["study_id", "plane", "fluid_sensitive"], as_index=False)
    .agg(
        series_count=("series_id", "nunique"),
        slice_count=("series_id", "size"),
        physical_fov_row_mm=("physical_fov_row_mm", "median"),
        physical_fov_column_mm=("physical_fov_column_mm", "median"),
        supported_fraction=("dicom_valid_support_fraction", "median"),
        valid_zero_fraction=("valid_zero_fraction", "median"),
        dicom_padding_fraction=("dicom_padding_fraction", "median"),
        square_pad_fraction_estimate=("square_pad_fraction_estimate", "median"),
        foreground_proxy_fraction=("foreground_proxy_fraction", "median"),
        foreground_bbox_area_fraction=("foreground_bbox_area_fraction", "median"),
        foreground_bbox_height_mm=("foreground_bbox_height_mm", "median"),
        foreground_bbox_width_mm=("foreground_bbox_width_mm", "median"),
        p01_shift=("valid_p01", lambda s: float(s.median())),
        p99_shift=("valid_p99", lambda s: float(s.median())),
    )
)
series_by_study_plane = (
    series_metrics.groupby(["study_id", "plane", "fluid_sensitive"], as_index=False)
    .agg(
        median_gap_mm=("median_inter_slice_gap_mm", "median"),
        physical_span_mm=("physical_span_mm", "median"),
        decode_errors=("decode_errors", "sum"),
    )
)
study_plane = study_plane.merge(
    series_by_study_plane,
    on=["study_id", "plane", "fluid_sensitive"],
    how="left",
)

status_rows = []
for study_id in selected:
    for target, status in study_labels[study_id]:
        for _, plane_row in study_plane[study_plane["study_id"] == study_id].iterrows():
            row = plane_row.to_dict()
            row.update({"target": target, "status": status})
            status_rows.append(row)
target_status = pd.DataFrame(status_rows)

metric_cols = [
    "series_count", "slice_count", "physical_fov_row_mm", "physical_fov_column_mm",
    "supported_fraction", "valid_zero_fraction", "dicom_padding_fraction",
    "square_pad_fraction_estimate", "foreground_proxy_fraction",
    "foreground_bbox_area_fraction", "foreground_bbox_height_mm",
    "foreground_bbox_width_mm", "median_gap_mm", "physical_span_mm", "decode_errors",
]
by_plane = (
    study_plane.groupby(["plane", "fluid_sensitive"], as_index=False)[metric_cols]
    .median(numeric_only=True)
)
by_target_status_plane = (
    target_status.groupby(["target", "status", "plane"], as_index=False)
    .agg(
        studies=("study_id", "nunique"),
        series_count=("series_count", "median"),
        physical_fov_row_mm=("physical_fov_row_mm", "median"),
        physical_fov_column_mm=("physical_fov_column_mm", "median"),
        valid_zero_fraction=("valid_zero_fraction", "median"),
        dicom_padding_fraction=("dicom_padding_fraction", "median"),
        square_pad_fraction_estimate=("square_pad_fraction_estimate", "median"),
        foreground_proxy_fraction=("foreground_proxy_fraction", "median"),
        foreground_bbox_area_fraction=("foreground_bbox_area_fraction", "median"),
        foreground_bbox_height_mm=("foreground_bbox_height_mm", "median"),
        foreground_bbox_width_mm=("foreground_bbox_width_mm", "median"),
        median_gap_mm=("median_gap_mm", "median"),
        physical_span_mm=("physical_span_mm", "median"),
    )
)

print("Median physical/input measurements by plane and fluid flag:")
display(by_plane.round(3))
print("Measurements grouped by report-derived target status and plane:")
display(by_target_status_plane.round(3))

# Small-sample descriptive difference only; no p-values or model-selection score.
present_absent = (
    target_status[target_status["status"].isin(("present", "absent"))]
    .groupby(["target", "status"])[[
        "foreground_bbox_area_fraction", "foreground_bbox_height_mm",
        "foreground_bbox_width_mm", "square_pad_fraction_estimate",
        "median_gap_mm", "physical_span_mm",
    ]].median()
)
print("Target-level present minus absent medians (descriptive; sampled studies only):")
display(present_absent.unstack("status").round(3))

OUT = Path("/kaggle/working/input_fidelity_audit")
OUT.mkdir(parents=True, exist_ok=True)
by_plane.to_csv(OUT / "input_metrics_by_plane.csv", index=False)
by_target_status_plane.to_csv(OUT / "input_metrics_by_target_status_plane.csv", index=False)
coverage.to_csv(OUT / "sample_status_coverage.csv", index=False)
'''
    ),
    code(
        '''# Plot coverage, scale, support, and report-state differences.
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
plot_metrics = [
    ("square_pad_fraction_estimate", "Estimated square-padding fraction"),
    ("physical_fov_row_mm", "Native physical FOV (row axis, mm)"),
    ("foreground_bbox_area_fraction", "Intensity foreground bbox / image area"),
    ("valid_zero_fraction", "Exact-zero fraction among valid pixels"),
]
for ax, (column, title) in zip(axes.flat, plot_metrics):
    values = [
        slice_metrics.loc[slice_metrics["plane"] == plane, column].dropna().to_numpy()
        for plane in PLANES
    ]
    boxplot = ax.boxplot(values, showfliers=False)
    ax.set_xticks(range(1, len(PLANES) + 1), PLANES)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Input fidelity audit: sampled report-only series")
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, column, title in [
    (axes[0], "foreground_bbox_area_fraction", "Foreground extent proxy by report status"),
    (axes[1], "square_pad_fraction_estimate", "Square padding by report status"),
]:
    plot = target_status[target_status["status"].isin(("present", "absent"))]
    for status, color in (("present", "#b0443c"), ("absent", "#3478a8")):
        subset = plot[plot["status"] == status]
        ax.scatter(
            subset[column],
            subset["target"].map({name: i for i, name in enumerate(TARGETS)}),
            s=12, alpha=0.35, label=status, color=color,
        )
    ax.set_yticks(range(len(TARGETS)), TARGETS)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
plt.show()

summary = {
    "audit": "report_only_input_fidelity_no_training",
    "seed": SEED,
    "maximum_studies": MAX_STUDIES,
    "selected_studies": int(len(selected)),
    "selected_series": int(len(series_metrics)),
    "decoded_slices": int(len(slice_metrics)),
    "dicom_decode_errors": int(series_metrics["decode_errors"].sum()),
    "report_status_coverage": coverage.to_dict(orient="records"),
    "official_gold_target_values_used": False,
    "official_label_nullness_used_only_to_filter_report_only_studies": True,
    "report_text_exported": False,
    "study_or_series_identifiers_exported": False,
    "model_loaded": False,
    "training_performed": False,
    "foreground_measurement": "Otsu intensity bounding-box proxy; not segmentation",
    "limitations": [
        "Small, status-stratified descriptive sample; not a clinical association test.",
        "DICOM support tags do not identify all scanner-generated background.",
        "Exact zero is treated as a valid value unless a DICOM padding tag says otherwise.",
        "Physical FOV and foreground proxy do not replace radiologist-reviewed anatomy coverage.",
    ],
}
(OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("Saved aggregate-only outputs to", OUT)
'''
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {
            "accelerator": "none",
            "dataSources": [],
            "dockerImageVersionId": 28755,
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": False,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

metadata = {
    "id": "tumis001/rsna-knee-input-fidelity-audit",
    "title": "RSNA Knee Input Fidelity Audit",
    "code_file": NOTEBOOK.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": ["tumis001/report-label-extraction"],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
}

NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
print(f"Wrote {NOTEBOOK}")
print(f"Wrote {METADATA}")
