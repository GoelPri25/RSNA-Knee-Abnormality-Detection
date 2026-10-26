import json
from pathlib import Path


HERE = Path(__file__).resolve().parent

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Pillar-0 support-path probe (inference inputs only)\n",
            "\n",
            "This CPU-only notebook replays the archived Pillar-0 preprocessing on one report-only study with axial, coronal, and sagittal series. It compares that historical path with a controlled support-aware variant using the same geometry, resampling, target shape, and channel repetition. It does not load a model, train, or export identifiers, reports, or labels.\n",
            "\n",
            "Only report-only studies are selected by checking that all official target cells are null. Their values are never read. Aggregates and a paired input tensor archive are saved under `/kaggle/working/pillar0_support_path_probe` for later review; all data remains in Kaggle.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": '''import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

SEED = 20260925
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_CANDIDATES = [
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
]
DATA = next((p for p in DATA_CANDIDATES if p.is_dir()), None)
if DATA is None:
    raise FileNotFoundError("Competition data is not attached to this kernel")

OUT = Path("/kaggle/working/pillar0_support_path_probe")
OUT.mkdir(parents=True, exist_ok=True)
TARGET_DHW = (192, 384, 384)
TARGET_SPACING_DHW_MM = (1.0, 1.0, 1.0)
PLANES = {"Axial", "Coronal", "Sagittal"}

print("Data mounted:", DATA)
print("pydicom:", pydicom.__version__)
print("Device: CPU; no model inference or training")
'''.splitlines(keepends=True),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": '''train = pd.read_csv(DATA / "train.csv")
series = pd.read_csv(DATA / "train_series.csv")
submission = pd.read_csv(DATA / "sample_submission.csv")
targets = [column for column in submission.columns if column != "StudyInstanceUID"]
train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)

# Read only the all-null predicate; no official label value enters this probe.
report_only_ids = set(
    train.loc[train[targets].isna().all(axis=1), "StudyInstanceUID"]
)
selected = None
selected_study = None
for study_id in sorted(report_only_ids):
    candidates = series[
        (series["StudyInstanceUID"] == study_id)
        & series["Anatomical_Plane"].isin(PLANES)
    ].copy()
    if candidates.empty:
        continue
    candidates = candidates.sort_values(
        ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression", "SeriesInstanceUID"],
        ascending=[True, False, False, True],
    )
    one_per_plane = candidates.groupby("Anatomical_Plane", as_index=False).head(1)
    if set(one_per_plane["Anatomical_Plane"]) == PLANES:
        selected = one_per_plane.sort_values("Anatomical_Plane").reset_index(drop=True)
        selected_study = study_id
        break

if selected is None:
    raise RuntimeError("No report-only study has all three standard planes")

print("Selected one report-only study with three standard planes.")
print("Series count:", len(selected), "(no identifiers or labels exported)")
'''.splitlines(keepends=True),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": '''def reconstruct_with_support(folder):
    paths = sorted(Path(folder).glob("*.dcm"))
    if len(paths) < 2:
        raise ValueError("A series needs at least two DICOM slices")
    headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in paths]
    iop = np.asarray(headers[0].ImageOrientationPatient, dtype=np.float64)
    col_direction, row_direction = iop[:3], iop[3:]
    normal = np.cross(col_direction, row_direction)
    normal /= np.linalg.norm(normal)
    positions = np.asarray([h.ImagePositionPatient for h in headers], dtype=np.float64)
    order = np.argsort(positions @ normal, kind="stable")
    projections = positions[order] @ normal
    gaps = np.diff(projections)
    if np.any(gaps <= 1e-3):
        raise ValueError("Repeated slice positions")
    dz = float(np.median(gaps))
    if not np.allclose(gaps, dz, atol=0.05, rtol=0.01):
        raise ValueError("Irregular slice spacing")
    spacing = np.asarray(headers[0].PixelSpacing, dtype=np.float64)

    old_pixels, valid_masks = [], []
    for index in order:
        ds = pydicom.dcmread(paths[index])
        raw = np.asarray(ds.pixel_array, dtype=np.float32)
        old = raw * float(getattr(ds, "RescaleSlope", 1.0))
        old += float(getattr(ds, "RescaleIntercept", 0.0))
        valid = np.isfinite(old)
        pad = getattr(ds, "PixelPaddingValue", None)
        pad_limit = getattr(ds, "PixelPaddingRangeLimit", pad)
        if pad is not None:
            lo, hi = sorted((float(pad), float(pad_limit)))
            valid &= ~((raw >= lo) & (raw <= hi))
        # The historical notebook used old pixels as-is; the boolean map is
        # retained only for the corrected path and is never inferred from zero.
        old_pixels.append(np.nan_to_num(old, nan=0.0, posinf=0.0, neginf=0.0))
        valid_masks.append(valid)

    old_volume = np.stack(old_pixels).astype(np.float32)
    support = np.stack(valid_masks).astype(np.float32)
    spacing_dhw = (dz, float(spacing[0]), float(spacing[1]))
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = normal * dz
    affine[:3, 1] = row_direction * spacing[0]
    affine[:3, 2] = col_direction * spacing[1]
    affine[:3, 3] = positions[order][0]
    return old_volume, support, spacing_dhw, affine


def center_pad_or_crop_pair(volume, support, target_shape):
    # Same centre crop/pad as the historical pipeline, applied to both arrays.
    values, mask = volume, support
    for axis, target in enumerate(target_shape):
        current = values.shape[axis]
        if current > target:
            start = (current - target) // 2
            sl = [slice(None)] * 3
            sl[axis] = slice(start, start + target)
            values, mask = values[tuple(sl)], mask[tuple(sl)]
    padding = []
    for current, target in reversed(list(zip(values.shape, target_shape))):
        amount = max(target - current, 0)
        padding.extend([amount // 2, amount - amount // 2])
    values = F.pad(values[None, None], padding, value=0).squeeze(0).squeeze(0)
    mask = F.pad(mask[None, None], padding, value=0).squeeze(0).squeeze(0)
    return values, mask


def percentile_window(values, low_q, high_q):
    lo, hi = torch.quantile(values.float(), torch.tensor([low_q, high_q]))
    return float(lo), float(hi)


def normalize(values, low, high, support=None):
    out = ((values - low) / max(high - low, 1e-6)).clamp(0, 1)
    if support is not None:
        out = torch.where(support > 1e-6, out, torch.zeros_like(out))
    return out


def stable_cosine(a, b, mask=None, chunk_size=1_000_000):
    # Double-precision chunked reductions avoid float32 cosine values slightly
    # above one on these large, nearly identical tensors.
    left, right = a.reshape(-1), b.reshape(-1)
    keep = mask.reshape(-1) if mask is not None else None
    dot = torch.zeros((), dtype=torch.float64)
    left_sq = torch.zeros((), dtype=torch.float64)
    right_sq = torch.zeros((), dtype=torch.float64)
    for start in range(0, left.numel(), chunk_size):
        stop = min(start + chunk_size, left.numel())
        x = left[start:stop].double()
        y = right[start:stop].double()
        if keep is not None:
            selected = keep[start:stop]
            x, y = x[selected], y[selected]
        dot += torch.dot(x, y)
        left_sq += torch.dot(x, x)
        right_sq += torch.dot(y, y)
    denominator = torch.sqrt(left_sq * right_sq)
    return (dot / denominator.clamp_min(1e-30)).item()


def build_pair(native, native_support, spacing_dhw):
    source = torch.from_numpy(native).float()[None, None]
    source_support = torch.from_numpy(native_support).float()[None, None]
    target_shape = [
        max(1, int(round(size * spacing / target_spacing)))
        for size, spacing, target_spacing in zip(
            native.shape, spacing_dhw, TARGET_SPACING_DHW_MM
        )
    ]

    # Historical path from pillar0_small_sample.ipynb: trilinear resize,
    # exclude all exact zeros from the p20/p99 window, then centre pad/crop.
    legacy_resampled = F.interpolate(
        source, size=target_shape, mode="trilinear", align_corners=False
    )[0, 0]
    legacy_values = legacy_resampled[legacy_resampled != 0]
    if legacy_values.numel() == 0:
        legacy_values = legacy_resampled.flatten()
    legacy_low, legacy_high = percentile_window(legacy_values, 0.20, 0.99)
    legacy_windowed = normalize(legacy_resampled, legacy_low, legacy_high)
    legacy_windowed, _ = center_pad_or_crop_pair(
        legacy_windowed, torch.ones_like(legacy_windowed), TARGET_DHW
    )

    # Controlled support-aware path: same native axes, target grid, channel
    # replication, and centre crop/pad. Resample signal and support separately,
    # divide out partial-support dilution, and exclude only invalid pixels.
    safe_signal = torch.where(source_support > 0, source, torch.zeros_like(source))
    numerator = F.interpolate(
        safe_signal, size=target_shape, mode="trilinear", align_corners=False
    )[0, 0]
    resampled_support = F.interpolate(
        source_support, size=target_shape, mode="trilinear", align_corners=False
    )[0, 0].clamp(0, 1)
    corrected = torch.where(
        resampled_support > 1e-6,
        numerator / resampled_support.clamp_min(1e-6),
        torch.zeros_like(numerator),
    )
    legacy_zero_fraction = float((legacy_resampled == 0).float().mean())
    support_aware_zero_fraction = float(
        ((corrected == 0) & (resampled_support >= 0.5)).float().sum()
        / (resampled_support >= 0.5).float().sum().clamp_min(1)
    )
    core = corrected[resampled_support >= 0.5]
    if core.numel() == 0:
        raise RuntimeError("No supported voxels remain after resampling")
    aware_low, aware_high = percentile_window(core, 0.20, 0.99)
    aware_windowed = normalize(corrected, aware_low, aware_high, resampled_support)
    aware_windowed, aware_support = center_pad_or_crop_pair(
        aware_windowed, resampled_support, TARGET_DHW
    )

    # Convert D,H,W -> H,W,D. Save grayscale inputs; repeat to three channels
    # only when/if the frozen checkpoint is invoked downstream.
    legacy_input = legacy_windowed.permute(1, 2, 0).contiguous()
    aware_input = aware_windowed.permute(1, 2, 0).contiguous()
    support_input = aware_support.permute(1, 2, 0).contiguous().clamp(0, 1)
    legacy_input = legacy_input[None].contiguous()
    aware_input = aware_input[None].contiguous()
    return {
        "legacy": legacy_input,
        "aware": aware_input,
        "support": support_input,
        "legacy_window": (legacy_low, legacy_high),
        "aware_window": (aware_low, aware_high),
        "resampled_shape_dhw": target_shape,
        "resampled_support_fraction": float((resampled_support >= 0.5).float().mean()),
        "resampled_partial_fraction": float(((resampled_support > 1e-6) & (resampled_support < 0.999)).float().mean()),
        "legacy_zero_fraction": legacy_zero_fraction,
        "support_aware_zero_fraction": support_aware_zero_fraction,
    }
'''.splitlines(keepends=True),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": '''rows = []
legacy_inputs, aware_inputs, support_grids = [], [], []
for row in selected.itertuples(index=False):
    plane = str(row.Anatomical_Plane)
    folder = DATA / "train_series" / str(row.StudyInstanceUID) / str(row.SeriesInstanceUID)
    if not folder.is_dir():
        # Kaggle competition mount layout fallback; no paths are exported.
        folder = DATA / "train_series" / str(row.SeriesInstanceUID)
    print("Processing", plane, "on Kaggle CPU", flush=True)
    native, native_support, spacing, affine = reconstruct_with_support(folder)
    pair = build_pair(native, native_support, spacing)
    a, b = pair["legacy"].float(), pair["aware"].float()
    difference = (a - b).abs()
    cos_all = stable_cosine(a, b)
    core = pair["support"] >= 0.95
    core_3 = core.unsqueeze(0).expand_as(a)
    core_a, core_b = a[core_3], b[core_3]
    cos_core = stable_cosine(a, b, core_3) if core_a.numel() else float("nan")
    support = pair["support"]
    flat_difference = difference.flatten()
    # PyTorch CPU quantile rejects very large tensors. A fixed-stride sample
    # bounds memory and is deterministic while preserving the tail estimate.
    quantile_stride = max(1, int(np.ceil(flat_difference.numel() / 1_000_000)))
    sampled_absdiff_p99 = torch.quantile(
        flat_difference[::quantile_stride], 0.99
    )
    rows.append({
        "plane": plane,
        "fluid_sensitive": int(bool(row.Fluid_Sensitive)),
        "native_shape_dhw": "x".join(map(str, native.shape)),
        "native_spacing_dhw_mm": ",".join(f"{x:.4f}" for x in spacing),
        "native_declared_invalid_fraction": float(1.0 - native_support.mean()),
        "native_exact_zero_fraction_among_supported": float(
            (((native == 0) & (native_support > 0)).sum())
            / max(float(native_support.sum()), 1.0)
        ),
        "resampled_shape_dhw": "x".join(map(str, pair["resampled_shape_dhw"])),
        "resampled_support_fraction": pair["resampled_support_fraction"],
        "resampled_partial_support_fraction": pair["resampled_partial_fraction"],
        "legacy_resampled_exact_zero_fraction": pair["legacy_zero_fraction"],
        "support_aware_resampled_exact_zero_fraction": pair["support_aware_zero_fraction"],
        "final_encoder_support_fraction": float(support.mean()),
        "final_partial_support_fraction": float(((support > 1e-6) & (support < 0.999)).float().mean()),
        "legacy_window_p20": pair["legacy_window"][0],
        "legacy_window_p99": pair["legacy_window"][1],
        "aware_window_p20": pair["aware_window"][0],
        "aware_window_p99": pair["aware_window"][1],
        "input_cosine_all": float(cos_all),
        "input_cosine_supported_core": float(cos_core),
        "input_mae_all": float(difference.mean()),
        "input_mae_supported_core": float(difference[core_3].mean()) if core_a.numel() else float("nan"),
        "input_absdiff_p99_sampled": float(sampled_absdiff_p99),
        "legacy_input_rms": float(a.square().mean().sqrt()),
        "aware_input_rms": float(b.square().mean().sqrt()),
    })
    # Store grayscale model inputs; the repeated channels are deterministic.
    legacy_inputs.append(a[0].half().cpu().numpy())
    aware_inputs.append(b[0].half().cpu().numpy())
    support_grid = F.adaptive_avg_pool3d(
        support[None, None].float(), (64, 32, 16)
    )[0, 0]
    support_grids.append(support_grid.half().cpu().numpy())
    print(
        f"{plane}: source invalid={rows[-1]['native_declared_invalid_fraction']:.6f}, "
        f"model support={rows[-1]['final_encoder_support_fraction']:.4f}, "
        f"input cosine={rows[-1]['input_cosine_all']:.6f}",
        flush=True,
    )
    del native, native_support, pair, a, b, difference

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "paired_input_metrics.csv", index=False)
np.savez_compressed(
    OUT / "paired_inputs.npz",
    legacy=np.stack(legacy_inputs),
    support_aware=np.stack(aware_inputs),
    support_grid=np.stack(support_grids),
    planes=metrics["plane"].astype(str).to_numpy(dtype="U16"),
)
with np.load(OUT / "paired_inputs.npz", allow_pickle=False) as saved:
    if saved["legacy"].shape != saved["support_aware"].shape:
        raise RuntimeError("Paired input shapes differ")
    if saved["support_grid"].shape != (3, 64, 32, 16):
        raise RuntimeError("Unexpected support-map shape")
    if saved["planes"].dtype.kind != "U":
        raise RuntimeError("Plane metadata must be safe Unicode, not pickle")

summary = {
    "audit": "pillar0_legacy_vs_support_aware_inputs_no_model_no_training",
    "seed": SEED,
    "study_count": 1,
    "series_count": len(rows),
    "planes": metrics["plane"].tolist(),
    "official_label_values_used": False,
    "only_official_nullness_used_to_select_report_only": True,
    "report_text_exported": False,
    "identifiers_exported": False,
    "model_loaded": False,
    "training_performed": False,
    "historical_preprocessing_source": "kaggle/experiment_kernels/pillar0_small_sample.ipynb, preprocess_for_pillar",
    "support_policy": "finite decoded values excluding PixelPaddingValue/PixelPaddingRangeLimit; exact zero remains valid",
    "input_tensor_shape_per_series": [1, 384, 384, 192],
    "input_dtype_in_archive": "float16, one channel per variant; support map pooled to 64x32x16",
    "archive_bytes": (OUT / "paired_inputs.npz").stat().st_size,
    "limitations": [
        "One report-only study; this is a preprocessing diagnostic, not a generalization test.",
        "PixelPadding tags cannot identify scanner background that is not declared as padding.",
        "Support is not an anatomical segmentation.",
        "No encoder embeddings were computed in this CPU preprocessing kernel.",
    ],
}
(OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))
print("\\nAggregate input comparison:")
display(metrics.round(6))
print("\\nSummary:")
print(json.dumps(summary, indent=2))
print("Saved identifier-free outputs to", OUT, flush=True)
'''.splitlines(keepends=True),
    },
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(HERE / "pillar0_support_path_probe.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
(HERE / "kernel-metadata.json").write_text(json.dumps({
    "id": "tumis001/rsna-knee-pillar0-support-path-probe",
    "title": "RSNA Knee Pillar0 Support Path Probe",
    "code_file": "pillar0_support_path_probe.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
}, indent=2) + "\n")
(HERE / "README.md").write_text("""# Pillar-0 support-path probe\n\nThis private, CPU-only Kaggle notebook replays the historical Pillar-0 transfer preprocessing next to a controlled support-aware variant on three report-only MRI series from one study. It reads official-label nullness only to select report-only data; it does not read label values, report text, load the frozen encoder, or train. The output archive contains paired one-channel FP16 inputs and a support map, with plane names only. It stays in Kaggle and is not uploaded to Modal by this notebook.\n\nRun `python build_notebook.py` to regenerate the notebook and kernel metadata.\n""")
print(f"Built notebook at {HERE / 'pillar0_support_path_probe.ipynb'}")
