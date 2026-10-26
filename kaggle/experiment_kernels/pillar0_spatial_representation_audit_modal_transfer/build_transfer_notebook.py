import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE_NOTEBOOK = REPOSITORY / "notebooks" / "kaggle_experiment" / "pillar0_spatial_representation_audit" / "pillar0_spatial_representation_audit.ipynb"
OUTPUT_DIR = Path(__file__).resolve().parent
notebook = json.loads(SOURCE_NOTEBOOK.read_text())

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Pillar-0 spatial audit: prepare transfer artifact\n",
            "\n",
            "This CPU-only Kaggle notebook selects one report-only study, reconstructs one series per standard plane, and writes the three preprocessed FP16 volumes to this private Kaggle notebook's output. The tensors stay in Kaggle storage until the separate Modal inference step downloads them into ephemeral memory. No training or official labels are used.\n",
        ],
    }
]

for index in [0, 1, 2, 3, 4, 5, 10]:
    source_cell = notebook["cells"][index]
    copied = {key: value for key, value in source_cell.items() if key in {"cell_type", "metadata", "source"}}
    if copied["cell_type"] == "code":
        source = "".join(copied["source"])
        if index == 0:
            source = source.replace("import torch\n", "")
            source = source.replace("torch.manual_seed(SEED)\n", "")
            source = source.replace("if torch.cuda.is_available():\n    torch.cuda.manual_seed_all(SEED)\n", "")
            source = source.replace(
                'print("CUDA available:", torch.cuda.is_available())\n'
                'if torch.cuda.is_available():\n'
                '    print("GPU:", torch.cuda.get_device_name(0))\n',
                'print("GPU use: none; preprocessing only")\n',
            )
            source = source.replace("display(train.head(3))\ndisplay(train_series.head(3))", "")
        if index == 2:
            source = source.replace(
                "%pip install -q pydicom==3.0.1",
                "import pydicom\nimport scipy\n"
                "print('Using preinstalled pydicom', pydicom.__version__, 'and scipy', scipy.__version__)",
            )
            source = source.replace("import matplotlib.pyplot as plt\n", "import matplotlib.pyplot as plt\nimport torch\n")
        copied["source"] = source.splitlines(keepends=True)
    cells.append(copied)

cells.append(
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Pack the three preprocessed volumes\n",
            "\n",
            "The archive contains only FP16 model inputs, low-resolution source-support fractions, and plane names. It contains no study/series identifiers or labels. The Modal job retrieves this small-kernel artifact from Kaggle, performs frozen FP32 inference, and deletes its ephemeral copy after loading.\n",
        ],
    }
)

cells.append(
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": '''import json
import numpy as np
import torch
import torch.nn.functional as F

OUTPUT_DIR = Path("/kaggle/working/pillar0_modal_transfer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

expected_planes = {"Axial", "Coronal", "Sagittal"}
actual_planes = set(representative_rows["Anatomical_Plane"].astype(str))
if actual_planes != expected_planes or len(representative_rows) != 3:
    raise RuntimeError(f"Expected exactly three standard planes; found {sorted(actual_planes)}")

volumes = []
supports = []
planes = []
support_fractions = []
native_finite_fractions = []

for row in representative_rows.itertuples(index=False):
    plane = str(row.Anatomical_Plane)
    print(f"Preparing {plane} volume on Kaggle CPU", flush=True)
    native_volume, native_geometry = reconstruct_series(
        row.StudyInstanceUID, row.SeriesInstanceUID
    )
    native_finite_fractions.append(float(np.isfinite(native_volume).mean()))
    canonical, support_canonical, _ = canonical_resample_lps(
        native_volume, native_geometry["affine_array_to_lps"]
    )
    windowed, window_low, window_high = high_contrast_mr_window(
        canonical, support_canonical
    )
    padded, _ = center_pad_or_crop_zyx(windowed, TARGET_SHAPE_ZYX)
    support_padded, _ = center_pad_or_crop_zyx(
        torch.from_numpy(support_canonical), TARGET_SHAPE_ZYX
    )

    # Existing verified checkpoint input convention: [C,H,W,D] = [3,384,384,192].
    model_hwd = padded.permute(1, 2, 0).contiguous()
    support_hwd = support_padded.permute(1, 2, 0).contiguous().clamp_(0, 1)
    support_grid = F.adaptive_avg_pool3d(
        support_hwd[None, None].float(), (64, 32, 16)
    )[0, 0]

    volume_np = model_hwd.numpy().astype(np.float16)
    support_np = support_grid.numpy().astype(np.float16)
    if volume_np.shape != (384, 384, 192) or support_np.shape != (64, 32, 16):
        raise RuntimeError(f"Unexpected {plane} tensor shapes: {volume_np.shape}, {support_np.shape}")
    if not np.isfinite(volume_np).all() or not np.isfinite(support_np).all():
        raise RuntimeError(f"Non-finite value in {plane} tensor")

    volumes.append(volume_np)
    supports.append(support_np)
    planes.append(plane)
    support_fractions.append(float(support_np.mean()))
    del native_volume, canonical, support_canonical, windowed, padded, support_padded

archive_path = OUTPUT_DIR / "pillar0_transfer.npz"
np.savez_compressed(
    archive_path,
    volumes=np.stack(volumes),
    supports=np.stack(supports),
    planes=np.asarray(planes),
)
manifest = {
    "scope": "three report-only series; preprocessed inputs only; no labels or identifiers",
    "preprocessing": "mask-aware linear resampling; intensity window estimated from voxels with support >= 0.5; unsupported voxels zeroed after normalization",
    "support_is_encoder_input": False,
    "planes": planes,
    "volume_shape_per_plane": [384, 384, 192],
    "volume_dtype": "float16",
    "support_shape_per_plane": [64, 32, 16],
    "support_dtype": "float16",
    "valid_source_support_fraction": support_fractions,
    "native_finite_voxel_fraction": native_finite_fractions,
    "native_nan_or_infinite_voxel_fraction": [1.0 - value for value in native_finite_fractions],
    "archive_bytes": archive_path.stat().st_size,
}
(OUTPUT_DIR / "transfer_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
print("Kaggle output artifact ready:", archive_path, flush=True)
'''.splitlines(keepends=True),
    }
)

result = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(OUTPUT_DIR / "pillar0_spatial_transfer.ipynb").write_text(json.dumps(result, indent=1) + "\n")

metadata = {
    "id": "tumis001/rsna-knee-pillar0-spatial-transfer",
    "title": "RSNA Knee Pillar0 Spatial Transfer",
    "code_file": "pillar0_spatial_transfer.ipynb",
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
}
(OUTPUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"Created {len(cells)} cells at {OUTPUT_DIR}")
