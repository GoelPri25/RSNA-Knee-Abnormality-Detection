import json
from pathlib import Path


experiment_dir = Path(__file__).resolve().parents[1]
source_path = experiment_dir / "pillar0_spatial_representation_audit" / "pillar0_spatial_representation_audit.ipynb"
output_dir = Path(__file__).resolve().parent
source = json.loads(source_path.read_text())

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Pillar-0 spatial audit: visual QA\n",
            "\n",
            "CPU-only preprocessing check for a report-only knee study. This notebook reconstructs the three acquired planes from DICOM geometry and displays central orthogonal views. It does not load a model, train, or read official labels. Review the images before running the separate frozen-feature audit.\n",
        ],
    }
]

for cell in source["cells"][:5]:
    copied = {key: value for key, value in cell.items() if key in {"cell_type", "metadata", "source"}}
    if copied["cell_type"] == "code":
        text = "".join(copied["source"])
        text = text.replace("import torch\n", "")
        text = text.replace("torch.manual_seed(SEED)\n", "")
        text = text.replace("if torch.cuda.is_available():\n    torch.cuda.manual_seed_all(SEED)\n", "")
        text = text.replace('print("CUDA available:", torch.cuda.is_available())\nif torch.cuda.is_available():\n    print("GPU:", torch.cuda.get_device_name(0))\n', 'print("GPU use: none (CPU-only visual QA)")\n')
        copied["source"] = text.splitlines(keepends=True)
    cells.append(copied)

cells.append(source["cells"][5])

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(output_dir / "pillar0_spatial_audit_visual_qa.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")

metadata = {
    "id": "tumis001/rsna-knee-pillar0-spatial-audit-visual-qa",
    "title": "RSNA Knee Pillar0 Spatial Audit Visual QA",
    "code_file": "pillar0_spatial_audit_visual_qa.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "kernel_sources": [],
    "model_sources": [],
}
(output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

print(f"Created notebook with {len(cells)} cells at {output_dir}")
