"""Generate the single-run Kaggle TPU notebook from the fixed-path runner."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run.py"
NOTEBOOK = ROOT / "notebook_kernel/rsna_hpo_blend_auc_control_tpu_fold0.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    md(r"""
    # HPO v1 +281 ResNet: AUC-aligned controller on Kaggle TPU

    This is one controlled follow-up to the Modal +281 run. It keeps the same
    report-only HPO v1 training labels, the same fixed 825-study fold-0 validation
    set scored against Steven v4, ResNet-34, 24 training windows per study,
    effective study batch 8, and the 58-study gold lock.

    The experiment changes only controller behavior: `ReduceLROnPlateau` now
    monitors validation macro AUC in max mode, and early stopping monitors that
    same metric. LR patience is 5 epochs with two-epoch cooldown; early stopping
    waits 12 epochs without a new AUC best. The checkpoint remains the raw
    highest-AUC epoch. Training evaluates 48 windows to match the full-window
    Modal evaluation, masking XLA padding copies.

    A private Kaggle dataset is initialized and Kaggle credentials are verified
    **before training starts**. The best and latest checkpoints, history, curve,
    and progress status are published every two epochs and once at completion.
    The runner uses explicit input paths and does not recursively search datasets.
    """),
    code(RUNNER.read_text(encoding="utf-8")),
    code("main()\n"),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13"},
        "kaggle": {"accelerator": "tpu", "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook", "isGpuEnabled": False},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
