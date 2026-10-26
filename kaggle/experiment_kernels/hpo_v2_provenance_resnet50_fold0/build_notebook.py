#!/usr/bin/env python3
"""Embed the fixed runner in a Kaggle notebook for interactive log visibility."""

import json
from pathlib import Path

root = Path(__file__).resolve().parent
runner = (root / "run.py").read_text(encoding="utf-8")
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "hpo-v2-description",
            "metadata": {},
            "source": [
                "# HPO v2 provenance label comparison\n",
                "\n",
                "One controlled change from the fixed HPO-v1 +281 ResNet-50 recipe: "
                "use the audited provenance-aware HPO-v2 probabilities for training. "
                "The 825-study Steven-v4 report validation fold remains fixed; gold labels stay excluded.\n",
                "\n",
                "The runner uses fixed Kaggle input paths and verifies the label, config, and split checksums. "
                "It saves model checkpoints and history in `/kaggle/working` after every epoch.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "hpo-v2-training",
            "metadata": {},
            "outputs": [],
            "source": runner.splitlines(keepends=True),
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(root / "rsna_hpo_v2_provenance_resnet50_fold0.ipynb").write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(root / "rsna_hpo_v2_provenance_resnet50_fold0.ipynb")
