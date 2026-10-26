"""CPU-only gold-holdout evaluation for the pairwise-AUC CNN checkpoint."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/kaggle/input")
SOURCE = ROOT / "datasets/tumis001/rsna-cnn-topk-support-source-v1/skanee"
CHECKPOINT = (
    ROOT / "notebooks/tumis001/rsna-cnn-top-k-pairwise-auc-fold-0/"
    "cnn_topk_pairwise_auc_fold0/checkpoint_best.pt"
)
TRAIN_CSV = ROOT / "competitions/rsna-knee-abnormality-detection/train.csv"
GOLD_CACHE = ROOT / "notebooks/tumis001/rsna-cnn-top-k-support-gold-cache/slice_cache"
OUTPUT = Path("/kaggle/working/cnn_pairwise_gold_eval")


def main() -> None:
    required = [SOURCE / "scripts/evaluate_gold_cnn.py", CHECKPOINT, TRAIN_CSV,
                GOLD_CACHE / "cache_summary.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing pinned evaluation input(s):\n" + "\n".join(missing))

    # This is an inference-only score against the existing 58-study development gold set.
    # Hide all accelerators explicitly, even if the kernel runtime exposes one unexpectedly.
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        sys.executable,
        "-u",
        str(SOURCE / "scripts/evaluate_gold_cnn.py"),
        "--cache",
        str(GOLD_CACHE),
        "--checkpoint",
        str(CHECKPOINT),
        "--train-csv",
        str(TRAIN_CSV),
        "--output-dir",
        str(OUTPUT),
    ]
    print("Pairwise checkpoint gold evaluation; training disabled; GPU disabled.", flush=True)
    print(f"Checkpoint: {CHECKPOINT}", flush=True)
    print("Running:", " ".join(command), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)


if __name__ == "__main__":
    main()
