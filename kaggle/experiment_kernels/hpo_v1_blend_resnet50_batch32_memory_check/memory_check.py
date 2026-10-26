#!/usr/bin/env python3
"""One-step T4x2 memory check for global batch 32; no epoch/training loop."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


INPUT = Path("/kaggle/input")
SOURCE = INPUT / "datasets/tumis001/rsna-cnn-hpo-blend-xla-source-v1/skanee"
LABELS = INPUT / "datasets/tumis001/rsna-knee-hpo-v1-blended-report-labels/soft_labels_hpo_v1_blend.json"
CACHE_DIRS = [
    INPUT / f"notebooks/tumis001/rsna-cnn-top-k-support-cache-shard-{i}/slice_cache"
    for i in range(3)
]
BATCH_STUDIES = 32
WINDOW_CAP = 24


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(f"Expected two Kaggle T4 GPUs; found {torch.cuda.device_count()} GPU(s)")
    for path in [SOURCE / "scripts/train_cnn_baseline.py", LABELS,
                 SOURCE / "configs/report_folds_v5.csv", *[p / "cache_summary.json" for p in CACHE_DIRS]]:
        if not path.is_file():
            raise FileNotFoundError(f"Fixed input is missing: {path}")

    torch.cuda.set_device(0)
    print(f"Device: {torch.cuda.get_device_name(0)} + {torch.cuda.get_device_name(1)}", flush=True)
    sys.path.insert(0, str(SOURCE))
    trainer_path = SOURCE / "scripts/train_cnn_baseline.py"
    spec = importlib.util.spec_from_file_location("train_cnn_baseline", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer: {trainer_path}")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)

    base_folds = trainer.read_folds(SOURCE / "configs/report_folds_v5.csv")
    labels = trainer.read_soft_labels(LABELS)
    folds = dict(base_folds)
    folds.update({study_id: 5 for study_id in labels if study_id not in folds})
    manifest = trainer.load_manifest(CACHE_DIRS, ["unordered"])
    cache = trainer.StudyCache(manifest)
    train_ids, validation_ids = trainer.build_studies(cache, labels, folds, 0)
    if (len(train_ids), len(validation_ids)) != (3524, 825):
        raise RuntimeError(f"Unexpected fixed split: {len(train_ids)} train / {len(validation_ids)} validation")

    config = json.loads((SOURCE / "configs/cnn_hpo_v1_blend.json").read_text())
    config["model"]["backbone"] = "resnet50"
    config["version"] = "cnn_topk_support_v1_resnet50_t4x2_memory_check"
    # Select the first shuffled batch from the same fixed seed and split.
    rng = np.random.default_rng(config["seed"])
    order = rng.permutation(len(train_ids))
    study_ids = [train_ids[int(i)] for i in order[:BATCH_STUDIES]]
    device = torch.device("cuda:0")
    batch, study_of_slice, target, weight, _, _ = trainer.prepare_batch(
        cache, labels, study_ids, WINDOW_CAP, rng, device,
        augmentation=config.get("augmentation"),
    )
    windows = int(batch.shape[0])
    print(f"Prepared global batch: {len(study_ids)} studies; {windows} slice windows; "
          f"two-GPU split approximately {windows // 2}/{windows - windows // 2} windows.", flush=True)
    print("Run mode: one mixed-precision forward/backward step; no optimizer update and no epoch loop.", flush=True)

    class TwoT4SliceModel(trainer.SliceModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[0, 1], output_device=0)

        def state_dict(self, *args, **kwargs):
            result = super().state_dict(*args, **kwargs)
            return OrderedDict((key.replace("backbone.module.", "backbone.", 1), value)
                               for key, value in result.items())

        def load_state_dict(self, state_dict, *args, **kwargs):
            parallel = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith("backbone.") and not key.startswith("backbone.module."):
                    key = key.replace("backbone.", "backbone.module.", 1)
                parallel[key] = value
            return super().load_state_dict(parallel, *args, **kwargs)

    # Pretrained values do not affect activation memory; skip downloading weights.
    model = TwoT4SliceModel("resnet50", False, config["model"]["dropout"],
                            len(trainer.TARGETS), config["model"]["top_k_fraction"]).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["mixed_precision"]))
    optimizer.zero_grad(set_to_none=True)
    for gpu in range(2):
        torch.cuda.reset_peak_memory_stats(gpu)

    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(config["mixed_precision"])):
            logits = model(batch, study_of_slice, len(study_ids))
            cells = F.binary_cross_entropy_with_logits(logits.float(), target,
                                                       reduction="none")
            loss = (cells * weight).sum() / weight.sum().clamp_min(1.0)
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        result = {
            "status": "fits",
            "global_studies": BATCH_STUDIES,
            "window_cap_per_study": WINDOW_CAP,
            "slice_windows": windows,
            "per_gpu_peak_allocated_mib": [round(torch.cuda.max_memory_allocated(i) / 2**20, 1)
                                             for i in range(2)],
            "per_gpu_peak_reserved_mib": [round(torch.cuda.max_memory_reserved(i) / 2**20, 1)
                                            for i in range(2)],
            "loss": float(loss.detach().cpu()),
            "training_epochs": 0,
            "optimizer_steps": 0,
        }
    except torch.cuda.OutOfMemoryError as error:
        for gpu in range(2):
            torch.cuda.synchronize(gpu)
        result = {
            "status": "out_of_memory",
            "global_studies": BATCH_STUDIES,
            "window_cap_per_study": WINDOW_CAP,
            "slice_windows": windows,
            "per_gpu_peak_allocated_mib": [round(torch.cuda.max_memory_allocated(i) / 2**20, 1)
                                             for i in range(2)],
            "per_gpu_peak_reserved_mib": [round(torch.cuda.max_memory_reserved(i) / 2**20, 1)
                                            for i in range(2)],
            "error": str(error).splitlines()[0],
            "training_epochs": 0,
            "optimizer_steps": 0,
        }
        print(f"Caught expected memory-limit result: {result['error']}", flush=True)
    finally:
        result_path = Path("/kaggle/working/batch32_memory_check.json")
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print("MEMORY_CHECK_RESULT " + json.dumps(result), flush=True)

    if result["status"] != "fits":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
