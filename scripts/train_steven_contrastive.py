"""Train the Steven-v4 Experiment-4 reader with target-aware contrastive loss.

This changes only the training objective and adds an auxiliary projection head.
The MedSigLIP image encoder, cached per-slice features, hierarchical MIL reader,
label source, folds, optimizer schedule, and checkpoint-selection metric remain
the established Steven-v4 configuration. Gold labels are not loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rsna_knee.contrastive import (
    TargetMemoryBank,
    soft_mass_contrastive_loss,
    target_aware_contrastive_loss,
)


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


class ContrastiveProjection(nn.Module):
    """Map one shared study representation into one space per finding."""

    def __init__(self, input_dim: int = 256, targets: int = 12,
                 projection_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.targets = targets
        self.projection_dim = projection_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, targets * projection_dim),
        )

    def forward(self, study_embedding: torch.Tensor) -> torch.Tensor:
        projected = self.network(study_embedding).reshape(
            len(study_embedding), self.targets, self.projection_dim)
        return nn.functional.normalize(projected, dim=-1)


def read_steven_labels(path: Path, excluded_ids: set[str]) -> dict:
    items, seen = [], set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "StudyInstanceUID" not in reader.fieldnames:
            raise ValueError("Steven v4 label CSV must contain StudyInstanceUID")
        missing = set(TARGETS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Steven v4 CSV is missing targets: {sorted(missing)}")
        for row in reader:
            study_id = str(row["StudyInstanceUID"])
            if study_id in excluded_ids:
                continue
            if study_id in seen:
                raise ValueError(f"Duplicate report-label study ID: {study_id}")
            seen.add(study_id)
            probabilities, weights = [], []
            for target in TARGETS:
                try:
                    value = float(row[target])
                except (TypeError, ValueError):
                    value = float("nan")
                valid = bool(np.isfinite(value) and 0.0 <= value <= 1.0)
                probabilities.append(value if valid else 0.5)
                weights.append(float(valid))
            items.append({"study_id": study_id, "probabilities": probabilities,
                          "weights": weights})
    return {"version": "steven_v4", "targets_in_order": TARGETS, "labels": items}


def read_development_ids(path: Path) -> list[str]:
    """Read only IDs/split assignments; never load any gold labels."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = set(reader.fieldnames or [])
        id_column = "StudyInstanceUID" if "StudyInstanceUID" in names else "study_id"
        split_column = "split" if "split" in names else "gold_split"
        if id_column not in names or split_column not in names:
            raise ValueError("Gold split CSV must contain study ID and split columns")
        rows = list(reader)
    ids = sorted({str(row[id_column]) for row in rows
                  if str(row[split_column]).lower() == "development"})
    if len(ids) != 29:
        raise ValueError(f"Expected 29 development IDs; found {len(ids)}")
    return ids


def write_learning_curve(path: Path, history: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_classification_bce"] for row in history],
                 label="Training BCE")
    axes[0].plot(epochs, [row["report_validation_bce"] for row in history],
                 label="Report-validation BCE")
    best = int(np.argmin([row["report_validation_bce"] for row in history]))
    axes[0].scatter([epochs[best]], [history[best]["report_validation_bce"]],
                    color="red", label=f"Selected checkpoint (epoch {best + 1})", zorder=3)
    axes[0].set(xlabel="Epoch", ylabel="Class-balanced BCE",
                title="Prediction learning curves")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_contrastive_loss"] for row in history],
                 color="purple", label="Target-aware contrastive loss")
    axes[1].plot(epochs, [row["eligible_anchors"] for row in history],
                 color="teal", label="Anchors with positive + negative keys")
    axes[1].set(xlabel="Epoch", title="Auxiliary objective and pair coverage")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.25)
    figure.tight_layout()
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=140)
    plt.close(figure)
    os.replace(temporary, path)


def train(model: nn.Module, projector: nn.Module, training: Sequence,
          validation: Sequence, config: dict, device: torch.device,
          output: Path) -> tuple[list[dict], int, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    from src.mil.data import collate
    from src.mil.train import epoch_examples, target_mean_bce, validation_loss

    rng = np.random.default_rng(config["seed"])
    torch.manual_seed(config["seed"])
    model.to(device)
    projector.to(device)
    parameters = list(model.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    batch_size = int(config["batch_size"])
    contrastive_config = config["contrastive"]
    contrastive_weight = float(contrastive_config["weight"])
    steps_per_epoch = math.ceil(len(training) / batch_size)
    total_steps = steps_per_epoch * int(config["epochs"])
    warmup = max(1, int(total_steps * config["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total_steps)))),
    )
    memory = TargetMemoryBank(contrastive_config["memory_capacity"])
    history: list[dict] = []
    best_loss, best_epoch = float("inf"), -1
    best_reader_state: dict[str, torch.Tensor] | None = None
    best_projector_state: dict[str, torch.Tensor] | None = None
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["epochs"]):
        started = time.monotonic()
        model.train()
        projector.train()
        examples = epoch_examples(training, config, rng)
        sums = {key: 0.0 for key in
                ("loss", "classification", "contrastive", "eligible", "pairs", "active_targets")}
        batch_count = 0
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            batch = {key: value.to(device) for key, value in collate(
                chunk, config["max_slices_train"], rng).items()}
            output_batch = model(
                batch["embeddings"], batch["positions"], batch["plane"],
                batch["fluid"], batch["slice_mask"], batch["series_mask"])
            classification_loss = target_mean_bce(
                output_batch["logits"], batch["target"], batch["weight"])
            if contrastive_weight > 0:
                projected = projector(output_batch["z_study"])
                common = {
                    "temperature": contrastive_config["temperature"],
                }
                if contrastive_config["objective"] == "absolute_mass_v2":
                    contrastive = soft_mass_contrastive_loss(
                        projected, batch["target"], batch["weight"],
                        [item.study_id for item in chunk], memory, **common,
                    )
                elif contrastive_config["objective"] == "thresholded_v1":
                    contrastive = target_aware_contrastive_loss(
                        projected, batch["target"], batch["weight"],
                        [item.study_id for item in chunk], memory,
                        positive_threshold=contrastive_config["positive_threshold"],
                        negative_threshold=contrastive_config["negative_threshold"],
                        **common,
                    )
                else:
                    raise ValueError(
                        f"Unknown contrastive objective: {contrastive_config['objective']}"
                    )
                contrastive_loss = contrastive.loss
                eligible_anchors = contrastive.eligible_anchors
                positive_pairs = contrastive.positive_pairs
                active_targets = contrastive.active_targets
            else:
                contrastive_loss = classification_loss.new_zeros(())
                eligible_anchors = positive_pairs = active_targets = 0
            loss = classification_loss + contrastive_weight * contrastive_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"])
            optimizer.step()
            scheduler.step()
            # Queue keys are detached and enter only after this batch's loss.
            if contrastive_weight > 0:
                memory.enqueue(projected.detach(), batch["target"], batch["weight"],
                               [item.study_id for item in chunk])
            sums["loss"] += float(loss.detach())
            sums["classification"] += float(classification_loss.detach())
            sums["contrastive"] += float(contrastive_loss.detach())
            sums["eligible"] += eligible_anchors
            sums["pairs"] += positive_pairs
            sums["active_targets"] += active_targets
            batch_count += 1

        report_val = validation_loss(model, validation, config, device)
        denominator = max(1, batch_count)
        record = {
            "epoch": epoch,
            "train_total_loss": sums["loss"] / denominator,
            "train_classification_bce": sums["classification"] / denominator,
            "train_contrastive_loss": sums["contrastive"] / denominator,
            "report_validation_bce": report_val,
            "eligible_anchors": int(sums["eligible"]),
            "positive_pairs": int(sums["pairs"]),
            "active_target_batches": int(sums["active_targets"]),
            "memory_bank_size": len(memory),
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.monotonic() - started, 2),
        }
        if not math.isfinite(report_val):
            raise FloatingPointError(f"Non-finite report-validation BCE at epoch {epoch}")
        history.append(record)
        if report_val < best_loss:
            best_loss, best_epoch = report_val, epoch
            best_reader_state = {key: value.detach().cpu().clone()
                                 for key, value in model.state_dict().items()}
            best_projector_state = {key: value.detach().cpu().clone()
                                    for key, value in projector.state_dict().items()}

        (output / "training_history.json").write_text(
            json.dumps({"history": history}, indent=2), encoding="utf-8")
        write_learning_curve(output / "learning_curves.png", history)
        save_checkpoint(output, config, model, projector, best_epoch, best_loss,
                        history, best_reader_state, best_projector_state, epoch)
        print(
            f"epoch {epoch + 1:02d}/{config['epochs']}: "
            f"train={record['train_classification_bce']:.4f} "
            f"contrastive={record['train_contrastive_loss']:.4f} "
            f"val={report_val:.4f} best={best_epoch + 1} "
            f"anchors={record['eligible_anchors']} "
            f"queue={record['memory_bank_size']} "
            f"lr={record['lr']:.2e} ({record['seconds']:.1f}s)",
            flush=True,
        )

    if best_reader_state is None or best_projector_state is None:
        raise RuntimeError("No valid checkpoint was selected")
    model.load_state_dict(best_reader_state)
    projector.load_state_dict(best_projector_state)
    return history, best_epoch, best_reader_state, best_projector_state


def save_checkpoint(output: Path, config: dict, model: nn.Module,
                    projector: nn.Module, best_epoch: int, best_loss: float,
                    history: list[dict], best_reader: dict[str, torch.Tensor] | None,
                    best_projector: dict[str, torch.Tensor] | None,
                    completed_epoch: int) -> None:
    """Save inference-compatible reader weights and separate objective weights."""
    metadata = {
        "config": config,
        "label_version": "steven_v4",
        "best_epoch": best_epoch,
        "best_report_validation_bce": best_loss,
        "completed_epochs": completed_epoch + 1,
        "gold_evaluation_used": False,
        "contrastive_head_is_inference_required": False,
        "history": history,
    }
    payload = {
        **metadata,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "contrastive_state_dict": {key: value.detach().cpu()
                                   for key, value in projector.state_dict().items()},
    }
    temporary = output / "latest_checkpoint.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, output / "latest_checkpoint.pt")
    if best_epoch == completed_epoch and best_reader is not None and best_projector is not None:
        best_payload = {
            **metadata,
            "state_dict": best_reader,
            "contrastive_state_dict": best_projector,
        }
        temporary = output / "best_checkpoint.pt.tmp"
        torch.save(best_payload, temporary)
        os.replace(temporary, output / "best_checkpoint.pt")
    (output / "training_progress.json").write_text(json.dumps({
        "completed_epochs": completed_epoch + 1,
        "best_epoch": best_epoch + 1,
        "best_report_validation_bce": best_loss,
        "latest_report_validation_bce": history[-1]["report_validation_bce"],
        "contrastive_head_inference_required": False,
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Experiment-4 source tree containing src/mil and configs/")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--steven-labels", type=Path, required=True)
    parser.add_argument("--gold-split", type=Path, required=True,
                        help="Split metadata only; no gold labels are read")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-objective", choices=("absolute_mass_v2", "thresholded_v1"),
                        default="absolute_mass_v2",
                        help="v2 uses confidence-weighted soft pairs and preserves their "
                             "absolute mass; "
                             "v1 reproduces the original thresholded objective")
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--memory-capacity", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.contrastive_weight < 0:
        raise ValueError("contrastive weight must be nonnegative")

    source_root = args.source_root.resolve()
    if not (source_root / "src/mil/train.py").is_file():
        raise FileNotFoundError(f"Experiment-4 source tree not found: {source_root}")
    sys.path.insert(0, str(source_root))
    from src.mil.data import Example, build_examples, load_cache
    from src.mil.train import (apply_class_balance, build_model,
                               class_balance_factors, predict_compact)
    import csv as csv_module

    feature_dirs = [args.cache_root / f"shard_{index}" / "features"
                    for index in range(3)]
    feature_dirs.append(args.cache_root / "gold" / "features")
    cache, cache_summary = load_cache(feature_dirs, exclude_order_sources=("unordered",))
    with (source_root / "configs/report_folds_v4.csv").open(
            newline="", encoding="utf-8") as stream:
        fold_by_study = {row["study_id"]: int(row["fold"])
                         for row in csv_module.DictReader(stream)}
    development_ids = read_development_ids(args.gold_split)
    # Only the 58 gold study IDs are needed to exclude them from the report
    # label file. The locked labels themselves are never loaded.
    with args.gold_split.open(newline="", encoding="utf-8-sig") as stream:
        split_reader = csv_module.DictReader(stream)
        split_rows = list(split_reader)
        id_column = "StudyInstanceUID" if "StudyInstanceUID" in split_reader.fieldnames else "study_id"
    excluded_gold_ids = {str(row[id_column]) for row in split_rows}
    labels = read_steven_labels(args.steven_labels, excluded_gold_ids)
    # Match the established Steven-v4 run: only studies represented in the
    # frozen five-fold assignment are eligible for train/report validation.
    # The public label table contains a small number of additional studies.
    fold_ids = set(fold_by_study)
    labels_without_fold = len(labels["labels"])
    labels["labels"] = [item for item in labels["labels"]
                        if item["study_id"] in fold_ids]
    excluded_missing_fold = labels_without_fold - len(labels["labels"])
    report_examples, _, provenance = build_examples({}, cache, labels)
    raw_train = [item for item in report_examples
                 if fold_by_study[item.study_id] != args.validation_fold]
    raw_validation = [item for item in report_examples
                      if fold_by_study[item.study_id] == args.validation_fold]
    factors = class_balance_factors(raw_train)
    training = apply_class_balance(raw_train, factors)
    validation = apply_class_balance(raw_validation, factors)
    missing_features = sorted({item.study_id for item in training + validation} - set(cache))
    if missing_features:
        raise ValueError(f"Missing MedSigLIP features for {len(missing_features)} labelled studies")

    dev_examples = [Example(study_id, cache[study_id], np.full(12, .5, np.float32),
                            np.ones(12, np.float32), "gold")
                    for study_id in development_ids if study_id in cache]
    if len(dev_examples) != 29:
        raise ValueError(f"Expected MedSigLIP features for 29 development studies; got {len(dev_examples)}")

    config = {
        "version": "mil_report_selected_v4_steven_target_contrastive",
        "seed": args.seed,
        "model": {"hidden": 256, "heads": 4, "context_layers": 1,
                  "dropout": .2, "z_dim": 256, "global_fusion": True,
                  "relative_position_bias": False},
        "epochs": args.epochs, "batch_size": 8, "lr": 3e-4,
        "weight_decay": .05, "warmup_fraction": .05, "grad_clip": 1.0,
        "max_slices_train": 48, "max_slices_eval": 96,
        "studies_per_epoch": None, "validation_fold": args.validation_fold,
        "loss": ("target_mean_class_balanced_bce_plus_"
                 f"{args.contrastive_objective}_contrastive"),
        "contrastive": {
            "objective": args.contrastive_objective,
            "weight": args.contrastive_weight, "temperature": args.temperature,
            "positive_threshold": .70, "negative_threshold": .20,
            "memory_capacity": args.memory_capacity, "projection_dim": 64,
            "projection_dropout": 0.0,
            "projection_head": "shared_study_to_target_specific_spaces",
            "memory_bank_keys_detached": True,
        },
        "checkpoint_selection": "lowest report_validation_bce",
        "gold_evaluation_used": False,
    }
    print(json.dumps({
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "feature_source": "precomputed frozen MedSigLIP slice embeddings",
        "cache": cache_summary,
        "training_studies": len(training),
        "report_validation_studies": len(validation),
        "development_prediction_studies": len(dev_examples),
        "evaluation_gold_labels_read": 0,
        "label_provenance": provenance,
        "label_rows_excluded_missing_fold": excluded_missing_fold,
        "config": config,
    }, indent=2), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    reader = build_model(config, cache_summary["feature_dim"])
    projector = ContrastiveProjection(config["model"]["z_dim"], len(TARGETS),
                                      config["contrastive"]["projection_dim"])
    args.output.mkdir(parents=True, exist_ok=True)
    history, best_epoch, best_reader, best_projector = train(
        reader, projector, training, validation, config, device, args.output)
    reader.load_state_dict(best_reader)
    ids, probabilities, embeddings = predict_compact(
        reader, dev_examples, config, device)
    np.savez_compressed(
        args.output / "development_predictions.npz",
        study_ids=np.asarray(ids), probabilities=np.asarray(probabilities, dtype=np.float32),
        study_embeddings=np.asarray(embeddings, dtype=np.float32),
    )
    checkpoint = torch.load(args.output / "best_checkpoint.pt", map_location="cpu",
                            weights_only=False)
    checkpoint.update({
        "state_dict": best_reader,
        "contrastive_state_dict": best_projector,
        "feature_dim": cache_summary["feature_dim"],
        "class_balance_factors": factors.tolist(),
        "label_version": "steven_v4",
    })
    torch.save(checkpoint, args.output / "best_checkpoint.pt")
    summary = {
        "experiment": "Steven v4 MedSigLIP target-aware contrastive objective",
        "config": config, "cache": cache_summary,
        "provenance": provenance,
        "training_studies": len(training), "report_validation_studies": len(validation),
        "best_epoch": best_epoch, "best_report_validation_bce": history[best_epoch]["report_validation_bce"],
        "development_prediction_rows": len(ids), "evaluation_gold_labels_read": 0,
        "gold_evaluation_opened": False,
        "runtime_seconds": round(sum(row["seconds"] for row in history), 2),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device)},
        "history": history,
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch + 1,
                      "best_report_validation_bce": summary["best_report_validation_bce"],
                      "development_prediction_rows": len(ids),
                      "evaluation_gold_labels_read": 0,
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
