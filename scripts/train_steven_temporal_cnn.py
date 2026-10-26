"""Paired Experiment 4 vs temporal-CNN reader training on Steven v4 labels.

Run inside the existing Experiment-4 source tree so cache parsing, study
collation, label balancing, and folds match the established reader.
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

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rsna_knee.temporal_cnn import TemporalCNNReader, count_trainable_parameters


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def read_split_ids(path: Path) -> set[str]:
    """Read gold membership only; never load official gold labels."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        id_column = "StudyInstanceUID" if "StudyInstanceUID" in columns else "study_id"
        if id_column not in columns:
            raise ValueError("Gold split file lacks a study-ID column")
        rows = list(reader)
    ids = {str(row[id_column]) for row in rows}
    if len(ids) != 58:
        raise ValueError(f"Expected 58 gold study IDs; found {len(ids)}")
    return ids


def read_steven_labels(path: Path, gold_ids: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read the twelve Steven v4 targets and drop every gold study by ID."""
    labels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        if "StudyInstanceUID" not in columns or not set(TARGETS) <= columns:
            raise ValueError("Steven v4 CSV is missing its study ID or target columns")
        for row in reader:
            study_id = str(row["StudyInstanceUID"])
            if study_id in gold_ids:
                continue
            if study_id in labels:
                raise ValueError(f"Duplicate report label row: {study_id}")
            values, weights = [], []
            for target in TARGETS:
                try:
                    value = float(row[target])
                except (TypeError, ValueError):
                    value = float("nan")
                valid = math.isfinite(value) and 0.0 <= value <= 1.0
                values.append(value if valid else 0.5)
                weights.append(float(valid))
            labels[study_id] = (np.asarray(values, np.float32), np.asarray(weights, np.float32))
    return labels


def report_auc(targets: np.ndarray, weights: np.ndarray, probabilities: np.ndarray) -> dict:
    """Macro AUC on confident soft-label cells; ambiguous cells are omitted."""
    from sklearn.metrics import roc_auc_score

    per_target: dict[str, float | None] = {}
    for index, name in enumerate(TARGETS):
        keep = (weights[:, index] > 0) & (
            (targets[:, index] <= 0.3) | (targets[:, index] >= 0.7)
        )
        truth = (targets[keep, index] >= 0.7).astype(np.int8)
        score = (float(roc_auc_score(truth, probabilities[keep, index]))
                 if len(truth) >= 10 and np.unique(truth).size == 2 else None)
        per_target[name] = score
    values = [value for value in per_target.values() if value is not None]
    return {"macro_auc": float(np.mean(values)) if values else None,
            "per_target_auc": per_target}


def save_history(output: Path, history: list[dict]) -> None:
    import pandas as pd

    frame = pd.DataFrame(history)
    frame.to_csv(output / "training_history.csv", index=False)
    temporary = output / "training_history.json.tmp"
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    os.replace(temporary, output / "training_history.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = frame["epoch"] + 1
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(epochs, frame["train_bce"], label="Training BCE")
    axes[0].plot(epochs, frame["validation_bce"], label="Report-validation BCE")
    axes[0].scatter([frame.loc[frame.validation_bce.idxmin(), "epoch"] + 1],
                    [frame.validation_bce.min()], color="#d1495b", zorder=3,
                    label="Best BCE checkpoint")
    axes[0].set(xlabel="Epoch", ylabel="Class-balanced BCE", title="Study-level loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, frame["validation_macro_auc"], color="#26734d",
                 marker=".", label="Report-validation macro AUC")
    axes[1].set(xlabel="Epoch", ylabel="Macro ROC AUC", ylim=(0, 1),
                title="Ranking quality")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.tight_layout()
    temporary_plot = output / "learning_curves.tmp.png"
    figure.savefig(temporary_plot, dpi=150)
    plt.close(figure)
    os.replace(temporary_plot, output / "learning_curves.png")


def evaluate(model, examples, config: dict, device: torch.device, collate):
    model.eval()
    numerator = torch.zeros(len(TARGETS), device=device)
    denominator = torch.zeros(len(TARGETS), device=device)
    scores, truths, weights, study_ids = [], [], [], []
    batch_size = int(config["batch_size"])
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            batch = {key: value.to(device) for key, value in
                     collate(chunk, config["max_slices_eval"]).items()}
            logits = model(batch["embeddings"], batch["positions"], batch["plane"],
                           batch["fluid"], batch["slice_mask"], batch["series_mask"])["logits"]
            cell_loss = F.binary_cross_entropy_with_logits(
                logits.float(), batch["target"], reduction="none")
            numerator += (cell_loss * batch["weight"]).sum(dim=0)
            denominator += batch["weight"].sum(dim=0)
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())
            truths.append(batch["target"].cpu().numpy())
            weights.append(batch["weight"].cpu().numpy())
            study_ids.extend(example.study_id for example in chunk)
    truth = np.concatenate(truths)
    weight = np.concatenate(weights)
    probabilities = np.concatenate(scores)
    active = denominator > 0
    bce = float((numerator[active] / denominator[active]).mean().cpu())
    return bce, report_auc(truth, weight, probabilities), study_ids, truth, weight, probabilities


def train_arm(arm: str, fold: int, model, train_examples, validation_examples,
              config: dict, device: torch.device, output: Path, collate,
              target_mean_bce) -> dict:
    rng = np.random.default_rng(config["seed"] + fold)
    torch.manual_seed(config["seed"] + fold)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    weight_decay = config.get(f"{arm}_weight_decay", config["weight_decay"])
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"],
                                  weight_decay=weight_decay)
    steps = math.ceil(len(train_examples) / config["batch_size"])
    total_steps = steps * config["epochs"]
    warmup = max(1, int(total_steps * config["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total_steps)))),
    )
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    best_scores = None
    best_ids = None
    best_truth = None
    best_weights = None
    patience_reference = float("inf")
    stale = 0

    for epoch in range(config["epochs"]):
        model.train()
        order = rng.permutation(len(train_examples))
        losses = []
        started = time.monotonic()
        for start in range(0, len(order), config["batch_size"]):
            chunk = [train_examples[index] for index in order[start:start + config["batch_size"]]]
            batch = {key: value.to(device) for key, value in
                     collate(chunk, config["max_slices_train"], rng).items()}
            logits = model(batch["embeddings"], batch["positions"], batch["plane"],
                           batch["fluid"], batch["slice_mask"], batch["series_mask"])["logits"]
            loss = target_mean_bce(logits, batch["target"], batch["weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {arm} loss at fold {fold}, epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"])
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))

        val_bce, auc, val_ids, val_truth, val_weights, val_scores = evaluate(
            model, validation_examples, config, device, collate)
        record = {
            "arm": arm, "fold": fold, "epoch": epoch,
            "train_bce": float(np.mean(losses)), "validation_bce": val_bce,
            "validation_macro_auc": auc["macro_auc"], "validation_per_target_auc": auc["per_target_auc"],
            "learning_rate": scheduler.get_last_lr()[0],
            "weight_decay": weight_decay,
            "seconds": round(time.monotonic() - started, 2),
        }
        history.append(record)
        if val_bce < best_loss:
            best_loss, best_epoch = val_bce, epoch
            best_state = {key: tensor.detach().cpu().clone()
                          for key, tensor in model.state_dict().items()}
            best_scores, best_ids = val_scores.copy(), list(val_ids)
            best_truth, best_weights = val_truth.copy(), val_weights.copy()

        if val_bce < patience_reference - config["min_delta"]:
            patience_reference, stale = val_bce, 0
        else:
            stale += 1

        payload = {
            "state_dict": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
            "best_state_dict": best_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "arm": arm, "fold": fold, "epoch": epoch + 1,
            "best_epoch": best_epoch + 1, "best_validation_bce": best_loss,
            "stale_epochs": stale, "config": config, "label_version": "steven_v4",
            "trainable_parameters": count_trainable_parameters(model),
        }
        _atomic_torch_save(payload, output / "latest_checkpoint.pt")
        if best_epoch == epoch:
            _atomic_torch_save({**payload, "state_dict": best_state}, output / "best_checkpoint.pt")
        save_history(output, history)
        print(
            f"epoch {epoch + 1:02d}/{config['epochs']} | {arm} | fold {fold} | "
            f"train_bce={record['train_bce']:.4f} val_bce={val_bce:.4f} "
            f"val_auc={auc['macro_auc']} best={best_epoch + 1} stale={stale}/"
            f"{config['patience']} lr={record['learning_rate']:.2e} "
            f"params={payload['trainable_parameters']:,} ({record['seconds']:.1f}s)",
            flush=True,
        )
        if stale >= config["patience"]:
            print(f"Early stopping {arm}, fold {fold}; restoring epoch {best_epoch + 1}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {arm}, fold {fold}")
    final_summary = {
        "arm": arm, "fold": fold, "train_studies": len(train_examples),
        "validation_studies": len(validation_examples), "selected_epoch": best_epoch + 1,
        "best_validation_bce": best_loss,
        "best_validation_macro_auc": report_auc(best_truth, best_weights, best_scores)["macro_auc"],
        "trainable_parameters": count_trainable_parameters(model),
        "epochs_completed": len(history), "history": history,
    }
    (output / "run_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    return {**final_summary, "study_ids": best_ids, "targets": best_truth,
            "weights": best_weights, "probabilities": best_scores}


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--steven-labels", type=Path)
    parser.add_argument("--gold-split", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reader-profile", choices=("original", "compact"),
                        default="original",
                        help="Use the original reader or its smaller, more regularized redesign")
    parser.add_argument("--cpu-smoke", action="store_true",
                        help="Run one synthetic one-epoch forward/backward smoke check")
    args = parser.parse_args()

    if args.cpu_smoke:
        _run_cpu_smoke()
        return
    missing = [name for name in
               ("source_root", "cache_root", "steven_labels", "gold_split", "output")
               if getattr(args, name) is None]
    if missing:
        parser.error("required outside --cpu-smoke: " + ", ".join("--" + name.replace("_", "-")
                                                                  for name in missing))
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    from src.mil.data import Example, collate, load_cache
    from src.mil.train import (apply_class_balance, build_model,
                               class_balance_factors, target_mean_bce)
    feature_dirs = [args.cache_root / f"shard_{index}" / "features" for index in range(3)]
    feature_dirs.append(args.cache_root / "gold" / "features")
    cache, cache_summary = load_cache(feature_dirs, exclude_order_sources=("unordered",))
    if cache_summary.get("feature_dim") != 1152:
        raise ValueError(f"Expected MedSigLIP width 1152; cache reports {cache_summary.get('feature_dim')}")
    gold_ids = read_split_ids(args.gold_split)
    labels = read_steven_labels(args.steven_labels, gold_ids)
    with (source_root / "configs/report_folds_v4.csv").open(
            newline="", encoding="utf-8") as stream:
        fold_by_study = {row["study_id"]: int(row["fold"]) for row in csv.DictReader(stream)}

    labeled_ids = {study_id for study_id, (_, weight) in labels.items() if np.any(weight > 0)}
    fold_eligible_ids = labeled_ids & set(fold_by_study)
    missing_cache = sorted(fold_eligible_ids - set(cache))
    labels_without_fold = labeled_ids - set(fold_by_study)
    if missing_cache:
        raise ValueError(
            f"Steven-v4 coverage mismatch: {len(missing_cache)} fold-eligible studies "
            "lack cached features"
        )

    examples = []
    for study_id, (target, weight) in labels.items():
        if study_id not in cache or study_id not in fold_by_study or not np.any(weight > 0):
            continue
        examples.append(Example(study_id, cache[study_id], target, weight, "report"))
    if not examples:
        raise RuntimeError("No Steven v4 report studies matched the feature cache and folds")
    if any(example.study_id in gold_ids for example in examples):
        raise RuntimeError("Gold study leakage detected before training")
    maximum_slices = max(len(series.embeddings) for example in examples for series in example.series)

    reader_config = (
        {"hidden_dim": 384, "residual_blocks": 2, "dropout": 0.2,
         "feature_dropout": 0.0, "factorized_stem": False,
         "linear_position": False, "weight_decay": 0.05}
        if args.reader_profile == "original" else
        {"hidden_dim": 128, "residual_blocks": 1, "dropout": 0.35,
         "feature_dropout": 0.10, "factorized_stem": True,
         "linear_position": True, "weight_decay": 0.10}
    )
    config = {
        "seed": args.seed, "epochs": args.epochs, "batch_size": 8,
        "lr": 3e-4, "weight_decay": 0.05,
        "temporal_cnn_weight_decay": reader_config["weight_decay"],
        "warmup_fraction": 0.05,
        "grad_clip": 1.0, "dropout": reader_config["dropout"],
        "feature_dropout": reader_config["feature_dropout"],
        "patience": 10, "min_delta": 0.001,
        "max_slices_train": maximum_slices, "max_slices_eval": maximum_slices,
        "all_cached_slices": True, "label_version": "steven_v4",
        "model_baseline": {"hidden": 256, "heads": 4, "context_layers": 1,
                           "dropout": 0.2, "z_dim": 256, "global_fusion": True,
                           "relative_position_bias": False},
        "reader_profile": args.reader_profile,
        "model_temporal_cnn": {**reader_config,
                               "stem_kernel": 3,
                               "block_kernel": 3, "pooling": "masked_mean_slice_logits"},
    }
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({
        "experiment": "Steven v4 matched Experiment 4 vs slice-preserving temporal CNN",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "cache": cache_summary, "matched_report_studies": len(examples),
        "labeled_studies_missing_features": len(missing_cache),
        "labeled_studies_without_report_fold": len(labels_without_fold),
        "excluded_gold_studies": len(gold_ids), "series_max_slices": maximum_slices,
        "folds": args.folds, "config": config,
    }, indent=2), flush=True)

    results: dict[str, list[dict]] = {"experiment4": [], "temporal_cnn": []}
    for fold in args.folds:
        training_raw = [example for example in examples if fold_by_study[example.study_id] != fold]
        validation_raw = [example for example in examples if fold_by_study[example.study_id] == fold]
        if not training_raw or not validation_raw:
            raise ValueError(f"Fold {fold} has an empty train or validation split")
        factors = class_balance_factors(training_raw)
        training = apply_class_balance(training_raw, factors)
        validation = apply_class_balance(validation_raw, factors)
        print(f"\nFold {fold}: {len(training)} train / {len(validation)} validation studies", flush=True)

        arms = {
            "experiment4": build_model({"model": config["model_baseline"]}, cache_summary["feature_dim"]),
            "temporal_cnn": TemporalCNNReader(
                input_dim=cache_summary["feature_dim"], targets=len(TARGETS),
                hidden_dim=config["model_temporal_cnn"]["hidden_dim"],
                dropout=config["model_temporal_cnn"]["dropout"],
                residual_blocks=config["model_temporal_cnn"]["residual_blocks"],
                feature_dropout=config["model_temporal_cnn"]["feature_dropout"],
                factorized_stem=config["model_temporal_cnn"]["factorized_stem"],
                linear_position=config["model_temporal_cnn"]["linear_position"],
            ),
        }
        for arm, model in arms.items():
            arm_output = output_root / arm / f"fold_{fold}"
            result = train_arm(arm, fold, model, training, validation, config,
                               device, arm_output, collate, target_mean_bce)
            results[arm].append(result)

    compact: dict[str, dict] = {}
    for arm, fold_results in results.items():
        rows = []
        for result in fold_results:
            for index, study_id in enumerate(result["study_ids"]):
                rows.append({"study_id": study_id, "fold": result["fold"],
                             "targets": result["targets"][index].tolist(),
                             "weights": result["weights"][index].tolist(),
                             "probabilities": result["probabilities"][index].tolist()})
        rows.sort(key=lambda item: item["study_id"])
        prediction_path = output_root / arm / "out_of_fold_predictions.json"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(json.dumps({"targets_in_order": TARGETS,
                                               "predictions": rows}, indent=2), encoding="utf-8")
        target_matrix = np.asarray([row["targets"] for row in rows], np.float32)
        weight_matrix = np.asarray([row["weights"] for row in rows], np.float32)
        probability_matrix = np.asarray([row["probabilities"] for row in rows], np.float32)
        compact[arm] = {
            "folds": [{key: value for key, value in result.items()
                       if key not in {"study_ids", "targets", "weights", "probabilities", "history"}}
                      for result in fold_results],
            "out_of_fold_macro_auc": report_auc(target_matrix, weight_matrix,
                                                 probability_matrix)["macro_auc"],
        }
    e4_folds = {result["fold"]: result for result in results["experiment4"]}
    cnn_folds = {result["fold"]: result for result in results["temporal_cnn"]}
    paired_folds = []
    for fold in sorted(set(e4_folds) & set(cnn_folds)):
        e4, cnn = e4_folds[fold], cnn_folds[fold]
        paired_folds.append({
            "fold": fold,
            "experiment4_validation_bce": e4["best_validation_bce"],
            "temporal_cnn_validation_bce": cnn["best_validation_bce"],
            "temporal_cnn_minus_experiment4_bce":
                cnn["best_validation_bce"] - e4["best_validation_bce"],
            "experiment4_validation_macro_auc": e4["best_validation_macro_auc"],
            "temporal_cnn_validation_macro_auc": cnn["best_validation_macro_auc"],
            "temporal_cnn_minus_experiment4_auc":
                (None if e4["best_validation_macro_auc"] is None
                 or cnn["best_validation_macro_auc"] is None else
                 cnn["best_validation_macro_auc"] - e4["best_validation_macro_auc"]),
        })
    bce_deltas = [row["temporal_cnn_minus_experiment4_bce"] for row in paired_folds]
    auc_deltas = [row["temporal_cnn_minus_experiment4_auc"] for row in paired_folds
                  if row["temporal_cnn_minus_experiment4_auc"] is not None]
    summary = {
        "experiment": config, "results": compact,
        "paired_fold_comparison": paired_folds,
        "paired_mean_bce_delta_temporal_cnn_minus_experiment4":
            float(np.mean(bce_deltas)) if bce_deltas else None,
        "paired_mean_auc_delta_temporal_cnn_minus_experiment4":
            float(np.mean(auc_deltas)) if auc_deltas else None,
        "paired_bce_wins_temporal_cnn": sum(value < 0 for value in bce_deltas),
        "paired_auc_wins_temporal_cnn": sum(value > 0 for value in auc_deltas),
        "gold_studies_used_for_training_or_selection": False,
        "runtime_environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "device": str(device)},
    }
    (output_root / "paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nPAIRED SUMMARY\n" + json.dumps(summary, indent=2), flush=True)


def _run_cpu_smoke() -> None:
    torch.manual_seed(7)
    model = TemporalCNNReader(dropout=0.0)
    lengths = torch.tensor([[6, 4], [3, 0]])
    slice_mask = torch.arange(6)[None, None, :] < lengths[:, :, None]
    embeddings = torch.randn(2, 2, 6, 1152)
    output = model(embeddings, torch.rand(2, 2, 6), torch.tensor([[0, 1], [2, 3]]),
                   torch.tensor([[0, 1], [2, 2]]), slice_mask, lengths > 0)
    target = torch.randint(0, 2, (2, 12)).float()
    loss = F.binary_cross_entropy_with_logits(output["logits"], target)
    loss.backward()
    gradients_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                           for parameter in model.parameters())
    if (output["slice_logits"].shape != (2, 2, 6, 12)
            or output["logits"].shape != (2, 12)
            or not torch.isfinite(loss)
            or not gradients_finite):
        raise RuntimeError("CPU model shape/loss smoke check failed")
    print(json.dumps({"smoke_check": "passed", "slice_logits": list(output["slice_logits"].shape),
                      "study_logits": list(output["logits"].shape), "loss": float(loss.detach()),
                      "trainable_parameters": count_trainable_parameters(model)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
