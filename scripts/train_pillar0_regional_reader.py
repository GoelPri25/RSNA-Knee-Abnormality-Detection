"""Train the scale-aligned regional reader on cached Pillar-0 features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rsna_knee.pillar0_reader import Pillar0RegionalTargetQueryReader, grid_coordinates, trainable_parameter_count
from train_pillar0_reader import (
    EXPECTED_SERIES,
    FEATURE_DIM,
    TARGETS,
    build_memmaps,
    read_folds,
    read_labels,
    read_csv_rows,
    validation_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-studies", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-series", type=int, default=EXPECTED_SERIES)
    return parser.parse_args()


def partition_8_to_2(grid: np.ndarray) -> np.ndarray:
    """Partition an 8³ flattened ZYX grid into eight 4³ ZYX blocks."""
    if grid.ndim < 3 or grid.shape[:3] != (8, 8, 8):
        raise ValueError(f"Expected an 8x8x8 grid with optional channels, got {grid.shape}")
    trailing = grid.shape[3:]
    reshaped = grid.reshape(2, 4, 2, 4, 2, 4, *trailing)
    axes = (0, 2, 4, 1, 3, 5, *range(6, 6 + len(trailing)))
    return reshaped.transpose(axes).reshape(8, 64, *trailing)


class RegionalTokenStudyDataset(Dataset):
    """Pack every series into eight geometry-aligned multiscale regions."""

    def __init__(self, store, by_study, study_ids, labels) -> None:
        self.store, self.by_study = store, by_study
        self.study_ids, self.labels = study_ids, labels
        self.coords8 = grid_coordinates(8).numpy()
        self.coords2 = grid_coordinates(2).numpy()

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        study_id = self.study_ids[index]
        regional_tokens, regional_scales = [], []
        regional_xyz, regional_protocol, regional_support = [], [], []

        for record in self.by_study[study_id]:
            row = record["index"]
            channel_scales = self.store["scales"][row]
            fine = self.store["fine8"][row].astype(np.float32) * channel_scales[0]
            native = self.store["native8"][row].astype(np.float32) * channel_scales[1]
            coarse = self.store["native2"][row].astype(np.float32) * channel_scales[2]
            fine_regions = partition_8_to_2(fine.reshape(8, 8, 8, FEATURE_DIM))
            native_regions = partition_8_to_2(native.reshape(8, 8, 8, FEATURE_DIM))
            xyz_regions = partition_8_to_2(self.coords8.reshape(8, 8, 8, 3))
            support_grid = self.store["support8"][row].astype(np.float32).reshape(8, 8, 8) / 255.0
            support_regions = partition_8_to_2(support_grid)
            coarse_regions = coarse.reshape(2, 2, 2, FEATURE_DIM).reshape(8, FEATURE_DIM)

            metadata = np.asarray(
                [record["plane"], record["fluid"], record["fat"]], dtype=np.int64
            )
            protocol_region = np.repeat(metadata[None, None, :], 8, axis=0)
            protocol_region = np.repeat(protocol_region, 129, axis=1)
            coarse_support = support_regions.mean(axis=1)
            support_region = np.concatenate(
                [support_regions, support_regions, coarse_support[:, None]], axis=1
            )
            xyz_region = np.concatenate(
                [xyz_regions, xyz_regions, self.coords2[:, None, :]], axis=1
            )
            token_region = np.concatenate(
                [fine_regions, native_regions, coarse_regions[:, None, :]], axis=1
            )
            scale_region = np.concatenate(
                [np.zeros((8, 64), np.int64), np.ones((8, 64), np.int64),
                 np.full((8, 1), 2, np.int64)], axis=1
            )

            regional_tokens.append(token_region)
            regional_scales.append(scale_region)
            regional_xyz.append(xyz_region)
            regional_protocol.append(protocol_region)
            regional_support.append(support_region)

        return {
            "study_id": study_id,
            "tokens": torch.from_numpy(np.concatenate(regional_tokens).astype(np.float16)),
            "scale_ids": torch.from_numpy(np.concatenate(regional_scales)),
            "xyz": torch.from_numpy(np.concatenate(regional_xyz).astype(np.float16)),
            "protocol_ids": torch.from_numpy(np.concatenate(regional_protocol)),
            "support": torch.from_numpy(np.concatenate(regional_support).astype(np.float16)),
            "target": torch.from_numpy(self.labels[study_id][0]),
            "weight": torch.from_numpy(self.labels[study_id][1]),
        }


def collate_regional(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(items)
    max_regions = max(item["tokens"].shape[0] for item in items)
    token_count = 129
    tokens = torch.zeros((batch_size, max_regions, token_count, FEATURE_DIM), dtype=torch.float16)
    scale_ids = torch.zeros((batch_size, max_regions, token_count), dtype=torch.long)
    xyz = torch.zeros((batch_size, max_regions, token_count, 3), dtype=torch.float16)
    protocol_ids = torch.zeros((batch_size, max_regions, token_count, 3), dtype=torch.long)
    support = torch.zeros((batch_size, max_regions, token_count), dtype=torch.float16)
    region_padding = torch.ones((batch_size, max_regions), dtype=torch.bool)
    for row, item in enumerate(items):
        count = item["tokens"].shape[0]
        tokens[row, :count] = item["tokens"]
        scale_ids[row, :count] = item["scale_ids"]
        xyz[row, :count] = item["xyz"]
        protocol_ids[row, :count] = item["protocol_ids"]
        support[row, :count] = item["support"]
        region_padding[row, :count] = False
    return {
        "study_ids": [item["study_id"] for item in items],
        "tokens": tokens,
        "scale_ids": scale_ids,
        "xyz": xyz,
        "protocol_ids": protocol_ids,
        "support": support,
        "region_padding_mask": region_padding,
        "target": torch.stack([item["target"] for item in items]),
        "weight": torch.stack([item["weight"] for item in items]),
    }


def save_fold_history(history: list[dict[str, Any]], output: Path) -> None:
    columns = ["epoch", "train_bce", "validation_bce", "validation_macro_auc", "lr", "seconds"]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in history)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=130)
    axes[0].plot(epochs, [r["train_bce"] for r in history], label="Train")
    axes[0].plot(epochs, [r["validation_bce"] for r in history], label="Report validation")
    axes[0].set(title="Masked soft-label BCE", xlabel="Epoch", ylabel="BCE")
    axes[0].legend()
    axes[1].plot(epochs, [r["validation_macro_auc"] for r in history], marker=".")
    axes[1].set(title="Report-validation ranking", xlabel="Epoch", ylabel="Macro AUC")
    axes[2].plot(epochs, [r["lr"] for r in history])
    axes[2].set(title="Learning rate", xlabel="Epoch", ylabel="LR")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "learning_curve.png")
    plt.close(fig)


def fit_fold(
    fold: int,
    all_ids: list[str],
    folds: dict[str, int],
    labels: dict,
    store: dict,
    by_study: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    output = args.output / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    train_ids = [sid for sid in all_ids if folds[sid] != fold]
    validation_ids = [sid for sid in all_ids if folds[sid] == fold]
    if not train_ids or not validation_ids:
        raise ValueError(f"Empty fold {fold}: train={len(train_ids)}, validation={len(validation_ids)}")
    train_values = np.stack([labels[sid][0] for sid in train_ids])
    train_weights = np.stack([labels[sid][1] for sid in train_ids])
    definitive = train_weights > 0
    positive = definitive & (train_values >= 0.7)
    negative = definitive & (train_values <= 0.3)
    prevalence = {
        target: {
            "positive_ge_0.7": int(positive[:, i].sum()),
            "negative_le_0.3": int(negative[:, i].sum()),
            "observed": int(definitive[:, i].sum()),
        }
        for i, target in enumerate(TARGETS)
    }
    train_data = RegionalTokenStudyDataset(store, by_study, train_ids, labels)
    validation_data = RegionalTokenStudyDataset(store, by_study, validation_ids, labels)
    train_loader = DataLoader(train_data, batch_size=args.batch_studies, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate_regional)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_studies, shuffle=False,
                                   num_workers=0, pin_memory=True, collate_fn=collate_regional)

    model = Pillar0RegionalTargetQueryReader().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, threshold=5e-4,
        threshold_mode="abs", cooldown=2, min_lr=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_raw_auc = -math.inf
    patience_reference = -math.inf
    best_epoch = 0
    stale = 0
    last_lr_drop_epoch = 0
    stopped_early = False

    def to_device(batch):
        return tuple(batch[key].to(device, non_blocking=True) for key in (
            "tokens", "scale_ids", "xyz", "protocol_ids", "support", "region_padding_mask"
        ))

    def evaluate():
        model.eval()
        weighted_loss = 0.0
        total_weight = 0.0
        logits_all, targets_all, weights_all = [], [], []
        with torch.no_grad():
            for batch in validation_loader:
                inputs = to_device(batch)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(*inputs)
                target = batch["target"].to(device)
                weight = batch["weight"].to(device)
                cells = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
                weighted_loss += float((cells * weight).sum())
                total_weight += float(weight.sum())
                logits_all.append(logits.float().cpu().numpy())
                targets_all.append(target.cpu().numpy())
                weights_all.append(weight.cpu().numpy())
        return (
            weighted_loss / max(total_weight, 1.0),
            np.concatenate(logits_all),
            np.concatenate(targets_all),
            np.concatenate(weights_all),
        )

    print(
        f"P0 regional fold {fold}: train={len(train_ids)} validation={len(validation_ids)} "
        f"parameters={trainable_parameter_count(model):,} batch={args.batch_studies} lr={args.lr:g}",
        flush=True,
    )
    print("Training label counts (definite positive/negative; soft middle values also enter BCE): "
          + json.dumps(prevalence), flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        loss_sum, weight_sum = 0.0, 0.0
        for batch in train_loader:
            inputs = to_device(batch)
            target = batch["target"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(*inputs)
            cells = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
            loss = (cells * weight).sum() / weight.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in fold {fold}, epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float((cells.detach() * weight).sum())
            weight_sum += float(weight.sum())

        val_bce, val_logits, val_targets, val_weights = evaluate()
        metrics = validation_metrics(val_logits, val_targets, val_weights)
        lr_before = float(optimizer.param_groups[0]["lr"])
        smoothed_auc = float(np.median([r["validation_macro_auc"] for r in history[-2:]] + [metrics["macro_auc"]]))
        scheduler.step(smoothed_auc)
        lr_after = float(optimizer.param_groups[0]["lr"])
        if lr_after < lr_before:
            last_lr_drop_epoch = epoch
        record = {
            "epoch": epoch,
            "train_bce": loss_sum / max(weight_sum, 1.0),
            "validation_bce": val_bce,
            "validation_macro_auc": metrics["macro_auc"],
            "lr": lr_before,
            "seconds": round(time.monotonic() - epoch_started, 1),
        }
        history.append(record)
        save_fold_history(history, output)

        # Keep the raw metric maximum as the checkpoint. The threshold governs
        # only patience so sub-threshold score changes do not hide raw values.
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_raw_auc": best_raw_auc,
            "best_epoch": best_epoch,
            "patience_reference_auc": patience_reference,
            "stale_epochs": stale,
            "history": history,
            "validation_per_target_auc": metrics["per_target_auc"],
            "architecture": "scale_aligned_regional_target_query",
            "model_config": {"width": 128, "heads": 4, "dropout": 0.15,
                             "feature_dim": FEATURE_DIM, "tokens_per_region": 129},
        }
        torch.save(checkpoint, output / "checkpoint_latest.pt")
        if metrics["macro_auc"] > best_raw_auc:
            best_raw_auc = metrics["macro_auc"]
            best_epoch = epoch
            checkpoint.update(best_raw_auc=best_raw_auc, best_epoch=best_epoch)
            torch.save(checkpoint, output / "checkpoint_best.pt")
        if metrics["macro_auc"] > patience_reference + args.min_delta:
            patience_reference = metrics["macro_auc"]
            stale = 0
        else:
            stale += 1
        print(
            f"P0 regional fold {fold} epoch {epoch}/{args.epochs}: "
            f"train_bce={record['train_bce']:.5f} report_val_bce={val_bce:.5f} "
            f"report_val_macro_auc={metrics['macro_auc']:.5f} "
            f"lr={lr_before:.3g}->{lr_after:.3g} best_epoch={best_epoch} "
            f"stale={stale}/{args.patience} ({record['seconds']}s)", flush=True,
        )
        print("Per-target report-validation AUC:", json.dumps(metrics["per_target_auc"]), flush=True)

        latest = torch.load(output / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
        latest.update(best_raw_auc=best_raw_auc, best_epoch=best_epoch,
                      patience_reference_auc=patience_reference, stale_epochs=stale)
        torch.save(latest, output / "checkpoint_latest.pt")
        fold_summary = {
            "fold": fold, "architecture": "scale_aligned_regional_target_query",
            "epochs_completed": epoch, "best_epoch": best_epoch,
            "best_report_validation_macro_auc": best_raw_auc,
            "last_report_validation_bce": val_bce,
            "last_report_validation_macro_auc": metrics["macro_auc"],
            "validation_per_target_auc": metrics["per_target_auc"],
            "train_studies": len(train_ids), "validation_studies": len(validation_ids),
            "trainable_parameters": trainable_parameter_count(model),
            "early_stopping_patience": args.patience,
            "last_lr_drop_epoch": last_lr_drop_epoch,
            "gold_labels_loaded": False,
        }
        (output / "run_summary.json").write_text(json.dumps(fold_summary, indent=2, allow_nan=True) + "\n")
        if stale >= args.patience and epoch - last_lr_drop_epoch >= 4:
            stopped_early = True
            print(f"P0 regional fold {fold}: early stop after epoch {epoch}; raw-best epoch {best_epoch}", flush=True)
            break

    best = torch.load(output / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    final_bce, logits, targets, weights = evaluate()
    final_auc = validation_metrics(logits, targets, weights)
    np.savez_compressed(output / "validation_predictions.npz", logits=logits,
                        targets=targets, weights=weights)
    summary = {
        "fold": fold, "architecture": "scale_aligned_regional_target_query",
        "selected_epoch": int(best["best_epoch"]),
        "best_report_validation_macro_auc": final_auc["macro_auc"],
        "best_report_validation_bce": final_bce,
        "epochs_completed": len(history), "stopped_early": stopped_early,
        "validation_per_target_auc": final_auc["per_target_auc"],
        "train_studies": len(train_ids), "validation_studies": len(validation_ids),
        "trainable_parameters": trainable_parameter_count(model),
        "last_lr_drop_epoch": last_lr_drop_epoch,
        "gold_labels_loaded": False,
        "training_label_counts": prevalence,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(f"P0 regional fold {fold} complete:", json.dumps(summary, allow_nan=True), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This experiment is configured to run on a Modal T4 GPU")

    labels = read_labels(args.labels)
    folds = read_folds(args.folds)
    store, _ = build_memmaps(args.cache, args.scratch, args.expected_series)
    by_study: dict[str, list[dict[str, Any]]] = {}
    for record in store["records"]:
        by_study.setdefault(record["study_id"], []).append(record)
    usable = sorted(sid for sid in by_study if sid in labels and sid in folds)
    selected_folds = sorted(set(folds[sid] for sid in usable)) if args.all_folds else [args.validation_fold]
    print(f"Device: {torch.cuda.get_device_name(0)} | studies={len(usable)} | folds={selected_folds}", flush=True)
    print("Gold studies loaded: 0 | encoder: frozen/precomputed | support: same soft prior as baseline", flush=True)

    summaries = []
    started = time.monotonic()
    for fold in selected_folds:
        summaries.append(fit_fold(fold, usable, folds, labels, store, by_study, args, device))
        (args.output / "cross_validation_progress.json").write_text(
            json.dumps({"completed_folds": summaries, "gold_labels_loaded": False}, indent=2, allow_nan=True) + "\n"
        )
    if args.all_folds:
        scores = [row["best_report_validation_macro_auc"] for row in summaries]
        aggregate = {
            "experiment": "pillar0_scale_aligned_regional_reader_cv",
            "architecture": "scale_aligned_regional_target_query",
            "folds": summaries,
            "mean_fold_macro_auc": float(np.mean(scores)),
            "std_fold_macro_auc": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "runtime_minutes": round((time.monotonic() - started) / 60, 2),
            "gold_labels_loaded": False,
        }
        (args.output / "cross_validation_summary.json").write_text(json.dumps(aggregate, indent=2, allow_nan=True) + "\n")
        print("P0 regional cross-validation complete:", json.dumps(aggregate, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
