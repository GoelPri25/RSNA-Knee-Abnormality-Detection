"""Train a single fold of the frozen-Pillar-0 multiscale token reader."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from rsna_knee.pillar0_reader import (
    Pillar0TargetQueryReader,
    grid_coordinates,
    trainable_parameter_count,
)


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
EXPECTED_SERIES = 24_035
TOKENS_PER_SERIES = 512 + 512 + 8
FEATURE_DIM = 384
SUPPORT_FLOOR = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-studies", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-series", type=int, default=EXPECTED_SERIES)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_labels(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No labels in {path}")
    fieldnames = set(rows[0])
    id_column = "StudyInstanceUID" if "StudyInstanceUID" in fieldnames else "study_id"
    missing = set(TARGETS) - fieldnames
    if missing or id_column not in fieldnames:
        raise ValueError(f"Label CSV schema mismatch; missing {sorted(missing)} or {id_column}")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        study_id = row[id_column].strip()
        if not study_id or study_id in result:
            raise ValueError(f"Empty or duplicate label study ID: {study_id!r}")
        values = np.full(len(TARGETS), 0.5, dtype=np.float32)
        weights = np.zeros(len(TARGETS), dtype=np.float32)
        for index, target in enumerate(TARGETS):
            text = (row.get(target) or "").strip()
            if text:
                try:
                    value = float(text)
                except ValueError as error:
                    raise ValueError(f"Invalid {target} label {text!r} for {study_id}") from error
                if math.isfinite(value) and 0.0 <= value <= 1.0:
                    values[index], weights[index] = value, 1.0
        result[study_id] = values, weights
    return result


def read_folds(path: Path) -> dict[str, int]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No folds in {path}")
    fields = set(rows[0])
    id_column = "study_id" if "study_id" in fields else "StudyInstanceUID"
    if id_column not in fields or "fold" not in fields:
        raise ValueError(f"Fold CSV must contain study_id and fold columns: {sorted(fields)}")
    result: dict[str, int] = {}
    for row in rows:
        study_id = row[id_column].strip()
        fold = int(row["fold"])
        if not study_id or study_id in result:
            raise ValueError(f"Empty or duplicate fold study ID: {study_id!r}")
        result[study_id] = fold
    return result


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _plane_id(value: str) -> int:
    return {"axial": 1, "coronal": 2, "sagittal": 3, "oblique": 4}.get(value.strip().lower(), 0)


def _bool_id(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return 2
    if normalized in {"false", "0", "no", "n"}:
        return 1
    return 0


def build_memmaps(cache: Path, scratch: Path, expected_series: int) -> tuple[dict[str, Any], list[str]]:
    """Validate compressed chunks and expand int8 payloads to temporary local memmaps."""
    chunks = sorted(cache.glob("spatial_batch_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"No spatial batches under {cache}")
    row_counts = []
    for path in chunks:
        with np.load(path, allow_pickle=False) as archive:
            row_counts.append(len(archive["study_uid"]))
    total = sum(row_counts)
    if total != expected_series:
        raise ValueError(f"Cache has {total:,} series; expected {expected_series:,}")

    scratch.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.memmap] = {
        "fine8": np.memmap(scratch / "fine8.int8", mode="w+", dtype=np.int8,
                            shape=(total, 512, FEATURE_DIM)),
        "native8": np.memmap(scratch / "native8.int8", mode="w+", dtype=np.int8,
                              shape=(total, 512, FEATURE_DIM)),
        "native2": np.memmap(scratch / "native2.int8", mode="w+", dtype=np.int8,
                              shape=(total, 8, FEATURE_DIM)),
        "scales": np.memmap(scratch / "scales.float32", mode="w+", dtype=np.float32,
                            shape=(total, 3, FEATURE_DIM)),
        "support8": np.memmap(scratch / "support8.uint8", mode="w+", dtype=np.uint8,
                              shape=(total, 512)),
    }
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    offset = 0
    for path, expected_rows in zip(chunks, row_counts):
        with np.load(path, allow_pickle=False) as archive:
            expected_shapes = {
                "fine8": (expected_rows, 512, FEATURE_DIM),
                "native8": (expected_rows, 512, FEATURE_DIM),
                "native2": (expected_rows, 8, FEATURE_DIM),
                "channel_scales": (expected_rows, 3, FEATURE_DIM),
                "support8": (expected_rows, 512),
            }
            for key, shape in expected_shapes.items():
                if archive[key].shape != shape:
                    raise ValueError(f"{path.name}: {key} shape {archive[key].shape}, expected {shape}")
            if not np.isfinite(archive["channel_scales"]).all():
                raise ValueError(f"Nonfinite quantization scale in {path.name}")
            keys = list(zip(archive["study_uid"].astype(str), archive["series_uid"].astype(str)))
            if len(set(keys)) != len(keys) or seen.intersection(keys):
                raise ValueError(f"Duplicate study/series key in {path.name}")
            seen.update(keys)
            end = offset + expected_rows
            for key, archive_key in (("fine8", "fine8"), ("native8", "native8"),
                                     ("native2", "native2"), ("scales", "channel_scales"),
                                     ("support8", "support8")):
                arrays[key][offset:end] = archive[archive_key]
            for row in range(expected_rows):
                records.append({
                    "index": offset + row,
                    "study_id": _text(archive["study_uid"][row]),
                    "series_id": _text(archive["series_uid"][row]),
                    "plane": _plane_id(_text(archive["plane"][row])),
                    "fluid": _bool_id(_text(archive["fluid_sensitive"][row])),
                    "fat": _bool_id(_text(archive["fat_suppression"][row])),
                })
            offset = end
        if len(seen) % 2048 < expected_rows:
            print(f"Indexed {len(seen):,}/{total:,} cached series", flush=True)
    for array in arrays.values():
        array.flush()
    study_ids = [record["study_id"] for record in records]
    print(f"Validated and memory-mapped {len(records):,} unique report-only series", flush=True)
    return {**arrays, "records": records}, study_ids


class TokenStudyDataset(Dataset):
    def __init__(self, store: dict[str, Any], study_indices: dict[str, list[dict[str, Any]]],
                 study_ids: list[str], labels: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self.store = store
        self.study_indices = study_indices
        self.study_ids = study_ids
        self.labels = labels
        self.coords8 = grid_coordinates(8).numpy()
        self.coords2 = grid_coordinates(2).numpy()

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, item: int) -> dict[str, Any]:
        study_id = self.study_ids[item]
        records = self.study_indices[study_id]
        token_blocks, scale_blocks, coord_blocks, protocol_blocks, support_blocks = [], [], [], [], []
        for record in records:
            index = record["index"]
            scales = self.store["scales"][index]
            token_blocks.extend((
                self.store["fine8"][index].astype(np.float32) * scales[0],
                self.store["native8"][index].astype(np.float32) * scales[1],
                self.store["native2"][index].astype(np.float32) * scales[2],
            ))
            scale_blocks.extend((
                np.zeros(512, dtype=np.int64), np.ones(512, dtype=np.int64),
                np.full(8, 2, dtype=np.int64),
            ))
            coord_blocks.extend((self.coords8, self.coords8, self.coords2))
            metadata = np.asarray([record["plane"], record["fluid"], record["fat"]], dtype=np.int64)
            protocol_blocks.extend((
                np.repeat(metadata[None, :], 512, axis=0),
                np.repeat(metadata[None, :], 512, axis=0),
                np.repeat(metadata[None, :], 8, axis=0),
            ))
            support8 = self.store["support8"][index].astype(np.float32).reshape(8, 8, 8) / 255.0
            support2 = support8.reshape(2, 4, 2, 4, 2, 4).mean(axis=(1, 3, 5)).reshape(8)
            support_blocks.extend((support8.reshape(512), support8.reshape(512), support2))
        tokens = np.concatenate(token_blocks).astype(np.float16, copy=False)
        return {
            "study_id": study_id,
            "tokens": torch.from_numpy(tokens),
            "scale_ids": torch.from_numpy(np.concatenate(scale_blocks)),
            "xyz": torch.from_numpy(np.concatenate(coord_blocks).astype(np.float16)),
            "protocol_ids": torch.from_numpy(np.concatenate(protocol_blocks)),
            "support": torch.from_numpy(np.concatenate(support_blocks).astype(np.float16)),
            "target": torch.from_numpy(self.labels[study_id][0]),
            "weight": torch.from_numpy(self.labels[study_id][1]),
        }


def collate_studies(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = len(items)
    max_tokens = max(len(item["scale_ids"]) for item in items)
    tokens = torch.zeros((batch, max_tokens, FEATURE_DIM), dtype=torch.float16)
    scale_ids = torch.zeros((batch, max_tokens), dtype=torch.long)
    xyz = torch.zeros((batch, max_tokens, 3), dtype=torch.float16)
    protocol_ids = torch.zeros((batch, max_tokens, 3), dtype=torch.long)
    support = torch.zeros((batch, max_tokens), dtype=torch.float16)
    padding_mask = torch.ones((batch, max_tokens), dtype=torch.bool)
    for row, item in enumerate(items):
        length = len(item["scale_ids"])
        tokens[row, :length] = item["tokens"]
        scale_ids[row, :length] = item["scale_ids"]
        xyz[row, :length] = item["xyz"]
        protocol_ids[row, :length] = item["protocol_ids"]
        support[row, :length] = item["support"]
        padding_mask[row, :length] = False
    return {
        "study_ids": [item["study_id"] for item in items],
        "tokens": tokens, "scale_ids": scale_ids, "xyz": xyz,
        "protocol_ids": protocol_ids, "support": support,
        "padding_mask": padding_mask,
        "target": torch.stack([item["target"] for item in items]),
        "weight": torch.stack([item["weight"] for item in items]),
    }


def validation_metrics(logits: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    per_target_auc: dict[str, float | None] = {}
    aucs = []
    for index, target in enumerate(TARGETS):
        keep = (weights[:, index] > 0) & ((targets[:, index] <= 0.3) | (targets[:, index] >= 0.7))
        binary = targets[keep, index] >= 0.7
        if len(np.unique(binary)) == 2:
            value = float(roc_auc_score(binary, probabilities[keep, index]))
            per_target_auc[target] = value
            aucs.append(value)
        else:
            per_target_auc[target] = None
    return {"macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "per_target_auc": per_target_auc}


def save_history(history: list[dict[str, Any]], output: Path) -> None:
    columns = ["epoch", "train_bce", "validation_bce", "validation_macro_auc", "lr", "seconds"]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in history)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if history:
        epochs = [row["epoch"] for row in history]
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=130)
        axes[0].plot(epochs, [row["train_bce"] for row in history], label="Training")
        axes[0].plot(epochs, [row["validation_bce"] for row in history], label="Report validation")
        axes[0].set(title="Masked soft-label BCE", xlabel="Epoch", ylabel="BCE")
        axes[0].legend()
        axes[1].plot(epochs, [row["validation_macro_auc"] for row in history], marker=".")
        axes[1].set(title="Report-validation ranking", xlabel="Epoch", ylabel="Macro AUC")
        axes[2].plot(epochs, [row["lr"] for row in history])
        axes[2].set(title="Learning rate", xlabel="Epoch", ylabel="LR")
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / "learning_curve.png")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This experiment is configured to run on the Modal T4 GPU")

    labels = read_labels(args.labels)
    folds = read_folds(args.folds)
    store, _ = build_memmaps(args.cache, args.scratch, args.expected_series)
    by_study: dict[str, list[dict[str, Any]]] = {}
    for record in store["records"]:
        by_study.setdefault(record["study_id"], []).append(record)
    usable = sorted(study_id for study_id in by_study if study_id in labels and study_id in folds)
    train_ids = [study_id for study_id in usable if folds[study_id] != args.validation_fold]
    validation_ids = [study_id for study_id in usable if folds[study_id] == args.validation_fold]
    if not train_ids or not validation_ids:
        raise ValueError(f"Empty fold {args.validation_fold}: train={len(train_ids)}, validation={len(validation_ids)}")
    print(f"Device: {torch.cuda.get_device_name(0)} | cache series: {len(store['records']):,}", flush=True)
    print(f"Report studies: {len(usable):,} | train: {len(train_ids):,} | validation: {len(validation_ids):,} | fold: {args.validation_fold}", flush=True)
    print("Gold studies loaded: 0 | encoder: frozen/precomputed | support prior: fractional", flush=True)

    index_by_study = by_study
    train_data = TokenStudyDataset(store, index_by_study, train_ids, labels)
    validation_data = TokenStudyDataset(store, index_by_study, validation_ids, labels)
    train_loader = DataLoader(train_data, batch_size=args.batch_studies, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate_studies)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_studies, shuffle=False,
                                   num_workers=0, pin_memory=True, collate_fn=collate_studies)

    model = Pillar0TargetQueryReader().to(device)
    print(f"Reader trainable parameters: {trainable_parameter_count(model):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=5e-4,
        threshold_mode="abs", cooldown=1, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history: list[dict[str, Any]] = []
    best_auc = -math.inf
    best_epoch = 0
    stale = 0
    stopped_early = False
    targets_train = np.stack([labels[study_id][0] for study_id in train_ids])
    weights_train = np.stack([labels[study_id][1] for study_id in train_ids])
    positives = ((targets_train >= 0.5) & (weights_train > 0)).sum(axis=0)
    totals = (weights_train > 0).sum(axis=0)
    print("Observed positive rate by target:", json.dumps({
        target: round(float(pos / max(int(total), 1)), 4)
        for target, pos, total in zip(TARGETS, positives, totals)
    }), flush=True)

    def model_inputs(batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        return tuple(batch[key].to(device, non_blocking=True) for key in (
            "tokens", "scale_ids", "xyz", "protocol_ids", "support", "padding_mask"
        ))

    def evaluate() -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        total_loss, total_weight = 0.0, 0.0
        all_logits, all_targets, all_weights = [], [], []
        with torch.no_grad():
            for batch in validation_loader:
                inputs = model_inputs(batch)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(*inputs)
                target = batch["target"].to(device)
                weight = batch["weight"].to(device)
                cells = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
                total_loss += float((cells * weight).sum())
                total_weight += float(weight.sum())
                all_logits.append(logits.float().cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_weights.append(weight.cpu().numpy())
        return (total_loss / max(total_weight, 1.0), np.concatenate(all_logits),
                np.concatenate(all_targets), np.concatenate(all_weights))

    for epoch in range(1, args.epochs + 1):
        model.train()
        started_epoch = time.monotonic()
        loss_sum, weight_sum = 0.0, 0.0
        for batch in train_loader:
            inputs = model_inputs(batch)
            target = batch["target"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(*inputs)
            cells = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
            denom = weight.sum().clamp_min(1.0)
            loss = (cells * weight).sum() / denom
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float((cells.detach() * weight).sum())
            weight_sum += float(weight.sum())

        val_bce, logits, val_targets, val_weights = evaluate()
        metrics = validation_metrics(logits, val_targets, val_weights)
        lr_used = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_bce)
        record = {
            "epoch": epoch,
            "train_bce": loss_sum / max(weight_sum, 1.0),
            "validation_bce": val_bce,
            "validation_macro_auc": metrics["macro_auc"],
            "lr": lr_used,
            "seconds": round(time.monotonic() - started_epoch, 1),
        }
        history.append(record)
        save_history(history, args.output)
        checkpoint = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": best_auc, "best_epoch": best_epoch,
            "stale_epochs": stale, "history": history,
            "validation_per_target_auc": metrics["per_target_auc"],
            "model_config": {"width": 128, "heads": 4, "dropout": 0.15,
                             "feature_dim": FEATURE_DIM, "support_floor": SUPPORT_FLOOR},
        }
        torch.save(checkpoint, args.output / "checkpoint_latest.pt")
        improved = metrics["macro_auc"] > best_auc + args.min_delta
        if improved:
            best_auc = metrics["macro_auc"]
            best_epoch = epoch
            stale = 0
            checkpoint["best_auc"] = best_auc
            checkpoint["best_epoch"] = best_epoch
            checkpoint["stale_epochs"] = stale
            torch.save(checkpoint, args.output / "checkpoint_best.pt")
        else:
            stale += 1

        print(
            f"P0 epoch {epoch}/{args.epochs}: train_bce={record['train_bce']:.5f} "
            f"report_val_bce={val_bce:.5f} report_val_macro_auc={metrics['macro_auc']:.5f} "
            f"lr={lr_used:.2g}->{optimizer.param_groups[0]['lr']:.2g} "
            f"best_epoch={best_epoch} stale={stale}/{args.patience} "
            f"({record['seconds']}s)", flush=True,
        )

        summary = {
            "experiment": "pillar0_multiscale_target_query_reader",
            "validation_fold": args.validation_fold,
            "best_epoch": best_epoch,
            "best_report_validation_macro_auc": best_auc,
            "best_report_validation_bce": min(row["validation_bce"] for row in history),
            "epochs_completed": epoch,
            "stopped_early": False,
            "gold_studies_loaded": 0,
            "cache_studies": len(by_study),
            "cache_series": len(store["records"]),
            "train_studies": len(train_ids),
            "validation_studies": len(validation_ids),
            "trainable_parameters": trainable_parameter_count(model),
            "last_validation_per_target_auc": metrics["per_target_auc"],
            "history": history,
        }
        (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

        if stale >= args.patience:
            stopped_early = True
            print(f"Early stopping after epoch {epoch}; restoring best epoch {best_epoch}", flush=True)
            break

    best_path = args.output / "checkpoint_best.pt"
    if not best_path.exists():
        raise RuntimeError("No finite best checkpoint was selected")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    final_bce, final_logits, final_targets, final_weights = evaluate()
    final_metrics = validation_metrics(final_logits, final_targets, final_weights)
    summary = {
        "experiment": "pillar0_multiscale_target_query_reader",
        "validation_fold": args.validation_fold,
        "selected_epoch": best_checkpoint["best_epoch"],
        "best_report_validation_macro_auc": final_metrics["macro_auc"],
        "best_report_validation_bce": final_bce,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "gold_studies_loaded": 0,
        "cache_studies": len(by_study),
        "cache_series": len(store["records"]),
        "train_studies": len(train_ids),
        "validation_studies": len(validation_ids),
        "trainable_parameters": trainable_parameter_count(model),
        "validation_per_target_auc": final_metrics["per_target_auc"],
        "runtime_minutes": round((time.monotonic() - started) / 60.0, 2),
        "checkpoint": "checkpoint_best.pt",
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print("Training complete:", json.dumps(summary, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
