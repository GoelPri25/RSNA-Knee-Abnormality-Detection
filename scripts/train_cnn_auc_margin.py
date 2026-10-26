"""Fine-tune the pinned HPO-v1 ResNet50 with soft BCE and masked AUC margin.

All inference layers and the existing cache/sampling/augmentation functions are
reused. Checkpoints contain the complete primal-dual, scheduler and RNG state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsna_knee.auc_margin import MaskedAUCMargin

CHECKPOINT_SHA = "7c76e8760b6989fb5bb3afa2fe14cf016ac766981ad612c94fc058acb703120a"
LABEL_SHA = "ffdffcb2a68e7dbdfbac14c97b1017c28ab7161628cf067040287fa9356c608f"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(value, path):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "soft-labels", "validation-soft-labels", "folds",
                 "evidence", "init-checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cache", type=Path, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--snapshot-batches", type=int, default=50)
    parser.add_argument("--commit-volume", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.snapshot_batches <= 0:
        raise ValueError("Epoch and snapshot limits must be positive")
    if digest(args.init_checkpoint) != CHECKPOINT_SHA or digest(args.soft_labels) != LABEL_SHA:
        raise ValueError("Starting checkpoint or HPO-v1 labels do not match the pinned artifacts")
    sys.path.insert(0, str(args.source))
    spec = importlib.util.spec_from_file_location("baseline_reader", args.source / "scripts/train_cnn_baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This full training run requires a GPU; use CPU loss tests separately")
    initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    config = initial["config"]
    if config["model"]["backbone"] != "resnet50" or initial["epoch"] != 17:
        raise ValueError("Expected epoch-17 ResNet50")
    # Explicit adaptations are recorded rather than silently pretending that
    # this hybrid weak-label objective is the paper's binary-label recipe.
    stage = dict(epochs=args.epochs, batch_studies=8, windows=24, seed=2026,
                 learning_rate=3.75e-5, auxiliary_lr=0.005, dual_lr=0.005,
                 auc_weight=0.1, margin=0.3, proximal_weight=0.0,
                 patience=8, min_delta=0.0005, snapshot_batches=args.snapshot_batches,
                 optimizer="AdamW primal; SGD centers; projected dual ascent",
                 classifier_reinitialized=False)
    manifest = baseline.load_manifest(args.cache, config["cache"]["exclude_order_sources"])
    cache = baseline.StudyCache(manifest)
    labels = baseline.read_soft_labels(args.soft_labels)
    validation_labels = baseline.read_soft_labels(args.validation_soft_labels)
    folds = baseline.read_folds(args.folds)
    train_ids, _ = baseline.build_studies(cache, labels, folds, 0)
    _, validation_ids = baseline.build_studies(cache, validation_labels, folds, 0)
    if (len(train_ids), len(validation_ids), len(manifest)) != (3524, 825, 4349):
        raise ValueError("Expected exactly 3524 train, 825 fixed validation and 4349 cached studies")
    evidence_payload = json.loads(args.evidence.read_text())
    if evidence_payload["targets"] != list(baseline.TARGETS):
        raise ValueError("Evidence target order differs")
    evidence = evidence_payload["evidence"]
    if set(evidence) != set(labels):
        raise ValueError("Evidence IDs differ from report-only HPO labels")
    cohort = np.array([evidence[s] for s in train_ids], dtype=np.int8)
    if not np.isin(cohort, [-1, 0, 1]).all():
        raise ValueError("Invalid evidence state")
    pos_rate, neg_rate = (cohort == 1).mean(0), (cohort == -1).mean(0)
    loss_fn = MaskedAUCMargin(pos_rate, neg_rate, stage["margin"]).to(device)
    print("Training evidence counts:", json.dumps({t: {"positive": int((cohort[:, j] == 1).sum()),
          "negative": int((cohort[:, j] == -1).sum())} for j, t in enumerate(baseline.TARGETS)}), flush=True)
    torch.manual_seed(stage["seed"])
    random.seed(stage["seed"])
    rng = np.random.default_rng(stage["seed"])
    model = baseline.SliceModel("resnet50", False, config["model"]["dropout"],
                               len(baseline.TARGETS), config["model"]["top_k_fraction"]).to(device)
    model.load_state_dict(initial["state_dict"], strict=True)
    reference = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage["learning_rate"],
                                  weight_decay=config["weight_decay"])
    auxiliary = torch.optim.SGD([loss_fn.a, loss_fn.b], lr=stage["auxiliary_lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=0.0005,
        threshold_mode="abs", cooldown=1, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    volume = None
    if args.commit_volume:
        import modal
        volume = modal.Volume.from_name("rsna-cnn-auc-margin-artifacts-v1")
    identity = {"checkpoint": CHECKPOINT_SHA, "labels": LABEL_SHA, "folds": digest(args.folds),
                "validation_labels": digest(args.validation_soft_labels), "evidence": digest(args.evidence),
                "trainer": digest(Path(__file__)), "loss_module": digest(Path(__file__).resolve().parents[1] / "src/rsna_knee/auc_margin.py"),
                "baseline_trainer": digest(args.source / "scripts/train_cnn_baseline.py"), "stage": stage}
    write_json(identity, output / "run_manifest.json")
    latest = output / "checkpoint_latest.pt"
    state = dict(epoch=0, cursor=0, order=None, history=[], best=None, meaningful_auc=None,
                 stalled=0, totals=[0.0, 0.0, 0], completed=False, initial_metrics=None)

    def commit():
        if volume is not None:
            volume.commit()

    def snapshot():
        atomic_save({"state_dict": model.state_dict(), "config": config, "stage": stage,
                     "epoch": 17 + state["epoch"] + int(state["cursor"] > 0), "identity": identity, "progress": state,
                     "optimizer": optimizer.state_dict(), "auxiliary_optimizer": auxiliary.state_dict(),
                     "auc_state": loss_fn.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": rng.bit_generator.state,
                     "python_rng": random.getstate()}, latest)
        commit()

    if latest.exists():
        resumed = torch.load(latest, map_location=device, weights_only=False)
        if resumed["identity"] != identity:
            raise ValueError("Resume inputs or implementation changed; use a new output directory")
        model.load_state_dict(resumed["state_dict"])
        optimizer.load_state_dict(resumed["optimizer"])
        auxiliary.load_state_dict(resumed["auxiliary_optimizer"])
        loss_fn.load_state_dict(resumed["auc_state"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
        state = resumed["progress"]
        torch.set_rng_state(resumed["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in resumed["cuda_rng"]])
        rng.bit_generator.state = resumed["numpy_rng"]
        random.setstate(resumed["python_rng"])
        print(f"RESUMED stage epoch {state['epoch'] + 1}, next study offset {state['cursor']}", flush=True)
        if state["completed"]:
            print("Run already completed; no training is repeated.", flush=True)
            return

    @torch.no_grad()
    def evaluate():
        model.eval()
        predictions, truth, weights = [], [], []
        numerator = denominator = 0.0
        for start in range(0, len(validation_ids), 8):
            chunk = validation_ids[start:start + 8]
            x, group, y, w, _, _ = baseline.prepare_batch(cache, validation_labels, chunk, None, None, device)
            with torch.autocast("cuda"):
                logits = model(x, group, len(chunk))
            bce = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
            numerator += float((bce * w).sum())
            denominator += float(w.sum())
            predictions.append(logits.float().sigmoid().cpu().numpy())
            truth.append(y.cpu().numpy()); weights.append(w.cpu().numpy())
        scores = np.concatenate(predictions)
        metrics = baseline.confident_auc(np.concatenate(truth), np.concatenate(weights), scores, 0.3, 0.7)
        metrics["validation_bce"] = numerator / denominator
        # Secondary ruler: agreeing confident source cells, same fixed studies.
        ev = np.array([evidence[s] for s in validation_ids])
        metrics["agreed_source_auc"] = baseline.confident_auc((ev == 1).astype(float),
                                                              (ev != 0).astype(float), scores, 0.3, 0.7)
        if metrics["targets_scored"] != 12 or not math.isfinite(metrics["macro_auc"]):
            raise ValueError("Validation must score all twelve targets")
        return metrics, scores

    if state["initial_metrics"] is None:
        print("Evaluating the pinned checkpoint on fixed report validation before any updates.", flush=True)
        metrics, scores = evaluate()
        state["initial_metrics"] = metrics
        state["best"] = {"epoch": 17, "macro_auc": metrics["macro_auc"], "metrics": metrics}
        state["meaningful_auc"] = metrics["macro_auc"]
        shutil.copyfile(args.init_checkpoint, output / "checkpoint_best.pt")
        write_json({"study_ids": validation_ids, "scores": scores.tolist()}, output / "baseline_validation_predictions.json")
        write_json(metrics, output / "baseline_metrics.json")
        scheduler.step(metrics["validation_bce"])
        snapshot()
        print(f"BASELINE AUC={metrics['macro_auc']:.6f} BCE={metrics['validation_bce']:.6f}", flush=True)

    while state["epoch"] < args.epochs:
        started = time.monotonic()
        model.train()
        if state["order"] is None:
            state["order"] = rng.permutation(len(train_ids)).tolist()
            state["totals"] = [0.0, 0.0, 0]
        for start in range(state["cursor"], len(train_ids), 8):
            chunk = [train_ids[i] for i in state["order"][start:start + 8]]
            x, group, y, w, _, _ = baseline.prepare_batch(cache, labels, chunk, 24, rng,
                                                         device, config["augmentation"])
            ev = torch.tensor([evidence[s] for s in chunk], device=device)
            optimizer.zero_grad(set_to_none=True); auxiliary.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                logits = model(x, group, len(chunk))
            bce = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none") * w).sum() / w.sum()
            auc, gap = loss_fn(logits, ev)
            loss = bce + stage["auc_weight"] * auc
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite hybrid loss; last durable snapshot is preserved")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not all(torch.isfinite(p.grad).all() for p in (loss_fn.a, loss_fn.b)):
                raise FloatingPointError("Non-finite AUC center gradient; durable state preserved")
            # Add a proximal gradient towards the starting representation.
            for p, ref in zip(model.parameters(), reference):
                if p.grad is not None:
                    p.grad.add_(p.detach() - ref, alpha=stage["proximal_weight"])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= scale_before:
                # Center updates use the unscaled, unweighted AUC gradient.
                factor = scale_before * stage["auc_weight"]
                loss_fn.a.grad.div_(factor); loss_fn.b.grad.div_(factor)
                auxiliary.step()
                loss_fn.ascend(gap, stage["dual_lr"])
            state["totals"][0] += float(bce.detach())
            state["totals"][1] += float(auc.detach())
            state["totals"][2] += 1
            state["cursor"] = min(start + 8, len(train_ids))
            batch = math.ceil(state["cursor"] / 8)
            if batch % args.snapshot_batches == 0 or state["cursor"] == len(train_ids):
                snapshot()
                print(f"DURABLE stage_epoch={state['epoch'] + 1} batch={batch}/441 "
                      f"BCE={state['totals'][0] / state['totals'][2]:.4f} "
                      f"AUC_saddle={state['totals'][1] / state['totals'][2]:.4f}", flush=True)
        metrics, scores = evaluate()
        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(metrics["validation_bce"])
        epoch = state["epoch"] + 18
        record = {"epoch": epoch, "stage_epoch": state["epoch"] + 1,
                  "train_bce": state["totals"][0] / state["totals"][2],
                  "train_auc_saddle": state["totals"][1] / state["totals"][2],
                  "validation_bce": metrics["validation_bce"], "validation_macro_auc": metrics["macro_auc"],
                  "agreed_source_macro_auc": metrics["agreed_source_auc"]["macro_auc"],
                  "lr": previous_lr, "lr_next": optimizer.param_groups[0]["lr"],
                  "alpha_mean": float(loss_fn.alpha.mean()), "positive_center_mean": float(loss_fn.a.mean()),
                  "negative_center_mean": float(loss_fn.b.mean()), "seconds_this_attempt": round(time.monotonic() - started, 1)}
        state["history"].append(record)
        if metrics["macro_auc"] > state["best"]["macro_auc"]:
            state["best"] = {"epoch": epoch, "macro_auc": metrics["macro_auc"], "metrics": metrics}
            atomic_save({"state_dict": model.state_dict(), "config": config, "epoch": epoch,
                         "stage": stage, "identity": identity}, output / "checkpoint_best.pt")
            write_json({"study_ids": validation_ids, "scores": scores.tolist()}, output / "best_validation_predictions.json")
        if metrics["macro_auc"] > state["meaningful_auc"] + stage["min_delta"]:
            state["meaningful_auc"] = metrics["macro_auc"]; state["stalled"] = 0
        else:
            state["stalled"] += 1
        state["epoch"] += 1; state["cursor"] = 0; state["order"] = None
        state["completed"] = state["stalled"] >= stage["patience"] or state["epoch"] >= args.epochs
        write_json(state["history"], output / "training_history.json")
        with (output / "training_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record)); writer.writeheader(); writer.writerows(state["history"])
        write_json({"best": state["best"], "initial_metrics": state["initial_metrics"],
                    "epochs_run": state["epoch"], "completed": state["completed"],
                    "stopped_early": state["stalled"] >= stage["patience"], "stage": stage,
                    "train_studies": 3524, "validation_studies": 825, "gold_labels_loaded": False}, output / "run_summary.json")
        snapshot()
        print("EPOCH_RESULT " + json.dumps(record), flush=True)
        print(f"CHECKPOINTS_READY epoch={epoch}", flush=True)
        if state["completed"]:
            break


if __name__ == "__main__":
    main()
