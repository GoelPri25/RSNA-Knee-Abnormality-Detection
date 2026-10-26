"""Pinned baseline loop, native data/model adapters, durable epoch recovery."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from efficientnet_v31_runner import prepare_trainer

TRAINER_SHA = "b8f8302319afbcf9970f0cf5b806d8fb78e72ef6e169b276274ddf7cd5e0ba85"


def prepare_local_global(raw, steven_audit=False):
    if hashlib.sha256(raw.encode()).hexdigest() != TRAINER_SHA:
        raise ValueError("Baseline trainer differs from the verified reference")
    text = prepare_trainer(raw)
    # Replace the memory adapter/backbone class, not the optimization/loss loop.
    start = text.index("class SliceModel("); end = text.index("\ndef augment(", start)
    text = text[:start] + "from local_global_model import SliceModel\n" + text[end:]
    start = text.index("def prepare_batch("); end = text.index("\ndef confident_auc(", start)
    text = text[:start] + "from local_global_data import NativeStudyCache as StudyCache, prepare_batch\n" + text[end:]
    marker = "    # Labels as free parameters"
    if text.count(marker) != 1: raise ValueError("Unexpected model construction boundary")
    # Preflight restores model/RNG state and persists its measured execution
    # choice before any training. Recovery resumes with that same choice.
    text = text.replace(marker, "    from local_global_speed import choose_execution\n"
        "    choose_execution(model, cache, labels, train_ids, config, device, args.output_dir)\n"
        "    DURABLE_COMMIT()\n" + marker)
    # Inference checkpoints cannot become half-written on interruption.
    text = text.replace("            torch.save(payload, path)",
        '            temporary = path.with_suffix(path.suffix + ".tmp")\n'
        '            torch.save(payload, temporary)\n'
        '            temporary.replace(path)')
    # The recovery file is authoritative. A first-epoch interruption between
    # writing history and writing recovery must be replayable, not a refusal.
    text = text.replace('    if history_path.exists() and not (args.output_dir / "recovery.pt").is_file():\n'
                        '        raise FileExistsError(f"Refusing to overwrite existing run history: {history_path}")',
                        '    if history_path.exists() and not (args.output_dir / "recovery.pt").is_file():\n'
                        '        print("Replaying uncommitted first epoch; no durable recovery exists.", flush=True)\n'
                        '        history_path.unlink()')
    marker = "            batches += 1"
    if text.count(marker) != 1: raise ValueError("Unexpected progress boundary")
    text = text.replace(marker, marker + '\n            if batches % 50 == 0 or batches == batch_count:\n'
        '                print(f"PROGRESS epoch={epoch} batch={batches}/{batch_count}", flush=True)')
    text = text.replace('"input": "3 adjacent slices + 3 acquired-support masks"',
        '"input": "3 native acquired neighbors + 3 support masks; study-wide contextual residual"')
    if steven_audit:
        # Score the same predictions; this reference never reaches the optimizer,
        # scheduler, checkpoint selection or early-stopping condition.
        marker = "        metrics, scores = evaluate()"
        if text.count(marker) != 1: raise ValueError("Unexpected evaluation boundary")
        text = text.replace(marker, marker + '\n'
            '        audit_targets = np.stack([STEVEN_AUDIT_LABELS[s][0] for s in validation_ids])\n'
            '        audit_weights = np.stack([STEVEN_AUDIT_LABELS[s][1] for s in validation_ids])\n'
            '        audit_metrics = confident_auc(audit_targets, audit_weights, scores,\n'
            '                                      config.get("confident_low", .3), config.get("confident_high", .7))\n'
            '        audit_p = np.clip(scores, 1e-7, 1 - 1e-7)\n'
            '        audit_bce = float(((-audit_targets * np.log(audit_p) -\n'
            '                           (1 - audit_targets) * np.log1p(-audit_p)) * audit_weights).sum() / audit_weights.sum())\n'
            '        print(f"STEVEN_AUDIT epoch={epoch} AUC={audit_metrics[\'macro_auc\']:.6f} BCE={audit_bce:.6f} (not used for controls)", flush=True)')
        marker = "        history.append(record)"
        if text.count(marker) != 1: raise ValueError("Unexpected history boundary")
        text = text.replace(marker,
            '        record["steven_audit_macro_auc"] = audit_metrics["macro_auc"]\n'
            '        record["steven_audit_bce"] = audit_bce\n' + marker)
    ast.parse(text)
    return text


def run(root=Path("/root/cache"), source=Path("/root/source"), assets=Path("/root/assets"),
        experiment="local_global_v31_d2_seed2026"):
    import modal
    if experiment not in {"local_global_v31_d2_seed2026", "local_global_v31_trainval_d2_seed2026"}:
        raise ValueError("Unknown local+global experiment")
    out = root / "experiments" / experiment / "run"; out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((assets / "asset_hashes.json").read_text())
    for name, sha in audit.items():
        if hashlib.sha256((assets / name).read_bytes()).hexdigest() != sha: raise ValueError(f"Asset changed: {name}")
    config = json.loads((assets / "config.json").read_text())
    if config.get("learnable_labels") or config.get("learned_weights") or config.get("pairwise_auc_weight", 0):
        raise ValueError("This architecture experiment uses only unchanged soft BCE")
    trainer_path = out.parent / "trainer.py"
    steven_audit = experiment == "local_global_v31_trainval_d2_seed2026"
    if steven_audit and not (out / "execution.json").exists():
        # Reuse only execution choices, never baseline weights or optimizer state.
        (out / "execution.json").write_bytes((assets / "execution.json").read_bytes())
    trainer_path.write_text(prepare_local_global((source / "scripts/train_cnn_baseline.py").read_text(), steven_audit))
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("local_global_trainer", trainer_path)
    trainer = importlib.util.module_from_spec(spec); sys.modules[spec.name] = trainer; spec.loader.exec_module(trainer)
    if steven_audit:
        trainer.STEVEN_AUDIT_LABELS = trainer.read_soft_labels(assets / "steven_audit_labels.json")
    trainer.DURABLE_COMMIT = modal.Volume.from_name("rsna-cnn-native-triplets-224px-v1").commit
    cache_dirs = [root / f"shard_{i}/slice_cache" for i in range(4)]
    for path in cache_dirs:
        if not json.loads((path / "cache_summary.json").read_text())["complete"]: raise ValueError(f"Incomplete cache: {path}")
    sys.argv = ["trainer", "--cache", *map(str, cache_dirs), "--soft-labels", str(assets / "labels.json"),
                "--validation-soft-labels", str(assets / "validation_labels.json"), "--folds", str(assets / "folds.csv"),
                "--config", str(assets / "config.json"), "--output-dir", str(out), "--validation-fold", "0",
                "--train-slices-per-study", "24"]
    trainer.main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="local_global_v31_d2_seed2026")
    run(experiment=parser.parse_args().experiment)
