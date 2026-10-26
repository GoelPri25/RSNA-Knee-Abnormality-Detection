"""Controlled EfficientNetV2-S runner with epoch-boundary recovery."""
from pathlib import Path
import ast
import json
import csv
import torch



MEMORY_ADAPTER = """
from contextlib import contextmanager, nullcontext
from torch.utils.checkpoint import checkpoint

@contextmanager
def isolated_bn_buffers(module):
    # Recompute against disposable running-stat buffers, then restore original
    # references. No double update and no in-place mutation of saved buffers.
    originals = []
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                value = getattr(child, name)
                if value is not None:
                    originals.append((child, name, value))
                    setattr(child, name, value.clone())
    try:
        yield
    finally:
        for child, name, value in originals:
            setattr(child, name, value)

class MemoryEfficientFeatures(torch.nn.Sequential):
    def forward(self, x):
        for block in self:
            if self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True,
                               context_fn=lambda block=block: (nullcontext(), isolated_bn_buffers(block)))
            else:
                x = block(x)
        return x
"""

def prepare_trainer(text: str) -> str:
    """Patch only the backbone adapter and durable epoch recovery hooks."""
    start = text.index('        old_conv = self.backbone.conv1')
    end = text.index('        self.head =', start)
    text = text[:start] + '''        self.backbone.features = MemoryEfficientFeatures(*self.backbone.features)
        old_conv = self.backbone.features[0][0]
        new_conv = torch.nn.Conv2d(6, old_conv.out_channels, old_conv.kernel_size,
                                  old_conv.stride, old_conv.padding,
                                  bias=old_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, :3].copy_(old_conv.weight)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.backbone.features[0][0] = new_conv
        features = self.backbone.classifier[1].in_features
        self.backbone.classifier = torch.nn.Identity()
''' + text[end:]
    text = text.replace('    if history_path.exists():\n        raise FileExistsError', '    if history_path.exists() and not (args.output_dir / "recovery.pt").is_file():\n        raise FileExistsError')
    boundary = '    for epoch in range(int(config["epochs"])):'
    assert text.count(boundary) == 1
    resume = '''    start_epoch = 0
    recovery_path = args.output_dir / "recovery.pt"
    if recovery_path.is_file():
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
        if recovery["config"] != config:
            raise ValueError("Recovery config differs from this experiment")
        model.load_state_dict(recovery["state_dict"], strict=True)
        optimizer.load_state_dict(recovery["optimizer"])
        scheduler.load_state_dict(recovery["scheduler"])
        scaler.load_state_dict(recovery["scaler"])
        global_step = recovery["global_step"]
        history, best = recovery["history"], recovery["best"]
        best_scores = recovery["best_scores"]
        stalled, best_loss = recovery["stalled"], recovery["best_loss"]
        rng.bit_generator.state = recovery["numpy_rng"]
        torch.set_rng_state(recovery["torch_rng"])
        torch.cuda.set_rng_state_all(recovery["cuda_rng"])
        start_epoch = recovery["epoch"] + 1
        # Roll back a CSV record written before a preemption but not committed.
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"RESUMED epoch={start_epoch} optimizer_step={global_step} stale={stalled}", flush=True)
        if (patience and stalled >= patience) or start_epoch >= int(config["epochs"]):
            print("Recovered a finished run; no additional training.", flush=True)
            start_epoch = int(config["epochs"])
            stopped_early = bool(patience and stalled >= patience)
    for epoch in range(start_epoch, int(config["epochs"])):
'''
    text = text.replace(boundary, resume.rstrip())
    # Save after checkpoint selection, scheduler stepping, and patience updates.
    boundary = '        if patience and stalled >= patience:'
    assert text.count(boundary) == 1
    save = '''        payload = {
            "state_dict": model.state_dict(), "config": config, "epoch": epoch,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "global_step": global_step,
            "history": history, "best": best, "best_scores": best_scores,
            "stalled": stalled, "best_loss": best_loss,
            "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        temporary = args.output_dir / "recovery.tmp"
        torch.save(payload, temporary)
        temporary.replace(args.output_dir / "recovery.pt")
        DURABLE_COMMIT()
        print(f"CHECKPOINTS_READY epoch={epoch}", flush=True)
'''
    text = text.replace(boundary, save + boundary)
    memory_adapter = '\n' + MEMORY_ADAPTER + '\n'
    text = text.replace('class SliceModel(torch.nn.Module):', memory_adapter + 'class SliceModel(torch.nn.Module):')
    ast.parse(text)
    return text


TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def prepare_steven_validation(steven_csv: Path, folds_path: Path, gold_split_path: Path,
                              output_path: Path) -> tuple[int, int]:
    import numpy as np

    with gold_split_path.open(newline="", encoding="utf-8") as stream:
        gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    with folds_path.open(newline="", encoding="utf-8") as stream:
        fold_map = {row["study_id"]: int(row["fold"]) for row in csv.DictReader(stream)}
    validation_ids = {study_id for study_id, fold in fold_map.items() if fold == 0}
    if len(gold_ids) != 58 or len(validation_ids) != 825 or gold_ids & validation_ids:
        raise RuntimeError("Fold/gold split dimensions or separation check failed")

    records = []
    seen: set[str] = set()
    with steven_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"StudyInstanceUID", *TARGETS}.issubset(reader.fieldnames or []):
            raise ValueError("Steven v4 label file has an unexpected target schema")
        for row in reader:
            study_id = row["StudyInstanceUID"]
            if study_id not in validation_ids:
                continue
            if study_id in seen:
                raise ValueError(f"Duplicate Steven v4 validation row: {study_id}")
            seen.add(study_id)
            probabilities, weights = [], []
            for target in TARGETS:
                try:
                    value = float(row[target])
                except (TypeError, ValueError):
                    value = float("nan")
                valid = bool(np.isfinite(value) and 0.0 <= value <= 1.0)
                probabilities.append(value if valid else None)
                weights.append(float(valid))
            records.append({"study_id": study_id, "probabilities": probabilities, "weights": weights})
    if seen != validation_ids:
        raise RuntimeError(f"Steven v4 validation coverage mismatch: {len(validation_ids - seen)} missing")
    output_path.write_text(json.dumps({
        "version": "steven_v4_fold0_validation_only",
        "targets_in_order": TARGETS,
        "labels": records,
    }, allow_nan=False) + "\n", encoding="utf-8")
    return len(gold_ids), len(validation_ids)

def run() -> None:
    import importlib.util
    import sys
    import csv
    import hashlib
    import modal
    root = Path('/root/rsna/persistent_cache/experiments/efficientnet_v31_d2_fold0')
    root.mkdir(parents=True, exist_ok=True)
    source = Path('/root/rsna/source')
    label_path = root / 'inputs/soft_labels_hpo_v3_1_d2_structured.json'
    assert hashlib.sha256(label_path.read_bytes()).hexdigest() == '38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d'
    labels = json.loads(label_path.read_text())
    ids = {r['study_id'] for r in labels['labels']}
    assert len(ids) == 4349 and all(all(w == 1 for w in r['weights']) for r in labels['labels'])
    with (source / 'configs/report_folds_v5.csv').open() as f:
        base = {r['study_id']: int(r['fold']) for r in csv.DictReader(f)}
    assert len(base) == 4068 and sum(v == 0 for v in base.values()) == 825
    extra = ids - base.keys()
    assert len(extra) == 281
    base.update({s: 5 for s in extra})
    folds = root / 'folds.csv'
    with folds.open('w') as f:
        w=csv.writer(f); w.writerow(['study_id','fold']); w.writerows(sorted(base.items()))
    with (source / 'configs/gold_split_v1.csv').open() as f:
        gold = {r['StudyInstanceUID'] for r in csv.DictReader(f)}
    assert len(gold) == 58 and not gold & ids
    validation = root / 'steven_v4_fold0_reference.json'
    prepare_steven_validation(root/'inputs/llm_labels_v4_blend.csv', folds,
                              source/'configs/gold_split_v1.csv', validation)
    config = json.loads(Path('/root/rsna/reference_config.json').read_text())
    config['model']['backbone'] = 'efficientnet_v2_s'
    config['version'] = 'hpo_v31_d2_efficientnet_v2_s_fold0'
    cp = root / 'config.json'; cp.write_text(json.dumps(config,indent=2)+'\n')
    raw=(source/'scripts/train_cnn_baseline.py').read_text()
    trainer_file=root/'train_efficientnet.py';trainer_file.write_text(prepare_trainer(raw))
    spec=importlib.util.spec_from_file_location('train_efficientnet',trainer_file)
    trainer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trainer
    sys.path.insert(0,str(source));spec.loader.exec_module(trainer)
    volume=modal.Volume.from_name('rsna-cnn-topk-support-cache-v1')
    trainer.DURABLE_COMMIT = volume.commit
    run_dir=root/'run';run_dir.mkdir(exist_ok=True)
    cache_dirs=[Path(f'/root/rsna/persistent_cache/shard_{i}/slice_cache') for i in range(3)]
    for d in cache_dirs:
        for name in ['manifest.jsonl','cache_summary.json']+[f'{kind}_{i:03d}.npy' for kind in ['slices','support'] for i in range(4)]:
            if not (d/name).is_file(): raise FileNotFoundError(d/name)
    print('EfficientNetV2-S ImageNet; HPO v3.1-D2; batch8; 3524 train / fixed825 validation; T4',flush=True)
    sys.argv=[str(trainer_file),'--cache',*map(str,cache_dirs),'--soft-labels',str(label_path),
              '--validation-soft-labels',str(validation),'--folds',str(folds),'--config',str(cp),
              '--output-dir',str(run_dir),'--validation-fold','0','--train-slices-per-study','24']
    trainer.main()

if __name__ == '__main__':
    run()
