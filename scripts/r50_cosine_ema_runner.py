"""ResNet50 v3.1-D2 224px with a fixed schedule: warmup + cosine over 20 epochs, no early stopping, no
validation-driven LR, plus an exponential moving average (EMA) of the weights.

Everything else matches the r50 v3.1-D2 reference (labels, cache, seed 2026, split, augmentation, batch, loss,
Steven-v4 validation). Reports three readouts: final raw weights, final EMA weights, and the legacy best-AUC epoch.
"""
from pathlib import Path
import ast, hashlib, importlib.util, json, sys
from synovitis_weight_candidate_runner import prepare_original

ROOT = Path('/root/cache'); ASSETS = Path('/root/assets')
RUN = ROOT / 'experiments/r50_cosine_ema_seed2026/run'

EMA_INIT = '''    import copy
    ema_decay = float(config.get("ema_decay", 0.0))
    ema_model = copy.deepcopy(model).eval() if ema_decay else None
    if ema_model is not None:
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
'''
EMA_UPDATE = '''                if ema_model is not None:
                    decay = min(ema_decay, (1 + global_step) / (10 + global_step))
                    with torch.no_grad():
                        for averaged, current in zip(ema_model.state_dict().values(), model.state_dict().values()):
                            if averaged.dtype.is_floating_point:
                                averaged.mul_(decay).add_(current.detach(), alpha=1 - decay)
                            else:
                                averaged.copy_(current)
'''
FINAL = '''    def write_predictions(path, values, epoch, kind):
        path.write_text(json.dumps({
            "validation_fold": config["validation_fold"], "checkpoint_epoch": epoch, "weights": kind,
            "targets_in_order": list(TARGETS),
            "predictions": [{"study_id": study_id, "probabilities": [round(float(p), 6) for p in row]}
                            for study_id, row in zip(validation_ids, values)],
        }, indent=2) + "\\n", encoding="utf-8")

    final_epoch = len(history) - 1
    final_metrics, final_scores = evaluate()
    write_predictions(args.output_dir / "oof_predictions_final.json", final_scores, final_epoch, "final_raw")
    readout = {"final_raw": {"epoch": final_epoch, "macro_auc": final_metrics["macro_auc"],
                             "validation_bce": final_metrics["validation_bce"]}}
    if ema_model is not None:
        ema_final_metrics, ema_final_scores = evaluate(ema_model)
        write_predictions(args.output_dir / "oof_predictions_ema.json", ema_final_scores, final_epoch, "ema_final")
        save_checkpoint({"state_dict": ema_model.state_dict(), "config": config, "epoch": final_epoch, "ema_decay": ema_decay},
                        args.output_dir / "checkpoint_ema.pt")
        readout["ema_final"] = {"epoch": final_epoch, "macro_auc": ema_final_metrics["macro_auc"],
                                "validation_bce": ema_final_metrics["validation_bce"]}
    readout["legacy_best_epoch"] = {"epoch": best["epoch"], "macro_auc": best["macro_auc"]}
    print("FINAL_READOUT " + json.dumps(readout), flush=True)
'''


def prepare_cosine_ema(raw: str) -> str:
    text = prepare_original(raw)

    def swap(old, new, count=1):
        nonlocal text
        assert text.count(old) == count, (old, text.count(old))
        text = text.replace(old, new)

    swap('    start_epoch = 0\n    recovery_path', EMA_INIT + '    start_epoch = 0\n    recovery_path')
    swap('        model.load_state_dict(recovery["state_dict"], strict=True)\n',
         '        model.load_state_dict(recovery["state_dict"], strict=True)\n'
         '        if ema_model is not None:\n'
         '            ema_model.load_state_dict(recovery["ema_state_dict"], strict=True)\n')
    swap('                global_step += 1\n', '                global_step += 1\n' + EMA_UPDATE)
    swap('    def evaluate() -> tuple[dict, np.ndarray]:', '    def evaluate(model=model) -> tuple[dict, np.ndarray]:')
    swap('        metrics, scores = evaluate()\n',
         '        metrics, scores = evaluate()\n'
         '        ema_metrics = evaluate(ema_model)[0] if ema_model is not None else {"macro_auc": None, "validation_bce": None}\n')
    swap('"seconds": round(time.monotonic() - epoch_started, 1)}',
         '"seconds": round(time.monotonic() - epoch_started, 1),\n'
         '                  "ema_validation_bce": ema_metrics["validation_bce"],\n'
         '                  "ema_validation_macro_auc": ema_metrics["macro_auc"]}')
    swap('        persist_epoch(record, epoch)\n',
         '        persist_epoch(record, epoch)\n'
         '        if ema_model is not None:\n'
         '            print(f"ema epoch {epoch}: val_bce={ema_metrics[\'validation_bce\']:.4f} '
         'report-fold AUC={ema_metrics[\'macro_auc\']:.4f}", flush=True)\n')
    swap('"optimizer": optimizer.state_dict(), "scheduler"',
         '"ema_state_dict": (ema_model.state_dict() if ema_model is not None else None),\n'
         '            "optimizer": optimizer.state_dict(), "scheduler"')
    swap('    summary = {\n', FINAL + '    summary = {\n')
    swap('    (args.output_dir / "run_summary.json").write_text',
         '    summary["fixed_schedule_readout"] = readout\n    (args.output_dir / "run_summary.json").write_text')
    ast.parse(text)
    return text


def run():
    import modal
    RUN.mkdir(parents=True, exist_ok=True)
    raw = Path('/root/source/scripts/train_cnn_baseline.py').read_text()
    assert hashlib.sha256(raw.encode()).hexdigest() == 'b8f8302319afbcf9970f0cf5b806d8fb78e72ef6e169b276274ddf7cd5e0ba85'
    audit = json.loads((ASSETS / 'build_report.json').read_text())
    for name, h in audit['files'].items():
        assert hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() == h, name
    p = RUN.parent / 'trainer.py'; p.write_text(prepare_cosine_ema(raw)); sys.path.insert(0, '/root/source')
    spec = importlib.util.spec_from_file_location('r50_cosine_ema_trainer', p)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    m.DURABLE_COMMIT = modal.Volume.from_name('rsna-cnn-topk-support-cache-v1').commit
    sys.argv = ['trainer', '--cache', *[str(ROOT / f'shard_{i}/slice_cache') for i in range(3)],
                '--soft-labels', str(ASSETS / 'train_labels.json'),
                '--validation-soft-labels', str(ASSETS / 'validation_labels.json'),
                '--folds', str(ASSETS / 'folds.csv'), '--config', str(ASSETS / 'config.json'),
                '--output-dir', str(RUN), '--validation-fold', '0', '--train-slices-per-study', '24']
    print('r50 v3.1-D2 224px; cosine 20 epochs peak 1.5e-4; EMA 0.999; no early stopping; seed2026; '
          'fixed Steven validation; durable epoch recovery', flush=True)
    m.main()


if __name__ == '__main__':
    run()
