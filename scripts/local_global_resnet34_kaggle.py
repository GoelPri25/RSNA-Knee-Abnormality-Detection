"""Kaggle native local+global ResNet34, HPO trainval, all 60 epochs."""
import ast
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

EXPERIMENT = 'local_global_resnet34_hpo60'
LABEL_SHA = '38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d'
TRAINER_SHA = 'b8f8302319afbcf9970f0cf5b806d8fb78e72ef6e169b276274ddf7cd5e0ba85'
EXPECTED_STUDIES = [1081,1047,1142,1079]


def exactly_one(paths, description):
    found = [p for p in paths if p.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f'{description}: expected one fixed input, found {found}; checked {[str(p) for p in paths]}')
    return found[0]


def prepare_resnet34_trainer(raw):
    from local_global_runner import prepare_local_global
    text = prepare_local_global(raw, steven_audit=True)
    # Observational checkpoint only: never used to step LR or stop training.
    marker = '        persist_epoch(record, epoch)'
    assert text.count(marker) == 1
    text = text.replace(marker, marker + '''
        previous_bce = min((r["validation_bce"] for r in history[:-1]), default=float("inf"))
        if record["validation_bce"] < previous_bce:
            save_checkpoint({"state_dict": model.state_dict(), "config": config, "epoch": epoch},
                            args.output_dir / "checkpoint_best_bce.pt")
'''.rstrip())
    marker = '        history.append(record)'
    assert text.count(marker) == 1
    text = text.replace(marker, '''        record["bce_generalization_gap"] = record["validation_bce"] - record["train_bce"]
''' + marker)
    # Full epoch-boundary recovery works with patience=0; all loop exits are
    # external failures or the explicit 60-epoch limit, not metric plateaus.
    assert 'if (patience and stalled >= patience) or start_epoch >= int(config["epochs"]):' in text
    ast.parse(text)
    return text


def main():
    print('STARTUP: local+global ResNet34/ImageNet; HPO v3.1-D2 train and validation; 60 epochs; early stopping DISABLED.', flush=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    import numpy as np
    import torch
    import torchvision
    from local_global_runner import TRAINER_SHA as pinned_sha
    assert pinned_sha == TRAINER_SHA
    inputs = Path('/kaggle/input')
    work = Path('/kaggle/working') / EXPERIMENT
    assets = work/'assets'; run = work/'run'
    source_file = exactly_one([
        inputs/'datasets/tumis001/rsna-cnn-hpo-blend-xla-source-v1/skanee/scripts/train_cnn_baseline.py',
        inputs/'rsna-cnn-hpo-blend-xla-source-v1/skanee/scripts/train_cnn_baseline.py'], 'baseline source')
    source = source_file.parents[1]
    label_path = exactly_one([
        inputs/'datasets/tumis001/rsna-knee-hpo-v31-d2-report-labels/soft_labels_hpo_v3_1_d2_structured.json',
        inputs/'rsna-knee-hpo-v31-d2-report-labels/soft_labels_hpo_v3_1_d2_structured.json'], 'HPO labels')
    if hashlib.sha256(label_path.read_bytes()).hexdigest() != LABEL_SHA:
        raise ValueError('HPO label hash mismatch')
    if hashlib.sha256(source_file.read_bytes()).hexdigest() != TRAINER_SHA:
        raise ValueError('Trainer hash mismatch')
    locks = json.loads((assets/'asset_hashes.json').read_text())
    for name,digest in locks.items():
        if hashlib.sha256((assets/name).read_bytes()).hexdigest() != digest: raise ValueError(f'Asset hash mismatch: {name}')
    config = json.loads((assets/'config.json').read_text())
    assert config['model']['backbone'] == 'resnet34' and config['model']['pretrained']
    assert config['epochs'] == 60 and config['early_stopping_patience'] == 0 and config['batch_studies'] == 8
    labels = json.loads(label_path.read_text())
    label_ids = {r['study_id'] for r in labels['labels']}
    folds = {r['study_id']:int(r['fold']) for r in csv.DictReader((assets/'folds.csv').open())}
    validation = json.loads((assets/'validation_labels.json').read_text())
    val_ids = {r['study_id'] for r in validation['labels']}
    assert len(label_ids) == len(labels['labels']) == len(folds) == 4349
    assert set(folds) == label_ids and val_ids == {s for s in folds if folds[s] == 0} and len(val_ids) == 825
    assert len(label_ids-val_ids) == 3524
    by_id = {r['study_id']:r for r in labels['labels']}
    assert all(r == by_id[r['study_id']] for r in validation['labels'])
    assert labels['targets_in_order'] == validation['targets_in_order']
    base_folds = {r['study_id']:int(r['fold']) for r in csv.DictReader((source/'configs/report_folds_v5.csv').open())}
    added = label_ids-set(base_folds)
    assert len(added) == 281 and all(folds[s] != 0 for s in added)
    assert all(folds[s] == f for s,f in base_folds.items())
    gold_ids = {r['StudyInstanceUID'] for r in csv.DictReader((source/'configs/gold_split_v1.csv').open())}
    assert len(gold_ids) == 58 and not gold_ids & label_ids
    cache_dirs=[];seen=set()
    for i in range(4):
        slug=f'rsna-cnn-native-triplet-cache-shard-{i}'
        tail=Path(f'native_triplets/shard_{i}/slice_cache')
        cache = exactly_one([inputs/'notebooks/tumis001'/slug/tail,
                             inputs/'kernels/tumis001'/slug/tail,inputs/slug/tail], f'native shard {i}')
        summary = json.loads((cache/'cache_summary.json').read_text())
        assert summary['complete'] and summary['schema'] == 'native_triplets_v1' and summary['decode_errors'] == 0
        assert summary['shard'] == [i,4] and summary['studies'] == EXPECTED_STUDIES[i]
        assert summary['cache'] == config['cache']
        records = [json.loads(line) for line in (cache/'manifest.jsonl').read_text().splitlines() if line.strip()]
        ids = {r['study_id'] for r in records}
        assert len(ids) == summary['studies'] and not ids & seen
        seen.update(ids)
        for name in {r[k] for r in records for k in ['part','support_part']}:
            array = np.load(cache/name,mmap_mode='r')
            assert array.dtype == np.uint8 and array.shape[1:] == (224,224)
            assert all(r['row_start']+r['stored_count'] <= len(array) for r in records if r['part']==name or r['support_part']==name)
            del array
        cache_dirs.append(cache)
        print(f'CACHE_READY shard={i} studies={len(ids)} path={cache}',flush=True)
    assert seen == label_ids
    if not torch.cuda.is_available(): raise RuntimeError('GPU is required for training')
    torch.cuda.set_device(0);torch.set_num_threads(4)
    print(f'GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; torchvision={torchvision.__version__}; one GPU / global batch 8',flush=True)
    print('SPLIT: 3524 train including extra281; fixed825 HPO validation;58gold excluded.',flush=True)
    print('CONTROLS: no early stopping; HPO BCE LR plateau (min_lr=1e-6); HPO AUC best checkpoint. Steven scores are audit-only.',flush=True)
    run.mkdir(parents=True,exist_ok=True)
    # Same execution choices as the Modal arm, no benchmarking or trial updates.
    execution={'gpu':torch.cuda.get_device_name(0),'batch':8,'windows_per_study':24,
               'selected':{'augment_on_gpu':True,'channels_last':False},
               'note':'Execution fixed to current Modal arm; no speed benchmark run on Kaggle.'}
    if not (run/'execution.json').exists(): (run/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    sys.path.insert(0,str(source))
    trainer_path=work/'trainer.py';trainer_path.write_text(prepare_resnet34_trainer(source_file.read_text()))
    spec=importlib.util.spec_from_file_location('resnet34_kaggle_trainer',trainer_path)
    trainer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trainer;spec.loader.exec_module(trainer)
    trainer.STEVEN_AUDIT_LABELS=trainer.read_soft_labels(assets/'steven_audit_labels.json')
    trainer.DURABLE_COMMIT=lambda: None  # Kaggle local files; no Modal client/server use.
    sys.argv=['trainer','--cache',*map(str,cache_dirs),'--soft-labels',str(label_path),
              '--validation-soft-labels',str(assets/'validation_labels.json'),'--folds',str(assets/'folds.csv'),
              '--config',str(assets/'config.json'),'--output-dir',str(run),'--validation-fold','0','--train-slices-per-study','24']
    trainer.main()
    summary=json.loads((run/'run_summary.json').read_text())
    assert summary['epochs_run'] == 60 and not summary['stopped_early']
    print('ALL_60_EPOCHS_COMPLETE',flush=True)

if __name__=='__main__':main()
