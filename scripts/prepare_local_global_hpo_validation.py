"""Build locked trainval assets from the completed local+global experiment."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/local_global/assets'
OUT = ROOT / 'data/local_global_hpo_validation/assets'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    locks = json.loads((BASE / 'asset_hashes.json').read_text())
    for name, digest in locks.items():
        assert sha(BASE / name) == digest, name
    labels = json.loads((BASE / 'labels.json').read_text())
    old_val = json.loads((BASE / 'validation_labels.json').read_text())
    by_id = {r['study_id']: r for r in labels['labels']}
    val_ids = [r['study_id'] for r in old_val['labels']]
    folds = {r['study_id']: int(r['fold']) for r in csv.DictReader((BASE / 'folds.csv').open())}
    assert len(by_id) == len(labels['labels']) == 4349
    assert len(val_ids) == len(set(val_ids)) == 825
    assert set(val_ids) == {s for s in by_id if folds[s] == 0}
    assert len(set(by_id) - set(val_ids)) == 3524
    assert labels['targets_in_order'] == old_val['targets_in_order']
    val = {'version': 'hpo_v31_d2_fixed_fold0_validation',
           'targets_in_order': labels['targets_in_order'],
           'labels': [by_id[s] for s in val_ids]}
    config = json.loads((BASE / 'config.json').read_text())
    config['version'] = 'hpo_v31_d2_local_global_native_neighbors_trainval_seed2026'
    config['note'] = ('Identical local+global architecture and training recipe; HPO-v3.1-D2 controls fixed '
                      'fold-0 validation. Steven-v4 is audit-only. Fresh ImageNet initialization.')
    execution = ROOT / 'data/local_global/results/execution.json'
    choices = json.loads(execution.read_text())
    assert choices['gpu'] == 'Tesla T4' and choices['batch'] == 8 and choices['windows_per_study'] == 24
    assert choices['selected'] == {'augment_on_gpu': True, 'channels_last': False}
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['labels.json', 'folds.csv']: shutil.copy2(BASE / name, OUT / name)
    shutil.copy2(BASE / 'validation_labels.json', OUT / 'steven_audit_labels.json')
    shutil.copy2(execution, OUT / 'execution.json')
    (OUT / 'validation_labels.json').write_text(json.dumps(val, separators=(',', ':')) + '\n')
    (OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    hashes = {name: sha(OUT / name) for name in ['labels.json', 'folds.csv', 'validation_labels.json',
                                              'steven_audit_labels.json', 'execution.json', 'config.json']}
    (OUT / 'asset_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    original = json.loads((BASE / 'config.json').read_text())
    assert {k: v for k,v in original.items() if k not in {'version','note'}} == {
        k: v for k,v in config.items() if k not in {'version','note'}}
    report = {'train_studies':3524, 'validation_studies':825, 'training_labels_byte_identical':True,
              'folds_byte_identical':True, 'core_config_identical':True,
              'changed_factor':'validation reference for LR / early stopping / checkpoint selection',
              'execution_choices_copied_from_completed_run':choices['selected'], 'sha256':hashes}
    (OUT.parent / 'asset_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
