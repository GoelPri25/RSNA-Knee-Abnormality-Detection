"""Phase 1 coverage run: the original 0.925 r50 trainer and recipe, reading the full-band 224px cache."""
from pathlib import Path
import hashlib, importlib.util, json, sys
from synovitis_weight_candidate_runner import prepare_original

ROOT = Path('/root/cache'); ASSETS = Path('/root/assets')
RUN = ROOT / 'experiments/fullband224_seed2026/run'
CACHES = [ROOT / f'fullband224/shard_{i}/slice_cache' for i in range(3)]


def run():
    import modal
    RUN.mkdir(parents=True, exist_ok=True)
    raw = Path('/root/source/scripts/train_cnn_baseline.py').read_text()
    assert hashlib.sha256(raw.encode()).hexdigest() == 'b8f8302319afbcf9970f0cf5b806d8fb78e72ef6e169b276274ddf7cd5e0ba85'
    audit = json.loads((ASSETS / 'build_report.json').read_text())
    for name, h in audit['files'].items():
        assert hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() == h, name
    p = RUN.parent / 'trainer.py'; p.write_text(prepare_original(raw)); sys.path.insert(0, '/root/source')
    spec = importlib.util.spec_from_file_location('fullband224_trainer', p)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    m.DURABLE_COMMIT = modal.Volume.from_name('rsna-cnn-topk-support-cache-v1').commit
    sys.argv = ['trainer', '--cache', *[str(c) for c in CACHES],
                '--soft-labels', str(ASSETS / 'train_labels.json'),
                '--validation-soft-labels', str(ASSETS / 'validation_labels.json'),
                '--folds', str(ASSETS / 'folds.csv'), '--config', str(ASSETS / 'config.json'),
                '--output-dir', str(RUN), '--validation-fold', '0', '--train-slices-per-study', '24']
    print('r50 v3.1-D2 224px FULL-BAND cache (0-100%, 20/series); original plateau recipe; seed2026; '
          'fixed Steven validation; durable epoch recovery', flush=True)
    m.main()


if __name__ == '__main__':
    run()
