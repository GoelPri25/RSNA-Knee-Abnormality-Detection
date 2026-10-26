"""Weight-averaged r50 checkpoints ("model soups"): fold-0 predictions with the original trainer's evaluation path.

Each soup is the uniform average of its checkpoints' state dicts (floating tensors averaged, including BatchNorm running
statistics; integer buffers copied from the first). Predictions use the trainer's prepare_batch and SliceModel on the
central-band 224px cache, all windows per study, CUDA autocast: the path that produced the saved oof_predictions.json.
The one-checkpoint soup of the 0.925 epoch-15 checkpoint must reproduce its saved predictions (parity check).
Soups are pre-registered; no selection happens here. Fixed input paths, every checkpoint pinned by SHA-256.
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def ds(owner, slug):
    for p in (Path('/kaggle/input/datasets') / owner / slug, Path('/kaggle/input') / slug):
        if p.is_dir():
            return p
    raise FileNotFoundError(f'{owner}/{slug} not attached')


SOUP = ds('tumis001', 'rsna-r50-soup-inputs')
SRC = SOUP / 'skanee'
TRAINER = SRC / 'scripts/train_cnn_baseline.py'
OUT = Path('/kaggle/working')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


assert sha(TRAINER) == 'b8f8302319afbcf9970f0cf5b806d8fb78e72ef6e169b276274ddf7cd5e0ba85', 'trainer changed'
assert sha(SOUP / 'validation_labels.json').startswith('10fdaaab'), 'validation labels changed'
assert sha(SOUP / 'folds.csv').startswith('2dd1044a'), 'folds changed'
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('trainer', TRAINER)
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)

COS = ds('tumis001', 'rsna-r50-v31-d2-cosine-ema-seed2026')
CK = {  # name: (path, sha256)
    'd2_ep15': (ds('tumis001', 'rsna-hpo-v31-d2-resnet50-best') / 'checkpoint_best.pt', '4091948e1ce9000e8f4ab7dbd7d859de9370a31d3fe09bd09bf99eaf45d8e48c'),
    'd2_ep23': (SOUP / 'checkpoints/d2_ep23.pt', 'dbcaf54e001f6046679a26687eb8d7625d8d1e14a3b0fd1b77a1b08f362a3b7c'),
    'syn_ep15': (SOUP / 'checkpoints/syn_ep15.pt', '2458452914a3e13db7685b646cd544b51d55cf57e76dfb0f2d55f62c86a629fa'),
    'syn_ep23': (SOUP / 'checkpoints/syn_ep23.pt', 'ff2ec61527b26618f445f7837a127657c579a3d607f7397e5c06cb635ceaead4'),
    'hpo_v2_ep15': (ds('tumis001', 'rsna-hpo-v2-provenance-resnet50-fold0-best') / 'checkpoint_best.pt', 'e5edb5ffa08a24037d6141cd9a4bcba45c3ab05b3082c6935533368822c6f076'),
    'hpo_v1_ep17': (SOUP / 'checkpoints/hpo_v1_ep17.pt', '7c76e8760b6989fb5bb3afa2fe14cf016ac766981ad612c94fc058acb703120a'),
    'cos_ep13': (COS / 'checkpoint_best.pt', 'c568d37324e96d85d171b7ed5a01cf0311262bdd35e25c3c19d630e06e969010'),
    'cos_ep19': (COS / 'checkpoint_latest.pt', 'fb145abe9edbe0909f19f7df20952e20f7bc1f5930637ebcbdc130c37d755ae3'),
    'cos_ema': (COS / 'checkpoint_ema.pt', '7d0cf81b1d3c2ae71950b997d577cdc74f3db1eba132294dbddd76df2890b257'),
}
for n, (p, h) in CK.items():
    assert sha(p) == h, f'{n}: unexpected checkpoint {p}'
SOUPS = {  # pre-registered
    'parity_d2_ep15': ['d2_ep15'],
    'within_d2': ['d2_ep15', 'd2_ep23'],
    'within_cos': ['cos_ep13', 'cos_ep19'],
    'd2_syn_best': ['d2_ep15', 'syn_ep15'],
    'plateau_recipe_4': ['d2_ep15', 'd2_ep23', 'syn_ep15', 'syn_ep23'],
    'plateau_labels_6': ['d2_ep15', 'd2_ep23', 'syn_ep15', 'syn_ep23', 'hpo_v2_ep15', 'hpo_v1_ep17'],
    'uniform_all_9': list(CK),
}
device = torch.device('cuda')
started = time.monotonic()
cache = tr.StudyCache(tr.load_manifest([ds('tumis001', f'rsna-cnn-topk-support-cache-shard-{i}-rehost') for i in range(3)], ['unordered']))
vlabels = tr.read_soft_labels(SOUP / 'validation_labels.json')
_, val_ids = tr.build_studies(cache, vlabels, tr.read_folds(SOUP / 'folds.csv'), 0)
assert len(val_ids) == 825, len(val_ids)


def load_sd(name):
    c = torch.load(CK[name][0], map_location='cpu', weights_only=False)
    m = c['config']['model']; assert m['backbone'] == 'resnet50' and c['config']['cache']['image_size'] == 224
    return c['state_dict'], m


def soup_state(names):
    sds = [load_sd(n) for n in names]
    assert all(sd.keys() == sds[0][0].keys() for sd, _ in sds)
    out = {}
    for k, v in sds[0][0].items():
        out[k] = torch.stack([sd[k].float() for sd, _ in sds]).mean(0).to(v.dtype) if v.is_floating_point() else v.clone()
    return out, sds[0][1]


summary = {}
for soup, names in SOUPS.items():
    sd, mc = soup_state(names)
    model = tr.SliceModel('resnet50', False, mc['dropout'], len(tr.TARGETS), mc.get('top_k_fraction', 0.1))
    model.load_state_dict(sd, strict=True); model.to(device).eval()
    scores = []
    with torch.no_grad():
        for s in range(0, len(val_ids), 4):
            chunk = val_ids[s:s + 4]
            x, owner, *_ = tr.prepare_batch(cache, vlabels, chunk, None, None, device)
            with torch.autocast('cuda', enabled=True):
                logits = model(x, owner, len(chunk))
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())
    P = np.concatenate(scores)
    (OUT / f'{soup}.json').write_text(json.dumps({
        'soup': soup, 'members': names, 'targets_in_order': list(tr.TARGETS),
        'predictions': [{'study_id': i, 'probabilities': [round(float(p), 6) for p in row]} for i, row in zip(val_ids, P)],
    }) + '\n')
    summary[soup] = {'members': names, 'studies': len(P)}
    print(f'{soup}: {len(names)} checkpoints -> {len(P)} studies; {time.monotonic() - started:.0f}s', flush=True)
    if soup == 'parity_d2_ep15':
        ref = json.loads((SOUP / 'd2_ep15_saved_oof_predictions.json').read_text())
        ref = {r['study_id']: r['probabilities'] for r in ref['predictions']}
        diff = float(np.abs(P - np.array([ref[i] for i in val_ids])).max())
        summary['parity_max_abs_diff'] = diff
        print(f'PARITY max |p - saved oof| = {diff:.5f}', flush=True)
        assert diff < 5e-3, 'evaluation path does not reproduce the saved predictions'
(OUT / 'soup_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
