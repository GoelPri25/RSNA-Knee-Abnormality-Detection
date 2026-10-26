"""Test-time augmentation for the r50 224 models on fold 0, with the original trainer's evaluation path.

Pre-registered (2026-10-10). The trainer's only geometric augmentation is a per-study translation drawn uniformly in
+-0.06 (affine-grid units; no zoom, no rotation, no flips), applied with grid_sample(bilinear, zeros) to all six
channels after normalisation; intensity jitter is x[0.85, 1.15]. Views reuse exactly that transform:
  none : identity (parity with the saved out-of-fold predictions)
  tta5 : identity + 4 axis shifts of 0.06
  tta9 : 3 x 3 grid of shifts in {-0.06, 0, +0.06}
  int3 : identity at intensity x0.85, x1, x1.15
Each set is the mean of its views' probabilities. Fixed input paths; every file pinned by SHA-256.
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def ds(owner, slug):
    for p in (Path('/kaggle/input/datasets') / owner / slug, Path('/kaggle/input') / slug):
        if p.is_dir():
            return p
    raise FileNotFoundError(f'{owner}/{slug} not attached')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


SOUP = ds('tumis001', 'rsna-r50-soup-inputs')
SRC = SOUP / 'skanee'
TRAINER = SRC / 'scripts/train_cnn_baseline.py'
OUT = Path('/kaggle/working')
assert sha(TRAINER) == 'b8f8302319afbcf9970f0cf5b806d8fb78e72ef6e169b276274ddf7cd5e0ba85', 'trainer changed'
assert sha(SOUP / 'validation_labels.json').startswith('10fdaaab'), 'validation labels changed'
assert sha(SOUP / 'folds.csv').startswith('2dd1044a'), 'folds changed'
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('trainer', TRAINER)
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)

MODELS = {
    'd2_ep15': (ds('tumis001', 'rsna-hpo-v31-d2-resnet50-best') / 'checkpoint_best.pt',
                '4091948e1ce9000e8f4ab7dbd7d859de9370a31d3fe09bd09bf99eaf45d8e48c'),
    'hpo_v1_ep17': (SOUP / 'checkpoints/hpo_v1_ep17.pt', '7c76e8760b6989fb5bb3afa2fe14cf016ac766981ad612c94fc058acb703120a'),
}
for n, (p, h) in MODELS.items():
    assert sha(p) == h, f'{n}: unexpected checkpoint {p}'
S = 0.06
SHIFTS = [(dx, dy) for dy in (-S, 0.0, S) for dx in (-S, 0.0, S)]           # tta9; index 4 is identity
VIEWS = {f'shift_{dx:+.2f}_{dy:+.2f}': ('shift', dx, dy) for dx, dy in SHIFTS}
VIEWS['int_0.85'] = ('int', 0.85, 0.0); VIEWS['int_1.15'] = ('int', 1.15, 0.0)
IDENT = 'shift_+0.00_+0.00'
SETS = {'none': [IDENT],
        'tta5': [IDENT, 'shift_-0.06_+0.00', 'shift_+0.06_+0.00', 'shift_+0.00_-0.06', 'shift_+0.00_+0.06'],
        'tta9': [f'shift_{dx:+.2f}_{dy:+.2f}' for dx, dy in SHIFTS],
        'int3': ['int_0.85', IDENT, 'int_1.15']}


def view(x, kind, a, b):
    if kind == 'int':
        x = x.clone(); x[:, :3] = x[:, :3] * a
        return x
    if a == 0.0 and b == 0.0:
        return x
    theta = torch.tensor([[1.0, 0.0, a], [0.0, 1.0, b]], device=x.device).expand(len(x), 2, 3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


device = torch.device('cuda')
started = time.monotonic()
cache = tr.StudyCache(tr.load_manifest([ds('tumis001', f'rsna-cnn-topk-support-cache-shard-{i}-rehost') for i in range(3)], ['unordered']))
vlabels = tr.read_soft_labels(SOUP / 'validation_labels.json')
_, val_ids = tr.build_studies(cache, vlabels, tr.read_folds(SOUP / 'folds.csv'), 0)
assert len(val_ids) == 825, len(val_ids)

nets = {}
for n, (p, _) in MODELS.items():
    c = torch.load(p, map_location='cpu', weights_only=False); mc = c['config']['model']
    assert mc['backbone'] == 'resnet50' and c['config']['cache']['image_size'] == 224
    m = tr.SliceModel('resnet50', False, mc['dropout'], len(tr.TARGETS), mc.get('top_k_fraction', 0.1))
    m.load_state_dict(c['state_dict'], strict=True); nets[n] = m.to(device).eval()

probs = {n: {v: [] for v in VIEWS} for n in MODELS}
with torch.no_grad():
    for s in range(0, len(val_ids), 4):
        chunk = val_ids[s:s + 4]
        x, owner, *_ = tr.prepare_batch(cache, vlabels, chunk, None, None, device)
        for v, (kind, a, b) in VIEWS.items():
            xv = view(x, kind, a, b)
            for n, m in nets.items():
                with torch.autocast('cuda', enabled=True):
                    logits = m(xv, owner, len(chunk))
                probs[n][v].append(torch.sigmoid(logits.float()).cpu().numpy())
        if s % 200 == 0:
            print(f'{s}/{len(val_ids)} studies; {time.monotonic() - started:.0f}s', flush=True)

summary = {}
for n in MODELS:
    P = {v: np.concatenate(probs[n][v]) for v in VIEWS}
    for name, members in SETS.items():
        A = np.mean([P[v] for v in members], 0)
        (OUT / f'{n}__{name}.json').write_text(json.dumps({
            'model': n, 'set': name, 'views': members, 'targets_in_order': list(tr.TARGETS),
            'predictions': [{'study_id': i, 'probabilities': [round(float(p), 6) for p in row]} for i, row in zip(val_ids, A)],
        }) + '\n')
    summary[n] = {'views': list(VIEWS)}
ref = json.loads((SOUP / 'd2_ep15_saved_oof_predictions.json').read_text())
ref = {r['study_id']: r['probabilities'] for r in ref['predictions']}
diff = float(np.abs(np.concatenate(probs['d2_ep15'][IDENT]) - np.array([ref[i] for i in val_ids])).max())
summary['parity_d2_ep15_max_abs_diff'] = diff
summary['seconds'] = round(time.monotonic() - started, 1)
(OUT / 'tta_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
assert diff < 5e-3, 'evaluation path does not reproduce the saved predictions'
