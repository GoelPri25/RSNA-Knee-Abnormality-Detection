"""Weight-averaged r50 checkpoints ("model soups"): fold-0 predictions with the trainer's own evaluation path.

Each soup is the uniform average of its checkpoints' state dicts (floating tensors averaged, including BatchNorm
running statistics; integer buffers copied from the first). Predictions use the original trainer's prepare_batch and
SliceModel on the central-band 224px cache, all windows per study, CUDA autocast — the same path that produced the
saved oof_predictions.json files. A one-checkpoint "soup" of the 0.925 epoch-15 checkpoint is the parity check.

Usage (container, GPU):
  podman run --rm --network=none --device nvidia.com/gpu=all -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z \
    docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime python -I /proj/repository/scripts/soup_fold0_predict.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path('/proj/repository'); OUT = Path('/out'); D = PROJ / 'data'
SRC = D / 'kaggle_reference/hpo_xla_actual/skanee'
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('trainer', SRC / 'scripts/train_cnn_baseline.py')
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)

CK = {
    'd2_ep15': 'hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/checkpoint_best.pt',
    'd2_ep23': 'hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/checkpoint_latest.pt',
    'syn_ep15': 'synovitis_weight_candidate_seed2026/published/checkpoint_best.pt',
    'syn_ep23': 'synovitis_weight_candidate_seed2026/published/checkpoint_latest.pt',
    'hpo_v2_ep15': 'hpo_v2_provenance_resnet50_fold0/kaggle_output/hpo_v2_provenance_resnet50_fold0/run/checkpoint_best.pt',
    'hpo_v1_ep17': 'hpo_blend_resnet50_submission/artifact/checkpoint_best.pt',
    'cos_ep13': 'r50_cosine_ema_seed2026/results/checkpoint_best.pt',
    'cos_ep19': 'r50_cosine_ema_seed2026/results/checkpoint_latest.pt',
    'cos_ema': 'r50_cosine_ema_seed2026/results/checkpoint_ema.pt',
}
# Pre-registered soups (no data-driven selection). parity_d2_ep15 must reproduce the saved 0.925 predictions.
SOUPS = {
    'parity_d2_ep15': ['d2_ep15'],
    'within_d2': ['d2_ep15', 'd2_ep23'],
    'within_cos': ['cos_ep13', 'cos_ep19'],
    'd2_syn_best': ['d2_ep15', 'syn_ep15'],
    'plateau_recipe_4': ['d2_ep15', 'd2_ep23', 'syn_ep15', 'syn_ep23'],
    'plateau_labels_6': ['d2_ep15', 'd2_ep23', 'syn_ep15', 'syn_ep23', 'hpo_v2_ep15', 'hpo_v1_ep17'],
    'uniform_all_9': list(CK),
}
only = sys.argv[1:]
device = torch.device('cuda')
A = D / 'r50_cosine_ema_seed2026/assets'
cache = tr.StudyCache(tr.load_manifest([D / f'external/topk-support-cache-shards/shard-{i}-rehost' for i in range(3)], ['unordered']))
vlabels = tr.read_soft_labels(A / 'validation_labels.json')
_, val_ids = tr.build_studies(cache, vlabels, tr.read_folds(A / 'folds.csv'), 0)
assert len(val_ids) == 825, len(val_ids)


def load_sd(name):
    c = torch.load(D / CK[name], map_location='cpu', weights_only=False)
    m = c['config']['model']; assert m['backbone'] == 'resnet50' and c['config']['cache']['image_size'] == 224
    return c['state_dict'], m


def soup_state(names):
    sds = [load_sd(n) for n in names]
    out = {}
    for k, v in sds[0][0].items():
        out[k] = torch.stack([sd[k].float() for sd, _ in sds]).mean(0).to(v.dtype) if v.is_floating_point() else v.clone()
    return out, sds[0][1]


for soup, names in SOUPS.items():
    if only and soup not in only:
        continue
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
    print(f'{soup}: {len(names)} checkpoints -> {len(P)} studies', flush=True)
    if soup == 'parity_d2_ep15':
        ref = json.loads((D / 'hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json').read_text())
        ref = {r['study_id']: r['probabilities'] for r in ref['predictions']}
        diff = np.abs(P - np.array([ref[i] for i in val_ids])).max()
        print(f'PARITY max |p - saved oof| = {diff:.5f}', flush=True)
        assert diff < 5e-3, 'evaluation path does not reproduce the saved predictions'
