"""CPU check of the pinned EMA checkpoint against the submission kernel's own expectations (no competition data).

Strict-loads the checkpoint into SliceModel exactly as submission.py does, re-applies the kernel's pins against the
published run summary, and runs one forward pass on a dummy 6-channel batch.
"""
import ast, hashlib, json, sys
from pathlib import Path

import torch

PROJ = Path('/proj/repository')
KERNEL = PROJ / 'kaggle/submission_kernels/r50_cosine_ema_submission/submission.py'
RES = PROJ / 'data/r50_cosine_ema_seed2026/results'
sys.path.insert(0, str(PROJ / 'data/kaggle_reference/hpo_xla_actual/skanee'))
from scripts.train_cnn_baseline import SliceModel
from src.reports.schema import TARGETS

pins = {n.targets[0].id: n.value.value for n in ast.parse(KERNEL.read_text()).body
        if isinstance(n, ast.Assign) and n.targets[0].id.startswith('EXPECTED_')}
path = RES / 'checkpoint_ema.pt'
assert hashlib.sha256(path.read_bytes()).hexdigest() == pins['EXPECTED_CHECKPOINT_SHA256']
ema = json.loads((RES / 'run_summary.json').read_text())['fixed_schedule_readout']['ema_final']
assert ema['epoch'] == pins['EXPECTED_CHECKPOINT_EPOCH'] and abs(ema['macro_auc'] - pins['EXPECTED_REPORT_VALIDATION_AUC']) < 1e-12
ck = torch.load(path, map_location='cpu', weights_only=False)
cfg, mc = ck['config'], ck['config']['model']
assert ck['epoch'] == pins['EXPECTED_CHECKPOINT_EPOCH'] and cfg['version'] == pins['EXPECTED_CONFIG_VERSION']
assert mc['backbone'] == 'resnet50' and mc.get('input_channels') in (None, 6)
model = SliceModel(mc['backbone'], False, mc['dropout'], len(TARGETS), mc.get('top_k_fraction', 0.1))
model.load_state_dict(ck['state_dict'], strict=True)
model.eval()
size = cfg['cache']['image_size']
with torch.inference_mode():
    out = model(torch.rand(8, 6, size, size), torch.zeros(8, dtype=torch.long), 1)
assert out.shape == (1, len(TARGETS)) and torch.isfinite(out).all(), out.shape
print(json.dumps({'strict_load': True, 'epoch': ck['epoch'], 'ema_decay': ck.get('ema_decay'), 'version': cfg['version'],
                  'image_size': size, 'forward_shape': list(out.shape),
                  'sigmoid': [round(float(v), 4) for v in torch.sigmoid(out[0])]}, indent=1))
print('CPU_CHECK_OK')
