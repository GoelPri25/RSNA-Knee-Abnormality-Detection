"""Offline checks for locked labels and audit-only per-epoch Steven scoring."""
import ast
import hashlib
import importlib.util
import json
import sys
import textwrap
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'data/kaggle_reference/hpo_xla_actual/skanee')]
from local_global_runner import prepare_local_global

def main():
    raw = (ROOT/'data/kaggle_reference/hpo_xla_actual/skanee/scripts/train_cnn_baseline.py').read_text()
    out = ROOT/'data/local_global_hpo_validation/checks'; out.mkdir(parents=True,exist_ok=True)
    old_namespace = {'__name__':'reference'}
    exec((out/'baseline_runner.py').read_text(),old_namespace)
    assert prepare_local_global(raw) == old_namespace['prepare_local_global'](raw)
    text = prepare_local_global(raw, steven_audit=True)
    ast.parse(text)
    a = text.index('        metrics, scores = evaluate()')
    b = text.index('        learning_rate_used', a)
    assert 'scheduler' not in text[a:b]
    audit_nodes = ast.parse(textwrap.dedent(text[a:b])).body[1:]
    assert all(not isinstance(node, ast.Assign) or all(
        not isinstance(target, ast.Name) or target.id != 'metrics' for target in node.targets)
        for node in audit_nodes)
    p = out/'generated_trainer.py';p.write_text(text)
    spec = importlib.util.spec_from_file_location('checked_trainer',p)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assets = ROOT/'data/local_global_hpo_validation/assets'
    locks = json.loads((assets/'asset_hashes.json').read_text())
    for name,digest in locks.items(): assert hashlib.sha256((assets/name).read_bytes()).hexdigest() == digest
    hpo = module.read_soft_labels(assets/'validation_labels.json')
    steven = module.read_soft_labels(assets/'steven_audit_labels.json')
    assert set(hpo) == set(steven) and len(hpo) == 825
    oof = json.loads((ROOT/'data/local_global/results/oof_predictions.json').read_text())
    ids = [r['study_id'] for r in oof['predictions']]
    scores = np.array([r['probabilities'] for r in oof['predictions']], dtype=np.float32)
    control_metrics = module.confident_auc(np.stack([hpo[s][0] for s in ids]),np.stack([hpo[s][1] for s in ids]),scores,.3,.7)
    control_before = dict(control_metrics)
    scope = dict(np=np,validation_ids=ids,STEVEN_AUDIT_LABELS=steven,confident_auc=module.confident_auc,
                 config={'confident_low':.3,'confident_high':.7},epoch=12,evaluate=lambda:(control_metrics,scores))
    exec(textwrap.dedent(text[a:b]),scope)
    assert control_metrics == control_before
    expected = json.loads((ROOT/'data/local_global/results/run_summary.json').read_text())['best']['macro_auc']
    assert abs(scope['audit_metrics']['macro_auc']-expected) < 1e-5
    for f in ['modal_local_global_hpo_validation.py','run_modal_personal_local_global_hpo_validation.py','prepare_local_global_hpo_validation.py']:
        ast.parse((ROOT/'scripts'/f).read_text())
    result = {'old_generated_trainer_unchanged':True,'validation_ids_identical':825,
              'audit_preserves_control_metrics':True,'audit_auc':scope['audit_metrics']['macro_auc'],
              'hpo_control_auc_on_saved_predictions':control_metrics['macro_auc'],'syntax_checks':True}
    (out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':main()
