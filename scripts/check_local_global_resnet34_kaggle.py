"""Offline gradient, disabled-stopping recovery, and notebook-lock checks."""
import ast
import hashlib
import json
from pathlib import Path
import torch
from check_local_global import check_cache,check_model,check_recovery
from local_global_model import SliceModel
from local_global_resnet34_kaggle import prepare_resnet34_trainer

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'data/local_global_resnet34_hpo60/checks';out.mkdir(parents=True,exist_ok=True)
    notebook=json.loads((ROOT/'kaggle/experiment_kernels/local_global_resnet34_hpo60/training.ipynb').read_text())
    bootstrap=''.join(notebook['cells'][1]['source']);ast.parse(bootstrap)
    assignments={node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(bootstrap).body
                 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in {'MODULES','ASSETS'}}
    modules,assets=assignments['MODULES'],assignments['ASSETS']
    for name,text in modules.items():
        assert text==(ROOT/'scripts'/name).read_text(),name
        ast.parse(text)
    locks=json.loads(assets['asset_hashes.json'])
    for name,digest in locks.items():assert hashlib.sha256(assets[name].encode()).hexdigest()==digest,name
    config=json.loads(assets['config.json']);reference=json.loads((ROOT/'data/local_global_hpo_validation/assets/config.json').read_text())
    restored=json.loads(json.dumps(config));restored['model']['backbone']='resnet50'
    restored['early_stopping_patience']=reference['early_stopping_patience']
    restored['version']=reference['version'];restored['note']=reference['note']
    assert restored==reference
    assert config['epochs']==60 and config['early_stopping_patience']==0
    raw=(ROOT/'data/kaggle_reference/hpo_xla_actual/skanee/scripts/train_cnn_baseline.py').read_text()
    trainer=prepare_resnet34_trainer(raw);ast.parse(trainer)
    assert 'checkpoint_best_bce.pt' in trainer
    cache=check_cache(out)
    model=check_model('resnet34')
    recovery=check_recovery(out,'resnet34',0,prepare_resnet34_trainer)
    summary=json.loads((out/'recovery_smoke/run_summary.json').read_text())
    assert summary['epochs_run']==2 and not summary['stopped_early'] and summary['early_stopping_patience']==0
    assert (out/'recovery_smoke/checkpoint_best_bce.pt').is_file()
    assert 'steven_audit_macro_auc' in summary['history'][0]
    report={'notebook_assets_and_modules_locked':True,'only_requested_recipe_changes':True,
            'cache':cache,'resnet34':model,'disabled_stopping_recovery':recovery,
            'best_bce_checkpoint_saved':True,'steven_audit_in_history':True,'device':'cpu','gold_read':False}
    (out/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
