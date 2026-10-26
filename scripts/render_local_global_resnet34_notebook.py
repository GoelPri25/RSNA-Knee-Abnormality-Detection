"""Build a fresh Kaggle notebook from the checked local+global modules/assets."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/local_global_hpo_validation/assets'
DEST=ROOT/'kaggle/experiment_kernels/local_global_resnet34_hpo60'
DATA=ROOT/'data/local_global_resnet34_hpo60'

def main():
    locks=json.loads((BASE/'asset_hashes.json').read_text())
    for name,digest in locks.items(): assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest,name
    config=json.loads((BASE/'config.json').read_text())
    config['model']['backbone']='resnet34'
    config['early_stopping_patience']=0
    config['version']='local_global_resnet34_hpo31_trainval_60epochs_seed2026'
    config['note']='Current local+global HPO-v3.1-D2 trainval recipe; ResNet34/ImageNet; no early stopping; all60 epochs. Steven audit-only.'
    previous=json.loads((BASE/'config.json').read_text())
    difference={k for k in config if config[k]!=previous[k]}
    assert difference=={'model','early_stopping_patience','version','note'}
    assert {k:v for k,v in config['model'].items() if k!='backbone'}=={k:v for k,v in previous['model'].items() if k!='backbone'}
    DATA.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
    (ROOT/'configs/cnn_local_global_resnet34_hpo60.json').write_text(json.dumps(config,indent=2)+'\n')
    assets={name:(BASE/name).read_bytes().decode('utf-8') for name in ['folds.csv','validation_labels.json','steven_audit_labels.json']}
    assets['config.json']=json.dumps(config,indent=2)+'\n'
    assets['asset_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in assets.items()},indent=2)+'\n'
    modules={name:(ROOT/'scripts'/name).read_text() for name in ['local_global_model.py','local_global_data.py',
             'local_global_speed.py','local_global_runner.py','efficientnet_v31_runner.py','local_global_resnet34_kaggle.py']}
    bootstrap='''print("STARTUP: writing pinned local+global modules and HPO validation assets", flush=True)
from pathlib import Path
import sys
WORK = Path('/kaggle/working/local_global_resnet34_hpo60')
WORK.mkdir(parents=True, exist_ok=True)
MODULES = __MODULES__
ASSETS = __ASSETS__
for name, text in MODULES.items(): (WORK / name).write_text(text)
(WORK / 'assets').mkdir(exist_ok=True)
for name, text in ASSETS.items(): (WORK / 'assets' / name).write_text(text)
sys.path.insert(0, str(WORK))
print("STARTUP: modules/assets ready; only fixed input paths will be checked", flush=True)
'''.replace('__MODULES__',repr(modules)).replace('__ASSETS__',repr(assets))
    run='from local_global_resnet34_kaggle import main\nmain()\n'
    ast.parse(bootstrap);ast.parse(run)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
              'language_info':{'name':'python'}},'cells':[
        {'cell_type':'markdown','metadata':{},'source':['# Local+global ResNet34: 60 epochs\n',
          'HPO v3.1-D2 for train and fixed fold-0 validation. 3,524 train / 825 validation, extra 281 training-only, gold excluded.\n',
          'Fresh ImageNet initialization. Early stopping disabled; BCE scheduler retained. Steven-v4 metrics are audit-only.\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':bootstrap.splitlines(keepends=True)},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':run.splitlines(keepends=True)}]}
    (DEST/'training.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    metadata={'id':'tumis001/rsna-local-global-resnet34-hpo-60-epochs','title':'RSNA Local Global ResNet34 HPO 60 Epochs',
              'code_file':'training.ipynb','language':'python','kernel_type':'notebook','is_private':True,
              'enable_gpu':True,'enable_tpu':False,'enable_internet':True,
              'dataset_sources':['tumis001/rsna-cnn-hpo-blend-xla-source-v1','tumis001/rsna-knee-hpo-v31-d2-report-labels'],
              'kernel_sources':[f'tumis001/rsna-cnn-native-triplet-cache-shard-{i}' for i in range(4)],
              'competition_sources':[],'model_sources':[],'machine_shape':'NvidiaTeslaT4'}
    (DEST/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    audit={'config_changes':sorted(difference),'training_label_sha256':locks['labels.json'],
           'fixed_folds_sha256':locks['folds.csv'],'modules_sha256':{n:hashlib.sha256(t.encode()).hexdigest() for n,t in modules.items()},
           'asset_sha256':json.loads(assets['asset_hashes.json']),'native_shards':metadata['kernel_sources']}
    (DATA/'build_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Built',DEST,'notebook bytes',(DEST/'training.ipynb').stat().st_size)
    print(json.dumps(audit['asset_sha256'],indent=2))

if __name__=='__main__':main()
