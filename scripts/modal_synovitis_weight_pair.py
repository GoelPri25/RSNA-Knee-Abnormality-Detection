"""One detached coordinator, sequential matched T4 runs, existing 224px cache."""
from pathlib import Path
import json,os,subprocess,shutil
import modal
LOCAL=Path(__file__).resolve().parents[1]
volume=modal.Volume.from_name('rsna-cnn-topk-support-cache-v1',create_if_missing=False)
secret=modal.Secret.from_dict({'KAGGLE_ACCESS_TOKEN':(Path.home()/'.kaggle/access_token').read_text().strip()}) if modal.is_local() else modal.Secret.from_dict({})
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1','torchvision==0.20.1','numpy','scikit-learn','matplotlib','kaggle==2.2.2','modal')
 .add_local_dir(str(LOCAL/'data/kaggle_reference/hpo_xla_actual/skanee'),'/root/source')
 .add_local_dir(str(LOCAL/'data/synovitis_weight_experiment/assets'),'/root/assets')
 .add_local_file(str(LOCAL/'scripts/efficientnet_v31_runner.py'),'/root/efficientnet_v31_runner.py')
 .add_local_file(str(LOCAL/'scripts/synovitis_weight_runner.py'),'/root/synovitis_weight_runner.py'))
app=modal.App('rsna-synovitis-weight-pair-seed2027')
ROOT=Path('/root/cache');PAIR=ROOT/'experiments/synovitis_weight_pair_seed2027'
@app.function(image=image,cpu=2,memory=4096,timeout=1800,volumes={'/root/cache':volume})
def prepare():
    import csv,numpy as np
    volume.reload();ids=set()
    for i in range(3):
        p=ROOT/f'shard_{i}/slice_cache';rows=[json.loads(x) for x in (p/'manifest.jsonl').read_text().splitlines()]
        si={r['study_id'] for r in rows};assert not ids&si;ids|=si
        for r in rows:
            for k in ('part','support_part'):
                assert (p/r[k]).is_file()
        a=np.load(p/rows[0]['part'],mmap_mode='r');assert a.shape[1:]==(224,224)
        print(f'224px volume shard{i}: {len(si)} studies',flush=True)
    labels=json.loads(Path('/root/assets/baseline_labels.json').read_text());assert ids=={r['study_id'] for r in labels['labels']} and len(ids)==4349
    folds=list(csv.DictReader(Path('/root/assets/folds.csv').open()));assert sum(r['fold']=='0' for r in folds)==825
    PAIR.mkdir(parents=True,exist_ok=True);shutil.copytree('/root/assets',PAIR/'assets',dirs_exist_ok=True);volume.commit()
    print('Matched pair CPU input checks passed; no downloads required.',flush=True)
@app.function(image=image,gpu='T4',cpu=4,memory=16384,timeout=43200,retries=3,volumes={'/root/cache':volume})
def train(arm):
    volume.reload();p=PAIR/arm;p.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONPATH']='/root/source:/root';env['CUDA_VISIBLE_DEVICES']='0'
    process=subprocess.Popen(['python','-u','/root/synovitis_weight_runner.py',arm],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (p/'training.log').open('a',buffering=1) as f:
        for line in process.stdout:print(line,end='',flush=True);f.write(line)
    code=process.wait();volume.commit()
    if code:raise RuntimeError(f'{arm} exited {code}; recovery retained')
    return arm
@app.function(image=image,cpu=1,memory=2048,timeout=3600,retries=2,secrets=[secret],volumes={'/root/cache':volume})
def publish(arm):
    volume.reload();os.environ['KAGGLE_API_TOKEN']=os.environ['KAGGLE_ACCESS_TOKEN'];k=Path.home()/'.kaggle';k.mkdir(exist_ok=True);(k/'access_token').write_text(os.environ['KAGGLE_ACCESS_TOKEN'])
    output=Path('/root/publish');output.mkdir(exist_ok=True)
    for name in ['checkpoint_best.pt','checkpoint_latest.pt','run_summary.json','training_history.json','training_history.csv','oof_predictions.json','learning_curve.png']:
        shutil.copy2(PAIR/arm/'run'/name,output/name)
    shutil.copy2(PAIR/'assets/build_report.json',output/'weight_experiment_build_report.json')
    slug=f'tumis001/rsna-synovitis-weight-{arm}-seed2027'
    (output/'dataset-metadata.json').write_text(json.dumps({'id':slug,'title':f'RSNA Synovitis Weight {arm.title()} Seed2027','licenses':[{'name':'other'}]}))
    subprocess.run(['kaggle','datasets','create','-p',str(output),'-r','zip'],check=True)
    return slug
@app.function(image=image,cpu=.25,memory=512,timeout=86400)
def coordinate():
    prepare.remote()
    for arm in ('baseline','candidate'):
        train.remote(arm)
        print('TRAINING_COMPLETE',arm,flush=True)
        try: print('PUBLISHED',publish.remote(arm),flush=True)
        except Exception as error: print(f'PUBLICATION_FAILED {arm}: {error}; checkpoints remain on volume',flush=True)
    print('MATCHED_PAIR_COMPLETE',flush=True)
@app.local_entrypoint()
def main():coordinate.remote()
