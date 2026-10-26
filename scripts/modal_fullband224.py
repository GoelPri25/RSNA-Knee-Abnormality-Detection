"""Detached T4 run: Phase 1 coverage. r50 v3.1-D2 224px with the original 0.925 recipe on the full-band cache
(central_band [0,1], 20 slices per series). Copy of modal_r50_cosine_ema.py; only names, cache paths, runner and
published files differ."""
from pathlib import Path
import json,os,subprocess,shutil
import modal
LOCAL=Path(__file__).resolve().parents[1]
volume=modal.Volume.from_name('rsna-cnn-topk-support-cache-v1',create_if_missing=False)
secret=modal.Secret.from_dict({'KAGGLE_ACCESS_TOKEN':(Path.home()/'.kaggle/access_token').read_text().strip()}) if modal.is_local() else modal.Secret.from_dict({})
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1','torchvision==0.20.1','numpy','scikit-learn','matplotlib','kaggle==2.2.2','modal')
 .add_local_dir(str(LOCAL/'data/kaggle_reference/hpo_xla_actual/skanee'),'/root/source')
 .add_local_dir(str(LOCAL/'data/fullband224_seed2026/assets'),'/root/assets')
 .add_local_file(str(LOCAL/'scripts/efficientnet_v31_runner.py'),'/root/efficientnet_v31_runner.py')
 .add_local_file(str(LOCAL/'scripts/synovitis_weight_candidate_runner.py'),'/root/synovitis_weight_candidate_runner.py')
 .add_local_file(str(LOCAL/'scripts/fullband224_runner.py'),'/root/fullband224_runner.py'))
app=modal.App('rsna-r50-fullband224-seed2026')
ROOT=Path('/root/cache');EXP=ROOT/'experiments/fullband224_seed2026'
@app.function(image=image,cpu=2,memory=4096,timeout=1800,volumes={'/root/cache':volume})
def prepare():
    import csv,numpy as np
    volume.reload();ids=set()
    for i in range(3):
        p=ROOT/f'fullband224/shard_{i}/slice_cache';rows=[json.loads(x) for x in (p/'manifest.jsonl').read_text().splitlines()]
        si={r['study_id'] for r in rows};assert not ids&si;ids|=si
        for r in rows:
            for k in ('part','support_part'):
                assert (p/r[k]).is_file()
        a=np.load(p/rows[0]['part'],mmap_mode='r');assert a.shape[1:]==(224,224)
        n=sum(r['count'] for r in rows);print(f'full-band 224px shard{i}: {len(si)} studies, {n} slices',flush=True)
    labels=json.loads(Path('/root/assets/train_labels.json').read_text());assert ids=={r['study_id'] for r in labels['labels']} and len(ids)==4349
    folds=list(csv.DictReader(Path('/root/assets/folds.csv').open()));assert sum(r['fold']=='0' for r in folds)==825
    EXP.mkdir(parents=True,exist_ok=True);shutil.copytree('/root/assets',EXP/'assets',dirs_exist_ok=True);volume.commit()
    recovery_path=EXP/'run/recovery.pt'
    if recovery_path.is_file():
        import torch
        recovery=torch.load(recovery_path,map_location='cpu',weights_only=False)
        assert recovery['config']==json.loads(Path('/root/assets/config.json').read_text())
        print('RECOVERY_VERIFIED '+json.dumps({'saved_epoch':recovery['epoch'],'resume_epoch':recovery['epoch']+1,
            'best':recovery['best'],'optimizer_steps':recovery['global_step']}),flush=True)
    print('CPU input checks passed; no downloads required.',flush=True)
@app.function(image=image,gpu='T4',cpu=4,memory=16384,timeout=43200,retries=3,volumes={'/root/cache':volume})
def train():
    volume.reload();EXP.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONPATH']='/root/source:/root';env['CUDA_VISIBLE_DEVICES']='0'
    process=subprocess.Popen(['python','-u','/root/fullband224_runner.py'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (EXP/'training.log').open('a',buffering=1) as f:
        for line in process.stdout:print(line,end='',flush=True);f.write(line)
    code=process.wait();volume.commit()
    if code:raise RuntimeError(f'trainer exited {code}; recovery retained')
@app.function(image=image,cpu=1,memory=2048,timeout=3600,retries=2,secrets=[secret],volumes={'/root/cache':volume})
def publish():
    volume.reload();os.environ['KAGGLE_API_TOKEN']=os.environ['KAGGLE_ACCESS_TOKEN'];k=Path.home()/'.kaggle';k.mkdir(exist_ok=True);(k/'access_token').write_text(os.environ['KAGGLE_ACCESS_TOKEN'])
    output=Path('/root/publish');output.mkdir(exist_ok=True)
    for name in ['checkpoint_best.pt','checkpoint_latest.pt','run_summary.json','training_history.json','training_history.csv',
                 'oof_predictions.json','learning_curve.png']:
        shutil.copy2(EXP/'run'/name,output/name)
    shutil.copy2(EXP/'assets/config.json',output/'config.json')
    slug='tumis001/rsna-r50-v31-d2-fullband224-seed2026'
    (output/'dataset-metadata.json').write_text(json.dumps({'id':slug,'title':'RSNA R50 v31 D2 Fullband224 Seed2026','licenses':[{'name':'other'}]}))
    subprocess.run(['kaggle','datasets','create','-p',str(output),'-r','zip'],check=True)
    return slug
@app.function(image=image,cpu=.25,memory=512,timeout=86400,max_containers=1)
def coordinate():
    prepare.remote()
    train.remote()
    print('TRAINING_COMPLETE',flush=True)
    try: print('PUBLISHED',publish.remote(),flush=True)
    except Exception as error: print(f'PUBLICATION_FAILED: {error}; checkpoints remain on volume',flush=True)
    print('RUN_COMPLETE',flush=True)
@app.local_entrypoint()
def main():
    job = modal.Function.from_name(app.name, 'coordinate').spawn()
    launch = {'app_name': app.name, 'function_call_id': job.object_id,
              'mode': 'deployed background job; no local result waiter'}
    destination = LOCAL/'data/fullband224_seed2026/latest_launch.json'
    destination.write_text(json.dumps(launch, indent=2)+'\n')
    print('BACKGROUND_JOB_SUBMITTED '+json.dumps(launch), flush=True)
