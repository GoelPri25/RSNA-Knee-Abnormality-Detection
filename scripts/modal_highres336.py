"""Detached, single-T4 336px ResNet50 training with persistent recovery."""
from pathlib import Path
import json,os,shutil,subprocess
import modal
ROOT=Path(__file__).resolve().parents[1]
NAME='rsna-resnet50-v31-trainval-336px-fold0'
DATASET='tumis001/rsna-resnet50-v31-trainval-336px-fold0'
volume=modal.Volume.from_name('rsna-cnn-140mm-336px-cache-v1',create_if_missing=False)
secret=modal.Secret.from_dict({'KAGGLE_ACCESS_TOKEN':(Path.home()/'.kaggle/access_token').read_text().strip()}) if modal.is_local() else modal.Secret.from_dict({})
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1','torchvision==0.20.1','numpy','scikit-learn','matplotlib','kaggle==2.2.2','modal')
 .add_local_dir(str(ROOT/'data/kaggle_reference/hpo_xla_actual/skanee'),'/root/source')
 .add_local_file(str(ROOT/'scripts/highres336_runner.py'),'/root/highres336_runner.py')
 .add_local_file(str(ROOT/'scripts/efficientnet_v31_runner.py'),'/root/efficientnet_v31_runner.py'))
app=modal.App(NAME)
RUN=Path('/root/cache/experiments/resnet50_v31_trainval_336')

def credentials():
    os.environ['KAGGLE_API_TOKEN']=os.environ['KAGGLE_ACCESS_TOKEN'];p=Path.home()/'.kaggle';p.mkdir(exist_ok=True);(p/'access_token').write_text(os.environ['KAGGLE_ACCESS_TOKEN'])

@app.function(image=image,cpu=2,memory=4096,timeout=3600,secrets=[secret],volumes={'/root/cache':volume})
def prepare():
    import numpy as np,csv,sys,torch
    volume.reload();credentials();root=Path('/root/cache');assets=root/'experiment_assets'
    labels=json.loads((assets/'soft_labels_hpo_v3_1_d2_structured.json').read_text());label_ids={r['study_id'] for r in labels['labels']};seen=set()
    for shard in range(3):
        cache=root/f'shard_{shard}/slice_cache';summary=json.loads((cache/'cache_summary.json').read_text());assert summary['cache']['image_size']==336 and summary['cache']['field_of_view_mm']==140
        rows=[json.loads(line) for line in (cache/'manifest.jsonl').read_text().splitlines()];ids={r['study_id'] for r in rows};assert not seen&ids;seen|=ids
        for kind in ['slices','support']:
            for name in {r['part' if kind=='slices' else 'support_part'] for r in rows}:
                a=np.load(cache/name,mmap_mode='r');assert a.shape[1:]==(336,336) and a.dtype==np.uint8
        print(f'Cache verified shard={shard} studies={len(ids)}',flush=True)
    assert seen==label_ids and len(seen)==4349
    folds=list(csv.DictReader((assets/'report_folds_hpo_add281_trainonly.csv').open()));assert len(folds)==4349 and sum(r['fold']=='0' for r in folds)==825
    inputs=root/'inputs';inputs.mkdir(exist_ok=True)
    if not (inputs/'llm_labels_v4_blend.csv').exists():
        subprocess.run(['kaggle','datasets','download','-d','stevenleehans/rsna-knee-llm-report-labels','-f','llm_labels_v4_blend.csv','-p',str(inputs),'--unzip'],check=True)
    sys.path.insert(0,'/root');from highres336_runner import load_trainer
    m=load_trainer().SliceModel('resnet50',False,.4,12,.1).eval();torch.set_num_threads(2)
    y=m(torch.zeros(2,6,336,336),torch.zeros(2,dtype=torch.long),1);assert y.shape==(1,12) and torch.isfinite(y).all()
    volume.commit();print('CPU preflight passed; 4349 cache IDs match labels; 3524 train / 825 validation',flush=True)

@app.function(image=image,gpu='T4',cpu=4,memory=16384,timeout=43200,retries=3,secrets=[secret],volumes={'/root/cache':volume})
def train():
    import sys,torch
    volume.reload();credentials();sys.path.insert(0,'/root');from highres336_runner import load_trainer
    torch.set_num_threads(4);m=load_trainer().SliceModel('resnet50',False,.4,12,.1).cuda().train()
    with torch.autocast('cuda'):
        y=m(torch.zeros(192,6,336,336,device='cuda'),torch.arange(8,device='cuda').repeat_interleave(24),8);loss=y.float().square().mean()
    loss.backward();print(f'T4 full-batch8 336px preflight passed; allocated MiB={torch.cuda.max_memory_allocated()/2**20:.1f}',flush=True)
    del m,y,loss;torch.cuda.empty_cache()
    RUN.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env['PYTHONPATH']='/root/source:/root';env['CUDA_VISIBLE_DEVICES']='0'
    process=subprocess.Popen(['python','-u','/root/highres336_runner.py'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (RUN/'training.log').open('a',buffering=1) as log:
        for line in process.stdout:print(line,end='',flush=True);log.write(line)
    code=process.wait();volume.commit()
    if code:raise RuntimeError(f'Trainer exit={code}; committed recovery preserved')
    publish.remote();return {'status':'complete','dataset':DATASET}

@app.function(image=image,cpu=1,memory=2048,timeout=3600,retries=2,secrets=[secret],volumes={'/root/cache':volume})
def publish():
    volume.reload();credentials();out=RUN/'upload';out.mkdir(exist_ok=True)
    for p in (RUN/'run').iterdir():
        if p.is_file() and p.name not in ['recovery.pt','recovery.tmp']:shutil.copy2(p,out/p.name)
    shutil.copy2(RUN/'training.log',out/'training.log')
    (out/'dataset-metadata.json').write_text(json.dumps({'id':DATASET,'title':'MRI ResNet50 V31 336px Fold0 Training Artifacts','licenses':[{'name':'other'}]}))
    probe=subprocess.run(['kaggle','datasets','files',DATASET],capture_output=True)
    args=['kaggle','datasets','create','-p',str(out)] if probe.returncode else ['kaggle','datasets','version','-p',str(out),'-m','Completed training']
    subprocess.run(args,check=True)

@app.local_entrypoint()
def main():
    prepare.remote();print(train.remote(),flush=True)
