"""Single detached T4 experiment; cached inputs and recoverable state on Volume."""
from pathlib import Path
import json
import os
import subprocess
import shutil
import modal

ROOT=Path(__file__).resolve().parents[1]
APP_NAME='rsna-hpo-v31-d2-convnext-tiny-fold0'
DATASET='tumis001/rsna-hpo-v31-d2-convnext-tiny-fold0'
VOLUME_ROOT=Path('/root/rsna/persistent_cache')
RUN_ROOT=VOLUME_ROOT/'experiments/convnext_tiny_v31_d2_fold0'
volume=modal.Volume.from_name('rsna-cnn-topk-support-cache-v1',create_if_missing=False)
secret=modal.Secret.from_dict({'KAGGLE_ACCESS_TOKEN':(Path.home()/'.kaggle/access_token').read_text().strip()}) if modal.is_local() else modal.Secret.from_dict({})
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch==2.5.1','torchvision==0.20.1','numpy','scikit-learn','matplotlib','kaggle==2.0.2','modal')
       .add_local_dir(str(ROOT/'data/kaggle_reference/hpo_xla_actual/skanee'),'/root/rsna/source')
       .add_local_file(str(ROOT/'configs/hpo_v31_d2_resnet50_reference.json'),'/root/rsna/reference_config.json')
       .add_local_file(str(ROOT/'scripts/convnext_v31_runner.py'),'/root/rsna/convnext_v31_runner.py')
       .add_local_file(str(ROOT/'scripts/modal_hpo_v2_provenance_candidate.py'),'/root/rsna/modal_hpo_v2_provenance_candidate.py'))
app=modal.App(APP_NAME)

def kaggle_env():
    os.environ['KAGGLE_API_TOKEN']=os.environ['KAGGLE_ACCESS_TOKEN']
    path=Path.home()/'.kaggle';path.mkdir(exist_ok=True)
    (path/'access_token').write_text(os.environ['KAGGLE_ACCESS_TOKEN'])

@app.function(image=image,cpu=2,memory=4096,timeout=3600,secrets=[secret],volumes={str(VOLUME_ROOT):volume})
def prepare():
    kaggle_env();volume.reload()
    for i in range(3):
        cache=VOLUME_ROOT/f'shard_{i}/slice_cache'
        if not (cache/'manifest.jsonl').is_file():
            raise FileNotFoundError(f'Existing Modal volume shard missing: {cache}; refusing GPU allocation')
    inputs=RUN_ROOT/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    for dataset,file in [('tumis001/rsna-knee-hpo-v31-d2-report-labels','soft_labels_hpo_v3_1_d2_structured.json'),('stevenleehans/rsna-knee-llm-report-labels','llm_labels_v4_blend.csv')]:
        if not (inputs/file).is_file():
            subprocess.run(['kaggle','datasets','download','-d',dataset,'-f',file,'-p',str(inputs),'--unzip'],check=True)
    import sys, ast, math, torch
    sys.path.insert(0, '/root/rsna')
    from convnext_v31_runner import prepare_trainer
    raw = Path('/root/rsna/source/scripts/train_cnn_baseline.py').read_text()
    tree = ast.parse(prepare_trainer(raw))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SliceModel')
    pool = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'target_topk_mean')
    from convnext_v31_runner import MEMORY_ADAPTER
    ns = {'torch': torch, 'math': math}
    exec(MEMORY_ADAPTER, ns)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pool, cls], type_ignores=[])), 'model-check', 'exec'), ns)
    torch.set_num_threads(2)
    model = ns['SliceModel']('convnext_tiny', False, .4, 12, .1)
    model.eval()
    scores = model(torch.randn(2, 6, 224, 224), torch.tensor([0, 0]), 1)
    assert scores.shape == (1, 12) and torch.isfinite(scores).all()
    scores.sum().backward()
    assert model.backbone.features[0][0].weight.grad is not None
    assert not torch.count_nonzero(model.backbone.features[0][0].weight[:, 3:])
    print('CPU preflight passed: 224px six-channel forward/backward and target pooling.', flush=True)
    volume.commit()
    print('Existing cache shards reused; small label inputs saved to volume.',flush=True)

@app.function(image=image,gpu='T4',cpu=4,memory=16384,timeout=43200,retries=3,secrets=[secret],volumes={str(VOLUME_ROOT):volume})
def train():
    kaggle_env();volume.reload()
    env=os.environ.copy();env['PYTHONPATH']='/root/rsna/source:/root/rsna';env['CUDA_VISIBLE_DEVICES']='0'
    # Verify actual T4 batch capacity before allocating a full training loop.
    import sys, ast, math, torch
    sys.path.insert(0, '/root/rsna')
    from convnext_v31_runner import prepare_trainer, MEMORY_ADAPTER
    tree = ast.parse(prepare_trainer(Path('/root/rsna/source/scripts/train_cnn_baseline.py').read_text()))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SliceModel')
    pool = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'target_topk_mean')
    ns = {'torch':torch, 'math':math};exec(MEMORY_ADAPTER, ns)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pool, cls], type_ignores=[])), 'gpu-preflight', 'exec'), ns)
    model=ns['SliceModel']('convnext_tiny',False,.4,12,.1).cuda().train()
    with torch.autocast('cuda'):
        logits=model(torch.randn(192,6,224,224,device='cuda'),torch.arange(8,device='cuda').repeat_interleave(24),8)
        loss=logits.float().square().mean()
    loss.backward()
    print(f'GPU batch8 preflight passed: peak allocated MiB={torch.cuda.max_memory_allocated()/2**20:.1f}',flush=True)
    del model,logits,loss;torch.cuda.empty_cache()
    process=subprocess.Popen(['python','-u','/root/rsna/convnext_v31_runner.py'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    log=RUN_ROOT/'training.log'
    with log.open('a',buffering=1) as f:
        for line in process.stdout:
            print(line,end='',flush=True);f.write(line)
    code=process.wait();volume.commit()
    if code: raise RuntimeError(f'Training exit {code}; durable epochs preserved')
    destination = publish.remote()
    return {'status':'completed','artifacts':destination,'summary_exists':(RUN_ROOT/'run/run_summary.json').is_file()}

@app.function(image=image,cpu=1,memory=2048,timeout=1800,retries=2,secrets=[secret],volumes={str(VOLUME_ROOT):volume})
def publish():
    kaggle_env();volume.reload()
    upload=RUN_ROOT/'kaggle_upload';upload.mkdir(exist_ok=True)
    for p in (RUN_ROOT/'run').iterdir():
        if p.is_file() and p.name not in {'recovery.pt','recovery.tmp'}:shutil.copy2(p,upload/p.name)
    for p in [RUN_ROOT/'training.log',RUN_ROOT/'config.json']:
        if p.exists():shutil.copy2(p,upload/p.name)
    (upload/'dataset-metadata.json').write_text(json.dumps({'id':DATASET,'title':'RSNA HPO v31 D2 ConvNeXt Tiny Fold0','licenses':[{'name':'other'}]}))
    probe=subprocess.run(['kaggle','datasets','files',DATASET],capture_output=True)
    command=['kaggle','datasets','create','-p',str(upload)] if probe.returncode else ['kaggle','datasets','version','-p',str(upload),'-m','Completed ConvNeXt-Tiny label-controlled experiment']
    subprocess.run(command,check=True)
    return DATASET

@app.local_entrypoint()
def main():
    prepare.remote()
    result=train.remote()
    print(result,flush=True)

