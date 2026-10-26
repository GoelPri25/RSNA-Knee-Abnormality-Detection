"""Original 224px ResNet50 with persistent epoch recovery and fixed validation."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys
from efficientnet_v31_runner import prepare_trainer
ROOT=Path('/root/cache');ASSETS=Path('/root/assets')
def prepare_original(raw):
    text=prepare_trainer(raw)
    a=text.index('class SliceModel(');b=text.index('\ndef augment(',a)
    c=raw.index('class SliceModel(');d=raw.index('\ndef augment(',c)
    text=text[:a]+raw[c:d]+text[b:]
    boundary='            batches += 1'
    assert text.count(boundary)==1
    text=text.replace(boundary,boundary+'\n            if batches % 50 == 0 or batches == batch_count:\n                print(f"PROGRESS epoch={epoch} batch={batches}/{batch_count}", flush=True)')
    ast.parse(text);return text

def run(arm):
    import modal
    assert arm == 'candidate'
    out=ROOT/f'experiments/synovitis_weight_candidate_seed2026/{arm}/run';out.mkdir(parents=True,exist_ok=True)
    raw=Path('/root/source/scripts/train_cnn_baseline.py').read_text()
    assert hashlib.sha256(raw.encode()).hexdigest()=='b8f8302319afbcf9970f0cf5b806d8fb78e72ef6e169b276274ddf7cd5e0ba85'
    audit=json.loads((ASSETS/'build_report.json').read_text())
    for name,h in audit['files'].items():assert hashlib.sha256((ASSETS/name).read_bytes()).hexdigest()==h,name
    p=out.parent/'trainer.py';p.write_text(prepare_original(raw));sys.path.insert(0,'/root/source')
    spec=importlib.util.spec_from_file_location('syn_weight_trainer',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    m.DURABLE_COMMIT=modal.Volume.from_name('rsna-cnn-topk-support-cache-v1').commit
    sys.argv=['trainer','--cache',*[str(ROOT/f'shard_{i}/slice_cache') for i in range(3)],'--soft-labels',str(ASSETS/f'{arm}_labels.json'),'--validation-soft-labels',str(ASSETS/'validation_labels.json'),'--folds',str(ASSETS/'folds.csv'),'--config',str(ASSETS/'config.json'),'--output-dir',str(out),'--validation-fold','0','--train-slices-per-study','24']
    print(f'ARM={arm}; seed2026; 224px; fixed Steven validation; durable epoch recovery',flush=True)
    m.main()
if __name__=='__main__':run(sys.argv[1])
