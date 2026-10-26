"""Resolution-only ResNet50 reader, durable epochs, dual validation references."""
from pathlib import Path
import ast,csv,hashlib,importlib.util,json,sys
import torch
from efficientnet_v31_runner import prepare_trainer,prepare_steven_validation

ROOT=Path('/root/cache')
RUN=ROOT/'experiments/resnet50_v31_trainval_336/run'


def prepared_trainer(raw):
    text=prepare_trainer(raw)
    # Retain the original ResNet adapter and forward instead of the EfficientNet adapter.
    start=raw.index('class SliceModel(');end=raw.index('\ndef augment(',start)
    a=text.index('class SliceModel(');b=text.index('\ndef augment(',a)
    text=text[:a]+raw[start:end]+text[b:]
    # Only recompute ResNet stages during training. Disposable BatchNorm buffers
    # prevent recomputation from updating running statistics a second time.
    text=text.replace('self.head(self.backbone(slices))','self.head(resnet_features(self.backbone, slices))')
    helper='''
def resnet_features(net,x):
    x=net.maxpool(net.relu(net.bn1(net.conv1(x))))
    for block in [net.layer1,net.layer2,net.layer3,net.layer4]:
        if net.training and torch.is_grad_enabled():
            x=checkpoint(block,x,use_reentrant=False,preserve_rng_state=True,
                         context_fn=lambda block=block:(nullcontext(),isolated_bn_buffers(block)))
        else: x=block(x)
    return net.fc(torch.flatten(net.avgpool(x),1))
'''
    text=text.replace('class SliceModel(',helper+'\nclass SliceModel(',1)
    boundary='        metrics, scores = evaluate()';assert text.count(boundary)==1
    text=text.replace(boundary,boundary+'\n        metrics.update(AUDIT_STEVEN(scores,validation_ids,epoch))')
    text=text.replace('        history.append(record)','        record.update({k: metrics[k] for k in ["steven_validation_macro_auc","steven_validation_bce"]})\n        history.append(record)')
    ast.parse(text);return text


def load_trainer():
    RUN.mkdir(parents=True,exist_ok=True)
    raw=Path('/root/source/scripts/train_cnn_baseline.py').read_text()
    assert hashlib.sha256(raw.encode()).hexdigest()=='b8f8302319afbcf9970f0cf5b806d8fb78e72ef6e169b276274ddf7cd5e0ba85'
    p=RUN.parent/'trainer.py';p.write_text(prepared_trainer(raw))
    sys.path.insert(0,'/root/source')
    spec=importlib.util.spec_from_file_location('highres_trainer',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def run():
    import modal
    trainer=load_trainer();assets=ROOT/'experiment_assets';labels=assets/'soft_labels_hpo_v3_1_d2_structured.json';folds=assets/'report_folds_hpo_add281_trainonly.csv'
    assert hashlib.sha256(labels.read_bytes()).hexdigest()=='38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d'
    cfg=json.loads((assets/'training_config.json').read_text());assert cfg['cache']['image_size']==336 and cfg['model']['backbone']=='resnet50'
    steven=RUN.parent/'steven_reference.json'
    prepare_steven_validation(ROOT/'inputs/llm_labels_v4_blend.csv',folds,Path('/root/source/configs/gold_split_v1.csv'),steven)
    refs={r['study_id']:r for r in json.loads(steven.read_text())['labels']}
    def audit(scores,ids,epoch):
        np=trainer.np;y=np.array([[.5 if v is None else v for v in refs[s]['probabilities']] for s in ids],dtype=np.float32);w=np.array([refs[s]['weights'] for s in ids],dtype=np.float32)
        metrics=trainer.confident_auc(y,w,scores,cfg['confident_low'],cfg['confident_high']);p=np.clip(scores.astype(np.float64),1e-7,1-1e-7);bce=float(((-y*np.log(p)-(1-y)*np.log1p(-p))*w).sum()/w.sum())
        print(f"STEVEN_AUDIT epoch={epoch} AUC={metrics['macro_auc']:.6f} BCE={bce:.6f} (not used for controls)",flush=True)
        return {'steven_validation_macro_auc':metrics['macro_auc'],'steven_validation_bce':bce}
    trainer.AUDIT_STEVEN=audit;trainer.DURABLE_COMMIT=modal.Volume.from_name('rsna-cnn-140mm-336px-cache-v1').commit
    sys.argv=['trainer','--cache',*[str(ROOT/f'shard_{i}/slice_cache') for i in range(3)],'--soft-labels',str(labels),'--validation-soft-labels',str(labels),'--folds',str(folds),'--config',str(assets/'training_config.json'),'--output-dir',str(RUN),'--validation-fold','0','--train-slices-per-study','24']
    trainer.main()

if __name__=='__main__':run()
