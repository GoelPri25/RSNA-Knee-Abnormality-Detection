"""CPU rescore of two pinned checkpoints on both fixed fold-0 label references."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'data/local_global/results'

def read(path):return json.loads(path.read_text())

def metrics(data,label_data):
    assert data['targets_in_order']==label_data['targets_in_order']
    labels={r['study_id']:r for r in label_data['labels']}
    ids=[r['study_id'] for r in data['predictions']]
    assert len(ids)==len(set(ids))==825
    truth=np.array([labels[s]['probabilities'] for s in ids],float)
    weight=np.array([labels[s]['weights'] for s in ids],float)
    score=np.array([r['probabilities'] for r in data['predictions']],float)
    keep=(weight>0)&((truth<=.3)|(truth>=.7))
    per={}
    for j,target in enumerate(data['targets_in_order']):
        y=(truth[keep[:,j],j]>=.7).astype(int)
        per[target]={'auc':float(roc_auc_score(y,score[keep[:,j],j])),
                     'labelled':len(y),'positives':int(y.sum())}
    p=np.clip(score,1e-7,1-1e-7)
    bce=(-truth*np.log(p)-(1-truth)*np.log1p(-p))*weight
    return {'macro_auc':float(np.mean([r['auc'] for r in per.values()])),
            'weighted_soft_bce':float(bce.sum()/weight.sum()),'per_target':per}

def main():
    paths={'baseline_0925':ROOT/'data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run',
           'local_global':OUTPUT}
    references={'Steven_v4':ROOT/'data/local_global/assets/validation_labels.json',
                'HPO_v31_D2':ROOT/'data/local_global/assets/labels.json'}
    report={'models':{},'input_sha256':{},'note':'Same 825 studies; each reference has its own confidence mask. Compare models within a reference, not AUC values across references.'}
    expected_ids=set()
    for name,folder in paths.items():
        p=folder/'oof_predictions.json';data=read(p);summary=read(folder/'run_summary.json')
        assert data['checkpoint_epoch']==summary['best']['epoch']
        ids={r['study_id'] for r in data['predictions']}
        if expected_ids:assert ids==expected_ids
        expected_ids=ids
        scores={ref:metrics(data,read(path)) for ref,path in references.items()}
        assert abs(scores['Steven_v4']['macro_auc']-summary['best']['macro_auc'])<1e-5
        report['models'][name]={'epoch':data['checkpoint_epoch'],**scores}
        report['input_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for path in references.values():report['input_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    report['deltas']={ref:report['models']['local_global'][ref]['macro_auc']-report['models']['baseline_0925'][ref]['macro_auc'] for ref in references}
    (OUTPUT/'dual_reference_validation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,r in report['models'].items():print(name,'epoch',r['epoch'],{ref:round(r[ref]['macro_auc'],6) for ref in references})
    print('Local+global minus baseline:',report['deltas'])
    print('Steven per-target deltas:')
    for target,r in report['models']['local_global']['Steven_v4']['per_target'].items():
        base=report['models']['baseline_0925']['Steven_v4']['per_target'][target]['auc']
        print(target,round(r['auc']-base,6))

if __name__=='__main__':main()
