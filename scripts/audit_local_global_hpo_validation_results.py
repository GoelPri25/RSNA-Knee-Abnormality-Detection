"""Rescore completed local+global validation-reference experiment on CPU."""
import hashlib
import json
from pathlib import Path
from audit_local_global_validation import metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/local_global_hpo_validation/results'

def main():
    folders={'baseline_0925':ROOT/'data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run',
             'local_global_steven_controls':ROOT/'data/local_global/results',
             'local_global_hpo_controls':OUT}
    references={'Steven_v4':ROOT/'data/local_global_hpo_validation/assets/steven_audit_labels.json',
                'HPO_v31_D2':ROOT/'data/local_global_hpo_validation/assets/validation_labels.json'}
    result={'models':{},'input_sha256':{},'note':'Same 825 studies; compare models within each reference. No image inference, GPU or gold.'}
    expected=None
    for name,folder in folders.items():
        p=folder/'oof_predictions.json';data=json.loads(p.read_text())
        summary=json.loads((folder/'run_summary.json').read_text())
        ids={r['study_id'] for r in data['predictions']}
        assert len(ids)==825 and data['checkpoint_epoch']==summary['best']['epoch']
        if expected is not None: assert ids==expected
        expected=ids
        scores={ref:metrics(data,json.loads(path.read_text())) for ref,path in references.items()}
        control='HPO_v31_D2' if name=='local_global_hpo_controls' else 'Steven_v4'
        assert abs(scores[control]['macro_auc']-summary['best']['macro_auc'])<1e-5
        result['models'][name]={'epoch':data['checkpoint_epoch'],**scores}
        result['input_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for path in references.values():result['input_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    current=result['models']['local_global_hpo_controls']
    result['deltas']={name:{ref:current[ref]['macro_auc']-result['models'][name][ref]['macro_auc'] for ref in references}
                      for name in ['baseline_0925','local_global_steven_controls']}
    summary=json.loads((OUT/'run_summary.json').read_text());history=summary['history']
    best_row=next(r for r in history if r['epoch']==summary['best']['epoch'])
    assert abs(best_row['steven_audit_macro_auc']-current['Steven_v4']['macro_auc'])<1e-5
    bce_best=min(history,key=lambda r:r['validation_bce'])
    steven_best=max(history,key=lambda r:r['steven_audit_macro_auc'])
    result['run']={'epochs':len(history),'seconds':summary['seconds'],'stopped_early':summary['stopped_early'],
                   'best_auc_epoch':summary['best']['epoch'],'best_auc_epoch_record':best_row,
                   'minimum_bce_epoch':bce_best['epoch'],'minimum_bce':bce_best['validation_bce'],
                   'maximum_steven_audit_epoch':steven_best['epoch'],'maximum_steven_audit_auc':steven_best['steven_audit_macro_auc']}
    (OUT/'dual_reference_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,model in result['models'].items():
        print(name,'epoch',model['epoch'],{ref:round(model[ref]['macro_auc'],6) for ref in references})
    print('DELTAS',json.dumps(result['deltas']))
    print('RUN',json.dumps(result['run']))
    print('Per target: new Steven AUC; change vs baseline .925; change vs old local+global')
    for target,r in current['Steven_v4']['per_target'].items():
        print(target,round(r['auc'],6),*[round(r['auc']-result['models'][name]['Steven_v4']['per_target'][target]['auc'],6)
                                      for name in ['baseline_0925','local_global_steven_controls']])

if __name__=='__main__':main()
