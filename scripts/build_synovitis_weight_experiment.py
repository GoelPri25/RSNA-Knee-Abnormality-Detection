"""Training-only Synovitis weight ablation; never modifies target probabilities."""
from pathlib import Path
import csv, hashlib, json, re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/synovitis_weight_experiment/assets'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base=ROOT/'data/label_datasets/hpo_v3_1_d2/kaggle/soft_labels_hpo_v3_1_d2_structured.json'
    assert sha(base)=='38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d'
    fold=ROOT/'data/highres336/experiment_assets/report_folds_hpo_add281_trainonly.csv'
    fm={r['study_id']:int(r['fold']) for r in csv.DictReader(fold.open())}
    reports={r['StudyInstanceUID']:r.get('Report','') for r in csv.DictReader((ROOT.parent/'reports_raw/train.csv').open())}
    states={r['StudyInstanceUID']:r['Synovitis'] for r in csv.DictReader((ROOT/'data/label_datasets/hpo_v3_1_d2/rules/s5_fx_site/rule_states.csv').open())}
    # Conservative preservation: keep every multilingual synovial-root mention,
    # including uncertain scopes; also preserve any non-silent rule evidence.
    term=re.compile(r's[iy]nov|синов|συνοβ|υμεν',re.I)
    d=json.loads(base.read_text()); candidate=json.loads(base.read_text());idx=d['targets_in_order'].index('Synovitis');changed=[];protected=[]
    for a,b in zip(d['labels'],candidate['labels']):
        sid=a['study_id'];assert sid in fm and sid in reports and sid in states
        if fm[sid]==0:continue
        direct=bool(term.search(reports[sid])) or states[sid] not in ('','SIL')
        if direct:protected.append(sid)
        else:
            b['weights'][idx]=a['weights'][idx]*.25;changed.append(sid)
        assert a['probabilities']==b['probabilities']
        assert all(a['weights'][j]==b['weights'][j] for j in range(12) if j!=idx)
    assert len(protected)+len(changed)==3524
    assert sum(f==0 for f in fm.values())==825
    (OUT/'baseline_labels.json').write_bytes(base.read_bytes())
    (OUT/'candidate_labels.json').write_text(json.dumps(candidate,allow_nan=False)+'\n')
    (OUT/'folds.csv').write_bytes(fold.read_bytes())
    ref=ROOT/'data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json'
    (OUT/'validation_labels.json').write_bytes(ref.read_bytes())
    cfg=json.loads((ROOT/'configs/hpo_v31_d2_resnet50_reference.json').read_text());cfg['seed']=2027
    cfg['version']='synovitis_weight_pair_seed2027';cfg['note']='Matched 224px ImageNet ResNet50 pair; fixed Steven-v4 validation; candidate reduces inferred Synovitis training weights only.'
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    report={'seed':2027,'training_studies':3524,'validation_studies':825,'protected_synovial_evidence':len(protected),'downweighted_training_cells':len(changed),'factor':.25,'changed_study_ids':changed,'probabilities_unchanged':True,'other_targets_unchanged':True,'validation_unchanged':True,'evidence_policy':'Preserve all multilingual synovial-root mentions or non-SIL rule evidence; conservative, not a clinical truth classifier.','loss_normalization':'Existing sum(weighted BCE)/sum(weights); lower Synovitis weights slightly increase normalized relative influence of other targets; no loss-code change.'}
    report['files']={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='build_report.json'}
    (OUT/'build_report.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ('changed_study_ids','files')})
if __name__=='__main__':main()
