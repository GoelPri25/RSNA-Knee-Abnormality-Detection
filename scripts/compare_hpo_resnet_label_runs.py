"""Paired CPU analysis of pinned HPO-v1 / v3.1-D2 validation predictions."""
from pathlib import Path
import json,csv
import numpy as np
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/hpo_v1_vs_v31_resnet_analysis'
OUT.mkdir(parents=True,exist_ok=True)
old=json.loads((ROOT/'data/hpo_v1_resnet50_auc_margin_fold0/results/baseline_validation_predictions.json').read_text())
new=json.loads((ROOT/'data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json').read_text())
ref=json.loads((ROOT/'data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json').read_text())
ids=old['study_ids'];targets=new['targets_in_order'];assert len(ids)==825 and len(set(ids))==825
nmap={r['study_id']:r['probabilities'] for r in new['predictions']}
rmap={r['study_id']:r for r in ref['labels']};assert set(ids)==nmap.keys()==rmap.keys()
a=np.asarray(old['scores']);b=np.asarray([nmap[s] for s in ids])
y=np.asarray([[.5 if p is None else p for p in rmap[s]['probabilities']] for s in ids]);w=np.asarray([rmap[s]['weights'] for s in ids])
assert a.shape==b.shape==y.shape==(825,12)
assert np.isfinite(a).all() and np.isfinite(b).all()
low,high=.3,.7
m=(w>0)&((y<=low)|(y>=high));truth=y>=high
# Original exact metrics avoid rounding ambiguity of exported OOF probabilities.
oldmetrics=json.loads((ROOT/'data/hpo_v1_resnet50_auc_margin_fold0/results/baseline_metrics.json').read_text())
newmetrics=json.loads((ROOT/'data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/run_summary.json').read_text())['best']['metrics']
rows=[];cellrows=[]
for k,t in enumerate(targets):
 mask=m[:,k];yy=truth[mask,k];aa=a[mask,k];bb=b[mask,k]
 oldauc=oldmetrics['per_target'][t]['auc'];newauc=newmetrics['per_target'][t]['auc']
 assert abs(roc_auc_score(yy,aa)-oldauc)<1e-5
 assert abs(roc_auc_score(yy,bb)-newauc)<1e-5
 wronga=(aa>=.5)!=yy;wrongb=(bb>=.5)!=yy
 valid=w[:,k]>0
 def bce(p):
  p=np.clip(p,1e-7,1-1e-7);return float(np.sum(w[:,k]*(-y[:,k]*np.log(p)-(1-y[:,k])*np.log1p(-p)))/w[:,k].sum())
 rows.append(dict(target=t,old_auc=oldauc,new_auc=newauc,delta_auc=newauc-oldauc,
  old_soft_bce=bce(a[:,k]),new_soft_bce=bce(b[:,k]),confident_cells=int(mask.sum()),
  old_mean_positive=float(aa[yy].mean()),new_mean_positive=float(bb[yy].mean()),
  old_mean_negative=float(aa[~yy].mean()),new_mean_negative=float(bb[~yy].mean()),
  prediction_rank_correlation=float(spearmanr(aa,bb).statistic),
  old_threshold_errors=int(wronga.sum()),new_threshold_errors=int(wrongb.sum()),
  shared_threshold_errors=int((wronga&wrongb).sum()),corrected=int((wronga&~wrongb).sum()),introduced=int((~wronga&wrongb).sum()),
  old_confident_false_negatives=int((yy&(aa<.1)).sum()),new_confident_false_negatives=int((yy&(bb<.1)).sum()),
  old_confident_false_positives=int((~yy&(aa>.9)).sum()),new_confident_false_positives=int((~yy&(bb>.9)).sum())))
 for i in np.flatnonzero(mask):
  cellrows.append(dict(study_id=ids[i],target=t,reference_probability=float(y[i,k]),old_prediction=float(a[i,k]),new_prediction=float(b[i,k]),delta=float(b[i,k]-a[i,k]),shared_error=bool(((a[i,k]>=.5)!=truth[i,k]) and ((b[i,k]>=.5)!=truth[i,k]))))
# Paired study bootstrap retains missingness and target co-occurrence.
rng=np.random.default_rng(2026);boots=[]
for _ in range(1000):
 sample=rng.integers(0,len(ids),len(ids));ds=[]
 for k in range(12):
  ix=sample[m[sample,k]];yy=truth[ix,k]
  if yy.all() or not yy.any(): break
  ds.append(roc_auc_score(yy,b[ix,k])-roc_auc_score(yy,a[ix,k]))
 if len(ds)==12:boots.append(ds)
boots=np.asarray(boots)
for k,r in enumerate(rows):r['paired_auc_delta_ci95']=np.quantile(boots[:,k],[.025,.975]).tolist()
summary={'reference':'825 fixed Steven-v4 report validation; confident probabilities <=0.3 or >=0.7',
 'old_checkpoint_epoch':17,'new_checkpoint_epoch':15,'old_macro_auc':oldmetrics['macro_auc'],'new_macro_auc':newmetrics['macro_auc'],
 'macro_delta':newmetrics['macro_auc']-oldmetrics['macro_auc'],
 'paired_macro_delta_ci95':np.quantile(boots.mean(1),[.025,.975]).tolist(),
 'bootstrap_replicates':len(boots),'per_target':rows,
 'limitations':['Study bootstrap measures uncertainty on these saved predictions, not seed/training variation or label causality.','Threshold errors are against report labels, not confirmed image findings.','Gold inference not performed; local available gold cache is Fit160 and not a matched-input comparison.']}
# Connect prediction differences to actual training-label changes.
oldlabels=json.loads((ROOT/'data/label_datasets/hpo_v1_blend/soft_labels_hpo_v1_blend.json').read_text())
newlabels=json.loads((ROOT/'data/label_datasets/hpo_v3_1_d2/kaggle/soft_labels_hpo_v3_1_d2_structured.json').read_text())
oldmap={r['study_id']:r['probabilities'] for r in oldlabels['labels']}
newmap={r['study_id']:r['probabilities'] for r in newlabels['labels']}
with (ROOT/'data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/report_folds_hpo_add281_trainonly.csv').open() as stream:
 trainids=[r['study_id'] for r in csv.DictReader(stream) if int(r['fold'])!=0]
assert len(trainids)==3524
for k,row in enumerate(rows):
 delta=np.asarray([newmap[s][k]-oldmap[s][k] for s in trainids])
 row['training_label_changed_cells']=int((np.abs(delta)>1e-6).sum())
 row['training_label_mean_delta']=float(delta.mean())

(OUT/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'per_target.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (OUT/'validation_cell_changes.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(cellrows[0]));writer.writeheader();writer.writerows(sorted(cellrows,key=lambda r:abs(r['delta']),reverse=True))
print('MACRO',summary['macro_delta'],summary['paired_macro_delta_ci95'])
for r in sorted(rows,key=lambda r:r['delta_auc'],reverse=True):print(r['target'],round(r['old_auc'],5),round(r['new_auc'],5),round(r['delta_auc'],5),'BCE',round(r['old_soft_bce'],4),round(r['new_soft_bce'],4),'errors',r['old_threshold_errors'],r['new_threshold_errors'])
