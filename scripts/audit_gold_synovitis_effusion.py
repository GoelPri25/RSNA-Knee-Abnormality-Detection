"""Diagnostic (evaluation only, aggregate): does gold Synovitis track Effusion, and does each model's Synovitis head carry
synovitis-specific signal beyond effusion? Gold is scored, never used to build labels.
  podman run --rm --network=none -v "$P":/proj:ro,Z localhost/rsna-label-ds python -I /proj/repository/scripts/audit_gold_synovitis_effusion.py /proj
"""
import sys, json; from pathlib import Path; import numpy as np, pandas as pd
R=Path(sys.argv[1]); O=R/"repository/data/image_detail_audit_2026-10-07/gold_image_model"
P=json.loads((O/"gold_image_model_predictions.json").read_text()); T=P["targets_in_order"]; ids=P["gold_study_ids"]
tr=pd.read_csv(R/"reports_raw/train.csv"); Y=tr.set_index("StudyInstanceUID").loc[ids,T].astype(int)
def auc(s,y):
    y=np.asarray(y,bool); r=pd.Series(np.asarray(s,float)).rank().to_numpy(); n1=y.sum(); n0=len(y)-n1; return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
print("gold Synovitis x Effusion:\n", pd.crosstab(Y.Synovitis, Y.Effusion, rownames=["Syn"], colnames=["Eff"]))
for m in ("r50_hpo_v31_d2_fold0","r50_hpo_blend_fold0"):
    M=pd.DataFrame(P["models"][m]["gold_probabilities"],index=ids,columns=T)
    print(m, "gold Syn AUC by model Synovitis=%.3f  by model Effusion=%.3f  | within gold-Effusion+ (n=%d) Syn AUC by Synovitis score=%.3f"%(
      auc(M.Synovitis,Y.Synovitis), auc(M.Effusion,Y.Synovitis), Y.Effusion.sum(), auc(M.Synovitis[Y.Effusion==1],Y.Synovitis[Y.Effusion==1])))
# report-label synovitis prevalence in training labels for scale
L=json.loads((R/"label_rules/hpo_v3_1/build/soft_labels_hpo_v3_1_structured.json").read_text()); k=T.index("Synovitis"); e=T.index("Effusion")
v=np.array([x["probabilities"][k] for x in L["labels"]]); ef=np.array([x["probabilities"][e] for x in L["labels"]])
print("train label mean Synovitis %.3f (>=0.5: %.3f)  Effusion >=0.5: %.3f | gold prevalence Syn %.3f Eff %.3f"%(v.mean(),(v>=.5).mean(),(ef>=.5).mean(),Y.Synovitis.mean(),Y.Effusion.mean()))
