from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


cells = [
    md("""# Matched Confidence-V2 vs Flight hybrid-v4 label A/B

**CPU-only, no image re-encoding, no GPU.** The same frozen MedSigLIP slice embeddings, hierarchical MIL architecture, training code, seed, schedule, and 30 epochs are used in both arms. Only the report-derived targets and their supervision weights differ.

All 58 gold studies are removed from report-label training. This notebook reads target values only for the 29 development IDs in the saved split and constructs no examples for the other 29. It does not open or score the prior evaluation extraction JSONL. Each arm's checkpoint and training history are saved after every epoch so a mid-run interruption leaves the most recent local progress in Kaggle working storage.

Primary comparison: development macro ROC AUC (12-target mean), paired across identical studies. Other thresholded metrics are diagnostics only; no threshold is tuned.
"""),
    code(r'''from pathlib import Path
import csv, json, math, sys, time, copy
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/label_ab")
WORK.mkdir(parents=True, exist_ok=True)
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SEED = 2026
print("Device: CPU-only | CUDA available but intentionally unused:", torch.cuda.is_available())

def iter_small_input_roots():
    return [p for p in INPUT.iterdir() if p.name != "competitions"]

def find_file(name, within=None):
    roots = [Path(within)] if within and Path(within).exists() else iter_small_input_roots()
    matches = []
    for root in roots:
        matches.extend(p for p in root.rglob(name) if p.is_file())
    if len(matches) != 1: raise FileNotFoundError(f"Expected one {name}, found {matches}")
    return matches[0]

def find_source_root():
    hits = [p.parent.parent for root in iter_small_input_roots() for p in root.rglob("scripts/train_mil.py") if p.is_file()]
    if len(hits) != 1: raise FileNotFoundError(f"Expected one baseline source tree, found {hits}")
    return hits[0]

def list_feature_dirs():
    manifests = []
    for root in iter_small_input_roots(): manifests.extend(p for p in root.rglob("manifest.jsonl") if p.is_file())
    records = []
    for manifest in manifests:
        summary_path = manifest.parent / "feature_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            if summary.get("subset") in {"weak", "gold"}: records.append((summary, manifest.parent))
    if len(records) != 4: raise RuntimeError(f"Expected 3 weak shards + 1 gold cache, found {[(s, str(p)) for s,p in records]}")
    weak = [p for s,p in records if s["subset"] == "weak"]
    gold = [p for s,p in records if s["subset"] == "gold"]
    weak_shards = {tuple(s["shard"]) for s,p in records if s["subset"] == "weak"}
    if weak_shards != {(0,3),(1,3),(2,3)} or len(gold) != 1: raise RuntimeError(f"Unexpected feature manifests: {weak_shards}, gold={len(gold)}")
    return weak + gold, [s for s,p in records]

source = find_source_root()
sys.path.insert(0, str(source))
from src.reports.schema import TARGETS as SOURCE_TARGETS
from src.mil.data import load_cache, build_examples
from src.mil.train import build_model, predict_compact, auc_report, epoch_examples, weighted_bce
from src.mil.data import collate
assert list(SOURCE_TARGETS) == TARGETS

split_path = find_file("report_gold_split.csv", "/kaggle/input/kernels/tumis001/report-label-audit")
split = pd.read_csv(split_path, dtype={"StudyInstanceUID": str})
dev_ids = set(split.loc[split["split"].eq("development"), "StudyInstanceUID"].astype(str))
assert len(dev_ids) == 29

# Parse train.csv line by line, and materialize official labels only for development IDs.
competition_csv = INPUT / "competitions" / "rsna-knee-abnormality-detection" / "train.csv"
dev_gold = {}
with competition_csv.open(newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        uid = str(row["StudyInstanceUID"])
        if uid in dev_ids:
            dev_gold[uid] = np.asarray([int(row[t]) for t in TARGETS], dtype=np.float32)
assert set(dev_gold) == dev_ids and len(dev_gold) == 29
print("Development gold loaded:", len(dev_gold), "| evaluation gold labels loaded: 0")

feature_dirs, feature_summaries = list_feature_dirs()
if len({(s["model_revision"], s["config_sha256"], s["feature_dim"]) for s in feature_summaries}) != 1:
    raise RuntimeError("Feature shards disagree on encoder/preprocessing")
config = json.loads((source / "configs/mil_v1.json").read_text())
cache, cache_summary = load_cache(feature_dirs, config["exclude_order_sources"])
print("Feature cache:", cache_summary)
print("Baseline:", config["version"], "| architecture: hierarchical MIL | epochs:", config["epochs"], "| batch size:", config["batch_size"], "| lr:", config["lr"])
assert config["seed"] == SEED and config["epochs"] == 30
'''),
    code(r'''def make_internal_confidence_v2():
    structured = pd.read_csv(find_file("report_structured_labels.csv", "/kaggle/input/kernels/tumis001/report-label-extraction"), dtype={"StudyInstanceUID": str})
    gold_ids = set(split["StudyInstanceUID"].astype(str))  # excludes all 58 gold studies, including evaluation
    present = {"severe": (.97,1.), "moderate_to_severe": (.95,.95), "moderate": (.90,.90),
               "mild_to_moderate": (.82,.75), "mild": (.70,.55), "unspecified": (.88,.75)}
    labels=[]
    for row in structured.itertuples(index=False):
        uid = str(row.StudyInstanceUID)
        if uid in gold_ids or str(row.extraction_success).lower() != "true": continue
        probs, weights = [], []
        for target in TARGETS:
            slug = target.lower().replace("'", "").replace(" ", "_")
            status = str(getattr(row, f"{slug}__status")).lower()
            severity = str(getattr(row, f"{slug}__severity")).lower().replace(" ", "_")
            if status == "absent": p,w=.05,1.
            elif status == "present": p,w=present.get(severity,present["unspecified"])
            else: p,w=.5,0.
            probs.append(float(p)); weights.append(float(w))
        if any(w > 0 for w in weights): labels.append({"study_id":uid,"probabilities":probs,"weights":weights})
    print("Confidence-V2 training labels:", len(labels))
    return {"version":"confidence_aware_v2","targets_in_order":TARGETS,"labels":labels}

def make_flight_hybrid_v4():
    table = pd.read_csv(find_file("report_labels_v4hybrid.csv"), dtype={"StudyInstanceUID": str}, low_memory=False)
    gold_ids = set(split["StudyInstanceUID"].astype(str))
    labels=[]
    for _, row in table.iterrows():
        uid=str(row["StudyInstanceUID"])
        if uid in gold_ids: continue
        probs=[]; weights=[]
        for target in TARGETS:
            value=pd.to_numeric(pd.Series([row[target]]), errors="coerce").iloc[0]
            if pd.isna(value) or not 0 <= float(value) <= 1: probs.append(.5); weights.append(0.)
            else: probs.append(float(value)); weights.append(1.)
        if any(w>0 for w in weights): labels.append({"study_id":uid,"probabilities":probs,"weights":weights})
    print("Flight hybrid-v4 training labels:", len(labels))
    return {"version":"flight_hybrid_v4","targets_in_order":TARGETS,"labels":labels}

label_sets = {"A_confidence_v2": make_internal_confidence_v2(), "B_flight_hybrid_v4": make_flight_hybrid_v4()}
all_gold_ids = set(split["StudyInstanceUID"].astype(str))
for arm, payload in label_sets.items():
    if any(x["study_id"] in all_gold_ids for x in payload["labels"]): raise RuntimeError(f"Gold label leakage in {arm}")
'''),
    code(r'''def save_history(arm, history):
    pd.DataFrame(history).to_csv(WORK / arm / "training_history.csv", index=False)
    (WORK / arm / "training_history.json").write_text(json.dumps(history, indent=2)+"\n")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot([r["epoch"]+1 for r in history], [r["train_bce"] for r in history], marker=".")
    ax.set(xlabel="Epoch", ylabel="Training BCE", title=f"{arm}: live training curve")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(WORK / arm / "learning_curve.png", dpi=130); plt.close(fig)

def fit_logged_checkpoint(model, train_examples, cfg, arm):
    """Same fixed-budget original trainer with per-epoch checkpoint/history writes."""
    import torch.nn.functional as F
    out = WORK / arm; out.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device=torch.device("cpu"); model.to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    batch=int(cfg["batch_size"])
    steps=math.ceil(len(epoch_examples(train_examples,cfg,np.random.default_rng(0)))/batch)
    total=steps*int(cfg["epochs"]); warmup=max(1,int(total*cfg["warmup_fraction"]))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/warmup)*.5*(1+math.cos(math.pi*min(1.,step/max(1,total)))))
    history=[]; started=time.monotonic()
    for epoch in range(int(cfg["epochs"])):
        epoch_start=time.monotonic(); examples=epoch_examples(train_examples,cfg,rng); loss_sum=0.; batches=0
        for start in range(0,len(examples),batch):
            chunk=examples[start:start+batch]
            b={k:v.to(device) for k,v in collate(chunk,cfg["max_slices_train"],rng).items()}
            outp=model(b["embeddings"],b["positions"],b["plane"],b["fluid"],b["slice_mask"],b["series_mask"])
            loss=weighted_bce(outp["logits"],b["target"],b["weight"])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"])
            optimizer.step(); scheduler.step(); loss_sum+=float(loss.detach()); batches+=1
        record={"epoch":epoch,"train_bce":loss_sum/max(batches,1),"lr":scheduler.get_last_lr()[0],"seconds":round(time.monotonic()-epoch_start,2)}
        if not math.isfinite(record["train_bce"]): raise FloatingPointError(f"Non-finite loss at {arm} epoch {epoch}")
        history.append(record)
        torch.save({"state_dict":model.state_dict(),"config":cfg,"feature_dim":cache_summary["feature_dim"],"epoch":epoch},out/"latest_checkpoint.pt")
        save_history(arm,history)
        print(f"{arm} epoch {epoch+1:02d}/{cfg['epochs']}: BCE={record['train_bce']:.4f} lr={record['lr']:.2e} ({record['seconds']:.1f}s)",flush=True)
    torch.save({"state_dict":model.state_dict(),"config":cfg,"feature_dim":cache_summary["feature_dim"],"epoch":cfg["epochs"]-1},out/"checkpoint.pt")
    print(f"{arm} complete in {(time.monotonic()-started)/60:.1f} min; checkpoints/history saved",flush=True)
    return model,history

def run_arm(arm,payload):
    train_examples,dev_examples,provenance=build_examples(dev_gold,cache,payload)
    if {e.study_id for e in dev_examples} != dev_ids: raise RuntimeError("Holdout is not exactly the development IDs")
    print(arm,"data provenance:",provenance,flush=True)
    model=build_model(config,cache_summary["feature_dim"])
    model,history=fit_logged_checkpoint(model,train_examples,config,arm)
    ids,probabilities,_=predict_compact(model,dev_examples,config,torch.device("cpu"))
    predictions=[{"study_id":uid,"probabilities":prob.tolist()} for uid,prob in zip(ids,probabilities)]
    metrics=auc_report(dev_gold,predictions)
    frame=pd.DataFrame([{"study_id":uid,**dict(zip(TARGETS,prob))} for uid,prob in zip(ids,probabilities)])
    frame.to_csv(WORK/arm/"development_predictions.csv",index=False)
    target_metrics=[]
    y=np.stack([dev_gold[uid] for uid in ids]); p=np.asarray(probabilities)
    for j,target in enumerate(TARGETS):
        pred=(p[:,j]>=.5).astype(int)
        target_metrics.append({"target":target,"auc":metrics["per_target"][target]["auc"],"accuracy_at_0_5":accuracy_score(y[:,j],pred),
                              "precision_at_0_5":precision_score(y[:,j],pred,zero_division=0),"recall_at_0_5":recall_score(y[:,j],pred,zero_division=0),
                              "f1_at_0_5":f1_score(y[:,j],pred,zero_division=0),"balanced_accuracy_at_0_5":balanced_accuracy_score(y[:,j],pred)})
    pd.DataFrame(target_metrics).to_csv(WORK/arm/"development_metrics_per_target.csv",index=False)
    return {"arm":arm,"dev_macro_auc":metrics["macro_auc"],"dev_studies":metrics["studies"],"per_target":metrics["per_target"],"history":history},dict(zip(ids,p))

results={}; predictions={}
for arm in ["A_confidence_v2","B_flight_hybrid_v4"]:
    results[arm],predictions[arm]=run_arm(arm,label_sets[arm])
    (WORK/arm/"run_summary.json").write_text(json.dumps(results[arm],indent=2)+"\n")
'''),
    code(r'''def macro_auc(y,p):
    vals=[]
    for j in range(len(TARGETS)):
        if 0 < y[:,j].sum() < len(y): vals.append(roc_auc_score(y[:,j],p[:,j]))
    return float(np.mean(vals))

dev_order=sorted(dev_ids)
y=np.stack([dev_gold[u] for u in dev_order])
p_a=np.stack([predictions["A_confidence_v2"][u] for u in dev_order])
p_b=np.stack([predictions["B_flight_hybrid_v4"][u] for u in dev_order])
delta=macro_auc(y,p_b)-macro_auc(y,p_a)
rng=np.random.default_rng(17); boot=[]
for _ in range(3000):
    ix=rng.integers(0,len(y),size=len(y)); ys=y[ix]
    if any(ys[:,j].sum() in (0,len(ys)) for j in range(ys.shape[1])): continue
    boot.append(macro_auc(ys,p_b[ix])-macro_auc(ys,p_a[ix]))
ci=np.quantile(boot,[.05,.95]).tolist() if boot else [None,None]
summary={"primary_metric":"development macro ROC AUC","development_n":len(dev_order),"evaluation_n_scored":0,
         "A_confidence_v2_macro_auc":macro_auc(y,p_a),"B_flight_hybrid_v4_macro_auc":macro_auc(y,p_b),
         "B_minus_A_macro_auc":delta,"paired_study_bootstrap_90pct_ci":ci,"valid_bootstrap_replicates":len(boot),
         "training_seed":SEED,"epochs_each":int(config["epochs"]),"gpu_used":False,
         "caveat":"Only 29 development studies; interval is descriptive and noisy, not a definitive model-selection guarantee."}
(WORK/"ab_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
per_target=[]
for j,t in enumerate(TARGETS):
    per_target.append({"target":t,"A_confidence_v2_auc":roc_auc_score(y[:,j],p_a[:,j]),"B_flight_hybrid_v4_auc":roc_auc_score(y[:,j],p_b[:,j]),
                       "B_minus_A_auc":roc_auc_score(y[:,j],p_b[:,j])-roc_auc_score(y[:,j],p_a[:,j])})
pd.DataFrame(per_target).to_csv(WORK/"ab_per_target_auc.csv",index=False)
print(json.dumps(summary,indent=2))
print("Per-target development AUC:"); display(pd.DataFrame(per_target).round(3))
'''),
]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12.13"},
   "kaggle":{"accelerator":"none","dataSources":[],"dockerImageVersionId":28755,"isInternetEnabled":False,"language":"python","sourceType":"notebook","isGpuEnabled":False}},"nbformat":4,"nbformat_minor":5}
(OUT/"report_label_ab.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1))
meta={"id":"tumis001/rsna-report-label-ab","title":"RSNA Report Label A-B on MedSigLIP Baseline","code_file":"report_label_ab.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":False,"enable_tpu":False,"enable_internet":False,"keywords":[],
      "dataset_sources":["tumis001/rsna-experiment1-baseline-source","flight0234/rsna-knee-hybrid-report-labels"],
      "kernel_sources":["tumis001/rsna-knee-medsiglip-features-shard-0","tumis001/rsna-knee-medsiglip-features-shard-1","tumis001/rsna-knee-medsiglip-features-shard-2","tumis001/rsna-knee-medsiglip-gold-features","tumis001/report-label-extraction","tumis001/report-label-audit"],
      "competition_sources":["rsna-knee-abnormality-detection"],"model_sources":[]}
(OUT/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
print("Wrote",OUT/"report_label_ab.ipynb")
