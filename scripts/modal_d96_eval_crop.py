"""Exp 1d (inference only): re-predict fold 0 + gold with the existing d96 SWA weights at eval crops 320 / 352 / 384.

The models train on the full 384 px (140 mm) field; the inherited evaluation takes the centre 320 crop (~117 mm).
320 reproduces the saved val_swa.json (checked locally); 352 and 384 are the new readouts. Scored locally by
scripts/audit_d96_eval_crop.py against the pre-registered rule in docs/experiment_ledger.md (Exp 1d).

  modal run --detach scripts/modal_d96_eval_crop.py
"""
from pathlib import Path
import json
import os
import sys

import modal

LOCAL = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("rsna-coatnet-d96-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "timm==1.0.29", "numpy")
         .add_local_dir(str(LOCAL / "data/coatnet_d96_fold0/assets"), "/root/assets")
         .add_local_file(str(LOCAL / "scripts/coatnet_d96_runner.py"), "/root/coatnet_d96_runner.py"))
app = modal.App("rsna-d96-eval-crop")
VOL = Path("/root/vol")
EXP = VOL / "experiments/coatnet_d96_fold0"
MODELS = {"coatnet_1b": ("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", EXP / "run_continue/weights_swa20_23.pt"),
          "r50_1c": ("resnet50.tv_in1k", EXP / "run_r50/weights_swa21_25.pt")}
RESOLUTIONS = (320, 352, 384)


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=2 * 3600, volumes={"/root/vol": volume})
def evaluate():
    import csv
    import torch
    os.environ["D96_CACHE"] = str(VOL / "d96")
    sys.path.insert(0, "/root")
    import coatnet_d96_runner as r
    volume.reload()
    out = EXP / "eval_crop"
    out.mkdir(exist_ok=True)
    ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    labels = r.load_labels(r.ASSETS / "soft_labels_hpo_v3_1_d2_structured.json", "labels")
    fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest()
    val_ids = sorted(s for s in labels if fold[s] == "0")
    gold_ids = sorted(s for s, m in meta.items() if m["gold"])
    assert len(val_ids) == 825 and len(gold_ids) == 58 and set(val_ids) == set(ref)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    summary = {}
    for name, (arch, weights) in MODELS.items():
        r.CFG["arch"] = arch
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        for res in RESOLUTIONS:
            r.CFG["eval_res"] = res
            model = r.Raptor(res, pretrained=False)
            model.load_state_dict(sd, strict=True)
            model = model.to(device).to(memory_format=torch.channels_last)
            for split, ids in (("val", val_ids), ("gold", gold_ids)):
                p = r.predict(model, ids, meta, device)
                r.save_predictions(out / f"{name}_{split}_{res}.json", ids, p, f"{name} swa, centre {res} crop, no TTA")
                if split == "val":
                    summary[f"{name}_{res}"] = r.macro_auc(p, val_ids, ref)[0]
            print(f"{name} {res}: val macro {summary[f'{name}_{res}']:.4f}", flush=True)
            volume.commit()
            del model; torch.cuda.empty_cache()
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    print("EVAL_CROP_DONE " + json.dumps(summary), flush=True)


@app.local_entrypoint()
def main():
    call = evaluate.spawn()
    print(f"SPAWNED {call.object_id}", flush=True)
