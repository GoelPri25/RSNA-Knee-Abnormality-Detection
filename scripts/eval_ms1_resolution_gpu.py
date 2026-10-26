"""Exp MS1-Q1: is input resolution MS1's thin-structure bottleneck? (inference only; Modal H100)

MS1 (MedSigLIP-448 fine-tuned, SWA 21-25) is re-scored on fold 0 (825) and gold (58) with the whole 384 px d96 crop
resized to R in {448, 560, 672} (32 / 40 / 48 patches a side; one patch covers 4.4 / 3.5 / 2.9 mm of the ~140 mm field).
The learned 32x32 position embeddings are resampled bicubically to the new grid (timm resample_abs_pos_embed); every
other weight is unchanged. No training. The cache is 384 px, so R > 384 adds no pixel information: it only cuts each
structure into more tokens. 448 must reproduce run_sig_lv2/val_swa.json. Scored on CPU by scripts/audit_ms1_q1.py.

Env: D96_CACHE, D96_PHASE=sig_lv2, MS1_WEIGHTS, Q1_OUT.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from timm.layers import resample_abs_pos_embed

import coatnet_d96_runner as r

assert os.environ["D96_PHASE"] == "sig_lv2" and r.CFG["arch"].startswith("vit")
OUT = Path(os.environ["Q1_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
RES = (448, 560, 672)


def load(res, sd):
    r.CFG["train_res"] = r.CFG["eval_res"] = res                   # Raptor builds the ViT at train_res
    m = r.build(res, pretrained=False)
    g = res // 14
    sd = dict(sd)
    sd["backbone.pos_embed"] = resample_abs_pos_embed(sd["backbone.pos_embed"].float(), new_size=(g, g),
                                                      num_prefix_tokens=0)
    m.load_state_dict(sd, strict=True)
    return m.to(torch.device("cuda")).eval()


def main():
    dev = torch.device("cuda")
    sd = torch.load(os.environ["MS1_WEIGHTS"], map_location="cpu", weights_only=True)
    assert tuple(sd["backbone.pos_embed"].shape) == (1, 1024, 1152), sd["backbone.pos_embed"].shape
    ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    meta = r.read_manifest()
    val_ids = sorted(ref)
    gold_ids = sorted(s for s, m in meta.items() if m["gold"])
    assert len(val_ids) == 825 and len(gold_ids) == 58
    log = {}
    for res in RES:
        t = time.time()
        m = load(res, sd)
        with torch.no_grad():
            pv = r.predict(m, val_ids, meta, dev, res=res)
            pg = r.predict(m, gold_ids, meta, dev, res=res)
        r.save_predictions(OUT / f"val_r{res}.json", val_ids, pv, f"MS1 SWA 21-25, whole crop -> {res}, pos-emb resampled")
        r.save_predictions(OUT / f"gold_r{res}.json", gold_ids, pg, f"MS1 SWA 21-25, whole crop -> {res}, pos-emb resampled")
        v12, per = r.macro_auc(pv, val_ids, ref)
        log[res] = {"val12": round(v12, 4), "per_target": per, "seconds": round(time.time() - t)}
        print("Q1", res, json.dumps(log[res]), flush=True)
        del m; torch.cuda.empty_cache()
    (OUT / "q1_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("Q1_DONE", flush=True)


if __name__ == "__main__":
    main()
