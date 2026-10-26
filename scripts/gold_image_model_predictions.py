"""Score saved image-model checkpoints on the 58 host gold studies (inference only; no training, no selection).

Every checkpoint here was trained on report folds only: the gold studies are absent from the training cache and folds
(checked: 0 of 58 in either). Gold-tuned runs (cnn_topk_gold_dpo, cnn_topk_gold_finetune) are deliberately excluded.
The model and batch code are imported unchanged from the hpo_v31_d2 run's own training script; inference uses every
cached window per study (the non-XLA evaluation path used in training-time validation and in its built-in gold hook).

Fidelity check first: for each checkpoint with saved fold-0 out-of-fold predictions, 32 fold-0 studies are re-scored
from the training cache and compared with the saved values. Gold labels are NOT read here; this script only writes
predictions. Scoring happens in audit_gold_image_model.py.

Container (GPU via CDI, no network, read-only project):
  podman run --rm --network=none --device nvidia.com/gpu=all -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z \
    docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
    python -I /proj/repository/scripts/gold_image_model_predictions.py /proj /out
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(R / "cnn_topk_experiment"))  # src/ with dataset.py sha c497fab7 (same as the runs)
RUN = R / "repository/data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0"
spec = importlib.util.spec_from_file_location("trainer", RUN / "train_cnn_baseline_instrumented.py")
trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
from src.images.dataset import StudyCache, load_manifest  # noqa: E402
from src.reports.schema import TARGETS  # noqa: E402

D = "repository/data/"
MODELS = {  # name: (checkpoint, saved fold-0 OOF predictions or None)
    "r50_hpo_v31_d2_fold0": (D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/checkpoint_best.pt",
                             D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json"),
    "r50_hpo_v31_trainval": (D + "hpo_v31_trainval_resnet50/results/hpo_v31_trainval_resnet50_fresh/run/checkpoint_best.pt", None),
    "r50_hpo_v2_fold0": (D + "hpo_v2_provenance_resnet50_fold0/kaggle_output/hpo_v2_provenance_resnet50_fold0/run/checkpoint_best.pt",
                         D + "hpo_v2_provenance_resnet50_fold0/kaggle_output/hpo_v2_provenance_resnet50_fold0/run/oof_predictions.json"),
    "r50_hpo_blend_fold0": (D + "hpo_blend_resnet50_submission/artifact/checkpoint_best.pt",
                            D + "hpo_v1_resnet50_auc_margin_fold0/results/baseline_validation_predictions.json"),
    "r34_plateau60_fold0": (D + "cnn_topk_plateau60/cnn_topk_support_v1_plateau60/checkpoint_best.pt",
                            D + "cnn_topk_plateau60/cnn_topk_support_v1_plateau60/oof_predictions.json"),
    "r34_topk_v1_fold0": ("cnn_topk_experiment/outputs/cnn_topk_support_v1/checkpoint_best.pt",
                          "cnn_topk_experiment/outputs/cnn_topk_support_v1/oof_predictions.json"),
    "r34_pairwise_fold0": ("data/kaggle_cli/cnn_pairwise_auc_fold0/cnn_topk_pairwise_auc_fold0/checkpoint_best.pt",
                           "data/kaggle_cli/cnn_pairwise_auc_fold0/cnn_topk_pairwise_auc_fold0/oof_predictions.json"),
    "r34_hpo_v1_auc_control_fold0": (D + "hpo_v1_blend_auc_control_fold0/artifacts/checkpoint_best.pt", None),
}
TRAIN_CACHE = [R / D / f"external/topk-support-cache-shards/shard-{s}-rehost" for s in range(3)]
GOLD_CACHE = R / "data/kaggle_cli/cnn_gold_cache_artifacts/slice_cache"
device = torch.device("cuda")


def load_oof(path):
    d = json.loads((R / path).read_text())
    if "scores" in d:
        m, tg = dict(zip(d["study_ids"], d["scores"])), d.get("targets_in_order", TARGETS)
    else:
        m, tg = {r["study_id"]: r["probabilities"] for r in d["predictions"]}, d.get("targets_in_order", d.get("targets", TARGETS))
    order = [list(tg).index(t) for t in TARGETS]
    return {k: np.asarray(v, float)[order] for k, v in m.items()}


def predict(model, cache, ids, cfg):
    dummy = (np.zeros(len(TARGETS), np.float32), np.ones(len(TARGETS), np.float32))
    labels, out = {s: dummy for s in ids}, []
    with torch.no_grad():
        for i in range(0, len(ids), 4):
            chunk = ids[i:i + 4]
            slices, sos, *_ = trainer.prepare_batch(cache, labels, chunk, None, None, device)
            with torch.autocast("cuda", enabled=bool(cfg.get("mixed_precision", True))):
                logits = model(slices, sos, len(chunk))
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


train_cache = StudyCache(load_manifest(TRAIN_CACHE, ["unordered"]))
gold_cache = StudyCache(load_manifest([GOLD_CACHE], ["unordered"]))
gold_ids = sorted(gold_cache.manifest)
assert len(gold_ids) == 58 and not set(gold_ids) & set(train_cache.manifest)
result = {"targets_in_order": list(TARGETS), "gold_study_ids": gold_ids, "models": {}}
for name, (ckpt, oof) in MODELS.items():
    t0 = time.time()
    payload = torch.load(R / ckpt, map_location="cpu", weights_only=True)
    cfg = payload["config"]; m = cfg["model"]
    model = trainer.SliceModel(m["backbone"], False, m["dropout"], len(TARGETS), m.get("top_k_fraction", 0.1))
    model.load_state_dict(payload["state_dict"]); model.to(device).eval()
    entry = {"checkpoint": ckpt, "sha256": hashlib.sha256((R / ckpt).read_bytes()).hexdigest(), "epoch": payload.get("epoch"),
             "config_version": cfg.get("version"), "backbone": m["backbone"], "train_validation_fold": cfg.get("validation_fold")}
    if oof:
        saved = load_oof(oof)
        ids = sorted(set(saved) & set(train_cache.manifest))
        ids = [ids[i] for i in np.random.default_rng(0).choice(len(ids), 32, replace=False)]
        mine, ref = predict(model, train_cache, ids, cfg), np.stack([saved[s] for s in ids])
        rk = lambda a: a.argsort(0).argsort(0)
        entry["fidelity_32_fold0"] = {
            "max_abs_diff": round(float(np.abs(mine - ref).max()), 5),
            "median_abs_diff": round(float(np.median(np.abs(mine - ref))), 5),
            "mean_spearman_over_targets": round(float(np.mean([np.corrcoef(rk(mine)[:, k], rk(ref)[:, k])[0, 1]
                                                              for k in range(len(TARGETS))])), 4)}
    entry["gold_probabilities"] = np.round(predict(model, gold_cache, gold_ids, cfg), 6).tolist()
    entry["seconds"] = round(time.time() - t0, 1)
    result["models"][name] = entry
    print(name, json.dumps({k: v for k, v in entry.items() if k != "gold_probabilities"}), flush=True)
    del model; torch.cuda.empty_cache()
(OUT / "gold_image_model_predictions.json").write_text(json.dumps(result) + "\n")
