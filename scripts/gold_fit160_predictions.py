"""Predict the 58 gold studies with the Fit160 r34 checkpoint (HPO-v1 +281, epoch 15, LB 0.903), CPU only.

Same model/batch code as scripts/gold_image_model_predictions.py (the hpo_v31_d2 run's training script, unchanged),
but reading the Fit160 gold cache (priyanjaliml/rsna-knee-fit160-cache-gold output). Gold labels are not read here;
scored by scripts/audit_gold_fit160.py.

Container (CPU, no network, read-only project):
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z \
    docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
    python -I /proj/repository/scripts/gold_fit160_predictions.py /proj /out
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.dont_write_bytecode = True
sys.path.insert(0, str(R / "cnn_topk_experiment"))
RUN = R / "repository/data/hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0"
spec = importlib.util.spec_from_file_location("trainer", RUN / "train_cnn_baseline_instrumented.py")
trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
from src.images.dataset import StudyCache, load_manifest  # noqa: E402
from src.reports.schema import TARGETS  # noqa: E402

CKPT = R / "repository/data/hpo_v1_blend_fit160_best_epoch15/checkpoint_best.pt"
GOLD = R / "repository/data/fit160_kaggle/kernel_outputs/gold_download/slice_cache"
assert hashlib.sha256(CKPT.read_bytes()).hexdigest() == "427dd72baa09185055b5b013e3b6a8e54c6f148e6a7dbbb630f2a9ed2a11a512"
summary = json.loads((GOLD / "cache_summary.json").read_text())
assert summary["cache"]["fov_mode"] == "fit" and summary["cache"]["field_of_view_mm"] == 160.0
torch.set_num_threads(8)
device = torch.device("cpu")

payload = torch.load(CKPT, map_location="cpu", weights_only=False)   # our own checkpoint (sha pinned above)
assert payload["epoch"] == 15
m = payload["config"]["model"]
model = trainer.SliceModel(m["backbone"], False, m["dropout"], len(TARGETS), m.get("top_k_fraction", 0.1))
model.load_state_dict(payload["state_dict"], strict=True); model.eval()

cache = StudyCache(load_manifest([GOLD], ["unordered"]))
ids = sorted(cache.manifest)
assert len(ids) == 58
dummy = (np.zeros(len(TARGETS), np.float32), np.ones(len(TARGETS), np.float32))
labels, out = {s: dummy for s in ids}, []
with torch.no_grad():
    for i in range(0, len(ids), 4):
        chunk = ids[i:i + 4]
        slices, sos, *_ = trainer.prepare_batch(cache, labels, chunk, None, None, device)
        out.append(torch.sigmoid(model(slices, sos, len(chunk)).float()).numpy())
        print(f"{i + len(chunk)}/58", flush=True)
p = np.concatenate(out)
(OUT / "gold_fit160_predictions.json").write_text(json.dumps({
    "model": "r34_hpo_v1_fit160_epoch15", "targets_in_order": list(TARGETS), "gold_study_ids": ids,
    "gold_probabilities": np.round(p, 6).tolist()}) + "\n")
print("done", p.shape)
