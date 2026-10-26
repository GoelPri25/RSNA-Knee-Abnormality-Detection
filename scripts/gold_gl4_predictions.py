"""Finished r50-GL4 students on the 58 host gold studies (inference only; no training, no selection).

The gold studies are in the d96 cache but in no fold and not in the MS1 crop table, so only the global head is scored
(predict_global, placeholder crop rows that the global path never reads). Fidelity check first: the same call on 16
fold-0 studies must reproduce each run's saved val_swa_global.json. Gold labels are NOT read here; this only writes
predictions (scored on the host).
Env: a lane's env file (CFG as in training). Usage (Colab VM, after colab_pubkd_setup.py):
  python -u scripts/gold_gl4_predictions.py OUT_DIR NAME=weights.pt[:val_swa_global.json] ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r50_gl4_runner as L  # noqa: E402

r = L.r
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
meta = r.read_manifest()
gold = sorted(s for s, m in meta.items() if m["gold"]); assert len(gold) == 58, len(gold)
for name, spec in (a.split("=", 1) for a in sys.argv[2:]):
    w, _, ref = spec.partition(":")
    model, _, _ = L.build(dev, pretrained=False)
    model.load_state_dict(torch.load(w, map_location="cpu", weights_only=True), strict=True)
    rec = {"model": name, "weights": w, "weights_sha256": r.sha(Path(w))}
    if ref:
        saved = json.loads(Path(ref).read_text())["predictions"][:16]
        ids = [p["study_id"] for p in saved]
        got = L.predict_global(model, ids, meta, L.ext_crop_table(ids, meta), dev)
        rec["fidelity_max_abs"] = float(np.abs(got - np.array([p["probabilities"] for p in saved])).max())
        assert rec["fidelity_max_abs"] < 5e-3, (name, rec["fidelity_max_abs"])
    p = L.predict_global(model, gold, meta, L.ext_crop_table(gold, meta), dev)
    rec.update({"targets_in_order": list(r.TARGETS), "head": "global",
                "predictions": [{"study_id": s, "probabilities": p[i].tolist()} for i, s in enumerate(gold)]})
    (OUT / f"gold_{name}.json").write_text(json.dumps(rec) + "\n")
    print(f"GOLD {name} done; fold-0 fidelity max|d| {rec.get('fidelity_max_abs')}", flush=True)
print("GOLD_DONE", flush=True)
