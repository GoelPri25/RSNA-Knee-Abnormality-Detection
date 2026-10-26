"""SER-0 (ledger row SER-0; no submission): which series group each v4 finding draws on. The LB v4 l4copy submission
code (v4_submission.py in the weights dataset, sha256 80f15c3b..., = kernel rsna-v4-l4copy-selfloc-submission v3)
end to end on the 825 fold-0 studies decoded from the train DICOMs by its own builder; each study built once, then
predicted under 7 series configs. Removing a slot = zeroing its rows, which is exactly what the builder leaves when a
study lacks that series (mask and the 94 uniform centres are recomputed from the zeroed volume). Self-located crops.
Out (/kaggle/working): ser0_{config}_{fused,global,local}.json, ser0_log.json. Scored by scripts/audit_series_ablation.py.
"""
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ART = next(p for p in (Path("/kaggle/input/datasets/tumis001/rsna-v4-l4copy-fold0"),
                       Path("/kaggle/input/rsna-v4-l4copy-fold0")) if p.exists())
assert hashlib.sha256((ART / "v4_submission.py").read_bytes()).hexdigest().startswith("80f15c3b36bd534f")
sys.path.insert(0, str(ART))                             # importable by name: Pool pickles sub.build_study
import v4_submission as sub  # noqa: E402

# slots: 0 SagF, 1 SagNF, 2 CorF, 3 CorNF, 4 Ax (sub.SLOTS order)
CONFIGS = {"all": [], "no_nonfs": [1, 3], "no_fs": [0, 2], "no_axial": [4], "no_sag": [0, 1], "no_cor": [2, 3],
           "sagf_only": [1, 2, 3, 4]}
STARTS = np.cumsum([0] + [k for _, _, k in sub.SLOTS])


def main():
    t0 = time.monotonic()
    data, work = sub.competition_dir(), Path("/kaggle/working")
    ids_path = next(p for p in (Path("/kaggle/input/datasets/tumis001/rsna-d96-fold0-ids/fold0_ids.json"),
                                Path("/kaggle/input/rsna-d96-fold0-ids/fold0_ids.json")) if p.exists())
    ids = json.loads(ids_path.read_text()); assert len(ids) == 825, len(ids)
    sub.ensure_timm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sub.load_model(ART / sub.WEIGHTS["file"], device, sub.WEIGHTS["sha256"])
    by_study = {}
    for row in sub.read_csv(data / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    out = {c: {} for c in CONFIGS}
    stats = {"empty_studies": 0, "decode_errors": 0, "fp32_retries": 0, "skipped": {c: 0 for c in CONFIGS},
             "slot_absent": [0] * 5, "seconds_per_config": {c: 0.0 for c in CONFIGS}}
    with Pool(4) as procs, torch.no_grad():
        for n, (sid, vol, mask, errors) in enumerate(
                procs.imap(sub.build_study, [(s, by_study.get(s, []), data / "train_series") for s in ids]), 1):
            stats["decode_errors"] += errors
            if not mask.any():
                stats["empty_studies"] += 1
                continue
            for j in range(5):
                stats["slot_absent"][j] += int(not mask[STARTS[j]:STARTS[j + 1]].any())
            for c, drop in CONFIGS.items():
                v = vol
                if drop:
                    v = vol.copy()
                    for j in drop:
                        v[STARTS[j]:STARTS[j + 1]] = 0
                m = v.reshape(v.shape[0], -1).max(1) > 0              # the builder's own mask rule
                if not m.any():
                    stats["skipped"][c] += 1
                    continue
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.monotonic()
                p = sub.predict_study(model, v, m, device)
                if not np.isfinite(p).all():
                    stats["fp32_retries"] += 1
                    p = sub.predict_study(model, v, m, device, amp=False)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                stats["seconds_per_config"][c] += time.monotonic() - t1
                out[c][sid] = p
            if n == 20:
                print("FIRST20 all fused " + json.dumps({s: np.round(out["all"][s][0], 3).tolist() for s in list(out["all"])[:3]}), flush=True)
            if n % 50 == 0 or n == len(ids):
                print(f"ser0 {n}/{len(ids)} studies; {time.monotonic() - t0:.0f}s; {json.dumps(stats)}", flush=True)
    for c in CONFIGS:
        for j, h in enumerate(("fused", "global", "local")):
            preds = [{"study_id": s, "probabilities": out[c][s][j].tolist()} for s in ids if s in out[c]]
            (work / f"ser0_{c}_{h}.json").write_text(json.dumps(
                {"targets_in_order": sub.TARGETS, "note": f"v4 l4copy fold 0, Kaggle path, self crops, config {c}, {h}",
                 "dropped_slots": CONFIGS[c], "predictions": preds}))
    stats["seconds_per_config"] = {c: round(v, 1) for c, v in stats["seconds_per_config"].items()}
    log = {"studies": len(out["all"]), "stats": stats, "configs": CONFIGS,
           "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
           "seconds": round(time.monotonic() - t0, 1)}
    (work / "ser0_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("SER0_DONE " + json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
