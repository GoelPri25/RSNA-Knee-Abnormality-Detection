"""Percentile-rank blend submission (as the public 0.941 notebook blends its branches): every model in MODELS predicts
every test study; per model and finding the predictions become percentile ranks over the test studies, and the
submission is the per-finding weighted mean of the ranks (BLEND = "prob" averages probabilities instead, for an A/B).
AUC depends only on order, so a rank blend lets models with different calibration count by their weights rather than by
their spread.

Default: ConvNeXt-S 22k global (KD-E4CNXS, LB 0.947) + the teammate's rap2-D global (CoAtNet-rmlp-2, LB 0.948), 0.5 / 0.5:
two different families at the same strength. Their weights cannot be fitted on fold 0 (rap2-D's Raptor init saw fold 0),
so they are equal. For reference, two E4-recipe students blend for little on fold 0 (same targets):
  ConvNeXt-S global alone            label-v2 0.9534, Steven-v4 0.9202
  0.8 ConvNeXt-S + 0.2 online E4     label-v2 0.9540, Steven-v4 0.9207 (rank = prob to 4 decimals)

MODE "test" writes submission.csv. MODE "fold0" predicts the fold-0 training studies from DICOM instead and writes one
val-format JSON per model plus the rank and probability blends, for score_fold0.py (a commit run, not a submission).
Before the test set each model re-predicts FID_N fold-0 studies (global head) and must match its run's saved
val_swa_global.json within FID_MAX.
Built by build_kernels.py (the GL4 core is inlined below); edit the source, not the built kernel.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

# <<GL4_CORE>>

MODE = "test"                     # "test" | "fold0"
FOLD0_LIMIT = None                # fold0 mode: first N studies only (None = all 825)
BLEND = "rank"                    # "rank" | "prob"
# weight: one number, or 12 per-finding numbers in TARGETS order. mirror: also predict the anatomical mirror and
# average the two probabilities (see mirror_tta_submission.py).
MODELS = [
    {"name": "cnxs", "arch": "cnxs", "head": "global", "weight": 0.5, "mirror": False,          # LB 0.947 alone
     "dataset": "priyanjaliml/rsna-pubkd-e4cnxs-fold0", "file": "weights_swa21_25_global_only.pt",
     "sha256": "1c21356b312082a4ad2ccc7470244f74f7a06686c25450e33a8d56dca546d64d"},
    {"name": "rap2", "arch": "rap2", "head": "global", "weight": 0.5, "mirror": False,          # LB 0.948 alone
     "dataset": "tumis001/rsna-pubkd-rap2d-fold0", "file": "weights_swa4_6.pt",
     "sha256": "82c46fc7c92ef96f1e675b40edffe91553abe427c82829f41d7bd9162b5ad594"},
]
# optional: BiT ResNet-101 21k, validated on fold 1 (fold-1 label-v2 0.9627 global; a different family and split):
#   {"name": "bit101f1", "arch": "bit101", "head": "global", "weight": 0.2, "mirror": False,
#    "dataset": "priyanjaliml/rsna-pubkd-e4bit101-fold1", "file": "weights_swa21_25_global_only.pt",
#    "sha256": "414cf43cc456f1906efaadc5854ac657c411e05c3cc356b3ff6c4011ff4eef6d"},
# optional third (fold 0: +0.0005 on ConvNeXt-S at weight 0.2):
#   {"name": "e4on", "arch": "r50", "head": "global", "weight": 0.2, "mirror": False,
#    "dataset": "priyanjaliml/rsna-pubkd-online-e4on-fold0", "file": "weights_swa21_25.pt",
#    "sha256": "5ad74809a0caccfe40b317148fa1559289c91765a9c42577eeae5b6eb75fd8d0"},
FOLD0_IDS = ("priyanjaliml/rsna-pubkd-e4cnxs-fold0", "fold0_predictions/val_swa_global.json")
FID_N, FID_MAX = 8, 0.01
FID = {}                          # <<FID>> model name -> {fold-0 study: saved global probabilities}


def model_view(out, spec):
    v = out[spec["name"]]
    return (v["plain"] + v["mirror"]) / 2 if "mirror" in v else v["plain"]


def main():
    started = time.monotonic()
    data, work = competition_dir(), Path("/kaggle/working")
    sample = read_csv(data / "sample_submission.csv")
    if not sample or list(sample[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("sample_submission.csv columns differ from the model target order")
    test_ids = [r["StudyInstanceUID"] for r in sample]
    fallback = {r["StudyInstanceUID"]: [float(r[t]) for t in TARGETS] for r in sample}
    write_submission(work / "submission.csv", test_ids, fallback)        # valid file even if inference is interrupted

    ensure_timm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {s["name"]: load_model({**s, "path": dataset_dir(s["dataset"]) / s["file"]}, device) for s in MODELS}
    t_load = time.monotonic() - started
    print(f"device {device} | models {[(s['name'], s['head'], s['weight'], s['mirror']) for s in MODELS]} | "
          f"blend {BLEND} | ready {t_load:.1f}s", flush=True)
    stats = {"empty_studies": 0, "decode_errors": 0, "fp32_retries": 0}

    if MODE == "fold0":
        ref = json.loads((dataset_dir(FOLD0_IDS[0]) / FOLD0_IDS[1]).read_text())["predictions"]
        ids = [r["study_id"] for r in ref][:FOLD0_LIMIT]
        by_study, wanted = {}, set(ids)
        for row in read_csv(data / "train_series.csv"):
            if row["StudyInstanceUID"] in wanted:
                by_study.setdefault(row["StudyInstanceUID"], []).append(row)
        out = run_models(models, MODELS, ids, by_study, data / "train_series", device, stats, started, "fold0 ")
        parts = [model_view(out, s) for s in MODELS]
        for s, p in zip(MODELS, parts):
            val_json(work / f"fold0_{s['name']}_{s['head']}.json", ids, p, f"{s['name']} {s['head']} mirror={s['mirror']}")
        w = [s["weight"] for s in MODELS]
        val_json(work / "fold0_blend_rank.json", ids, weighted([pct_rank(p) for p in parts], w), "rank blend")
        val_json(work / "fold0_blend_prob.json", ids, weighted(parts, w), "probability blend")
        summary = {"mode": "fold0", "studies": len(ids), "models": MODELS, "stats": stats,
                   "seconds": round(time.monotonic() - started, 1)}
        (work / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return

    t0 = time.monotonic()
    fid = fidelity(models, {n: r for n, r in FID.items() if n in models}, data, device, FID_N)
    print(f"FIDELITY max |dp| per model {fid} ({time.monotonic() - t0:.1f}s)", flush=True)
    if any(v >= FID_MAX for v in fid.values()):
        raise ValueError(f"fidelity gate failed: {fid}")
    t_test = time.monotonic()
    by_study = {}
    for row in read_csv(data / "test_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    out = run_models(models, MODELS, test_ids, by_study, data / "test_series", device, stats, started, "test ")
    parts = [model_view(out, s) for s in MODELS]
    w = [s["weight"] for s in MODELS]
    P = weighted([pct_rank(p) for p in parts] if BLEND == "rank" else parts, w)
    final = {s: (P[i].tolist() if np.isfinite(P[i]).all() else fallback[s]) for i, s in enumerate(test_ids)}
    write_submission(work / "submission.csv", test_ids, final)
    check_submission(work / "submission.csv", test_ids)
    summary = {"mode": "test", "blend": BLEND, "models": MODELS, "fidelity_max_abs": fid, "stats": stats,
               "studies": len(test_ids), "seconds_model_load": round(t_load, 1),
               "seconds_test": round(time.monotonic() - t_test, 1), "seconds": round(time.monotonic() - started, 1)}
    (work / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
