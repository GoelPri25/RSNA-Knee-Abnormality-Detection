"""Anatomical-mirror TTA submission for one GL4 student: each test study is read twice, as is and anatomically mirrored
(the left / right mirror the model saw in training with probability 0.5: sagittal windows reverse their 3 slice
channels, coronal / axial windows flip width; the fused head also self-locates and crops on the mirrored view), and the
submission is the mean of the two probabilities, as the runner's val_swa_mirror.

Earlier evidence: Exp 1e (r50 d96) mirror TTA was +0.0009 on fold 0 and unchanged on the LB (0.936). Not yet measured
for the E4 recipe or ConvNeXt-S: run MODE "fold0" first and score the three JSONs with score_fold0.py. The mirror
doubles the inference time (efficiency track: ~0.00087 AUC per extra minute).

MODE "test" writes submission.csv (COMBINE "prob" = mean of probabilities, "logit" = mean of logits). MODE "fold0"
predicts the fold-0 training studies from DICOM and writes fold0_<name>_{plain,mirror,tta}.json (a commit run).
Before the test set the model re-predicts FID_N fold-0 studies (plain view, global head) against its run's saved
val_swa_global.json (max |dp| < FID_MAX).
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
COMBINE = "prob"                  # "prob" | "logit"
MODEL = {"name": "cnxs", "arch": "cnxs", "head": "global", "mirror": True,                     # LB 0.947 without mirror
         "dataset": "priyanjaliml/rsna-pubkd-e4cnxs-fold0", "file": "weights_swa21_25_global_only.pt",
         "sha256": "1c21356b312082a4ad2ccc7470244f74f7a06686c25450e33a8d56dca546d64d"}
# fused head instead: "head": "fused", "file": "weights_swa21_25.pt",
#   "sha256": "29af81b432de5b339aa016b965709bb06b0045d2db089a2fc23e764e9265e2f3"
FOLD0_IDS = ("priyanjaliml/rsna-pubkd-e4cnxs-fold0", "fold0_predictions/val_swa_global.json")
FID_N, FID_MAX = 8, 0.01
FID = {}                          # <<FID>> model name -> {fold-0 study: saved global probabilities}


def combine(plain, mirror):
    if COMBINE == "logit":
        z = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))  # noqa: E731
        return 1 / (1 + np.exp(-(z(plain) + z(mirror)) / 2))
    return (plain + mirror) / 2


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
    spec = {**MODEL, "mirror": True}
    models = {spec["name"]: load_model({**spec, "path": dataset_dir(spec["dataset"]) / spec["file"]}, device)}
    t_load = time.monotonic() - started
    print(f"device {device} | {spec['name']} {spec['head']} head + anatomical mirror ({COMBINE}) | ready {t_load:.1f}s",
          flush=True)
    stats = {"empty_studies": 0, "decode_errors": 0, "fp32_retries": 0}

    if MODE == "fold0":
        ref = json.loads((dataset_dir(FOLD0_IDS[0]) / FOLD0_IDS[1]).read_text())["predictions"]
        ids = [r["study_id"] for r in ref][:FOLD0_LIMIT]
        by_study, wanted = {}, set(ids)
        for row in read_csv(data / "train_series.csv"):
            if row["StudyInstanceUID"] in wanted:
                by_study.setdefault(row["StudyInstanceUID"], []).append(row)
        v = run_models(models, [spec], ids, by_study, data / "train_series", device, stats, started, "fold0 ")[spec["name"]]
        tag = f"{spec['name']}_{spec['head']}"
        val_json(work / f"fold0_{tag}_plain.json", ids, v["plain"], "plain view")
        val_json(work / f"fold0_{tag}_mirror.json", ids, v["mirror"], "anatomical mirror only")
        val_json(work / f"fold0_{tag}_tta.json", ids, combine(v["plain"], v["mirror"]), f"mirror TTA ({COMBINE})")
        summary = {"mode": "fold0", "studies": len(ids), "model": spec, "combine": COMBINE, "stats": stats,
                   "seconds": round(time.monotonic() - started, 1)}
        (work / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return

    t0 = time.monotonic()
    fid = fidelity(models, {n: r for n, r in FID.items() if n in models}, data, device, FID_N)
    print(f"FIDELITY max |dp| {fid} ({time.monotonic() - t0:.1f}s)", flush=True)
    if any(x >= FID_MAX for x in fid.values()):
        raise ValueError(f"fidelity gate failed: {fid}")
    t_test = time.monotonic()
    by_study = {}
    for row in read_csv(data / "test_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    v = run_models(models, [spec], test_ids, by_study, data / "test_series", device, stats, started, "test ")[spec["name"]]
    P = combine(v["plain"], v["mirror"])
    final = {s: (P[i].tolist() if np.isfinite(P[i]).all() else fallback[s]) for i, s in enumerate(test_ids)}
    write_submission(work / "submission.csv", test_ids, final)
    check_submission(work / "submission.csv", test_ids)
    summary = {"mode": "test", "model": spec, "combine": COMBINE, "fidelity_max_abs": fid, "stats": stats,
               "studies": len(test_ids), "seconds_model_load": round(t_load, 1),
               "seconds_test": round(time.monotonic() - t_test, 1), "seconds": round(time.monotonic() - started, 1)}
    (work / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
