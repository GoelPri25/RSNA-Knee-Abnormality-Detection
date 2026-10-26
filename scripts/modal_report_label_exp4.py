"""Run a matched label-source screen with the original Experiment 4 trainer on Modal.

Raw competition CSV/DICOM files and evaluation labels stay on Kaggle. Modal
downloads precomputed features and report-derived supervision, then publishes
only checkpoints, learning curves, metrics, and development-only predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
import modal

APP_NAME = "rsna-report-label-exp4"
ARTIFACT_ID = "tumis001/rsna-exp4-report-label-comparison"
FLIGHT_ARTIFACT_ID = "tumis001/rsna-exp4-alternate-label-runs-v1"
ALTERNATE_ARTIFACT_ID = "tumis001/rsna-exp4-steven-label-run-v1"
ROOT = Path("/tmp/rsna_report_label_exp4")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
KERNELS = ["tumis001/rsna-knee-medsiglip-features-shard-0",
           "tumis001/rsna-knee-medsiglip-features-shard-1",
           "tumis001/rsna-knee-medsiglip-features-shard-2",
           "tumis001/rsna-knee-medsiglip-gold-features",
           "tumis001/report-label-extraction", "tumis001/report-label-audit"]

app = modal.App(APP_NAME)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "pandas", "scikit-learn",
                      "matplotlib", "kaggle==2.2.2"))
if modal.is_local():
    token_path = Path.home() / ".kaggle/access_token"
    if not token_path.is_file():
        raise FileNotFoundError(f"Kaggle API token missing: {token_path}")
    kaggle_secret = modal.Secret.from_dict({"KAGGLE_ACCESS_TOKEN": token_path.read_text().strip()})
else:
    kaggle_secret = modal.Secret.from_dict({})


@app.function(image=image, gpu="T4", timeout=28800, secrets=[kaggle_secret], scaledown_window=60)
def train_label_arms():
    import csv, json, os, subprocess, sys, time
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = ROOT
    inputs, artifacts = root / "inputs", root / "kaggle_upload"
    inputs.mkdir(parents=True, exist_ok=True); artifacts.mkdir(parents=True, exist_ok=True)
    kdir = Path.home() / ".kaggle"; kdir.mkdir(mode=0o700, parents=True, exist_ok=True)
    credential = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not credential:
        raise RuntimeError("Kaggle API token secret was not injected into Modal")
    (kdir / "access_token").write_text(credential + "\n"); (kdir / "access_token").chmod(0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kdir)

    def run(cmd, cwd=None):
        print("$", " ".join(map(str, cmd)), flush=True)
        p = subprocess.run(list(map(str, cmd)), cwd=cwd, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.stdout: print(p.stdout, end="", flush=True)
        if p.returncode: raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
        return p.stdout

    def kernel_output(kernel, dest):
        dest.mkdir(parents=True, exist_ok=True)
        run(["kaggle", "kernels", "output", kernel, "-p", dest])

    # Fetch the baseline source and cached derived embeddings from Kaggle.
    src_dir = inputs / "source"; src_dir.mkdir(exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", "tumis001/rsna-experiment1-baseline-source",
         "-p", src_dir, "--unzip"])
    scripts = list(src_dir.rglob("scripts/train_experiment4.py"))
    if len(scripts) != 1: raise RuntimeError(f"Expected one Experiment 4 source tree: {scripts}")
    source = scripts[0].parents[1]; sys.path.insert(0, str(source))
    for kernel in KERNELS:
        dest = inputs / kernel.replace("/", "__")
        if not list(dest.rglob("manifest.jsonl")) and not list(dest.rglob("report_structured_labels.csv")):
            kernel_output(kernel, dest)
    for dataset, slug in [("flight0234/rsna-knee-hybrid-report-labels", "flight"),
                          ("stevenleehans/rsna-knee-llm-report-labels", "steven")]:
        dest = inputs / f"{slug}_labels"; dest.mkdir(exist_ok=True)
        run(["kaggle", "datasets", "download", "-d", dataset, "-p", dest, "--unzip"])

    from src.mil.data import Example, load_cache
    from src.mil import train as train_module
    import importlib
    from src.mil.train import build_model, apply_class_balance, class_balance_factors, predict_compact

    def find(folder, filename):
        hits = list(folder.rglob(filename))
        if len(hits) != 1: raise RuntimeError(f"Expected one {filename} under {folder}, got {hits}")
        return hits[0]

    def kernel_dir(kernel): return inputs / kernel.replace("/", "__")
    split = pd.read_csv(find(kernel_dir(KERNELS[-1]), "report_gold_split.csv"), dtype={"StudyInstanceUID": str})
    gold_ids = set(split.StudyInstanceUID.astype(str))
    dev_ids = sorted(split.loc[split.split.eq("development"), "StudyInstanceUID"].astype(str))
    if len(gold_ids) != 58 or len(dev_ids) != 29: raise RuntimeError("Unexpected gold split dimensions")
    with (source / "configs/report_folds_v4.csv").open(newline="", encoding="utf-8") as f:
        fold_by_study = {r["study_id"]: int(r["fold"]) for r in csv.DictReader(f)}

    structured = pd.read_csv(find(kernel_dir(KERNELS[-2]), "report_structured_labels.csv"),
                              dtype={"StudyInstanceUID": str})
    severity_map = {"severe": (.97, 1.), "moderate_to_severe": (.95, .95),
                    "moderate": (.90, .90), "mild_to_moderate": (.82, .75),
                    "mild": (.70, .55), "unspecified": (.88, .75)}
    internal_items = []
    for r in structured.itertuples(index=False):
        uid = str(r.StudyInstanceUID)
        if uid in gold_ids or str(r.extraction_success).lower() != "true": continue
        probs, weights = [], []
        for target in TARGETS:
            slug = target.lower().replace("'", "").replace(" ", "_")
            status = str(getattr(r, f"{slug}__status")).lower()
            severity = str(getattr(r, f"{slug}__severity")).lower().replace(" ", "_")
            p, w = ((.05, 1.) if status == "absent" else
                    severity_map.get(severity, severity_map["unspecified"]) if status == "present" else (.5, 0.))
            probs.append(p); weights.append(w)
        if any(weights): internal_items.append({"study_id": uid, "probabilities": probs, "weights": weights})

    def csv_labels(slug, filename, name, has_conf):
        path = find(inputs / f"{slug}_labels", filename)
        frame = pd.read_csv(path, dtype={"StudyInstanceUID": str}, low_memory=False)
        if frame.StudyInstanceUID.duplicated().any() or not set(TARGETS).issubset(frame.columns):
            raise RuntimeError(f"Invalid label table: {path}")
        items = []
        for r in frame.to_dict("records"):
            uid = str(r["StudyInstanceUID"])
            if uid in gold_ids: continue
            probs, weights = [], []
            for t in TARGETS:
                value = pd.to_numeric(pd.Series([r[t]]), errors="coerce").iloc[0]
                conf = (pd.to_numeric(pd.Series([r.get(f"{t}__conf")]), errors="coerce").iloc[0]
                        if has_conf else 1.)
                if pd.isna(value) or pd.isna(conf) or not 0 <= float(value) <= 1:
                    probs.append(.5); weights.append(0.)
                else: probs.append(float(value)); weights.append(float(conf))
            if any(weights): items.append({"study_id": uid, "probabilities": probs, "weights": weights})
        return {"version": name, "labels": items}

    label_sets = {
        "confidence_v2_control": {"version": "confidence_v2", "labels": internal_items},
        # Match the prior clean-label audit recipe: scores are soft targets,
        # with unit supervision weight (the optional confidence column is ignored).
        "flight_v4": csv_labels("flight", "report_labels_v4hybrid.csv", "flight_v4", False),
        "steven_v4": csv_labels("steven", "llm_labels_v4_blend.csv", "steven_v4", False),
    }
    for arm, payload in label_sets.items():
        ids = [x["study_id"] for x in payload["labels"]]
        if len(ids) != len(set(ids)) or set(ids) & gold_ids: raise RuntimeError(f"Gold leakage/repeated IDs: {arm}")
        print(f"{arm}: {len(ids)} report-only label rows; all 58 gold IDs excluded", flush=True)

    feature_dirs = []
    for kernel in KERNELS[:4]:
        manifests = list(kernel_dir(kernel).rglob("manifest.jsonl"))
        dirs = [p.parent for p in manifests if (p.parent / "feature_summary.json").is_file()]
        if len(dirs) != 1: raise RuntimeError(f"Bad feature cache from {kernel}: {dirs}")
        feature_dirs.extend(dirs)
    cache, cache_summary = load_cache(feature_dirs, exclude_order_sources=("unordered",))
    print("Cache", cache_summary, "GPU", torch.cuda.get_device_name(0), flush=True)

    config = {"version": "mil_report_selected_v4", "seed": 2026,
              "model": {"hidden": 256, "heads": 4, "context_layers": 1, "dropout": .2,
                        "z_dim": 256, "global_fusion": True, "relative_position_bias": False},
              "epochs": 60, "batch_size": 8, "lr": 3e-4, "weight_decay": .05,
              "warmup_fraction": .05, "grad_clip": 1., "max_slices_train": 48,
              "max_slices_eval": 96, "studies_per_epoch": None, "validation_fold": 0,
              "loss": "target_mean_class_balanced_bce"}
    if cache_summary["feature_dim"] != 1152 or set(dev_ids) - set(cache):
        raise RuntimeError("Cache dimensions/dev coverage do not match Experiment 4")
    dev_examples = [Example(uid, cache[uid], np.full(12, .5, np.float32), np.ones(12, np.float32), "gold")
                    for uid in dev_ids]

    # The callback adds persistence only; Experiment 4's optimizer, loss,
    # schedule, sampling, validation and checkpoint selection remain unchanged.
    train_file = source / "src/mil/train.py"
    source_text = train_file.read_text()
    signature = "device: torch.device, log=print) -> tuple[list[dict], int]:"
    marker = "        if patience is not None and stale_epochs >= int(patience):"
    if source_text.count(signature) != 1 or source_text.count(marker) != 1:
        raise RuntimeError("Unexpected trainer code; refusing to patch unknown Experiment 4 source")
    source_text = source_text.replace(signature, "device: torch.device, log=print, on_epoch=None) -> tuple[list[dict], int]:")
    source_text = source_text.replace(marker,
        "        if on_epoch is not None:\n            on_epoch(model, record, best_epoch, best_loss)\n" + marker)
    train_file.write_text(source_text)
    importlib.reload(train_module)
    train_fn = train_module.train_model_with_validation
    build_model = train_module.build_model
    apply_class_balance = train_module.apply_class_balance
    class_balance_factors = train_module.class_balance_factors
    predict_compact = train_module.predict_compact

    upload = artifacts; upload.mkdir(exist_ok=True)
    (upload / "dataset-metadata.json").write_text(json.dumps({
        "id": ALTERNATE_ARTIFACT_ID, "title": "RSNA Experiment 4 alternate label runs",
        "licenses": [{"name": "CC0-1.0"}], "keywords": ["rsna", "experiment4", "private"]
    }, indent=2))
    (upload / "README.md").write_text(
        "Private derived artifacts for the Steven v4 label arm. "
        "The completed Confidence-V2 control is preserved in "
        f"https://www.kaggle.com/datasets/{ARTIFACT_ID}. "
        "The completed Flight v4 arm is preserved in "
        f"https://www.kaggle.com/datasets/{FLIGHT_ARTIFACT_ID}. "
        "This dataset contains checkpoints, histories, curves, summaries, and 29 development predictions only. "
        "No DICOM, competition CSV, or evaluation labels are included.\n")
    # Create a private dataset for Steven v4 at its first durable checkpoint.
    # Existing Confidence-V2 and Flight v4 datasets remain unchanged.
    created = False
    def publish(note):
        nonlocal created
        if created:
            run(["kaggle", "datasets", "version", "-p", upload, "-m", note, "-r", "zip"])
        else:
            # Kaggle datasets are private by default; do not pass --public.
            run(["kaggle", "datasets", "create", "-p", upload]); created = True

    def verify_uploaded_arm(arm):
        for attempt in range(10):
            result = subprocess.run(["kaggle", "datasets", "files", ARTIFACT_ID], text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Kaggle file listing failed: {result.stdout}")
            if (f"{arm}/best_checkpoint.pt" in result.stdout and
                    f"{arm}/latest_checkpoint.pt" in result.stdout):
                print(f"Kaggle confirmed best/latest checkpoints for {arm} (index check {attempt + 1})", flush=True)
                return
            print(f"Kaggle version index is still updating for {arm}; retry {attempt + 1}/10", flush=True)
            time.sleep(10)
        # Kaggle can accept an upload before its listing endpoint has indexed it.
        # The successful create/version command above is the durable-write signal;
        # report indexing lag without throwing away the live training process.
        print(f"WARNING: Kaggle accepted the upload for {arm}, but its file listing is still indexing; "
              "check the dataset after this run.", flush=True)

    # Confidence-V2 is already complete. Flight v4 is complete in its own
    # dataset; only rerun Steven v4, whose earlier attempt stopped after epoch 15.
    training_arms = {"steven_v4": label_sets["steven_v4"]}
    overall, start_all = {}, time.monotonic()
    for arm, payload in training_arms.items():
        raw = [Example(x["study_id"], cache[x["study_id"]], np.asarray(x["probabilities"], np.float32),
                       np.asarray(x["weights"], np.float32), "report")
               for x in payload["labels"] if x["study_id"] in cache and x["study_id"] in fold_by_study]
        train_raw = [x for x in raw if fold_by_study[x.study_id] != 0]
        val_raw = [x for x in raw if fold_by_study[x.study_id] == 0]
        factors = class_balance_factors(train_raw)
        train = apply_class_balance(train_raw, factors)
        validation = apply_class_balance(val_raw, factors)
        out = artifacts / arm; out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(config["seed"]); model = build_model(config, cache_summary["feature_dim"])
        hist = []
        def on_epoch(live_model, record, best_epoch, best_loss):
            hist.append(dict(record))
            state = {"state_dict": {k: v.detach().cpu() for k, v in live_model.state_dict().items()}, "config": config,
                     "label_version": payload["version"], "selected_epoch": best_epoch + 1,
                     "best_report_validation_bce": float(best_loss),
                     "feature_dim": cache_summary["feature_dim"], "class_balance_factors": factors.tolist()}
            torch.save(state, out / "latest_checkpoint.pt")
            if best_epoch == record["epoch"]: torch.save(state, out / "best_checkpoint.pt")
            df = pd.DataFrame(hist); df.to_csv(out / "training_history.csv", index=False)
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            ax.plot(df.epoch + 1, df.train_classification_bce, label="Train BCE")
            ax.plot(df.epoch + 1, df.report_validation_bce, label="Report-validation BCE")
            ax.set(xlabel="Epoch", ylabel="Class-balanced BCE", title=f"{arm}: Experiment 4")
            ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=130); plt.close(fig)
            if (record["epoch"] + 1) % 15 == 0:
                (upload / "run_progress.json").write_text(json.dumps({"arm": arm, "epoch": record["epoch"] + 1,
                    "best_epoch": best_epoch + 1, "best_report_validation_bce": float(best_loss),
                    "gold_evaluation_touched": False}, indent=2))
                if not (out / "best_checkpoint.pt").is_file():
                    raise RuntimeError(f"Checkpoint missing before upload for {arm}")
                publish(f"{arm}, epoch {record['epoch'] + 1}")
                verify_uploaded_arm(arm)

        print(f"\n=== {arm}: train {len(train)}, report-validation {len(validation)}, 60 epochs ===", flush=True)
        history, best_epoch = train_fn(model, train, validation, config, torch.device("cuda"), on_epoch=on_epoch)
        ids, probs, _ = predict_compact(model, dev_examples, config, torch.device("cuda"))
        pd.DataFrame(probs, columns=TARGETS).assign(StudyInstanceUID=ids).to_csv(out / "development_predictions.csv", index=False)
        summary = {"arm": arm, "label_version": payload["version"], "label_rows": len(payload["labels"]),
                   "train_studies": len(train), "report_validation_studies": len(validation),
                   "selected_epoch": best_epoch + 1,
                   "best_report_validation_bce": float(history[best_epoch]["report_validation_bce"]),
                   "best_train_classification_bce": float(history[best_epoch]["train_classification_bce"]),
                   "development_prediction_rows": len(ids), "evaluation_gold_labels_read": 0,
                   "history": history}
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2)); overall[arm] = summary
        print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2), flush=True)

    result = {"experiment": "experiment4_label_source_comparison", "config": config,
              "cache": cache_summary, "arms": overall,
              "confidence_v2_control_reference": {
                  "dataset": ARTIFACT_ID, "selected_epoch": 18,
                  "best_report_validation_bce": 0.567200779914856,
                  "best_train_classification_bce": 0.5095585599923745,
                  "development_prediction_rows": 29,
                  "evaluation_gold_labels_read": 0},
              "flight_v4_reference": {
                  "dataset": FLIGHT_ARTIFACT_ID, "selected_epoch": 19,
                  "best_report_validation_bce": 0.5694433450698853,
                  "best_train_classification_bce": 0.527398198773533,
                  "development_prediction_rows": 29,
                  "evaluation_gold_labels_read": 0},
              "gold_studies_excluded": 58,
              "development_predictions_only": 29, "evaluation_gold_labels_read": 0,
              "runtime_seconds": round(time.monotonic() - start_all, 1)}
    (upload / "final_summary.json").write_text(json.dumps(result, indent=2))
    publish("Completed Experiment 4 report label source comparison")
    for arm in training_arms:
        verify_uploaded_arm(arm)
    return result


@app.function(image=image, timeout=900, secrets=[kaggle_secret])
def kaggle_preflight():
    import os, subprocess
    from pathlib import Path
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Kaggle API token secret was not injected")
    (kdir / "access_token").write_text(token + "\n")
    (kdir / "access_token").chmod(0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kdir)
    dest = Path("/tmp/kaggle-preflight")
    dest.mkdir(exist_ok=True)
    for cmd in (["kaggle", "datasets", "download", "-d", "tumis001/rsna-experiment1-baseline-source", "-p", str(dest), "--unzip"],
                ["kaggle", "kernels", "output", "tumis001/report-label-audit", "-p", str(dest / "audit")]):
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            raise RuntimeError(f"Kaggle preflight failed ({result.returncode}): {' '.join(cmd)}")
    listing = subprocess.run(["kaggle", "datasets", "files", ARTIFACT_ID], text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(listing.stdout, end="", flush=True)
    if listing.returncode or "confidence_v2_control" not in listing.stdout:
        raise RuntimeError("Kaggle checkpoint artifact listing preflight failed")
    return {"dataset_source": "downloaded", "report_split_kernel_output": "downloaded"}


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(train_label_arms.remote(), indent=2))


@app.local_entrypoint()
def preflight():
    import json
    print(json.dumps(kaggle_preflight.remote(), indent=2))
