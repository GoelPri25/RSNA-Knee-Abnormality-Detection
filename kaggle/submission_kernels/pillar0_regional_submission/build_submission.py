"""Build the private Kaggle inference kernel for the selected Pillar-0 reader."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVAL_RUNNER = HERE.parent / "pillar0_regional_gold_eval" / "run.py"
OUTPUT = HERE / "run.py"


SUBMISSION_MAIN = r'''
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle T4 GPU for frozen Pillar-0 inference")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    data = locate_data()
    sample = pd.read_csv(data / "sample_submission.csv", dtype={"StudyInstanceUID": str})
    official_test = pd.read_csv(data / "test.csv", dtype={"StudyInstanceUID": str})
    test_series = pd.read_csv(data / "test_series.csv", dtype=str)
    expected_columns = ["StudyInstanceUID", *TARGETS]
    if list(sample.columns) != expected_columns:
        raise RuntimeError(f"Unexpected sample-submission columns: {list(sample.columns)}")
    if list(official_test.columns) != ["StudyInstanceUID"]:
        raise RuntimeError(f"Unexpected hidden-test schema: {list(official_test.columns)}")
    study_ids = sample["StudyInstanceUID"].astype(str).tolist()
    test_ids = official_test["StudyInstanceUID"].astype(str).tolist()
    if len(study_ids) != len(set(study_ids)) or set(study_ids) != set(test_ids):
        raise RuntimeError("Sample submission and official test studies do not match exactly")
    if test_series["SeriesInstanceUID"].duplicated().any():
        raise RuntimeError("Duplicate SeriesInstanceUID in test_series.csv")
    test_series = test_series[test_series.StudyInstanceUID.astype(str).isin(set(study_ids))]
    if set(test_series.StudyInstanceUID.astype(str)) != set(study_ids):
        raise RuntimeError("At least one test study has no series metadata")
    test_series = test_series.sort_values(["StudyInstanceUID", "SeriesInstanceUID"], kind="stable")
    by_study = {sid: ([], []) for sid in study_ids}
    for row in test_series.itertuples(index=False):
        by_study[str(row.StudyInstanceUID)][1].append(row)
    if any(not rows for _, rows in by_study.values()):
        raise RuntimeError("At least one hidden test study has an empty series list")

    bundle = locate_bundle()
    pillar = load_pillar(bundle)
    visual = pillar.model.visual
    cfg = visual.model_config["modalities"][MODEL_INPUT_KEY]
    layout = visual.prepare_multiscale_layout(
        img_size=cfg["image_size"], merge_ratio=cfg["merge_ratio"],
        local2global=cfg["local2global"], patch_size=cfg["patch_size"])
    train_root = data / "test_series"
    for index, row in enumerate(test_series.itertuples(index=False), 1):
        record = capture_token_record(pillar, visual, layout, train_root, row)
        by_study[str(row.StudyInstanceUID)][0].append(record)
        if index % 4 == 0 or index == len(test_series):
            print(f"Encoded test series {index}/{len(test_series)}", flush=True)
        del record
        if index % 16 == 0:
            gc.collect()

    artifacts = dataset_dir("rsna-pillar0-regional-reader-cv-v1")
    selected_checkpoint = checkpoint_path(artifacts, 0)
    checkpoint = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "scale_aligned_regional_target_query":
        raise RuntimeError("Selected checkpoint architecture does not match this submission reader")
    if int(checkpoint.get("best_epoch", -1)) != 36:
        raise RuntimeError(f"Unexpected fold-0 selected epoch: {checkpoint.get('best_epoch')}")
    reader = Pillar0RegionalTargetQueryReader().to(DEVICE)
    reader.load_state_dict(checkpoint["model_state_dict"], strict=True)
    reader.eval()
    print(f"Reader checkpoint: fold 0, selected epoch {checkpoint['best_epoch']}, "
          f"report-validation macro AUC {checkpoint['best_raw_auc']:.6f}", flush=True)

    def collate_with_ids(items):
        inputs = collate([{key: value for key, value in item.items() if key != "study_id"}
                          for item in items])
        return [item["study_id"] for item in items], inputs

    loader = DataLoader(GoldStudies(study_ids, by_study), batch_size=2, shuffle=False,
                        num_workers=0, collate_fn=collate_with_ids)
    predictions = {}
    with torch.inference_mode():
        for ids, batch in loader:
            values = [tensor.to(DEVICE, non_blocking=True) for tensor in batch]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = reader(*values)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            if probs.shape != (len(ids), len(TARGETS)) or not np.isfinite(probs).all():
                raise RuntimeError(f"Invalid prediction array shape or values: {probs.shape}")
            for sid, row in zip(ids, probs):
                predictions[sid] = row

    if set(predictions) != set(study_ids):
        raise RuntimeError("Predictions are missing or contain extra studies")
    output = pd.DataFrame({
        "StudyInstanceUID": study_ids,
        **{name: [float(predictions[sid][j]) for sid in study_ids]
           for j, name in enumerate(TARGETS)},
    })
    if list(output.columns) != list(sample.columns):
        raise RuntimeError("Generated prediction columns do not exactly match sample_submission.csv")
    values = output[TARGETS].to_numpy(dtype=np.float64)
    if output.shape != sample.shape or not np.isfinite(values).all() \
            or np.any(values < 0) or np.any(values > 1):
        raise RuntimeError("Submission failed shape, finite-value, or probability-range validation")
    if output["StudyInstanceUID"].tolist() != study_ids:
        raise RuntimeError("Submission row order does not match sample_submission.csv")

    output_path = Path("/kaggle/working/submission.csv")
    output.to_csv(output_path, index=False, float_format="%.9g")
    # Round-trip the actual file before it can be used for official submission.
    check = pd.read_csv(output_path, dtype={"StudyInstanceUID": str})
    if list(check.columns) != list(sample.columns) or check.shape != sample.shape \
            or check["StudyInstanceUID"].tolist() != study_ids:
        raise RuntimeError("Saved CSV does not match the competition sample-submission schema")
    check_values = check[TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(check_values).all() or np.any(check_values < 0) or np.any(check_values > 1):
        raise RuntimeError("Saved CSV contains invalid probability values")

    summary = {
        "experiment": "pillar0_regional_reader_single_model_submission",
        "training_performed": False,
        "checkpoint_fold": 0,
        "checkpoint_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_report_validation_macro_auc": float(checkpoint["best_raw_auc"]),
        "gold_evaluation_macro_auc": 0.7856011957598646,
        "test_studies": len(study_ids),
        "test_series": int(len(test_series)),
        "submission_shape": list(check.shape),
        "submission_columns_match_sample": True,
        "all_predictions_finite_in_0_1": True,
        "output": str(output_path),
        "raw_dicom_transferred": False,
    }
    (WORK / "submission_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Submission file passed schema and value checks:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    source = EVAL_RUNNER.read_text(encoding="utf-8")
    marker = "\ndef main():\n"
    if marker not in source:
        raise RuntimeError("Could not find reusable inference helpers in gold-evaluation runner")
    helper_source = source.split(marker, 1)[0]
    helper_source = helper_source.replace(
        'WORK = Path("/kaggle/working/pillar0_regional_gold_eval")',
        'WORK = Path("/kaggle/working/pillar0_regional_submission")',
    )
    OUTPUT.write_text(helper_source.rstrip() + "\n\n" + SUBMISSION_MAIN.lstrip(), encoding="utf-8")
    metadata = {
        "id": "tumis001/rsna-pillar-0-regional-reader-submission",
        "title": "RSNA Pillar-0 Regional Reader Submission",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["tumis001/rsna-pillar0-regional-reader-cv-v1"],
        "kernel_sources": ["tumis001/pillar0-breastmri-offline-bundle"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (HERE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
