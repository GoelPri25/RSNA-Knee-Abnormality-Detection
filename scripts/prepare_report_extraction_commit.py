"""Prepare the committed Kaggle report extraction notebook.

Usage: python scripts/prepare_report_extraction_commit.py INPUT_IPYNB OUTPUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


CELL_6 = r'''from collections import Counter
import gc
import os
import sys
import time


# Expand deterministic schema normalization using the vocabulary observed
# during the completed audit run. The frozen prompt remains unchanged.
SEVERITY_ALIASES.update(
    {
        "trace": "mild",
        "minimal": "mild",
        "minor": "mild",
        "small": "mild",
        "low": "mild",
        "incipient": "mild",
        "early": "mild",
        "small_to_moderate": "mild_to_moderate",
        "moderate_to_large": "moderate_to_severe",
        "mild_to_severe": "moderate_to_severe",
        "large": "severe",
        "massive": "severe",
        "high_grade_partial_tear": "severe",
        "complete_radial_tear": "severe",
        "chronic": "unspecified",
        "degenerative": "unspecified",
        "ruptured": "unspecified",
        "subacute": "unspecified",
    }
)

STATUS_SEVERITY_ALIASES = {
    "mild": "mild",
    "mild_to_moderate": "mild_to_moderate",
    "moderate": "moderate",
    "moderate_to_severe": "moderate_to_severe",
    "severe": "severe",
}


def validate_and_normalize_extraction(parsed):
    if not isinstance(parsed, dict):
        raise ValueError("Response JSON must be an object")
    if set(parsed) != set(TARGETS):
        raise ValueError("Target keys do not match the required target set")

    normalized = {}
    corrections = []

    for target in TARGETS:
        result = parsed[target]
        if not isinstance(result, dict):
            raise ValueError(f"Result for {target} must be an object")
        if set(result) != {"status", "severity", "negated"}:
            raise ValueError(f"Invalid fields for {target}: {result}")

        original_status = str(result["status"]).strip().lower()
        original_severity = result["severity"]
        severity = normalize_severity(original_severity)
        negated = result["negated"]
        status = original_status

        if status in STATUS_SEVERITY_ALIASES:
            derived_severity = STATUS_SEVERITY_ALIASES[status]
            corrections.append(
                {
                    "target": target,
                    "field": "status",
                    "original": original_status,
                    "normalized": "present",
                    "reason": "severity value was emitted in the status field",
                }
            )
            status = "present"
            if severity in {"none", "unspecified"}:
                corrections.append(
                    {
                        "target": target,
                        "field": "severity",
                        "original": severity,
                        "normalized": derived_severity,
                        "reason": "severity recovered from the status field",
                    }
                )
                severity = derived_severity

        if status not in STATUS_VALUES:
            raise ValueError(f"Invalid status for {target}: {original_status}")
        if severity not in SEVERITY_VALUES:
            raise ValueError(
                f"Invalid severity for {target}: {original_severity}"
            )
        if not isinstance(negated, bool):
            raise ValueError(f"Invalid negated value for {target}")

        if severity != original_severity:
            corrections.append(
                {
                    "target": target,
                    "field": "severity",
                    "original": original_severity,
                    "normalized": severity,
                    "reason": "normalized severity vocabulary",
                }
            )
        if negated and status != "absent":
            corrections.append(
                {
                    "target": target,
                    "field": "negated",
                    "original": True,
                    "normalized": False,
                    "reason": f"negated=True is incompatible with status={status}",
                }
            )
            negated = False
        if status in {"absent", "not_mentioned"} and severity != "none":
            corrections.append(
                {
                    "target": target,
                    "field": "severity",
                    "original": severity,
                    "normalized": "none",
                    "reason": f"severity is incompatible with status={status}",
                }
            )
            severity = "none"
        if status == "present" and severity == "none":
            corrections.append(
                {
                    "target": target,
                    "field": "severity",
                    "original": "none",
                    "normalized": "unspecified",
                    "reason": "present pathology cannot have severity=none",
                }
            )
            severity = "unspecified"

        normalized[target] = {
            "status": status,
            "severity": severity,
            "negated": negated,
        }

    return normalized, corrections


CHECKPOINT_PATH = OUTPUT_DIR / "report_extractions.jsonl"
PROGRESS_PATH = OUTPUT_DIR / "progress.json"
FIRST_PASS_PATH = OUTPUT_DIR / "report_extractions_first_pass.jsonl"
PRIMARY_CHUNK_SIZE = 16


def write_records_atomic(records, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def write_progress(progress):
    temporary_path = PROGRESS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    temporary_path.replace(PROGRESS_PATH)


adaptive_split_count = 0


def clear_oom_state():
    for attribute_name in ("last_type", "last_value", "last_traceback", "last_exc"):
        if hasattr(sys, attribute_name):
            setattr(sys, attribute_name, None)
    gc.collect()
    for device_index in range(2):
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()


def run_adaptive_chunk(chunk_frame):
    global adaptive_split_count
    try:
        return run_dual_gpu_chunk(chunk_frame)
    except torch.OutOfMemoryError:
        clear_oom_state()
        if len(chunk_frame) <= 2:
            raise
        adaptive_split_count += 1
        split_at = len(chunk_frame) // 2

    # Recurse after leaving the exception block so the failed generation
    # traceback cannot retain its CUDA tensors during the smaller retry.
    clear_oom_state()
    print(
        f"OOM for {len(chunk_frame)} studies; retrying as "
        f"{split_at} + {len(chunk_frame) - split_at}",
        flush=True,
    )
    left = run_adaptive_chunk(chunk_frame.iloc[:split_at])
    right = run_adaptive_chunk(chunk_frame.iloc[split_at:])
    return left + right


all_records = read_checkpoint(CHECKPOINT_PATH)
completed_uids = {record["study_uid"] for record in all_records}
remaining_frame = work_frame[
    ~work_frame["StudyInstanceUID"].isin(completed_uids)
].reset_index(drop=True)

production_start = time.perf_counter()
total_chunks = (
    len(remaining_frame) + PRIMARY_CHUNK_SIZE - 1
) // PRIMARY_CHUNK_SIZE

print("Committed production extraction")
print("Already checkpointed:", len(all_records))
print("Remaining studies:", len(remaining_frame))
print("Primary reports per GPU:", PRIMARY_CHUNK_SIZE // 2)
print("Primary chunks:", total_chunks)
print(flush=True)

for chunk_start in range(0, len(remaining_frame), PRIMARY_CHUNK_SIZE):
    chunk_number = chunk_start // PRIMARY_CHUNK_SIZE + 1
    chunk_frame = remaining_frame.iloc[
        chunk_start:chunk_start + PRIMARY_CHUNK_SIZE
    ]
    chunk_records = run_adaptive_chunk(chunk_frame)

    if {r["study_uid"] for r in chunk_records} != set(
        chunk_frame["StudyInstanceUID"]
    ):
        raise RuntimeError("Chunk returned incorrect study IDs")

    all_records.extend(chunk_records)
    write_records_atomic(all_records, CHECKPOINT_PATH)

    if chunk_number == 1 or chunk_number % 10 == 0 or chunk_number == total_chunks:
        elapsed = time.perf_counter() - production_start
        failures = sum(record["error"] is not None for record in all_records)
        rate = elapsed / max(chunk_start + len(chunk_frame), 1)
        remaining = len(remaining_frame) - chunk_start - len(chunk_frame)
        print(
            f"Chunk {chunk_number:03d}/{total_chunks:03d} | "
            f"checkpointed {len(all_records)}/{len(work_frame)} | "
            f"failures {failures} | remaining {rate * remaining / 3600:.2f}h",
            flush=True,
        )
        write_progress(
            {
                "total_studies": len(work_frame),
                "completed_studies": len(all_records),
                "failed_studies": failures,
                "estimated_remaining_hours": rate * remaining / 3600,
                "adaptive_split_count": adaptive_split_count,
                "complete": False,
            }
        )

if len(all_records) != len(work_frame):
    raise RuntimeError("First pass did not produce one record per study")

write_records_atomic(all_records, FIRST_PASS_PATH)

# Retry only parse failures, one report per GPU. This changes batch context but
# keeps the frozen prompt and model revision unchanged.
failed_uids = {
    record["study_uid"] for record in all_records if record["error"] is not None
}
retry_records = []
if failed_uids:
    retry_frame = work_frame[
        work_frame["StudyInstanceUID"].isin(failed_uids)
    ].reset_index(drop=True)
    print("Retrying parse failures:", len(retry_frame), flush=True)
    for start in range(0, len(retry_frame), 2):
        records = run_adaptive_chunk(retry_frame.iloc[start:start + 2])
        for record in records:
            record["retry_attempt"] = 1
            record["parser_version"] = "expanded_clinical_severity_v1"
        retry_records.extend(records)

    retry_by_uid = {record["study_uid"]: record for record in retry_records}
    all_records = [
        retry_by_uid.get(record["study_uid"], record) for record in all_records
    ]
    write_records_atomic(all_records, CHECKPOINT_PATH)

production_elapsed_seconds = time.perf_counter() - production_start
successful_records = [r for r in all_records if r["error"] is None]
failed_records = [r for r in all_records if r["error"] is not None]

extractor_config["parser_version"] = "expanded_clinical_severity_v1"
extractor_config["severity_aliases"] = SEVERITY_ALIASES
extractor_config["status_severity_aliases"] = STATUS_SEVERITY_ALIASES
extractor_config["adaptive_oom_splitting"] = True
config_path.write_text(json.dumps(extractor_config, indent=2), encoding="utf-8")

write_progress(
    {
        "total_studies": len(all_records),
        "completed_studies": len(all_records),
        "successful_studies": len(successful_records),
        "failed_studies": len(failed_records),
        "semantic_corrections": sum(
            len(record["corrections"]) for record in successful_records
        ),
        "runtime_seconds": production_elapsed_seconds,
        "adaptive_split_count": adaptive_split_count,
        "retried_studies": len(retry_records),
        "complete": True,
    }
)

print()
print("Production extraction complete.")
print("Checkpoint records:", len(all_records))
print("Successful reports:", len(successful_records))
print("Failed reports:", len(failed_records))
print("Adaptive OOM splits:", adaptive_split_count)
print("Runtime hours:", round(production_elapsed_seconds / 3600, 2))
'''


CELL_7 = r'''from collections import Counter
import shutil


STRUCTURED_CSV_PATH = OUTPUT_DIR / "report_structured_labels.csv"
FAILURES_CSV_PATH = OUTPUT_DIR / "failed_extractions.csv"
SUMMARY_PATH = OUTPUT_DIR / "extraction_summary.json"
RUN_METADATA_PATH = OUTPUT_DIR / "extraction_run.json"
ARCHIVE_BASE = Path("/kaggle/working/report_label_extraction_complete")

records = read_checkpoint(CHECKPOINT_PATH)
if len(records) != len(train):
    raise RuntimeError(f"Expected {len(train)} records, found {len(records)}")
if len({record["study_uid"] for record in records}) != len(records):
    raise RuntimeError("Checkpoint contains duplicate study IDs")

target_slugs = {
    "ACL": "acl",
    "MCL": "mcl",
    "Medial Meniscus": "medial_meniscus",
    "Lateral Meniscus": "lateral_meniscus",
    "Medial OA": "medial_oa",
    "Lateral OA": "lateral_oa",
    "PF OA": "pf_oa",
    "Effusion": "effusion",
    "Synovitis": "synovitis",
    "Baker's": "bakers",
    "Contusion": "contusion",
    "Fracture": "fracture",
}

flat_rows = []
failure_rows = []
status_counts = {target: Counter() for target in TARGETS}
error_counts = Counter()

for record in records:
    success = record["error"] is None
    row = {
        "StudyInstanceUID": record["study_uid"],
        "extraction_success": success,
        "error": record["error"],
        "response_sha256": record["response_sha256"],
        "prompt_tokens": record["prompt_tokens"],
        "device_index": record["device_index"],
        "generation_batch_size": record["batch_size"],
        "correction_count": len(record["corrections"]),
        "prompt_name": record["prompt_name"],
        "prompt_sha256": record["prompt_sha256"],
        "model_revision": record["model_revision"],
        "retry_attempt": record.get("retry_attempt", 0),
    }
    if not success:
        error_counts[record["error"]] += 1
        failure_rows.append(
            {
                "StudyInstanceUID": record["study_uid"],
                "error": record["error"],
                "response_sha256": record["response_sha256"],
                "prompt_tokens": record["prompt_tokens"],
            }
        )

    for target in TARGETS:
        slug = target_slugs[target]
        if success:
            prediction = record["predictions"][target]
            status = prediction["status"]
            status_counts[target][status] += 1
            label = 1.0 if status == "present" else 0.0 if status == "absent" else float("nan")
            mask = int(status in {"present", "absent"})
            row[f"{slug}__status"] = status
            row[f"{slug}__severity"] = prediction["severity"]
            row[f"{slug}__negated"] = prediction["negated"]
            row[f"{slug}__pseudo_label"] = label
            row[f"{slug}__label_mask"] = mask
        else:
            row[f"{slug}__status"] = None
            row[f"{slug}__severity"] = None
            row[f"{slug}__negated"] = None
            row[f"{slug}__pseudo_label"] = float("nan")
            row[f"{slug}__label_mask"] = 0
    flat_rows.append(row)

structured_frame = pd.DataFrame(flat_rows)
failure_frame = pd.DataFrame(failure_rows)
structured_frame.to_csv(STRUCTURED_CSV_PATH, index=False)
failure_frame.to_csv(FAILURES_CSV_PATH, index=False)

summary = {
    "total_studies": len(records),
    "successful_studies": int(structured_frame["extraction_success"].sum()),
    "failed_studies": int((~structured_frame["extraction_success"]).sum()),
    "failure_rate": float((~structured_frame["extraction_success"]).mean()),
    "semantic_corrections": int(structured_frame["correction_count"].sum()),
    "status_counts_by_target": {
        target: dict(status_counts[target]) for target in TARGETS
    },
    "error_counts": dict(error_counts),
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

batch_distribution = Counter(record["batch_size"] for record in records)
run_metadata = {
    "model_id": EXPECTED_MODEL_ID,
    "model_revision": EXPECTED_REVISION,
    "prompt_name": PROMPT_NAME,
    "prompt_sha256": PROMPT_SHA256,
    "parser_version": "expanded_clinical_severity_v1",
    "total_studies": len(records),
    "gpu_count": 2,
    "gpu_model": torch.cuda.get_device_name(0),
    "runtime_seconds": production_elapsed_seconds,
    "runtime_hours": production_elapsed_seconds / 3600,
    "adaptive_oom_splits": adaptive_split_count,
    "per_gpu_batch_distribution": dict(batch_distribution),
    "stores_report_text": False,
    "stores_raw_model_response": False,
}
RUN_METADATA_PATH.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

archive_path = shutil.make_archive(
    str(ARCHIVE_BASE), "zip", root_dir=OUTPUT_DIR
)

print("Checkpoint records:", len(records))
print("Successful reports:", summary["successful_studies"])
print("Failed reports:", summary["failed_studies"])
print("Failure rate:", f"{100 * summary['failure_rate']:.2f}%")
print("Semantic corrections:", summary["semantic_corrections"])
print("Runtime hours:", round(production_elapsed_seconds / 3600, 2))
print("Adaptive OOM splits:", adaptive_split_count)
print("Archive:", archive_path)
print("Archive size (MB):", round(Path(archive_path).stat().st_size / 1024**2, 2))
if error_counts:
    print("Failure types:")
    for error, count in error_counts.most_common():
        print(f" - {count} x {error}")
'''


CELL_8 = r'''import zipfile


archive_path = Path("/kaggle/working/report_label_extraction_complete.zip")
required_outputs = {
    "extractor_config.json",
    "report_extractions.jsonl",
    "report_structured_labels.csv",
    "failed_extractions.csv",
    "extraction_summary.json",
    "extraction_run.json",
    "progress.json",
}

if not archive_path.is_file():
    raise FileNotFoundError(archive_path)

with zipfile.ZipFile(archive_path) as archive:
    archived_names = set(archive.namelist())

missing_outputs = required_outputs - archived_names
if missing_outputs:
    raise RuntimeError(f"Archive is missing outputs: {sorted(missing_outputs)}")

print("Committed artifact verification passed.")
print("Archive:", archive_path)
print("Archive size (MB):", round(archive_path.stat().st_size / 1024**2, 2))
print("Archived files:", sorted(archived_names))
'''


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepare_report_extraction_commit.py INPUT OUTPUT_DIR")
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    notebook = json.loads(input_path.read_text(encoding="utf-8"))
    if len(notebook["cells"]) < 8:
        raise RuntimeError("Expected at least eight notebook cells")

    replacements = {5: CELL_6, 6: CELL_7, 7: CELL_8}
    notebook["cells"] = notebook["cells"][:8]
    for index, text in replacements.items():
        notebook["cells"][index]["cell_type"] = "code"
        notebook["cells"][index]["source"] = source_lines(text)
        notebook["cells"][index]["outputs"] = []
        notebook["cells"][index]["execution_count"] = None
        notebook["cells"][index]["metadata"] = {}

    for cell in notebook["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report-label-extraction.ipynb"
    output_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    metadata = json.loads(
        (input_path.parent / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    metadata["enable_internet"] = False
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
