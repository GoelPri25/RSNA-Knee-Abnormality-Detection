from __future__ import annotations

import json
import csv
from pathlib import Path


OUT = Path(__file__).parent
REPO_ROOT = OUT.parents[2]
SPLIT_CONFIG = (
    REPO_ROOT
    / "kaggle/experiment_kernels/cnn_slice_series_reader/source/skanee/configs/gold_split_v1.csv"
)
with SPLIT_CONFIG.open("r", encoding="utf-8", newline="") as handle:
    split_records = [
        (row["StudyInstanceUID"], row["split"])
        for row in csv.DictReader(handle)
    ]
if len(split_records) != 58 or sum(split == "development" for _, split in split_records) != 29:
    raise ValueError(f"Unexpected fixed gold split in {SPLIT_CONFIG}")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    md(r'''# Gold-calibrated report-label audit for Steven v4

**CPU notebook; cloud API inference; no DICOM access and no model training.** This first stage tests whether few-shot examples from real development-gold report/label pairs help a report reader predict the challenge's image-assessed labels on unseen development-gold reports. The 29 locked evaluation-gold studies are excluded before their reports or labels are accessed.

The model is **DeepSeek V4.1 Flash on Ollama Cloud**, called directly at `https://ollama.com/api/chat`. It is not downloaded or run inside Kaggle. Add the API key in Kaggle Notebook Secrets as `OLLAMA_API_KEY`; never paste it into a cell. Internet must be enabled for this CPU notebook. The LLM is an offline label-generation tool only; it is not part of the competition submission model.

This run skips the already completed development-gold cross-fit and generates a separate candidate-label file for the 4,349 report-only studies. It also compares those candidates directly with Steven v4, especially where the report does not mention a target. Steven v4 remains untouched.

**Data handling:** report text and development-gold examples are sent to Ollama Cloud for inference. The notebook does not send DICOMs, study IDs, locked evaluation rows, or locked evaluation labels.'''),
    code(r'''from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
MODEL = "deepseek-v4.1-flash:cloud"
API_URL = "https://ollama.com/api/chat"
WORK = Path("/kaggle/working/steven_v4_fewshot_calibration")
WORK.mkdir(parents=True, exist_ok=True)
CANDIDATE_JSONL = WORK / "report_only_candidate_labels.jsonl"
PREDICTIONS_JSONL = WORK / "development_oof_predictions.jsonl"
METRICS_JSON = WORK / "development_oof_metrics.json"
METRICS_CSV = WORK / "development_oof_per_target.csv"
REQUEST_TIMEOUT_SECONDS = 240
MAX_RETRIES = 6
TEMPERATURE = 0.0

print("Execution: Kaggle CPU | local Ollama runtime: not used")
print("Inference endpoint:", API_URL, "| model:", MODEL)
print("Accelerator requested: CPU only")

try:
    from kaggle_secrets import UserSecretsClient
    OLLAMA_API_KEY = UserSecretsClient().get_secret("OLLAMA_API_KEY")
except Exception as exc:
    raise RuntimeError(
        "Add an OLLAMA_API_KEY secret in Kaggle Notebook settings, then rerun. "
        "Do not paste the key into the notebook."
    ) from exc
if not OLLAMA_API_KEY or len(OLLAMA_API_KEY.strip()) < 10:
    raise RuntimeError("Kaggle secret OLLAMA_API_KEY is empty or malformed.")

INPUT = Path("/kaggle/input")
RESUME_CANDIDATE_PATHS = [
    INPUT / "datasets/tumis001/rsna-fewshot-candidate-label-checkpoints-v1/report_only_candidate_labels.jsonl",
    INPUT / "rsna-fewshot-candidate-label-checkpoints-v1/report_only_candidate_labels.jsonl",
]
if not CANDIDATE_JSONL.is_file():
    resume_source = next((p for p in RESUME_CANDIDATE_PATHS if p.is_file()), None)
    if resume_source is not None:
        CANDIDATE_JSONL.write_bytes(resume_source.read_bytes())
        print("Restored prior private Kaggle checkpoint:", resume_source)
    else:
        print("No prior candidate checkpoint mounted; this will start a fresh candidate pass.")
COMPETITION_ROOTS = [
    INPUT / "competitions" / "rsna-knee-abnormality-detection",
    INPUT / "rsna-knee-abnormality-detection",
]
def exactly_one_existing(paths, description):
    found = [p for p in paths if p.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {description}; found: {found}")
    return found[0]

TRAIN_CSV = exactly_one_existing(
    [p / "train.csv" for p in COMPETITION_ROOTS], "competition train.csv"
)
# These fixed IDs come from the versioned gold_split_v1.csv. Embedding only
# assignments avoids depending on Kaggle's mounted notebook-output path.
SPLIT_RECORDS = GOLD_SPLIT_RECORDS_PLACEHOLDER
split = pd.DataFrame(SPLIT_RECORDS, columns=["StudyInstanceUID", "split"])
required_split = {"StudyInstanceUID", "split"}
if not required_split.issubset(split.columns):
    raise ValueError(f"Gold split is missing columns: {required_split - set(split.columns)}")
dev_ids = set(split.loc[split["split"].eq("development"), "StudyInstanceUID"].astype(str))
locked_eval_ids = set(split.loc[split["split"].eq("evaluation"), "StudyInstanceUID"].astype(str))
if len(dev_ids) != 29 or len(locked_eval_ids) != 29 or dev_ids & locked_eval_ids:
    raise ValueError(
        f"Unexpected gold split: development={len(dev_ids)}, locked={len(locked_eval_ids)}, "
        f"overlap={len(dev_ids & locked_eval_ids)}"
    )

# Stream train.csv by row. Locked evaluation rows are skipped before reading their report
# or target values. We load only development gold targets and report-only training reports.
development_gold = {}
report_only = []
with TRAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    required = {"StudyInstanceUID", "Report", *TARGETS}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"train.csv is missing fields: {required - set(reader.fieldnames or [])}")
    for row in reader:
        uid = str(row["StudyInstanceUID"])
        if uid in locked_eval_ids:
            continue
        report = (row.get("Report") or "").strip()
        if uid in dev_ids:
            if not report:
                raise ValueError(f"Development gold report is empty for {uid}")
            values = []
            for target in TARGETS:
                raw = (row.get(target) or "").strip()
                if raw not in {"0", "0.0", "1", "1.0"}:
                    raise ValueError(f"Development gold target {target} is not binary for {uid}")
                values.append(int(float(raw)))
            development_gold[uid] = {"report": report, "labels": values}
            continue
        values = [(row.get(target) or "").strip() for target in TARGETS]
        if any(values):
            raise ValueError(f"Unexpected labeled non-development study encountered: {uid}")
        if report:
            report_only.append({"uid": uid, "report": report})

if set(development_gold) != dev_ids:
    raise ValueError(
        f"Development reports/labels mismatch: missing={len(dev_ids-set(development_gold))}, "
        f"extra={len(set(development_gold)-dev_ids)}"
    )
if len(report_only) != 4349:
    raise ValueError(f"Expected 4,349 report-only studies after gold lock; found {len(report_only)}")
print("Development gold examples:", len(development_gold))
print("Locked evaluation gold rows accessed: 0")
print("Report-only studies available for a later candidate pass:", len(report_only))
print("Reports are held in memory only for this run; IDs are never sent to Ollama.")

STEVEN_LABELS_FILENAME = "llm_labels_v4_blend.csv"
STEVEN_LABELS_CANDIDATES = [
    INPUT / "datasets/stevenleehans/rsna-knee-llm-report-labels" / STEVEN_LABELS_FILENAME,
    INPUT / "stevenleehans/rsna-knee-llm-report-labels" / STEVEN_LABELS_FILENAME,
    INPUT / "rsna-knee-llm-report-labels" / STEVEN_LABELS_FILENAME,
]
STEVEN_LABELS_CSV = next((p for p in STEVEN_LABELS_CANDIDATES if p.is_file()), None)
if STEVEN_LABELS_CSV is None:
    mounted_inputs = sorted(p.name for p in INPUT.iterdir()) if INPUT.is_dir() else []
    raise FileNotFoundError(
        "Steven v4 CSV was not found at known Kaggle dataset mount paths. "
        f"Checked: {[str(p) for p in STEVEN_LABELS_CANDIDATES]}; "
        f"top-level /kaggle/input entries: {mounted_inputs}"
    )
print("Steven v4 comparison labels:", STEVEN_LABELS_CSV)
'''),
    md(r'''## Cloud API smoke test

This request contains no competition data. It confirms the secret, internet connection, exact cloud model tag, and JSON response before any report is sent.'''),
    code(r'''def ollama_chat(messages, *, max_tokens=1600, return_usage=False):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        # We need the final structured answer, not a separate reasoning trace.
        # With thinking enabled, the model can spend its output budget in
        # message.thinking and return an empty message.content.
        "think": False,
        "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
    }
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
    }
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
            if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
                wait = min(60, 2 ** attempt)
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = min(120, int(retry_after))
                print(f"Transient Ollama HTTP {response.status_code}; retrying in {wait}s")
                time.sleep(wait)
                continue
            response.raise_for_status()
            body = response.json()
            message = body.get("message") or {}
            content = message.get("content", "")
            if not isinstance(content, str) or not content.strip():
                diagnostics = {
                    "message_fields": sorted(message.keys()),
                    "done_reason": body.get("done_reason"),
                    "eval_count": body.get("eval_count"),
                    "thinking_chars": len(message.get("thinking") or ""),
                }
                raise ValueError(f"Ollama returned empty final content: {diagnostics}")
            usage = {
                "input_tokens": int(body.get("prompt_eval_count") or 0),
                "cached_input_tokens": int(body.get("prompt_eval_cached_count") or 0),
                "output_tokens": int(body.get("eval_count") or 0),
            }
            return (content, usage) if return_usage else content
        except requests.HTTPError as exc:
            # The retryable HTTP statuses above are handled explicitly. Other
            # client errors (e.g. auth/model/schema) need correction, not retry.
            raise RuntimeError(
                f"Ollama returned non-retryable HTTP {response.status_code}: "
                f"{response.text[:500]}"
            ) from exc
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 == MAX_RETRIES:
                break
            wait = min(60, 2 ** attempt)
            print(f"Transient Ollama transport error ({type(exc).__name__}); retrying in {wait}s")
            time.sleep(wait)
        except ValueError:
            # An empty/malformed model response is not a transport failure.
            # Surface it immediately instead of repeating a billable request.
            raise
    raise RuntimeError(f"Ollama API failed after {MAX_RETRIES} attempts") from last_error

smoke = ollama_chat(
    [{"role": "user", "content": 'Return exactly this JSON object: {"ok": true}'}],
    max_tokens=80,
)
print("Cloud API smoke-test response:", smoke[:300])
'''),
    md(r'''## Cross-fitted few-shot predictions on development gold

For each development case, the prompt contains the other development-gold report/label pairs, but never that case or any identical normalized report. The locked 29-case evaluation split is not used. The prompt asks for challenge-target probabilities, explicit textual status, and short evidence snippets; it does not ask the model to reveal private chain-of-thought.'''),
    code(r'''LABEL_GUIDE = """
Target rubric (apply the challenge's image-label definitions as closely as report text permits):
- ACL: high-grade partial/full tear; mild signal or degeneration alone is negative.
- MCL: high-grade acute tear; low-grade sprain or chronic change is negative.
- Medial/Lateral Meniscus: definite surface-reaching tear or displaced/truncated morphology; intrasubstance degeneration alone is negative.
- Medial/Lateral/PF OA: moderate/large region of high-grade cartilage loss (>50% thickness, roughly >=1 cm).
- Effusion: moderate or large joint fluid, not merely trace/small fluid.
- Synovitis: explicitly described synovial inflammation/thickening.
- Baker's: moderate/large Baker/popliteal cyst.
- Contusion: bone contusion / impact-related marrow edema-like finding, not nonspecific edema alone.
- Fracture: acute fracture line/cortical break; do not equate every osteochondral or chronic lesion with acute fracture.
Borderline findings are negative under the challenge rubric. Reports are clinical text and may omit findings or use different thresholds than the image labels.
""".strip()

STATUSES = {"affirmed", "negated", "uncertain", "not_mentioned"}

def normalized_report(text):
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)

def stable_report_hash(text):
    return hashlib.sha256(normalized_report(text).encode("utf-8")).hexdigest()

def build_messages(examples, report_items):
    system = """You map multilingual knee MRI report text to the competition's twelve image-assessed targets.
Use the label rubric below and the labeled examples as calibration demonstrations. The gold targets are image-derived; a report can omit a finding or use different thresholds.
Do not assume not-mentioned means absent. Use the examples to calibrate how report language maps to the official target. Do not invent evidence. Do not output chain-of-thought. Return only JSON matching the requested shape.

""" + LABEL_GUIDE + """

For every queried report return one object with: report_key; labels, containing every target, each with p (probability 0..1 that the official challenge target is positive), status (affirmed|negated|uncertain|not_mentioned), and evidence (short exact phrase or empty string). For not-mentioned findings, use the few-shot calibration to estimate p, but keep evidence empty."""
    demonstrations = []
    for i, example in enumerate(examples, 1):
        target_map = {t: int(v) for t, v in zip(TARGETS, example["labels"])}
        demonstrations.append(
            f"Example {i}\nREPORT:\n{example['report']}\nOFFICIAL_GOLD_LABELS:\n"
            + json.dumps(target_map, ensure_ascii=False, separators=(",", ":"))
        )
    queries = []
    for item in report_items:
        queries.append(f"REPORT_KEY: {item['key']}\nREPORT:\n{item['report']}")
    user = (
        "Calibration examples:\n\n" + "\n\n".join(demonstrations)
        + "\n\nNow label these unseen reports. Return a JSON object with a results array. "
        + "Each result must include its exact report_key and a labels object containing every target.\n\n"
        + "\n\n".join(queries)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def parse_json_object(content):
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Could not locate JSON in model response")
        value = json.loads(content[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from Ollama")
    return value

def validate_result(result, expected_keys):
    if "results" not in result or not isinstance(result["results"], list):
        raise ValueError("Response must contain a results array")
    validated = {}
    for record in result["results"]:
        key = record.get("report_key")
        if key not in expected_keys or key in validated:
            raise ValueError(f"Unexpected or duplicate report_key: {key}")
        labels = record.get("labels")
        if not isinstance(labels, dict) or set(labels) != set(TARGETS):
            raise ValueError(f"Response target set mismatch for {key}")
        clean = {}
        for target in TARGETS:
            item = labels[target]
            p = float(item["p"])
            status = str(item["status"]).strip().lower()
            evidence = str(item.get("evidence", ""))[:500]
            if not math.isfinite(p) or not 0 <= p <= 1 or status not in STATUSES:
                raise ValueError(f"Invalid p/status for {key}, {target}")
            clean[target] = {"p": p, "status": status, "evidence": evidence}
        validated[key] = clean
    if set(validated) != set(expected_keys):
        raise ValueError(f"Missing report keys: {set(expected_keys)-set(validated)}")
    return validated

def load_completed_predictions():
    rows = {}
    if not PREDICTIONS_JSONL.exists():
        return rows
    for line in PREDICTIONS_JSONL.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        rows[str(record["uid"])] = record
    return rows

RUN_GOLD_CROSSFIT = False  # Version 5 completed this stage; avoid duplicate API calls.
completed = load_completed_predictions() if RUN_GOLD_CROSSFIT else {}
if RUN_GOLD_CROSSFIT:
    with PREDICTIONS_JSONL.open("a", encoding="utf-8") as progress:
        for index, (uid, item) in enumerate(development_gold.items(), 1):
            if uid in completed:
                print(f"Resume: already predicted {index}/{len(development_gold)}")
                continue
            own_hash = stable_report_hash(item["report"])
            examples = [
                {"report": other["report"], "labels": other["labels"]}
                for other_uid, other in development_gold.items()
                if other_uid != uid and stable_report_hash(other["report"]) != own_hash
            ]
            if len(examples) < 5:
                raise RuntimeError(f"Too few distinct few-shot examples for {uid}: {len(examples)}")
            report_key = "QUERY_1"
            messages = build_messages(examples, [{"key": report_key, "report": item["report"]}])
            content = ollama_chat(messages)
            parsed = parse_json_object(content)
            output = validate_result(parsed, {report_key})[report_key]
            record = {
                "uid": uid,
                "gold": {t: int(v) for t, v in zip(TARGETS, item["labels"])},
                "prediction": output,
                "model": MODEL,
                "few_shot_examples": len(examples),
                "report_hash": own_hash,
            }
            progress.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
            completed[uid] = record
            print(f"Few-shot OOF {index}/{len(development_gold)} completed")
else:
    print("Skipping completed development-gold cross-fit; no duplicate requests.")
'''),
    code(r'''def score_development_oof(records):
    rows = []
    for target in TARGETS:
        y = np.asarray([r["gold"][target] for r in records], dtype=int)
        p = np.asarray([r["prediction"][target]["p"] for r in records], dtype=float)
        auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
        rows.append({
            "target": target,
            "n": int(len(y)),
            "positive_n": int(y.sum()),
            "negative_n": int((1-y).sum()),
            "auc": auc,
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1-1e-6), labels=[0, 1])),
            "accuracy_at_0_5": float(((p >= .5).astype(int) == y).mean()),
        })
    per_target = pd.DataFrame(rows)
    result = {
        "model": MODEL,
        "split": "29 development-gold reports; leave-one-report-group-out few-shot prompts",
        "locked_evaluation_gold_rows_accessed": 0,
        "macro_auc": float(per_target["auc"].mean()),
        "macro_brier": float(per_target["brier"].mean()),
        "macro_log_loss": float(per_target["log_loss"].mean()),
        "development_studies": len(records),
        "targets_with_defined_auc": int(per_target["auc"].notna().sum()),
        "per_target": rows,
    }
    return result, per_target

if RUN_GOLD_CROSSFIT:
    if set(completed) != set(development_gold):
        raise RuntimeError(
            f"Incomplete development predictions: {len(completed)}/{len(development_gold)}. "
            "Rerun this cell; completed requests are resumed from JSONL."
        )
    ordered_records = [completed[uid] for uid in development_gold]
    summary, per_target = score_development_oof(ordered_records)
    METRICS_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_target.to_csv(METRICS_CSV, index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_target"}, indent=2))
    display(per_target)
    print("Saved resumable predictions and metrics under:", WORK)
else:
    print("Gold cross-fit scoring skipped; completed metrics remain in version 5 output.")
'''),
    md(r'''## Candidate labels and direct diff against Steven v4

This run makes cloud API calls for all 4,349 report-only studies, using the 29 development-gold report/label pairs as demonstrations and batching four reports per request. Every completed response is flushed to a resumable JSONL checkpoint. UIDs stay local; only reports and gold demonstrations go to Ollama. The locked evaluation gold stays excluded. The outputs remain separate from Steven v4. At the end, the notebook compares soft-label deltas by target and highlights `not_mentioned` cases.'''),
    code(r'''RUN_FULL_REPORT_ONLY_CANDIDATE = True
CANDIDATE_JSONL = WORK / "report_only_candidate_labels.jsonl"
CANDIDATE_CSV = WORK / "report_only_candidate_labels.csv"
BATCH_SIZE = 4

if not RUN_FULL_REPORT_ONLY_CANDIDATE:
    print("Full report-only pass is disabled. Review the gold cross-fit metrics first.")
else:
    candidate_done = {}
    if CANDIDATE_JSONL.exists():
        for line in CANDIDATE_JSONL.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            candidate_done[str(row["uid"])] = row
    expected_report_ids = {r["uid"] for r in report_only}
    if not set(candidate_done).issubset(expected_report_ids):
        raise RuntimeError("Resume checkpoint contains IDs outside the report-only set")
    print(
        f"Resuming candidate generation: {len(candidate_done)}/{len(report_only)} "
        "report-only studies already have saved labels."
    )

    USAGE_JSONL = WORK / "ollama_usage.jsonl"
    USAGE_TOTALS = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
                    "output_tokens": 0, "estimated_cost_usd": 0.0}

    def request_once(keyed):
        messages = build_messages(
            [{"report": x["report"], "labels": x["labels"]}
             for x in development_gold.values()],
            keyed,
        )
        content, usage = ollama_chat(messages, max_tokens=3600, return_usage=True)
        cached = min(usage["input_tokens"], usage["cached_input_tokens"])
        uncached = usage["input_tokens"] - cached
        cost = (uncached * 0.15 + cached * 0.003 + usage["output_tokens"] * 0.60) / 1_000_000
        usage_record = {**usage, "estimated_cost_usd": cost}
        with USAGE_JSONL.open("a", encoding="utf-8") as usage_file:
            usage_file.write(json.dumps(usage_record) + "\\n")
            usage_file.flush()
            os.fsync(usage_file.fileno())
        for key in ("calls", "input_tokens", "cached_input_tokens", "output_tokens"):
            USAGE_TOTALS[key] += 1 if key == "calls" else usage[key]
        USAGE_TOTALS["estimated_cost_usd"] += cost
        print(
            "Ollama usage this run: "
            f"calls={USAGE_TOTALS['calls']} input={USAGE_TOTALS['input_tokens']} "
            f"cached_input={USAGE_TOTALS['cached_input_tokens']} "
            f"output={USAGE_TOTALS['output_tokens']} "
            f"estimated_cost=${USAGE_TOTALS['estimated_cost_usd']:.4f}"
        )
        parsed = parse_json_object(content)
        return validate_result(parsed, {x["key"] for x in keyed})

    def request_with_json_recovery(keyed, retries_left=1):
        try:
            return request_once(keyed)
        except (json.JSONDecodeError, ValueError) as error:
            if len(keyed) > 1:
                midpoint = len(keyed) // 2
                print(
                    f"Malformed/invalid JSON for a {len(keyed)}-report response; "
                    "retrying as smaller report batches."
                )
                left = request_with_json_recovery(keyed[:midpoint], retries_left)
                right = request_with_json_recovery(keyed[midpoint:], retries_left)
                return {**left, **right}
            if retries_left > 0:
                print("Malformed/invalid single-report JSON; retrying that report once.")
                return request_with_json_recovery(keyed, retries_left - 1)
            raise RuntimeError(
                f"Could not obtain valid JSON for {keyed[0]['key']} after bounded retries"
            ) from error

    with CANDIDATE_JSONL.open("a", encoding="utf-8") as progress:
        remaining = [r for r in report_only if r["uid"] not in candidate_done]
        total_batches = math.ceil(len(remaining) / BATCH_SIZE)
        for batch_index in range(total_batches):
            batch = remaining[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
            keyed = [
                {"key": f"REPORT_{i+1:02d}", "report": item["report"]}
                for i, item in enumerate(batch)
            ]
            predictions = request_with_json_recovery(keyed)
            for item, query in zip(batch, keyed):
                record = {
                    "uid": item["uid"],
                    "prediction": predictions[query["key"]],
                    "model": MODEL,
                    "few_shot_examples": len(development_gold),
                    "report_hash": stable_report_hash(item["report"]),
                }
                progress.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.flush()
                os.fsync(progress.fileno())
                candidate_done[item["uid"]] = record
            print(
                f"Candidate batches {batch_index+1}/{total_batches}; "
                f"saved {len(candidate_done)}/{len(report_only)} report-only studies"
            )

    if set(candidate_done) != {r["uid"] for r in report_only}:
        raise RuntimeError(
            f"Candidate pass incomplete: {len(candidate_done)}/{len(report_only)}. "
            "Rerun this cell to resume from the JSONL checkpoint."
        )

    candidate_rows = []
    for record in candidate_done.values():
        row = {"StudyInstanceUID": record["uid"]}
        for target in TARGETS:
            row[target] = record["prediction"][target]["p"]
            row[f"{target}__status"] = record["prediction"][target]["status"]
            row[f"{target}__evidence"] = record["prediction"][target]["evidence"]
        candidate_rows.append(row)
    pd.DataFrame(candidate_rows).sort_values("StudyInstanceUID").to_csv(
        CANDIDATE_CSV, index=False
    )
    print("Separate candidate labels saved to:", CANDIDATE_CSV)
    print("Raw resumable model responses saved to:", CANDIDATE_JSONL)

    # Compare soft targets by exact StudyInstanceUID; never modify Steven v4.
    # The status is the LLM's report-reading category, not an image-derived truth.
    steven = pd.read_csv(STEVEN_LABELS_CSV, dtype={"StudyInstanceUID": str})
    missing = {"StudyInstanceUID", *TARGETS} - set(steven.columns)
    if missing:
        raise ValueError(f"Steven v4 CSV is missing columns: {sorted(missing)}")
    if steven["StudyInstanceUID"].duplicated().any():
        raise ValueError("Steven v4 CSV contains duplicate study IDs")
    if set(steven["StudyInstanceUID"]) & (dev_ids | locked_eval_ids):
        raise ValueError("Steven v4 comparison file unexpectedly contains gold IDs")
    candidate = pd.read_csv(CANDIDATE_CSV, dtype={"StudyInstanceUID": str})
    joined = steven[["StudyInstanceUID", *TARGETS]].merge(
        candidate, on="StudyInstanceUID", how="outer",
        suffixes=("__steven_v4", "__candidate"), indicator=True,
    )
    if not joined["_merge"].eq("both").all() or len(joined) != len(report_only):
        raise RuntimeError(
            f"Candidate/Steven ID coverage differs: {joined['_merge'].value_counts().to_dict()}"
        )
    summary_rows, diff_rows = [], []
    for target in TARGETS:
        old = joined[f"{target}__steven_v4"].astype(float)
        new = joined[target].astype(float)
        status = joined[f"{target}__status"].astype(str)
        delta = new - old
        silent = status.eq("not_mentioned")
        summary_rows.append({
            "target": target,
            "mean_steven_v4": float(old.mean()),
            "mean_candidate": float(new.mean()),
            "mean_delta_candidate_minus_steven": float(delta.mean()),
            "mean_abs_delta": float(delta.abs().mean()),
            "changed_by_at_least_0_10": int((delta.abs() >= 0.10).sum()),
            "not_mentioned_count": int(silent.sum()),
            "silent_candidate_positive_count": int((silent & new.ge(0.5)).sum()),
            "silent_mean_steven_v4": float(old[silent].mean()) if silent.any() else float("nan"),
            "silent_mean_candidate": float(new[silent].mean()) if silent.any() else float("nan"),
        })
        for i in range(len(joined)):
            if abs(float(delta.iloc[i])) >= 0.10 or silent.iloc[i]:
                diff_rows.append({
                    "StudyInstanceUID": joined["StudyInstanceUID"].iloc[i],
                    "target": target,
                    "steven_v4": float(old.iloc[i]),
                    "candidate": float(new.iloc[i]),
                    "delta": float(delta.iloc[i]),
                    "status": status.iloc[i],
                    "evidence": joined[f"{target}__evidence"].iloc[i],
                })
    diff_summary = pd.DataFrame(summary_rows)
    diff_summary.to_csv(WORK / "steven_v4_candidate_diff_summary.csv", index=False)
    diff_cases_path = WORK / "steven_v4_candidate_diff_cases.csv"
    pd.DataFrame(diff_rows).to_csv(diff_cases_path, index=False)
    print("Candidate vs Steven v4 ID coverage:", len(joined), "studies; exact match")
    print("Per-target deltas, including the not-mentioned group:")
    display(diff_summary)
    print("Changed and not-mentioned case details:", diff_cases_path)
'''),
    md(r'''## Interpretation

This is a small, direct test of the few-shot labeler against official image-assessed labels. A strong score does not prove the report-only predictions are correct: the gold set is small, and some image findings are absent from reports. The candidate file is a proposal with probabilities, statuses, and evidence, not a replacement truth set. Review its differences and per-target gold performance before using it for retraining. This notebook contains no cell that overwrites Steven v4.'''),
]

for cell in cells:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "GOLD_SPLIT_RECORDS_PLACEHOLDER" in source:
            cell["source"] = source.replace(
                "GOLD_SPLIT_RECORDS_PLACEHOLDER", repr(split_records)
            ).splitlines(True)

for index, cell in enumerate(cells, 1):
    cell["id"] = f"cell-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(OUT / "steven_v4_fewshot_calibration.ipynb").write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
)
print("Wrote", OUT / "steven_v4_fewshot_calibration.ipynb")
