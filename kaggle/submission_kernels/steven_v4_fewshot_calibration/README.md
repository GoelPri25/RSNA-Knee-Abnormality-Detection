# Steven v4 few-shot report-label calibration

This private Kaggle CPU notebook uses a cloud reasoning model to generate a separate candidate label file for the 4,349 report-only studies, calibrated by the 29 development-gold report/label pairs. It also compares candidate probabilities and report statuses directly with Steven v4, with a focused view of `not_mentioned` targets. It does not train a model or make LLM calls during competition submission inference.

## Inputs and safeguards

- Competition `train.csv` supplies reports and official gold labels.
- The fixed 29-development / 29-locked-evaluation split is embedded from the repository's versioned `gold_split_v1.csv`; no audit notebook output needs to be attached.
- Locked evaluation studies are excluded before their report and targets are read. Their IDs are used only to skip rows.
- Development-gold evaluation is leave-one-report-group-out: the target report and any normalized exact duplicate are removed from its demonstrations.
- The LLM receives report text and gold demonstrations through Ollama Cloud. Study UIDs and DICOM data are never sent.
- Version 5 already completed the 29-case development-gold cross-fit; this run skips repeating those billable calls.
- Steven v4 is attached as an input only for a UID-aligned probability diff. It never edits that dataset or its label file.
- The recovered 1,324-study prefix is stored in the private Kaggle dataset `tumis001/rsna-fewshot-candidate-label-checkpoints-v1`; the run resumes from it and avoids repeating those requests.
- Each new candidate response is flushed to a local JSONL checkpoint. The full CSV and comparison tables are written after all 4,349 studies finish. Ollama input, cached-input, and output token counts are logged for the resumed portion so actual token pricing can be estimated.

## Run on Kaggle

1. Create a private Kaggle notebook from this notebook, with the RSNA competition data attached.
2. Enable Internet and keep the accelerator set to **None** (CPU).
3. Add the secret `OLLAMA_API_KEY` in the notebook's Secrets settings. This is the Ollama Cloud API key; do not put it in notebook source or output.
4. Run the notebook top to bottom. It checks the competition, Steven v4, and private checkpoint inputs, performs a data-free API smoke test, skips the completed development-gold cross-fit, resumes the saved prefix, and labels the remaining reports in batches of four. Malformed JSON triggers bounded smaller-batch retries.
5. Review `report_only_candidate_labels.csv`, `steven_v4_candidate_diff_summary.csv`, and `steven_v4_candidate_diff_cases.csv` under `/kaggle/working/steven_v4_fewshot_calibration/`. The diff is descriptive; the follow-on CNN training and Kaggle submission test whether these candidate labels help.

## Build and publish

From this directory:

```bash
python build_notebook.py
kaggle kernels push -p .
```

`kaggle kernels push -p .` publishes a new private version and starts its run. Add the Kaggle secret before pushing; otherwise the run stops at the secret check.
