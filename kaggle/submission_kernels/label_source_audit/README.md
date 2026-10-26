# Report-label source audit

This private, CPU-only Kaggle notebook compares published report-derived label tables with the existing 29-study development-gold split. It is an **audit only**: it reads no DICOM data, does not train, and does not score or summarize the locked evaluation split.

It scores source coverage, per-target ROC AUC, Brier score, and thresholded classification metrics. It flags a source if all observed targets exactly reproduce the development gold labels. A source with a high agreement score is not automatically a valid independent teacher: study-level leakage, source provenance, target definition, and use of any gold rows for source creation still matter.

The internal comparison row reuses the saved conservative-v2 extraction on development-gold reports and applies our Confidence-V2 probability map. That is the closest saved comparison for the report-to-label logic; it is not the same as claiming those gold rows were among the 4,349 report-only training labels.

The teammate dataset `priyanjaliml/skanee-blended-soft-labels` is currently inaccessible to this Kaggle account (file manifest returns HTTP 403), so it is excluded from the attached inputs. Once accessible, add it to `dataset_sources` in `kernel-metadata.json` and regenerate the notebook.

Run with:

```bash
python build_notebook.py
kaggle kernels push -p .
```

All competition inputs and derived CSV outputs remain within Kaggle. The notebook has internet and GPU disabled.
