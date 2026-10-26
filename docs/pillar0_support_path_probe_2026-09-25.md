# Correction to the Pillar-0 support-path probe

**Probe:** private Kaggle kernel `tumis001/rsna-knee-pillar0-support-path-probe`, version 4  
**Data:** one report-only study; one axial, coronal, and sagittal series  
**Training:** none; the encoder was not loaded

The paired probe completed, but its “legacy” branch came from `preprocess_for_pillar` in [`pillar0_small_sample.ipynb`](../kaggle/experiment_kernels/pillar0_small_sample.ipynb). That notebook is marked failed and superseded in [`kaggle/notebooks/README.md`](../kaggle/notebooks/README.md). Its preprocessing is not the baseline used in the authoritative Pillar-0 run. The probe's numerical comparison must not be treated as a production-baseline result.

The authoritative code in Cell 10 of [`pillar0_milestone.ipynb`](../kaggle/notebooks/milestone/pillar0_milestone.ipynb) already handles support explicitly before the frozen encoder. It converts declared DICOM padding to `NaN`, builds a finite-value mask, resamples signal and support separately onto canonical DICOM LPS coordinates, corrects partial-support interpolation, estimates the p20–p99 intensity window only over voxels with support at least 0.5, and zeros unsupported voxels after normalization. Exact-zero intensities with valid support are retained. So the earlier claim that this preprocessing turns padding into zero before carrying a mask was incorrect for the authoritative Pillar-0 path.

The earlier one-study probe found no DICOM-tagged invalid pixels in its three series. The final fixed-size canvas was 10.3–12.7% source-supported because of centre-padding to `(192, 384, 384)`. That is explicit model-input padding, not evidence of NaNs or invalid DICOM pixels. The probe's large p20 changes came from its outdated “nonzero means supported” percentile rule; this does not imply the verified baseline has that error.

The checkpoint notebook confirms the exact offline source is already available at `tumis001/pillar0-breastmri-offline-bundle`, with pinned revision `be5050ba482607f90e607e044f03a2ebc38f98c3` and expected weight SHA256 `d658fa8aad744c66d1abf43c2bf6bf594e75e834186dce9aadd1b7a9e4381bf0`. A separate Hugging Face token is not needed to load this Kaggle bundle.

The next useful audit is therefore not another comparison against the failed preprocessing. It should use the authoritative inputs and inspect whether the frozen encoder's spatial activations or final embedding are sensitive to the large centre-padded region. That can test whether passing support information into a later feature-pooling or aggregation stage is worthwhile. It remains a representation diagnostic on one study; it cannot establish an AUC improvement.

## Probe artifacts

- [`paired_input_metrics.csv`](../reports/pillar0_support_path_probe_2026-09-25/paired_input_metrics.csv)
- [`run_summary.json`](../reports/pillar0_support_path_probe_2026-09-25/run_summary.json)
- Paired inputs and support map remain in the private Kaggle kernel output. Do not use them as a production-baseline comparison.
- Probe source: [`pillar0_support_path_probe`](../kaggle/experiment_kernels/pillar0_support_path_probe/).
