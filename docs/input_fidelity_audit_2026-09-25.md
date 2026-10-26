# Pillar-0 input-fidelity audit

**Run:** `tumis001/rsna-knee-input-fidelity-audit`, private Kaggle kernel, version 2

**Scope:** report-only studies; native DICOM inspection; no model loading or training

**Sample:** 36 studies selected to cover report-derived present/absent states across 12 targets

## What was checked

The audit inspected native DICOM pixel arrays and geometry for the selected studies' five MRI series. Pixel support was defined from finite decoded values, excluding `PixelPaddingValue` and its optional range when those tags exist. Exact zero was retained as a valid pixel unless a DICOM padding tag said otherwise. It measured physical field of view and inter-slice spacing, estimated the area that would be padding if a rectangular slice were placed in a square without cropping, and calculated an Otsu-threshold foreground extent as a coarse intensity proxy. That proxy is not an anatomy segmentation.

The sample was stratified with report-derived statuses. Those statuses are noisy and correlated: a study can contribute to many target rows, and the report is not an independent image adjudication. The audit only used official-label nullness to filter to report-only studies; no official gold target values were read. It exported no report text or study/series identifiers.

## Results

- All 36 studies were processed: **180 series, 6,609 slices, zero DICOM decode errors**.
- DICOM-tag-defined supported fraction was **1.0** in every plane/sequence summary; the declared padding fraction was **0**. No declared native pixel padding was found in this sample.
- Exact zero values were present in the stored images. Median zero fraction ranged from **1.1% to 5.0%** across the six plane/sequence groups. Zero is therefore a real image value in these inputs and cannot serve as a missing-pixel sentinel.
- The native square-padding estimate was **0** in all six plane/sequence groups. Median in-plane physical field of view was about **150–160 mm**. Median inter-slice gap ranged from **3.3 to 4.0 mm**; the sampled axial series had different median through-plane spans by sequence (about **89 mm** versus **133 mm**), while coronal and sagittal spans were about **104–106 mm**.
- The coarse foreground proxy did not separate report-present from report-absent states consistently. Across targets, the median present-minus-absent change in foreground bounding-box area was about **+3.2 percentage points** for axial, **+0.6 points** for coronal, and **+2.1 points** for sagittal, with target-specific changes crossing zero in every plane. This is not evidence of a useful diagnostic signal.

The raw aggregate outputs are checked in alongside this note under [`reports/input_fidelity_audit_2026-09-25/`](../reports/input_fidelity_audit_2026-09-25/). The Kaggle run summary records the scope and limitations.

## What this says about missing values

This audit weakens one specific hypothesis: the native DICOM files in this sample do not show widespread NaNs or DICOM-tagged padding that would explain the model's poor score. It also confirms that zero-valued pixels occur naturally.

It does **not** prove that the tensors sent into MedSigLIP were free of unsupported regions. Spatial resampling, volume alignment, resizing, or intensity-windowing can create unsupported pixels after DICOM decoding. This audit measured the source images, not the complete historical preprocessing path or the encoder inputs. Therefore it cannot yet establish whether the old preprocessing replaced resampling NaNs with zero, whether support was tracked through each transform, or whether intensity percentiles were computed only over supported pixels.

## Next experiment: trace support through the actual encoder input path

Before another model-training run, replay the exact preprocessing used to create the MedSigLIP inputs on this fixed report-only sample. Carry a separate boolean/fractional support map through every orientation change, resample, crop, and resize. Keep real zero-valued pixels valid. Recompute intensity statistics only over supported pixels and set unsupported pixels to the model's chosen fill value after normalization. For each plane and sequence, report the support fraction at the final encoder input, the change in intensity window, and the change in embedding norm and cosine similarity between the historical and support-aware preprocessing.

This is an inference/data-pipeline probe, **not training**. It should decide whether there is a material representation defect before any retraining is considered. If the two encoder inputs are effectively unchanged and support is near-complete, the padding hypothesis is not a reason to redo the MedSigLIP baseline. If support loss or normalization shifts are substantial, fix and validate that preprocessing first, then train only a controlled comparison using the existing split and early stopping.

## Limits

This is a small, deliberately status-balanced descriptive sample, not a prevalence sample or clinical association study. It cannot establish causality, model performance, or whether any difference generalizes. DICOM padding tags do not describe every scanner-generated background region; the foreground proxy is intensity-based and not anatomically validated. Report status is not equivalent to the competition's official label.
