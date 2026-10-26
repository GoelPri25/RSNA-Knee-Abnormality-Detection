# Frozen Pillar-0 knee MRI transfer: first milestone

## Verified status (2026-09-18)

The first real-data milestone is complete on Kaggle. The audit found 4,407 training
studies, 24,371 series and 819,635 DICOM files. Only 58 studies have complete official
labels for all 12 targets; the other 4,349 have reports. Geometry-based reconstruction,
canonical LPS preprocessing, visual checks, frozen Pillar-0 embedding extraction, a
24-series cache, variable-series aggregation and a four-study overfit check all ran.
The overfit result validates wiring only and is not evidence of generalisation.

Kaggle confirms that the competition metric is macro ROC-AUC across the 12 targets.
Submission notebooks have a nine-hour CPU/GPU limit and internet must be disabled.
Pillar-0 revision `be5050ba482607f90e607e044f03a2ebc38f98c3` is mirrored as a private
Kaggle notebook output with its ECL-2.0 license, provenance and SHA-256 inventory.
An internet-disabled T4 notebook loaded all 94,130,369 parameters with no missing or
unexpected keys and completed a synthetic forward pass in 1.744 seconds at 2,806.4 MB
peak allocated GPU memory. The output was finite, unit norm and shaped `(1, 1152)`.

The report-supervision audit is also complete. Reports exist for all 4,407 training
studies, while only 58 studies have complete official labels. A fixed seed-42 split
reserves 29 gold studies for prompt development and 29 for locked evaluation. The
frozen conservative extractor represents `present`, `absent`, `uncertain` and
`not_mentioned` separately, retains severity and negation, and masks the two unknown
states rather than turning them into binary labels. On locked evaluation its micro
pseudo-positive precision was 0.7029, micro pseudo-negative precision was 0.9130,
resolved coverage was 0.6609 and macro neutral-unknown AUROC was 0.8124. These results
support structured, confidence-aware auxiliary supervision; they do not support
unconditional hard pseudo-labels.

The report extractor uses the offline Apache-2.0 Qwen3-4B-Instruct-2507 checkpoint at
revision `cdbee75f17c01a7cc42f958dc650907174af0554`. A two-T4, batch-eight-per-GPU
benchmark processed 16 reports in 85.59 seconds, projecting 6.55 hours for all 4,407
reports. Internet remained disabled. The production extraction run will live in a
separate resumable notebook and will store structured outputs and provenance without
report text.

## Architecture (planned, after the data gate)

Physically reconstructed MRI series -> documented MRI preprocessing -> frozen
Pillar-0 BreastMRI -> cached series embedding plus metadata -> compact masked
attention fusion -> study embedding -> linear 12-logit head. No softmax; sigmoid
probabilities for evaluation. Start with unweighted BCEWithLogitsLoss on known gold
labels. Review prevalence before deciding on weighting. Unknown is not negative.

Use one small attention block (provisionally width 128, four heads) with a learned
study query, padding mask and no arbitrary series-order positional encoding. Derive
input embedding dimension from the actual verified encoder output. Preserve unknown
metadata explicitly. No replacement backbone, full fine-tuning, JEPA or report heuristics.

## Implemented first: inventory and native geometry

`knee-audit` inventories filenames and CSVs, hashes tables, reports missing values,
duplicate rows and configured gold-label distributions. It samples 32 series directories
with seed 42, reads all headers within each sampled directory, and records available
fields, sequence/scanner metadata, patient/study/series UIDs and geometry failures.
It does not decode image pixels. Counts based on directories are explicitly candidates.
Archives are listed, not expanded. Inspect all file suffixes before deciding inventory
coverage. The JSON contains potentially identifying metadata and stays outside Git.

`knee-volume` accepts one single-frame MR series. Slice order is the projection of
ImagePositionPatient onto cross(IOP first triplet, IOP second triplet). It validates
orthonormal directions, UIDs, pixel spacing, dimensions, repeated positions, acquisition
consistency, slice gaps and in-plane drift. Unsupported/mixed geometry raises an error;
there is no filename or InstanceNumber fallback. Enhanced multi-frame MR requires a
separate implementation if found. Compressed images may require codec dependencies,
to be selected after the transfer-syntax audit.

Output is `(slice,row,column)` float32 plus an affine from those indices to DICOM
LPS millimetres, ordered source paths, positions and metadata. Per-slice rescale slope
and intercept are applied; DICOM pixel padding becomes NaN. Display uses a separate
1–99 percentile window and physical aspect ratios. Native axes are explicitly labelled;
plots are not misrepresented as canonical axial/coronal/sagittal views.

No anatomy-changing resize, crop or interpolation is applied yet. After inspecting
real volumes and the official checkpoint preprocessing, choose physical output spacing,
orientation, field of view, intensity/window normalization and padding. Preserve the
native reconstruction and record every affine/grid transformation and normalization
parameter. Reject a proposed crop if it truncates knee anatomy; do not independently
stretch axes to force a cube. Validate several protocols visually before encoder use.

## Next gates, in order

1. Run the frozen report extractor over all 4,407 reports in a separate resumable,
   internet-disabled notebook. Preserve structured states and provenance; do not emit
   report text or collapse unknown states into negatives.
2. Compare the final 1152-D CLIP projection with intermediate Atlas vision features.
3. Define and visually validate a physically meaningful knee crop so that zero padding
   does not dominate global pooling. Compare it with the current full-FOV preprocessing.
4. Revisit the 1 mm isotropic resampling policy without sacrificing native in-plane
   detail, and keep the current transform as the reproducible baseline.
5. Scale embedding extraction only after the report supervision and cheap encoder probes
   justify the GPU cost. Preserve the series-fusion hypothesis as the primary experiment.

## Validation boundary

Synthetic DICOM tests check ordering, affine coordinates (including oblique planes),
invalid geometry rejection, actual pixel decoding/rescaling, audit repeatability and
visualization generation. These tests cannot establish correct knee preprocessing,
Pillar-0 transfer or completion of the real-data end-to-end milestone.

Executed in a disposable Python 3.11 container: **11 tests passed**. Matplotlib emitted
14 dependency deprecation warnings; there were no test failures. Real-data execution is
recorded in `notebooks/kaggle_milestone`; offline deployment verification is recorded in
`notebooks/kaggle_offline`.

## Remote run

Access-token authentication verified for `tumis001`. Private CPU notebook version 1
was uploaded and execution requested with the competition attached:
https://www.kaggle.com/code/tumis001/rsna-knee-pillar0-data-audit .
This supersedes the initial access blocker above. Actual dataset findings are still
pending execution and review. Source is bundled from `feat/pillar0-volumetric-pipeline`;
credentials are mounted read-only in the CLI container and never embedded in the notebook.
All competition data and reconstructed volumes remain on Kaggle.

## Verified first Kaggle run (version 1, completed)

Execution completed successfully in approximately 18 minutes. Only the operational
log and aggregate findings were read remotely; competition files and volume outputs
remain on Kaggle.

| Item | Observed result |
| --- | --- |
| train.csv | 4,407 rows; StudyInstanceUID, Report, 12 target columns |
| train_series.csv | 24,371 rows |
| test.csv | 3 visible example rows; hidden evaluation size is not inferred |
| test_series.csv | 15 rows |
| DICOM files | 819,635 |
| Series directory candidates | 24,386 |
| Missing labels | 4,349 per target; 58 observed values per target |
| Missing reports | None under the audit's missing-value policy |
| Exact duplicate CSV rows | None across the five CSVs |
| Geometry sample | 32/32 passed; zero duplicate SOP UIDs in sample |
| Sample planes from geometry | 10 coronal, 9 sagittal, 9 axial, 4 oblique |
| Pixel reconstructions | 3/3 succeeded |

The series CSVs provide Fluid_Sensitive, Fat_Suppression and Anatomical_Plane directly,
with no missing values reported. Use these supplied metadata fields instead of report
or sequence-description heuristics. The 12 target names are now configured explicitly
in configs/audit.json based on the actual table schemas.

Three reconstructed shapes (slice,row,column): (15,528,528), (29,560,560),
(24,384,384). Their measured spacing in mm was approximately (3.30,0.303,0.303),
(4.40,0.286,0.286), (4.95,0.443,0.443), respectively. This strong anisotropy must
inform preprocessing. Reconstruction success is not yet visual confirmation of anatomy.

Still pending: unique study/series key counts and joins, whether the 58 known labels
per target refer to the same studies, label frequencies, patient-ID usability and
scanner/protocol distributions. No report pseudo-labels, encoder outputs or training
results exist yet. Earlier unverified-status statements describe the pre-run state;
this section supersedes them only for the specific observations listed above.

## Small-sample experiment notebook

A clean 21-cell private GPU notebook was generated from
`scripts/prepare_kaggle_experiment.py` and launched as version 1:
https://www.kaggle.com/code/tumis001/rsna-knee-pillar0-small-sample . It is currently
running. Cells separately cover environment, supervision audit, DICOM reconstruction,
visual checks, physical resampling, checkpoint loading, series embedding cache,
variable-series aggregation, tiny overfit and `z_study` export.

The BreastMRI checkpoint uses three DCE phase channels. A knee series supplies one
acquisition, so this first transfer probe repeats the same high-contrast windowed series
three times. This is recorded as an experimental input adapter and a major domain
mismatch. It does not invent DCE information. The notebook requires a Kaggle secret
named `HF_TOKEN` whose Hugging Face account has accepted the gated checkpoint terms.

Version 1 stopped before image decoding because its hard-coded image directory pattern
did not match the competition mount. It nevertheless verified that all 58 gold-labeled
studies have all 12 labels, all 24,371 series join to train studies, and study/series IDs
are unique in their respective tables. Version 2 infers the UID directory levels from
an actual DICOM path, validates those levels against the CSV IDs, adds stable notebook
cell IDs, and is running. No model result is claimed from version 1.
