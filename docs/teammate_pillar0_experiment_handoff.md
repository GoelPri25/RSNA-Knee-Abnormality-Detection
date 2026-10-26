# Pillar‑0 experiment handoff

This is the short route into the frozen Pillar‑0 spatial-token reader. It covers
the current five-fold Modal run, the source files, and the Kaggle inputs a
teammate needs to reproduce or modify the experiment.

## Start here

- **Reader and trainer:** [`src/rsna_knee/pillar0_reader.py`](../src/rsna_knee/pillar0_reader.py), [`scripts/train_pillar0_reader.py`](../scripts/train_pillar0_reader.py), and [`scripts/train_pillar0_regional_reader.py`](../scripts/train_pillar0_regional_reader.py).
- **Modal launcher/data movement:** [`scripts/modal_pillar0_regional_reader.py`](../scripts/modal_pillar0_regional_reader.py).
- **Design and findings:** [`pillar0_target_query_experiment.md`](pillar0_target_query_experiment.md), [`pillar0_reader_xai_and_stopping.md`](pillar0_reader_xai_and_stopping.md), and [`project_learning_synthesis.md`](project_learning_synthesis.md).
- **Cache-generation provenance:** the `pillar0_milestone` notebook, listed in [`kaggle/notebooks/README.md`](../kaggle/notebooks/README.md).

## What the reader trains

Pillar‑0's encoder is frozen. It has already generated a spatial cache for
24,035 series; the Modal trainer does not decode raw DICOM or update the
encoder. Each series has three cached scales: 512 fine 8³ tokens, 512 native
8³ tokens, and 8 coarse 2³ tokens. Each token contains a 384-dimensional
feature vector, quantized to int8 with saved scales. Fractional acquired-image
support is stored separately.

The regional reader partitions each aligned 8³ map into eight 4³ regions. For
each region it combines 64 fine, 64 native, and one aligned coarse token.
Learned target queries first read within regions, then read region summaries
across all series in a study. A 12-output head predicts the findings. Only the
reader and head train; this is a single-model experiment, not an ensemble.

The current Modal run is configured for five report-fold validation runs,
sequentially on one T4. Each fold trains an independent model, with up to 60
epochs, four studies per batch, and early stopping on report-validation macro
AUC. Checkpoint archives are published to Kaggle every five epochs so they
survive Modal container shutdown. Check for an existing Modal app before
starting another instance.

## Inputs and access

Modal downloads only derived tokens, report labels, and fold metadata. Raw
competition DICOM remains on Kaggle.

| Input/output | Kaggle identifier | Contents/use |
| --- | --- | --- |
| Spatial-token cache | `tumis001/rsna-pillar0-spatial-token-cache-v1` | Private cache (6.88 GiB in the recorded version), support maps, and run config for 24,035 series. |
| Soft labels | `stevenleehans/rsna-knee-llm-report-labels` | `llm_labels_v4_blend.csv` (Steven v4 report labels). |
| Fold/source config | `tumis001/rsna-cnn-topk-support-source-v1` | `skanee/configs/report_folds_v5.csv`. |
| Run artifacts | `tumis001/rsna-pillar0-regional-reader-cv-v1` | Private checkpoint/history/curve archives, published at intervals. |
| Competition | `rsna-knee-abnormality-detection` | Must be accepted under Kaggle rules; raw DICOM is not copied to Modal for this run. |

Repository access does not grant Kaggle access. The owner must grant the
teammate access to each private Kaggle dataset. The teammate also needs their
own Modal account and Kaggle API credentials. Never share the work Modal
profile, `/home/.../.config/rsna/modal-personal.env`, or API tokens. The
launcher reads a local Kaggle token and injects it into the Modal function;
credentials are not committed.

## Run safely

From the repository root, use `scripts/modal_pillar0_regional_reader.py` after
configuring the teammate's own Modal account and Kaggle token. Before starting,
confirm the prior Modal app is finished and use a private output dataset that
the teammate owns or can version. The launcher currently names its artifact
dataset in a constant; change that slug and the run ID in a local branch so
separate work does not mix or overwrite artifacts. Do not start two Modal
instances for the same run.

Use report-fold metrics for development. The 58 official gold studies remain
excluded and locked for a final assessment.
