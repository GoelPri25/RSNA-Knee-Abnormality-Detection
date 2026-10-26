# CNN experiment handoff

This is the short route into the trainable ResNet study reader. It identifies
the current 48-window fold-1 run, the reusable source tree, the input datasets,
and what a teammate needs to run an independent experiment.

## Start here

- **Training source branch:** `experiment/cnn-topk` (development branch, not part of this public snapshot). Read `README.md` and `docs/cnn_topk_support_v1.md` first. Main entry points are `scripts/train_cnn_baseline.py`, `src/images/`, `src/reports/`, and `configs/cnn_window_coverage_full48.json`.
- **Current Kaggle wrapper:** [`kaggle/experiment_kernels/cnn_full48_window_coverage_fold1/`](../kaggle/experiment_kernels/cnn_full48_window_coverage_fold1/). Its metadata names the run `tumis001/rsna-cnn-full-48-windows-fold-1`.
- **Source package attached to that kernel:** `tumis001/rsna-cnn-window-coverage-source-v1` (contains `skanee.zip`). It is a packaged copy of the training source, not the cache or labels.

The current kernel runs **only the full-48-window variant**. A different older
wrapper contains both a 24-window control and a 48-window candidate; do not use
it to reproduce this run.

## What the reader trains

Each study contributes four series slots with twelve physical-order windows per
slot: 48 windows in total. Each window is a 2.5D triplet of neighboring slices
from the same series. ResNet-34 receives three image channels plus three
acquired-pixel support-mask channels. It emits twelve finding logits per
window; each target pools its own highest-scoring 10% of windows into a
study-level logit. The backbone starts from ImageNet weights and is trainable.

The current fold-1 run uses 224 × 224 crops, a 140 mm field of view, Steven v4
report-derived soft labels, microbatches of four studies, and gradient
accumulation over two microbatches (about eight studies per optimizer update).
ReduceLROnPlateau monitors report-validation BCE; checkpoint selection and
early stopping monitor report-validation macro AUC. It produces one
fold-specific model, not an ensemble. The 58 official gold studies are excluded
from training and model selection.

## Inputs and access

Attach these inputs to a Kaggle notebook under an account that has accepted the
competition rules and has access to the private resources:

| Input | Kaggle identifier | Contents/use |
| --- | --- | --- |
| Competition | `rsna-knee-abnormality-detection` | DICOMs and `train.csv`; the wrapper reads official labels only to identify and exclude the 58 gold studies. |
| Training source | `tumis001/rsna-cnn-window-coverage-source-v1` | Packaged training script and configs (`skanee.zip`). |
| Soft labels | `stevenleehans/rsna-knee-llm-report-labels` | `llm_labels_v4_blend.csv` (Steven v4). |
| Slice cache shard 0 | `tumis001/rsna-cnn-top-k-support-cache-shard-0` | Cached image windows and acquired-pixel support masks. |
| Slice cache shard 1 | `tumis001/rsna-cnn-top-k-support-cache-shard-1` | Cached image windows and acquired-pixel support masks. |
| Slice cache shard 2 | `tumis001/rsna-cnn-top-k-support-cache-shard-2` | Cached image windows and acquired-pixel support masks. |

Repository access does not grant Kaggle access. The owner must grant the
teammate access to each private Kaggle input. Do not put DICOMs, labels, caches,
checkpoints, or API tokens in Git or chat.

## Run safely

For a separate experiment, duplicate the Kaggle kernel under the teammate's
account, retain the required inputs, and change the kernel ID/output folder.
Check status before launching; do not launch another CNN instance while one is
already running on that account. The current kernel writes history, plots,
summary, and checkpoints to
`/kaggle/working/cnn_full48_window_coverage_fold1/full48`.

For code changes, edit the source branch and refresh the packaged source dataset
before running a new kernel version. Record source commit, config, label file,
fold, and cache identifiers in the run summary. Keep the 58 gold studies locked
except for a separately approved final evaluation.
