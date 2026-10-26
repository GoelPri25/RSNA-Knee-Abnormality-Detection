# HPO-v1 training recipes: 0.918 ResNet-34 and 0.924 ResNet-50

This note shares the two single-model HPO-v1 recipes so they can be reproduced and modified. The 0.918 and 0.924 figures are Kaggle public leaderboard scores reported for the respective submissions. They are separate from the held-out report-validation AUCs below.

## At a glance

| | 0.918 recipe | 0.924 recipe |
|---|---|---|
| Backbone | ImageNet-pretrained ResNet-34 | ImageNet-pretrained ResNet-50 |
| Training labels | HPO v1 blended report labels | Same HPO v1 labels |
| Training set | 3,243 original fold-0 train studies + 281 additional report-only studies = 3,524 | Same 3,524 studies |
| Validation | Same fixed 825 report-only fold-0 studies; scored against Steven v4 | Same fixed 825 studies and Steven v4 reference |
| Gold set | 58 gold studies excluded from training and validation | Same |
| Input / reader | 140 mm cache; six-channel 2.5D input; 24 windows per study; target-specific top-10% pooling | Same |
| Selected checkpoint | Epoch 15; report-fold macro AUC 0.885516 | Epoch 17; report-fold macro AUC 0.890904 |
| Public score | 0.918 | 0.924 |

The controlled difference between these two runs is the backbone: ResNet-50 replaces ResNet-34. Both start from their respective ImageNet-pretrained weights and train from scratch for this task; the ResNet-50 run does not warm-start from the ResNet-34 checkpoint.

## Shared data and model pipeline

Each study contributes 24 physically ordered slice windows during training. A window is made from three neighboring grayscale slices plus their three acquired-support masks, giving a `[6, 224, 224]` input. The CNN predicts 12 finding logits for each window. For each finding independently, the reader averages the highest-scoring 10% of that study’s window logits.

The training labels are the 4,349-study HPO-v1 blended report-label file. The original grouped fold split assigns 3,243 studies to training and 825 to fold-0 validation. The remaining 281 labeled studies have no original fold assignment; they were added to training only. The validation IDs did not change. Validation metrics use Steven v4 report labels, not HPO-v1 training labels. The 58 gold studies are locked out and are not loaded by either training run.

The training objective is confidence-weighted soft-label binary cross entropy. Both runs use AdamW, a 60-epoch maximum, 5% learning-rate warmup, and the same augmentation and optimizer settings. `ReduceLROnPlateau` monitors report-validation BCE (factor 0.5, patience 2, threshold 0.0005, cooldown 1, minimum LR 1e-6). Checkpoint selection and early stopping monitor report-validation macro AUC; early stopping uses patience 8 and minimum meaningful improvement 0.0005. The checkpoint is saved whenever AUC improves, including improvements smaller than the early-stop threshold.

Full base config: `kaggle/experiment_kernels/hpo_v1_blend_resnet_tpu/source/skanee/configs/cnn_hpo_v1_blend.json`. The ResNet-50 runner changes the backbone to `resnet50` and uses the matching config version; its run manifest records the same training controls.

## Code and artifacts

The training orchestration scripts are:

- `scripts/modal_hpo_blend_add281_trainonly.py` — HPO-v1 +281 ResNet-34 run that produced the 0.918 submission.
- `scripts/modal_hpo_blend_add281_resnet50.py` — same recipe with ResNet-50, associated with the 0.924 submission.
- Shared model, data loader, and config: `kaggle/experiment_kernels/hpo_v1_blend_resnet_tpu/source/skanee/`.

The submission code and pinned checkpoint checks are in:

- `kaggle/submission_kernels/hpo_blend_add281_submission/`
- `kaggle/submission_kernels/hpo_blend_resnet50_submission/`

The ResNet-50 best-checkpoint record is preserved in `data/hpo_blend_resnet50_submission/artifact/training_history.json`; the local checkpoint file is `data/hpo_blend_resnet50_submission/artifact/checkpoint_best.pt`. These data artifacts are not tracked in Git. For the ResNet-34 run, the pinned epoch and checkpoint hash are recorded in `kaggle/submission_kernels/hpo_blend_add281_submission/README.md` and its submission script; the original private training artifacts are on Kaggle.

## Kaggle input datasets

Attach the inputs below to the training notebook or runner. These datasets may be private; access must be granted separately by their owner.

| Purpose | Kaggle source |
|---|---|
| Training source package, configs, and split files | [`tumis001/rsna-cnn-hpo-blend-xla-source-v1`](https://www.kaggle.com/datasets/tumis001/rsna-cnn-hpo-blend-xla-source-v1) |
| HPO-v1 training labels | [`tumis001/rsna-knee-hpo-v1-blended-report-labels`](https://www.kaggle.com/datasets/tumis001/rsna-knee-hpo-v1-blended-report-labels) |
| Steven v4 validation reference | [`stevenleehans/rsna-knee-llm-report-labels`](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) |
| Original slice-cache shard outputs | `tumis001/rsna-cnn-top-k-support-cache-shard-0`, `...-shard-1`, `...-shard-2` |
| Dataset-mounted cache copies | `tumis001/rsna-cnn-topk-support-cache-shard-0-rehost`, `...-shard-1-rehost`, `...-shard-2-rehost` |
| ResNet-34 training artifacts | `tumis001/rsna-hpo-blend-add281-trainonly-fold0` |
| ResNet-50 training artifacts | `tumis001/rsna-hpo-blend-add281-resnet50-fold0` |

The cache shards are derived 140 mm, 224-pixel slice/support arrays. Use all three shards together. They are not raw DICOMs. The Kaggle competition data also needs to be attached for training-time cache validation or for submission inference, depending on the runner.

## Reproducing and modifying the recipes

For a clean comparison, start from the matching runner and its pinned config, keep the fold-0 validation IDs fixed, and change one training variable at a time. Record the resulting config, label-file hash, split counts, best epoch, report-validation macro AUC and BCE, per-target AUC, runtime, and Kaggle public score. Do not use the 58 gold labels to tune this report-validation comparison.

If changing the model or data loader, preserve the six-channel image/support contract and verify the checkpoint’s inference preprocessing matches training. When creating a new Kaggle kernel, attach all three cache shards as inputs and use a new output directory/kernel slug so artifacts from the reference runs remain intact. The original Modal runners expect Kaggle credentials for downloading source inputs and publishing checkpoints; never put API credentials in this repository or notebook.

Access to the private Kaggle datasets is separate from access to this source code. The Git repository also needs to be shared with the teammate if she does not already have repository access.
