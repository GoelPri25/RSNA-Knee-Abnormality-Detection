# EfficientNetV2-S versus HPO v3.1-D2 ResNet50

Active detached app: https://modal.com/apps/sumitgsk/main/ap-cActW1OeIe54EpIu5UhGyP

One T4, batch 8, ImageNet initialization, 224px original 140mm cache, 24 sampled windows for training and all48 for validation. Same top10% target pooling, BCE, AdamW, augmentation, seed2026, scheduler and stopping controls as the v3.1-D2 ResNet50 run (best fixed report AUC0.896327 at epoch15). Max60 epochs; BCE plateau factor0.5/patience2/cooldown1/threshold0.0005/minlr1e-6; AUC early stopping patience8/min_delta0.0005. Gold excluded, 3524train and fixed825 Steven-v4 validation.

Backbone changes from ResNet50 to torchvision EfficientNetV2-S. Its classifier is replaced by the same dropout0.4 + twelve-logit head. ImageNet RGB stem weights copied to image channels; three support-mask kernels initialized zero. Native EfficientNet blocks, including their intrinsic stochastic-depth behavior, remain part of the architecture. No progressive resizing, new loss, label changes, or ensemble.

Labels dataset `tumis001/rsna-knee-hpo-v31-d2-report-labels`, sha25638367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d.

Launch: `python3 scripts/run_modal_personal_efficientnet_v31.py` uses the personal Modal profile and --detach, checks against duplicate active apps. Preparation runs on CPU, confirms existing volume cache paths, fetches only small labels, and verifies six-channel224px forward/backward + pooling before GPU allocation. Initial CPU preparation had a missing test pooling helper; it was corrected before any GPU training started.

Cache volume: `rsna-cnn-topk-support-cache-v1`, reused `shard_{0,1,2}/slice_cache`. Persistent run: `experiments/efficientnet_v31_d2_fold0/run`. Each completed epoch stores recovery state including model, AdamW, scheduler, AMP scaler, optimizer-step count, numpy/torch/CUDA RNG, history, best predictions and early-stop counters, then commits the volume. Automatic retries3; timeout12hours per function attempt. An interrupted epoch is replayed from its last completed predecessor. No claim of batch-level recovery.

Artifacts are published after completion to private Kaggle dataset `tumis001/rsna-hpo-v31-d2-efficientnet-v2-s-fold0`; publication is invoked remotely from the training task, so parent disconnection cannot skip it. Intermediate checkpoints remain on the volume, not published to Kaggle every epoch.

Scripts: modal_efficientnet_v31.py (compute/persistence), efficientnet_v31_runner.py (adapter + recovery), run_modal_personal_efficientnet_v31.py (account-safe launch). Source trainer downloaded from `tumis001/rsna-cnn-hpo-blend-xla-source-v1` into `data/kaggle_reference/hpo_xla_actual/skanee`. Fixed reference config committed in configs/hpo_v31_d2_resnet50_reference.json. Model weights downloaded from torchvision's official ImageNet URL.

The initial full batch8 attempt exhausted T4 memory before completing an epoch. Stage-wise activation checkpointing now reduces stored activations without reducing batch8 or224px resolution. It preserves RNG during recomputation and swaps BatchNorm running-stat buffers to disposable clones for replay, restoring original references afterward. This avoids double updates to running statistics. A full192-window AMP forward/backward GPU capacity check is required before training. Failed preparation and failed memory attempt apps are stopped, not parallel training runs.
