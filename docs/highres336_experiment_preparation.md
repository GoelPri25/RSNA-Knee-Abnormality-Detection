# ResNet50 336 px experiment preparation

Training will run on one Modal T4, detached, using persistent volume `rsna-cnn-140mm-336px-cache-v1`. Kaggle CPU only builds the three shards sequentially; no GPU training is started by preparation.

The matched control is the ongoing 224 px HPO v3.1-D2 train-and-validation run, not the 0.925 run whose controls used Steven-v4. Use v3.1-D2 BCE for scheduling and v3.1-D2 AUC for checkpoint selection/early stopping; log Steven-v4 metrics separately. Resolution is the only intended change from that matched control. Preserve effective batch 8; run an actual T4 memory preflight before deciding whether microbatch accumulation is needed and document any execution differences.

Cache: 140 mm FOV, 336 px, four series slots, twelve selected slices per slot, central 20–80% band, 1st–99th percentile intensity clipping, uint8 images and support masks. Rebuilt from DICOMs, not upsampled from the old cache. The original builder is reused, with only image size changed plus an empty-slot return-arity fix and more frequent progress logs. No recursive discovery of Kaggle inputs. 100 studies per part limits RAM usage. SHA-256 modulo 3 assigns shards exactly as before; report-only IDs must equal the 4,349 label IDs. Gold is excluded.

Pinned private assets: https://www.kaggle.com/datasets/tumis001/rsna-hpo-v31-336px-experiment-assets

Contains unchanged v3.1-D2 labels (SHA-256 `38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d`), the 3,524 train/825 validation split, training config, and checksums. No label refinement occurs here.

First CPU builder: https://www.kaggle.com/code/tumis001/rsna-cnn-140mm-336px-cache

Other builders are prepared in `kaggle/experiment_kernels/highres336_cache_shard1/` and `highres336_cache_shard2/`. The coordinator `scripts/prepare_highres336_datasets.py` waits for each shard, downloads outputs under `data/highres336/`, checks completion, publishes each as a private dataset, and uploads to Modal `/shard_N/slice_cache`. Dataset IDs are `tumis001/rsna-cnn-140mm-336px-cache-shard-N`.

Preparation state/log: `data/highres336/preparation.pid`, `preparation.log`, and `shard_N_published.json`. Inspect these before rerunning the coordinator; do not launch duplicate builders. Preparation has started, but the shards are not yet all ready. Cache payload upper bound is roughly 47 GB decimal (4,349 × 48 × 336² × 2); actual size depends on decoded slices. Local disk had 249 GB free at launch.

Kaggle output is published only after completion. Completed local shards and Modal uploads persist. No claim of mid-study cloud recovery: an interrupted initial CPU builder needs its partial output attached before continuation. Modal training recovery must preserve optimizer, scheduler, AMP and RNG state, using the existing durable runner pattern.

## Training launch

Modal app: `rsna-resnet50-v31-trainval-336px-fold0` (`ap-hRm4mqlEvg5YqQygvc9zdF`). Entry point `scripts/modal_highres336.py`; reader/recovery/audit adapter `scripts/highres336_runner.py`. Started with `modal run --detach` on the personal account. CPU checks precede GPU allocation; one T4, batch 8, ImageNet ResNet50, v3.1-D2 train/validation, Steven audit. Maximum 60 epochs, AUC early stopping patience 8 and min delta 0.0005; BCE ReduceLROnPlateau factor 0.5/patience 2/cooldown 1/min LR 1e-6. Each GPU attempt has a 12-hour timeout and three retries, resuming completed epochs from durable recovery.

To fit 336px batch eight, ResNet stages use activation checkpointing with disposable BatchNorm buffers during recomputation. This preserves architecture, pooling and state keys; CPU forward/backward and submission-key parity passed. A full 192-window AMP GPU forward/backward preflight must pass before training starts. This execution change should be accounted for when interpreting the resolution comparison.

Recovery commits at epoch boundaries: model, AdamW moments, scheduler, scaler, global step, history, best predictions/checkpoint selection, patience and NumPy/Torch/CUDA RNG. Preemption can replay the incomplete epoch; completed epochs are not intentionally restarted. Intermediate checkpoints remain on Modal. Final artifacts are automatically published to `tumis001/rsna-resnet50-v31-trainval-336px-fold0`.

All three cache shards are uploaded. Shard 2 private dataset publication hit a title collision; its original Kaggle notebook output remains durable and its direct Modal upload succeeded. This does not block training.

## L4 migration after epoch 2

User authorized migration after epoch 2 completed and epoch 3 started. A read-only synthetic benchmark loaded the committed epoch-2 weights and timed five AMP forward/backward steps after two warmups, batch eight/192 windows, 336 px, identical activation checkpointing. Median T4 2.9295 sec; L4 1.5894 sec (1.843× compute speedup). This excludes augmentation, data loading, optimizer and validation; do not infer an identical end-to-end speedup.

Verified recovery epoch 2, global step 1323, stale counter 0 and exact saved config. T4 app `ap-hRm4mqlEvg5YqQygvc9zdF` was stopped explicitly before launching L4. Resume app `ap-DgCUsC41U1yuz1IXsl7F4Y` launched with `modal run --detach scripts/modal_highres336_l4.py`. Same model/batch/AMP/checkpointing/controls/volume; epoch 3 is replayed from its beginning. GPU change can alter floating-point results; this is not guaranteed bit-identical continuation.

Local benchmark evidence: `data/highres336/l4_migration_benchmark.json`. Resumed entry point uses the same recovery loader and final Kaggle publication path. Benchmark app `ap-rOWpKiqYgww0kSUgbbNQFJ` finished without changing saved training state.
