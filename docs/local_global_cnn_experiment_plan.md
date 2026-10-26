# Combined local and global MRI-context experiment

## Aim and comparison

One shared ImageNet-initialized ResNet50 learns local acquired-slice context and study-wide context jointly. This is a single model, not an ensemble. The user requested both changes in one experiment, so any gain tests the combined design and cannot isolate the contribution of each component.

Anchor: the 0.925 ResNet50/HPO-v3.1-D2 model at 224 px. Keep the same 3,524 training studies, fixed 825 Steven-v4 validation studies, extra 281 training-only studies, excluded 58 gold studies, seed 2026, batch 8, 24 training windows and up to 48 evaluation windows. Training probabilities and weights are unchanged. AdamW LR 0.0003, decay 0.03, original warmup, BCE plateau scheduler and AUC checkpoint/early stopping remain unchanged. The separate Synovitis weighting candidate is not part of this run.

## Local and global reader

Preserve the baseline 12 selected centres per acquired series and four-series selection policy. Retrieve each centre's actual preceding and following acquired DICOM slices. Old caches contain sparse centres and cannot recreate missing native neighbors. Normalize using the original selected-centre percentile reference; retain 140 mm/224 px preprocessing. Store unique native slices once, with triplet row mappings, source indices, within-series positions, physical gaps and support masks. Failed neighbor decodes remain zero with zero support; they are never replaced by a more distant slice.

A shared ResNet50 produces each window's feature vector once. Retain target-specific top-10% local logit pooling. Project the same features to 256 dimensions, add plane/contrast/series/position metadata, apply one study-wide attention layer with four heads, and pool with 12 finding-specific queries. Add a small contextual residual to the local logits. Residual output weights start small and nonzero so attention and queries receive BCE gradients immediately.

Plane metadata comes from DICOM orientation when available. Fluid-sensitive status retains the competition metadata. Fat suppression uses explicit metadata/tags; missing tags mean unknown. Coordinates are relative to their own series; no cross-plane coordinate alignment is assumed. Padded tokens are excluded from attention and pooling, and are never run through the encoder.

The acquired support masks now receive the same geometric transformation as intensities (nearest interpolation for masks). Inspection found that the old augmentation transformed a separate support copy without putting it back into the six-channel input. This correction is part of the local-input implementation, and is another reason not to describe the experiment as a single isolated architectural change.

Earlier readers used within-series convolution/attention or acquisition-specific logit corrections. This design combines native acquired neighbors with study-wide feature interaction.

## Speed and memory

Select windows before constructing their triplets. Keep host arrays as uint8 when GPU augmentation is selected, reducing that transfer by four times versus float32. Avoid a second encoder, permanently cached embeddings, larger batches and prefetched GPU batches.

On eight real baseline-cache studies, a byte-identical NumPy assembly comparison measured median 49.84 ms before selection optimization versus 27.42 ms afterward (1.82x). Peak traced allocations were essentially unchanged at 115.6 MB. This measures only assembly, not an epoch; about 22 ms per batch is roughly ten seconds over 441 training batches.

Before training, a five-step-per-variant GPU preflight compares CPU/GPU augmentation and contiguous/channels-last layout using the same model, batch and RNG. Select only a faster variant whose peak allocated AND reserved VRAM do not exceed the reference variant. Restore model, BatchNorm buffers and RNG afterward, persist the choice, and retain that choice on recovery. No epoch-speedup promise is justified yet.

The live T4 preflight selected GPU augmentation with contiguous layout: median step 0.781 s versus 0.997 s for CPU augmentation, about 22% less measured time (1.277x throughput). Allocated peaks were 9,159,001,088 versus 9,162,474,496 bytes; reserved peaks were 10,773,069,824 versus 10,798,235,648 bytes. Channels-last was faster but slightly exceeded the allocated-memory reference, so it was rejected. This remains a short step benchmark, not a measured epoch improvement.

PyTorch references: [channels-last memory format](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) and [performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html). The actual runtime is pinned to PyTorch 2.5.1 and torchvision 0.20.1; benchmarks decide which options are retained.

## Preparation and checks

The builder, dataset, model, speed preflight and Modal runner are in `scripts/`. Four CPU cache notebooks are under `kaggle/experiment_kernels/local_global_native_cache/shard_0` through `shard_3`. Four shards bound output size even before deduplication. Fixed Kaggle input paths are used; there is no recursive input search. Refresh embedded modules with `python scripts/render_local_global_cache_notebooks.py`.

All four version-1 CPU cache notebooks completed on 2026-10-08. Their completion receipts confirm 1,081 / 1,047 / 1,142 / 1,079 studies, totaling 4,349, with identical settings and zero decode errors. All arrays were validated and transferred to the new Modal volume; transfer app `ap-AVtjWxOQx5Rh3nem1b9W3J` finished normally. Receipts are in `data/local_global/transfer_receipts/`. CPU checks passed for native adjacency, exact baseline centre pixels and support, metadata, finite gradients through local/contextual paths, permutation invariance and evaluation batch isolation. A toy interruption/resume check restored epoch 1 and Adam state without duplicating history. These are implementation checks, not model-performance evidence.

All subsequent checks use the reusable container defined in `containers/local_global_checks/Dockerfile`:

```bash
podman build -t localhost/rsna-local-global-checks:latest containers/local_global_checks
podman run --rm --network none -v "$PWD:/workspace:rw" localhost/rsna-local-global-checks:latest
```

Results are under `data/local_global/verification/`. An earlier CPU check used the project virtual environment and installed pydicom there; no host installers remain running. Container tests use PyTorch 2.5.1, matching Modal. They download no model weights and read no gold labels.

## Modal lifecycle

New persistent volume: `rsna-cnn-native-triplets-224px-v1`. `scripts/modal_local_global.py` has explicit transfer and train stages, and defaults to CPU transfer. Transfers skip already validated arrays and commit each completed file. Training is T4/4 CPUs/16 GiB RAM with a 12-hour per-attempt timeout and three retries. Use personal credentials and always launch with `modal run --detach`.

Launch with `python scripts/run_modal_personal_local_global.py --stage train`. The wrapper refuses an active duplicate, deploys the app, and submits a background coordinator from a detached entrypoint that exits immediately. Training has no local synchronous result waiter. The user authorized starting this separate experiment alongside the Synovitis candidate on 2026-10-08. Verified app `ap-Dwn1elCgg17kzUiQp6fGbE`, call `fc-01M4EP2YC5GFCDV15JA5RGJV8X`, passed the GPU preflight and reached epoch 0 batch 300/441. Launch receipts and startup logs are in `data/local_global/`.

Training keeps atomic inference checkpoints and full epoch recovery (Adam, scheduler, scaler, RNG, history and selection counters). An interrupted incomplete epoch is replayed. A completed run is not restarted. Publish final artifacts privately to Kaggle on completion; publication failure preserves the volume checkpoint and does not rerun training. Do not start this GPU experiment while the Synovitis candidate is still active without authorization for concurrent GPU jobs.

## Interpretation

Compare fixed-reference macro and per-target AUC with the saved baseline. Contextual corrections or attention maps are diagnostic, not proof of improvement. Judge the total design with public scoring. Repeatedly inspected gold is development evidence, not independent final validation. No justified prediction of attainable 0.970 or a theoretical maximum is available.

The earlier three-checkpoint probability ensemble now scores 0.935 publicly. This supports complementary checkpoint errors, but does not establish that this single reader can reproduce the ensemble gain.

## Completed run (2026-10-09)

Training ended normally after zero-based epoch 20 (21 epochs), with eight stale epochs following the kept epoch-12 checkpoint. Max epochs remained 60. Best Steven-v4 macro AUC: 0.8924575986885784; its weighted validation BCE: 0.4960571535669192. Best/latest checkpoints, predictions, history and inference code were published privately to [Kaggle](https://www.kaggle.com/datasets/tumis001/rsna-local-global-resnet50-v31-d2); the Kaggle file list confirms both checkpoints. Full recovery remains on the native-triplet volume. The deployment shows zero active tasks.

`scripts/audit_local_global_validation.py` rescores the same 825 held-out studies on CPU, with exact target-order, ID and checkpoint-epoch checks. Saved rounded predictions reproduce each original Steven score within 1e-5. Each reference uses its own confidence mask (weight > 0 and probability ≤0.3 or ≥0.7), so comparisons are within a reference.

| Steven-selected checkpoint | Steven v4 AUC | HPO v3.1-D2 AUC |
| --- | ---: | ---: |
| Original 0.925 baseline, epoch 15 | 0.896327 | 0.946741 |
| Local+global, epoch 12 | 0.892458 | 0.941066 |
| Local+global minus baseline | -0.003870 | -0.005675 |

On Steven-v4, the largest improvements are Medial OA (+0.014407), Lateral OA (+0.008982) and Medial Meniscus (+0.005771). Largest declines are MCL (-0.034351), Fracture (-0.019263), Synovitis (-0.008775) and Contusion (-0.008717).

The combined change has not beaten this baseline on either reference. That does not isolate native-neighbour input, corrected support augmentation or the Transformer as the cause. It also does not establish which checkpoint an HPO-controlled run would select; only each run's Steven-selected best predictions were rescored. One seed per design and checkpoint selection on this fold limit causal claims. No leaderboard result exists for this checkpoint yet.

Validation BCE reached 0.482576 at epoch 15 while AUC was 0.891732, below the kept checkpoint. The BCE scheduler therefore held LR at 0.00015 until after epoch 18; AUC early stopping still triggered after epoch 20. This followed the configured controls. Most post-startup epochs took about 404–409 seconds on this Modal T4; the first took 551.7 seconds. Total reported time was 8,880.4 seconds, about 2.47 hours.

Local summaries, completion log, best predictions and the hashed dual-reference audit are in `data/local_global/results/`. The rescore uses no model inference, GPU, network or gold labels.
