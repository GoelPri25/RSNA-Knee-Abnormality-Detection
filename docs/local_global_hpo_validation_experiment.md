# Local+global model with HPO v3.1-D2 validation

The completed local+global run used HPO v3.1-D2 training labels but Steven-v4 labels for LR scheduling, early stopping and best-checkpoint selection. This comparison changes only that validation reference to HPO v3.1-D2. It starts fresh from ImageNet, with fresh task heads and AdamW state. It does not continue from the completed checkpoint.

| Setting | Both local+global runs |
|---|---|
| Training studies | 3,524, including the extra 281 training-only studies |
| Validation studies | Same fixed 825 fold-0 IDs |
| Gold | Excluded from training and validation |
| Training labels | HPO v3.1-D2, unchanged probabilities and weights |
| Architecture | Shared ResNet50, native adjacent triplets, local top-k + study-wide contextual residual |
| Cache | Four native-triplet shards, 140 mm / 224 px, existing Modal volume |
| Sampling | 24 training windows; up to 48 evaluation windows |
| Batch / seed | 8 / 2026 |
| Loss | Unchanged weighted soft BCE; no ranking term |
| AdamW | LR 0.0003; weight decay 0.03; warmup fraction 0.05 |
| Scheduler | Validation BCE, factor 0.5, patience 2, absolute threshold 0.0005, cooldown 1, minimum LR 1e-6 |
| Best checkpoint | Validation macro AUC |
| Early stopping | AUC, patience 8, minimum improvement 0.0005 |
| Maximum epochs | 60 |
| Execution | One T4, four CPUs, GPU augmentation, contiguous layout |

For the new run, **HPO v3.1-D2 alone controls the scheduler, best checkpoint and early stopping**. `STEVEN_AUDIT` prints AUC and BCE each epoch, and the history stores these values. The audit uses the same predictions and requires no second model inference. It does not modify the control metrics or enter any loss.

Different reference labels also change which confident cells enter the AUC calculation (probability <=0.3 or >=0.7, positive weight). Compare model scores within each reference, rather than interpreting the higher HPO AUC as a model gain. Validation affects the LR trajectory and checkpoint selection; its labels do not directly provide training gradients. Until the LR decisions diverge, the matched runs should broadly follow the same training trajectory, subject to GPU nondeterminism.

## Input locks and checks

`prepare_local_global_hpo_validation.py` validates the original input hashes, checks all 825 validation IDs against the fixed folds, copies training labels and folds byte-for-byte, extracts validation probabilities directly from HPO v3.1-D2, and asserts that the core config is unchanged. Version and note metadata are the only config edits. It copies the original execution selection so a new timing benchmark cannot change augmentation placement or memory layout.

| Asset | SHA256 |
|---|---|
| Training labels | `38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d` |
| Folds | `2dd1044adcc54ca4d9bcb9c9fd3b910a6784bd2b99e792bbeaf84a42d417f06a` |
| HPO validation subset | `7323326a7bfab1c8649561b4a0543852eb0d454fdaf83b4a6e03d055aa619009` |
| Steven audit subset | `10fdaaab54e67300b91415fb3a5d4f7567d21cceb71ae973df7f3fd7e545a7e5` |
| New config | `051562ca4abe9844a528455d7b7d5e5e0fe45155018cd73c03b4cb59ca0739c3` |
| Execution selection | `8f5cbacce83f28ee5b03078ecc1188b9505bbc5150d04e1bac3b5f4121e81353` |

Offline container checks confirmed unchanged generated code for the old arm, identical validation IDs, syntactically valid new code, and preservation of HPO control metrics by the injected Steven audit. On the completed run's saved epoch-12 predictions, that audit exactly reproduces Steven AUC 0.8924575986885784; HPO AUC is 0.9410661244940745. These are reference values, not results of the new run.

## Execution and recovery

App: [`rsna-local-global-hpo-validation`](https://modal.com/apps/sumitgsk/main/ap-KUSNH339BmjPU6w0uInxiU). Launched 2026-10-09 using function call `fc-01M4F0FV8BSZSFAFDXTAHJNFDX`; startup verified one GPU training task and its CPU coordinator, with epoch-0 batch progress. The short launcher app has already stopped normally. The local CLI deploys the app and invokes a short entrypoint using `modal run --detach`. The entrypoint spawns the deployed coordinator and returns, leaving no local training result waiter. The launcher refuses another copy of this app and checks for active personal-account tasks.

Reuse volume `rsna-cnn-native-triplets-224px-v1`, with fixed `shard_0` through `shard_3` paths. No data transfer or recursive search is performed. New outputs are under `/experiments/local_global_v31_trainval_d2_seed2026/run`, separate from the completed run. Only execution settings are copied; no previous training checkpoint is copied.

Each epoch saves best/latest inference checkpoints and full recovery state (model, optimizer, scheduler, scaler, RNG, counters and history), then commits the volume. A retry resumes from the last committed epoch; an interrupted uncommitted epoch is replayed. The GPU function allows 12 hours and three retries. After completion, a CPU function publishes outputs to the private dataset `tumis001/rsna-local-global-resnet50-v31-trainval`. Publication failure retains checkpoints on the volume and does not rerun training.

Local assets, checks and launch receipts: `data/local_global_hpo_validation/`.

Reproduce setup and launch using the existing inputs/container:

```bash
python3 scripts/prepare_local_global_hpo_validation.py
mkdir -p data/local_global_hpo_validation/checks
git show 211d5e6:scripts/local_global_runner.py > data/local_global_hpo_validation/checks/baseline_runner.py
podman run --rm --network none -v "$PWD:/workspace:rw" -w /workspace localhost/rsna-local-global-checks:latest python scripts/check_local_global_hpo_validation.py
python scripts/run_modal_personal_local_global_hpo_validation.py
```

Assess the HPO-selected best checkpoint against both references, per target as well as macro, and compare its public submission with the prior local+global checkpoint. One matched seed does not establish a causal gain beyond run-to-run variation.


## Completed result, 2026-10-09

Normal early stop after 24 epochs (0–23), eight stale epochs after the best epoch 15. Total recorded training time 12,094.2 seconds, approximately 3 hours 22 minutes. Best HPO validation macro AUC 0.9468035669939453, BCE 0.43851241911300504; training BCE at that epoch 0.4039153014721514. Lowest validation BCE is epoch 20, 0.43566968513257576, without a new AUC best.

The best/latest checkpoints, histories, best held-out predictions, curve, execution settings and architecture/cache modules were published to the private [RSNA Local Global ResNet50 V31 Trainval dataset](https://www.kaggle.com/datasets/tumis001/rsna-local-global-resnet50-v31-trainval). The file list confirms best checkpoint 99,806,325 bytes and latest checkpoint 99,807,035 bytes. The completed deployed app has zero active tasks.

CPU rescoring of the same 825 saved predictions gives:

| Selected checkpoint | Steven-v4 AUC | HPO v3.1-D2 AUC |
|---|---:|---:|
| Original 224 px ResNet50, public 0.925, epoch 15 | 0.896327 | 0.946741 |
| Local+global, Steven controls, epoch 12 | 0.892458 | 0.941066 |
| Local+global, HPO controls, epoch 15 | 0.895697 | 0.946804 |

Relative to the prior local+global run, gains are +0.003240 on Steven and +0.005737 on HPO. Relative to the original 0.925 checkpoint, deltas are −0.000630 on Steven and +0.000062 on HPO. The new run largely recovers the earlier local+global deficit and approximately ties the original single CNN on macro validation metrics; it does not establish a leaderboard improvement. This remains one matched-seed comparison.

Steven per-target changes against the original 0.925 checkpoint include Medial OA +0.016167, Baker's +0.011869, Lateral OA +0.008235; MCL −0.031550, Fracture −0.028154 and Contusion −0.011906. Improvement is redistributed across targets rather than uniformly better. The HPO-selected epoch 15 is also the highest Steven audit AUC epoch in this run, so its selection did not miss a better Steven-scored epoch here.

Reproduce the comparison in the existing offline CPU container with `python scripts/audit_local_global_hpo_validation_results.py`. Hashed inputs, complete per-target tables and run records are in `data/local_global_hpo_validation/results/dual_reference_comparison.json`. No GPU inference or gold labels are used.
