# Local+global ResNet34, HPO validation, 60 epochs

Notebook: [RSNA Local Global ResNet34 HPO 60 Epochs](https://www.kaggle.com/code/tumis001/rsna-local-global-resnet34-hpo-60-epochs), private, version 1.

This parallel experiment takes the current HPO-controlled local+global setup, replaces ImageNet ResNet50 with ImageNet ResNet34, and disables early stopping. It trains from task initialization for 60 epochs to observe the training/validation BCE trajectories beyond the usual stopping point. It does not resume or alter the running Modal ResNet50 job.

The narrower ResNet34 output has 512 features rather than 2048. The existing local classification head and projection into the unchanged 256-wide contextual branch adapt to that width. The one-layer, four-head Transformer, target queries, attention pooling and 0.1 contextual residual remain unchanged. Total trainable parameters: 22,232,077.

## Fixed recipe

| Item | Setting |
|---|---|
| Training / validation labels | HPO v3.1-D2 |
| Training / validation IDs | 3,524 / same fixed 825 fold-0 studies |
| Extra 281 | All training-only |
| Gold | 58 excluded; no gold targets loaded |
| Inputs | Four native-triplet cache shards, 140 mm / 224 px; acquired adjacent slices and support masks |
| Sampling | 24 training windows; up to 48 validation windows |
| Batch / initialization / seed | 8 studies / ImageNet / 2026 |
| Loss | Same weighted soft BCE, no ranking loss |
| Optimizer | AdamW, LR 0.0003, weight decay 0.03, gradient clipping 1 |
| Warmup | 5% of the 60-epoch optimizer-step budget |
| LR controller | HPO validation BCE; factor 0.5, patience 2, absolute threshold 0.0005, cooldown 1, floor 1e-6 |
| Early stopping | Disabled (`early_stopping_patience=0`) |
| Epoch limit | 60 |
| Execution | One T4, GPU augmentation, contiguous memory layout, AMP |
| Steven-v4 | AUC/BCE audit every epoch, never used for controls |

This is an intentional combined backbone and training-duration change. Kaggle is also a different execution environment from Modal; runtime package versions are printed at startup. Do not attribute a result solely to backbone capacity, or treat the higher HPO-reference AUC as a gain against Steven-reference AUC.

Training BCE uses augmented images, while validation uses unaugmented held-out studies. Their gap is a useful trajectory diagnostic; equality is not the objective or proof of clinical generalization. Inspect validation BCE and AUC together, including after the LR reaches its floor.

## Outputs and recovery

Per-epoch history records training BCE, HPO validation BCE/AUC, LR used/next, Steven audit BCE/AUC and the BCE difference. The curve is updated each epoch. Save:

- `checkpoint_latest.pt`: most recent epoch.
- `checkpoint_best.pt`: highest HPO validation AUC.
- `checkpoint_best_bce.pt`: lowest HPO validation BCE, observational only.
- `recovery.pt`: complete optimizer, scheduler, scaler, model, RNG, history and counters.
- `run_summary.json`, histories and best-AUC held-out predictions on completion.

Outputs: `/kaggle/working/local_global_resnet34_hpo60/run`. The notebook can resume an existing recovery file in that directory. The generic recovery guard was corrected so patience=0 cannot falsely declare an unfinished run complete. A Kaggle timeout/restart does not automatically remount a prior `/kaggle/working`; continuation after session loss requires recovering its output and restoring `recovery.pt` plus the inference checkpoints. This is not a Modal volume guarantee.

The CLI requested a 43,200-second limit; Kaggle's platform limit still applies. No metric-based stop is configured. If the platform interrupts before epoch 60, continue from the saved full recovery state rather than restarting weights/AdamW.

## Input checks and reproducibility

Use fixed paths only, never recursive dataset discovery. Check source/label/asset hashes, exact fold assignments, all 281 additions, gold separation, four disjoint cache shard study sets, schema, completion, cache geometry, zero decode errors and mmap array headers/row coverage before training. No preprocessing or data download inside the notebook. Notebook modules/assets are embedded from the checked local files; the pinned source and HPO label dataset remain attached inputs.

Training-label SHA256: `38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d`.
Folds SHA256: `2dd1044adcc54ca4d9bcb9c9fd3b910a6784bd2b99e792bbeaf84a42d417f06a`.
HPO validation SHA256: `7323326a7bfab1c8649561b4a0543852eb0d454fdaf83b4a6e03d055aa619009`.
Config SHA256: `3503fdc949929c6c276d351609c2e54d46d304f37151829fd5ba76dcffe325b6`.

Offline CPU checks passed for native neighbor mapping, gradients through CNN and context branches, token permutation invariance, isolation between studies, per-epoch best-BCE checkpoint, Steven audit history, and simulated interruption/resumption with early stopping disabled. The two-epoch smoke test resumed epoch 1 with retained optimizer state, did not duplicate history, and completed despite a flat AUC. No GPU or gold targets were used in checks.

Live startup confirmed all four native cache shards, the 3,524/825 split, ImageNet ResNet34 download, retained GPU-augmentation/contiguous execution, and epoch-0 training progress through batch 400/441. Kaggle runs PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128; the Modal comparison uses 2.5.1 / 0.20.1, so software environment is an additional comparison limitation.

After pushing version 1, the CLI pulled it back: every cell's code and all input/accelerator metadata matched the local notebook. Receipts and test results are in `data/local_global_resnet34_hpo60/`.

## User-requested stop and checkpoint export (2026-10-09)

The user stopped this run to preserve Kaggle GPU quota after 24 completed epochs (indices 0–23), rather than finishing the configured 60. Kaggle confirmed `CANCEL_ACKNOWLEDGED` for version 1, session 356582309. No timed-stop worker was launched; the stop/export script ran directly only after the user's explicit stop instruction.

Before stopping training, a separate private CPU-only probe verified that cancellation followed by a version-pinned output download preserved written files and their SHA256 hashes. The normal CLI output listing remained empty even after cancellation; the version-pinned SDK `download_kernel_output` endpoint retrieved the training output ZIP. Initial output downloads were empty or incomplete while Kaggle finalized the files, so only the subsequent complete ZIP was accepted.

The export contains 27 files and is saved in `data/local_global_resnet34_hpo60/salvaged_output.zip` (563,244,815 bytes), extracted under `data/local_global_resnet34_hpo60/salvaged/`. Per-file hashes are in `salvaged_SHA256SUMS.json`. Offline CPU validation loaded all checkpoints, checked the exact configuration and tensor keys, confirmed that latest weights equal recovery weights, and checked history alignment and full Adam/scheduler/scaler/RNG state. The validation receipt is `salvage_validation.json`.

| Saved checkpoint | Epoch index | HPO validation AUC | HPO validation BCE | Steven audit AUC |
| --- | ---: | ---: | ---: | ---: |
| Best AUC | 15 | 0.935626 | 0.456317 | 0.879452 |
| Best BCE | 16 | 0.931230 | 0.442336 | 0.875658 |
| Latest / recovery | 23 | 0.927617 | 0.446341 | 0.875128 |

Recovery records optimizer step 10,584 and resumes at epoch index 24. Its next learning rate is 0.000075. Resume requires restoring the saved run files into the new session, including both best checkpoints; merely attaching the original notebook as an input does not restore `/kaggle/working` automatically. The sum of completed epoch durations is 21,203 seconds (5 hours 53 minutes 23 seconds), excluding setup and any unfinished work after the final completed epoch. This export is not a completed 60-epoch experiment, and no submission was made.

```bash
python3 scripts/render_local_global_resnet34_notebook.py
podman run --rm --network none -v "$PWD:/workspace:rw" -w /workspace localhost/rsna-local-global-checks:latest python scripts/check_local_global_resnet34_kaggle.py
kaggle kernels push -p kaggle/experiment_kernels/local_global_resnet34_hpo60 --timeout 43200 --accelerator NvidiaTeslaT4
kaggle kernels logs tumis001/rsna-local-global-resnet34-hpo-60-epochs
```
