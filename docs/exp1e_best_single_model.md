# Exp 1e: best single model (public LB 0.936)

This document covers only the best single model, Exp 1e: a ResNet-50 on the d96 input, trained on the per-target
label set. It explains where the weights, data and code are, and how to run inference or retrain.
The full experiment history is in `docs/experiment_ledger.md` (rows 1c and 1e).

## 1. Results

| | Value |
|---|---|
| Public LB | **0.936** (the same recipe with seed 2027 scored 0.934, so the recipe is about 0.935 ± 0.001) |
| Fold 0, 10-target macro AUC | 0.9073 (seed 2027: 0.9097) |
| Fold 0, 12-target macro AUC | 0.9037 |
| Fold 0, clear-band 6-target AUC | 0.9719 |
| Gold-58 | 0.905 (development check only) |

**What the targets mean:**
- Fold 0 has 825 studies and the training folds have 3,524.
- The 10-target macro leaves out Lateral OA and Baker's cyst. In the per-target label set those two come from
  Steven-v4, the same source as the fold-0 labels, so scoring them would be circular.
- The headline fold-0 macro does not track the public LB. The clear-band AUC does: it compares moderate-or-worse
  positives against confident negatives, over the 6 targets with at least 20 such positives.

## 2. Kaggle resources (private, owner `tumis001`)

| What | Link | Contents |
|---|---|---|
| Model weights | https://www.kaggle.com/datasets/tumis001/rsna-d96-r50-exp1e-pertarget-fold0 | `weights_swa21_25.pt` (sha256 `cb2382ec…`, 97 MB) and the timm 1.0.29 wheel for offline install |
| Submission notebook | https://www.kaggle.com/code/tumis001/rsna-d96-r50-exp1e-pertarget-submission | Inference on T4, no internet. Its only inputs are the weights dataset above and the competition data |
| d96 training cache | https://www.kaggle.com/code/tumis001/rsna-coatnet-d96-cache-shard-0 (also `-1`, `-2`, `-3`) | Notebook outputs: `manifest.jsonl` plus uint8 slice arrays for every training study. Needed only for retraining |
| Labels | `priyanjaliml/rsna-knee-final-per-target-labels` (your dataset) | Saved locally as `soft_labels_final_per_target_any_model.json` (sha256 `d40803d0…`) |

The submission notebook builds the d96 input from the DICOMs itself, so `submission.py` is also the reference
implementation of the full test-time pipeline.

## 3. Code in this repository

| File | Role |
|---|---|
| `kaggle/submission_kernels/d96_r50_exp1e_pertarget_submission/submission.py` (+ `kernel-metadata.json`) | Inference: DICOM → d96 → model → `submission.csv` |
| `scripts/coatnet_d96_runner.py` | Training. 1e is `D96_PHASE=r50_pertarget`. That block (near the top) changes the base config; everything else in the file is shared with other experiments |
| `scripts/d96_positions.py` | Helper imported by the runner. 1e does not use its output; it only needs to be importable |
| `scripts/modal_coatnet_d96.py` | Modal launcher: copies the 4 cache shards from Kaggle to a Modal volume, runs a CPU preflight, then trains on one H100 |
| `kaggle/experiment_kernels/coatnet_d96_cache/` and `coatnet_d96_cache_shard{0..3}/` | Source of the cache-building notebooks, for rebuilding the cache |
| `scripts/audit_d96_native.py` | Scorer: per target, clear and mild bands, paired bootstrap against 1e |

## 4. Recipe

| Part | Setting |
|---|---|
| Input (d96) | 96 slices over 5 series slots: sagittal fluid 26, sagittal non-fluid 22, coronal fluid 18, coronal non-fluid 12, axial 18. 384 px over a 140 mm field; per-series 2/98% intensity window; uint8 |
| Windows | 3 adjacent slices form the 3 channels. Training: 16 stratified random windows per study. Evaluation: 94 evenly spaced |
| Backbone | `resnet50.tv_in1k` (timm), ImageNet initialisation |
| Head | "Raptor", from the public notebook: global average pool per window → LayerNorm → per-finding attention over windows (Linear 2048→256, tanh, dropout 0.2, →12; softmax over windows for each finding) → per-finding linear classifier |
| Labels / loss | Per-target soft labels; every cell weight 1.0; weighted BCE |
| Optimiser | AdamW: lr 1e-4 for the backbone, 1e-3 for the head; weight decay 0.05; gradient clip 1.0; bf16 |
| Schedule | 25 epochs: 1 warm-up epoch, then cosine decay to 1% of the peak |
| Batch | 2 studies × 16 windows per step, 4 accumulation steps (8 studies per update) |
| Augmentation | Anatomical left/right mirror (p 0.5), ±10° rotation, ±10% scale, ±6% shift, gamma and gain jitter |
| Final weights | Weight average (SWA) of epochs 21–25, then BatchNorm statistics recomputed on 256 training studies |
| Inference | Centre 320 crop (a crop, not a resize), 94 windows, no TTA |
| Seed / folds | Seed 2026. Train on every non-zero fold, validate on fold 0 |

## 5. Running it

**Inference.** Fork the submission notebook, or attach the weights dataset to your own notebook and run
`submission.py`.

**Retraining.**
1. **Get the cache.** Download the 4 cache-shard notebook outputs into `<cache>/shard_0` … `<cache>/shard_3`. The
   Modal launcher does this itself with `kaggle kernels output`, using your token in `~/.kaggle/access_token`.
2. **Set up the assets directory.** The runner checks each file's sha256 and stops on a mismatch.

   | File | sha256 prefix | Source |
   |---|---|---|
   | `soft_labels_final_per_target_any_model.json` | `d40803d0` | your per-target label dataset |
   | `folds.csv` (`study_id,fold`; fold `0` = validation) | `2dd1044a` | not in git; ask for a copy |
   | `validation_labels.json` (fold-0 Steven-v4 confident cells) | `10fdaaab` | not in git; ask for a copy |

3. **Run the training.**
   ```bash
   D96_PHASE=r50_pertarget D96_CACHE=<cache> D96_ASSETS=<assets> D96_OUT=<out> python scripts/coatnet_d96_runner.py
   ```
   - Requirements: torch, timm 1.0.29, numpy.
   - On one H100 the run takes about 40 min (about 88 s per epoch); on a Kaggle T4, expect several hours.
   - With Modal, put the assets in `data/coatnet_d96_fold0/assets/`, run `modal token new`, then
     `modal run --detach scripts/modal_coatnet_d96.py --phase r50_pertarget`.
4. **Check the outputs.** `<out>` should contain `weights_swa21_25.pt` and `val_swa.json` (fold-0 predictions).
   Score them with `python scripts/audit_d96_native.py <root> <run_dir_name> <label>`.

## 6. Results of changes tried on top of 1e

None of these beat 1e on the LB or on the clear band. Details are in the ledger.
- Cross-window transformer (2a).
- Per-plane heads (2c, 2d).
- Native slice density (4b): +0.003 on fold 0, all of it in the mild band.
- Label re-valuing (3a, 3b).
- Self-supervised pretraining on our MRI (5a).
- Cyclic SWA (6a).
- Series-allocated windows (6b): MCL +0.014, but OA, meniscus and contusion fall.

## 7. Rules

- Fold-0 and gold-58 labels are never changed. Gold-58 is never used for training or for choosing models.
- Model predictions may shape training labels only out-of-fold: the model that produces a study's label must not
  have trained on that study.
- Raw reports and DICOM never leave Kaggle. No OAI data.
