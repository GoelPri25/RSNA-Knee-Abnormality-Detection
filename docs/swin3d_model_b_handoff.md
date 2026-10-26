# Handoff: Model B (Swin3D-T on d96n), as of 2026-10-15

For review by a teammate. Model B is the second model family next to the 2.5D CNNs (Exp 1e and relatives): a whole-series
3D video transformer instead of a 2D backbone over slice windows. The full record, with every pre-registration and
number, is the B1, B2 and B3-abl rows of `docs/experiment_ledger.md`; this document is the map.

---

## 1. Where things stand

| Model | Labels | Fold 0 clear6 / 10-target | Public LB | Status |
|---|---|---|---|---|
| **B1** Swin3D-T, per-plane stage-4 branches | per-target (1e, sha d40803d0…) | 0.9499 / 0.8866 | **0.906** | failed floor; overfit after epoch 10 |
| **B2** Swin3D-T, frozen early stages, shared stage 4, token attention, MR augmentation | label-v2 (sha dd0ab9ab…) | 0.9542 / 0.8941 | **0.920** | failed fold-0 gates; overfit fixed; **in the best ensemble**: L4 + B2 = 0.944 (2026-10-15, ledger E-ref) |
| B3-abl: A0 (B2 recipe) vs A1 (nothing frozen), 6 epochs, Kaggle T4 | label-v2 | A0 0.9298 / 0.8675; A1 0.9261 / 0.8631 | – | **unfreezing lost** (fine-4 −0.0069 [−0.0113, −0.0023]); B3 not run, **Model B closed** (2026-10-15) |
| Comparators: Exp 1e / 1e on label-v2 | per-target / label-v2 | 0.9719 / 0.9073; 0.9739 / 0.9074 | 0.936 / 0.937 | best single models |

clear6 = clear-band AUC (report moderate+ positives vs confident negatives) on ACL, Medial Meniscus, Medial OA,
Lateral OA, PF OA, Effusion. 10-target = all-cell macro AUC without Lateral OA and Baker's. Both are on fold 0 (825 studies)
against the unchanged Steven-v4 validation labels.

**Main findings so far**

1. The clear-band screen (LB ≈ −0.687 + 1.671 × clear6, fitted on 2.5D CNNs) **underrates Swin3D**: it predicted
   0.900 for B1 (actual 0.906) and 0.907 for B2 (actual 0.920). Fold 0 implied B2 was ~0.033 LB behind 1e-label-v2; the
   real gap is 0.017.
2. B2's deficit is concentrated in fine structures. Vs 1e-label-v2 on the same labels (all cells): ACL −0.055,
   Lateral Meniscus −0.038, PF OA −0.041, Lateral OA −0.029; level or ahead on Effusion, Synovitis, Baker's, Contusion,
   Fracture.
3. B1 (nothing frozen) at its epoch 10 was better on ACL (0.898 vs B2 0.855) and PF OA clear band (0.935 vs 0.910),
   so freezing patch embedding + stages 1–2 was the first suspect. The ablation (A0 vs A1, 6 epochs) rejected it: unfreezing was worse on all four fine targets (fine-4 −0.0069 [−0.0113, −0.0023]), so B1's advantage came from something else (per-plane branches, flat LR or labels). Scored by `scripts/score_swin3d_b3_ablation.py`; ledger row B3-abl.
4. B2's gap does not depend on slice thickness; B1-ep10's ACL gap does (thin −0.002, thick −0.041).
5. As a partner: mean(B2, 1e-label-v2) clear6 0.9713 vs mean(ConvNeXt P1-cnx, 1e-label-v2) 0.9763. Rank correlation
   with 1e-label-v2: B2 0.845, P1-cnx 0.854. On the LB (ledger E-ref, 2026-10-15) the opposite held: **L4 + B2 scored 0.944**
   (new best, kernel `tumis001/rsna-lean4-b2-d96-probability-ensemble`), L4 + P1-cnx 0.943 (= L4). The +0.001 is inside LB
   noise (~0.003), but by the pre-registered rule B2 stays in the ensemble. "Model B closed" means no new Swin3D training;
   the B2 weights remain in use. The fold-0 screen had L4 + B2 at −0.0005 vs L4 (third time it underrated Swin3D).

## 2. Recipes

**Input (d96n cache, both models).** Every native slice, ≤ 40 per series, 4 series slots (sagittal fluid, sagittal
non-fluid, coronal fluid, axial fluid-preferred); each series is one volume [T, 320, 320] (centre 320 crop of 384 px over
140 mm), grey repeated to 3 channels, Kinetics normalisation. Built by `kaggle/experiment_kernels/d96_native_cache/`
(6 shard kernels `tumis001/rsna-d96-native-cache-shard-{0..5}`).

**B1** (`SWIN_PHASE=swin3d_b`). torchvision Swin3D-T, Kinetics-400 weights (sha 7615ae03…). Patch embedding + stages
1–3 shared, patch merging 3 + stage 4 + norm one branch per plane (59.0M params). Head: stage-4 tokens averaged over
space (one per depth), LayerNorm + series embedding, per-finding attention over all tokens, per-finding read-out.
AdamW backbone 5e-5 / head 1e-3, wd 0.05, 25 epochs, 1 warm-up, cosine to 1%, SWA 21–25, drop path 0.1, bf16, one study
per forward, 8 per step, seed 2026. Augmentation at 1e's values (mirror, affine, gamma, gain). val12 peaked at epoch 10
(0.889) and val BCE rose 0.492 → 0.532 by SWA.

**B2** (`SWIN_PHASE=swin3d_b2`). B1 plus: label-v2; patch embedding + stages 1–2 **frozen**; **one stage-4 branch
shared by all planes** (26.5M of 28.1M trainable); drop path 0.25; layer-wise lr decay 0.9; per-finding attention over
**every stage-4 token** (T/2 × 10 × 10 per series); 15 epochs, SWA 11–15. Augmentation per series: affine / gamma /
gain, series dropout p 0.1, every-second-slice p 0.25, bias field p 0.3, blur / unsharp p 0.2 each, noise p 0.2. Val
BCE min 0.507 at epoch 11, SWA 0.512. Train time 1.43 h on an H100.

Eval for both: centre 320, all series, no TTA (mirror TTA reported as secondary).

**B3-abl** (Kaggle T4 × 2, fp16 + GradScaler, `SWIN_CFG` overrides). A0 = B2 recipe; A1 = nothing frozen (lr decay
continued to the patch embedding, gradient checkpointing for memory); A2 (temporal patch 1) does not fit Kaggle and was
not run. 6 epochs, warm-up 1, cosine, SWA 5–6. Pre-registered rules (ledger B3-abl): unfreeze adopted if
A1 − A0 ≥ +0.010 on fine-4 (mean all-cell AUC of ACL, both menisci, PF OA) and val12 not worse by > 0.002; B3 go only
if projected clear6 ≥ 0.965 (partner) or ≥ 0.9745 (single model).

## 3. Code (this repository)

| File | Purpose |
|---|---|
| `scripts/swin3d_d96n_runner.py` | Model B runner: model, augmentation, training, eval, SWA; `SWIN_MODE=profile` runs preflight checks + timing. Env: `SWIN_PHASE` (`swin3d_b` / `swin3d_b2`), `D96_CACHE`, `D96_ASSETS`, `D96_OUT`, `SWIN_AMP=fp16` (T4), `SWIN_CFG` (JSON overrides, B2 only), `SWIN_WEIGHTS` |
| `scripts/modal_coatnet_d96.py` | Modal launcher: `--phase swin3d_b2` trains on one H100 (`train_b`); `--swin-profile-run --phase swin3d_b2` profiles; `--bench-b-run` runs the encoder benchmark |
| `scripts/bench_model_b_encoders.py` | Encoder benchmark used to pick Swin3D-T (speed, memory, fused attention parity) |
| `scripts/audit_phase1.py` | Standard scorer (clear6, 10-target, per target, bootstrap vs 1e, partner check, curve) |
| `scripts/diag_swin3d_b3.py` | B3 diagnostics: clear band per target, pairs, rank correlation, gap vs slice thickness. `python scripts/diag_swin3d_b3.py <project root>` (expects the run folders under `data/coatnet_d96_fold0/`) |
| `scripts/check_swin3d_tpatch.py` | CPU check that the temporal-patch-1 inflation is exact on repeated slices |
| `kaggle/submission_kernels/d96n_swin3d_b{1,2}_submission/` | LB inference kernels (rebuild d96n from DICOM, fp32, parity on 8 fold-0 studies) |
| `kaggle/experiment_kernels/swin3d_ablation/run.py` | Kaggle ablation driver (one arm per GPU, `MODE` = profile / train) |

Modal: the runs used a personal Modal account; with your own account the launcher works as is after `modal token new`
(it copies the d96n shard outputs from Kaggle to a volume first, which needs your Kaggle token).

## 4. Sources: data, labels, checkpoints

Everything below is private under `tumis001` unless marked public or yours. Hashes are sha256 (prefix shown; full
values in `manifest.json` of the review bundle). The runner checks the label, fold and pretrained-weight hashes at start
and refuses to train on a mismatch.

### 4.1 Image data

| Item | Source | Notes |
|---|---|---|
| Raw DICOM + `train.csv` | competition [`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | Read only by the cache kernels and the submission kernels |
| d96n training cache (4,407 studies = 4,349 training + 58 gold, 6 shards by study-id hash) | outputs of kernels [`rsna-d96-native-cache-shard-0`](https://www.kaggle.com/code/tumis001/rsna-d96-native-cache-shard-0) … [`-shard-5`](https://www.kaggle.com/code/tumis001/rsna-d96-native-cache-shard-5) | Built from the competition DICOM by `kaggle/experiment_kernels/d96_native_cache/run_template.py` (rendered per shard by `render.py`); CPU, no internet. Each shard: `manifest.jsonl` + uint8 arrays, plus `cache_summary.json`. The kernels also mount `tumis001/rsna-knee-hpo-v31-d2-report-labels`, used only for the list of labelled study ids |
| d96n on Modal | volume `rsna-coatnet-d96-cache`, folder `d96n/` | Copied from the six shard outputs by `scripts/modal_coatnet_d96.py` (needs a Kaggle token); identical content |
| d96n at test time | rebuilt from DICOM inside the submission kernels | Same code path as the cache (parity checked on 8 fold-0 studies, max \|Δp\| 0.0062 for B2) |

### 4.2 Labels and splits

| File | sha256 | Used by | Origin |
|---|---|---|---|
| `soft_labels_final_per_target_any_model.json` (per-target set, "1e labels") | d40803d0… | B1 training | Your [`priyanjaliml/rsna-knee-final-per-target-labels`](https://www.kaggle.com/datasets/priyanjaliml/rsna-knee-final-per-target-labels); local copy `data/coatnet_d96_fold0/assets/` |
| `soft_labels_final_per_target_latD2.json` (label-v2) | dd0ab9ab… | B2 and B3-abl training | Your [`priyanjaliml/rsna-knee-labelv2-checkpoint`](https://www.kaggle.com/datasets/priyanjaliml/rsna-knee-labelv2-checkpoint) (2026-10-14); local copies `data/teammate_labelv2/` and `data/coatnet_d96_fold0/assets/`; re-shared in [`rsna-swin3d-ablation-code`](https://www.kaggle.com/datasets/tumis001/rsna-swin3d-ablation-code) |
| `validation_labels.json` (version `steven_v4_fold0_validation_only`) | 10fdaaab… | fold-0 scoring for every run (never modified) | Steven-v4 confident cells for fold 0, from the public [`stevenleehans/rsna-knee-llm-report-labels`](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) (`llm_labels_v4_blend.csv`); the same file every d96 run since Exp 1 is scored on (`docs/handoff_d96_2026-10-12.md`); copy in `rsna-swin3d-ablation-code` |
| `folds.csv` (study_id → fold; fold 0 = 825 validation studies, folds 1–4 = 3,243, fold 5 = 281 training-only additions) | 2dd1044a… | all runs (train on folds 1–5) | The d96 project split, unchanged since Exp 1; copy in `rsna-swin3d-ablation-code` |
| Gold-58 labels | – | dev-only gold predictions in the review bundle | Competition `train.csv` (58 studies with image-level labels); never used for training or selection |

### 4.3 Pretrained weights

| Item | Source | sha256 |
|---|---|---|
| Swin3D-T Kinetics-400 | torchvision `Swin3D_T_Weights.KINETICS400_V1` (download URL in `torchvision.models.video`) | 7615ae03… (checked by the runner and the Kaggle ablation driver; loaded `weights_only=True`) |

### 4.4 Checkpoints and run outputs

| Run | Kaggle (weights) | Modal volume `rsna-coatnet-d96-cache` | Local (`data/coatnet_d96_fold0/`) |
|---|---|---|---|
| B1 | [`rsna-d96n-swin3d-b1-fold0`](https://www.kaggle.com/datasets/tumis001/rsna-d96n-swin3d-b1-fold0): `weights_swa21_25.pt` (SWA of epochs 21–25), sha 9c0d4444…, + `parity_fold0.json` | `experiments/coatnet_d96_fold0/run_swin3d_b/` | `run_swin3d_b/` |
| B2 | [`rsna-d96n-swin3d-b2-fold0`](https://www.kaggle.com/datasets/tumis001/rsna-d96n-swin3d-b2-fold0): `weights_swa11_15.pt` (SWA of epochs 11–15), sha 7853700f…, + `parity_fold0.json` | `experiments/coatnet_d96_fold0/run_swin3d_b2/` | `run_swin3d_b2/` |
| B3-abl A0 / A1 | kernel [`rsna-swin3d-b3-ablation`](https://www.kaggle.com/code/tumis001/rsna-swin3d-b3-ablation) output v3 (complete): `A0_frozen/` and `A1_unfrozen_ckpt/` `weights_swa5_6.pt` (sha 297e66e9… / b96d41ae…), per-epoch weights deleted; predictions and logs also in the review bundle `ablation_train_v3/` | – | `data/kaggle_kernel_outputs/swin3d_b3_ablation_train_v3/` |
| 1e-label-v2 (comparator) | your `priyanjaliml/rsna-knee-labelv2-checkpoint`: `r50_labelv2_swa21_25.pt`, sha 1c6006b1… | – | `data/teammate_labelv2/` |

Only the SWA weights are on Kaggle. The per-epoch weights (`weights_epochNN.pt`) and `recovery.pt` of B1 / B2 stay in the Modal run folders (personal account; ask if you need a specific epoch, e.g. B1 epoch 10). Predictions, logs and histories for every run are in the review
bundle [`rsna-swin3d-model-b-review`](https://www.kaggle.com/datasets/tumis001/rsna-swin3d-model-b-review) (layout below).

### 4.5 Kernels and the review bundle

| Resource | Contents |
|---|---|
| Kernels [`rsna-d96n-swin3d-b1-submission`](https://www.kaggle.com/code/tumis001/rsna-d96n-swin3d-b1-submission), [`-b2-submission`](https://www.kaggle.com/code/tumis001/rsna-d96n-swin3d-b2-submission) | LB 0.906 / 0.920; source in `kaggle/submission_kernels/d96n_swin3d_b{1,2}_submission/` |
| Kernel [`rsna-swin3d-b3-ablation`](https://www.kaggle.com/code/tumis001/rsna-swin3d-b3-ablation) | ablation (profile v1 / v2, training v3); source `kaggle/experiment_kernels/swin3d_ablation/` |
| Dataset [`rsna-swin3d-ablation-code`](https://www.kaggle.com/datasets/tumis001/rsna-swin3d-ablation-code) | runner snapshot (a84ecfbe…, = repo version at commit efcb1a8), `coatnet_d96_runner.py`, `d96_positions.py`, `folds.csv`, label-v2, `validation_labels.json` |
| Dataset [`rsna-swin3d-model-b-review`](https://www.kaggle.com/datasets/tumis001/rsna-swin3d-model-b-review) | `run_swin3d_b/`, `run_swin3d_b2/`: `val_*.json` (fold 0: SWA, SWA + mirror, best epoch, each scored epoch), `gold_*.json`, `history.json`, `run_summary.json` (full config), `training.log`; `comparators/`: fold-0 predictions of 1e and 1e-label-v2; `audits/`; `profiles/` (Modal B1 profile + encoder bench, Kaggle ablation profiles); `lb_submission_b2/`; `diagnostics/b3_diag.txt`; `SHA256SUMS`, `manifest.json` |

Weights load with `torch.load(..., weights_only=True)`; the state dict keys match `SwinB` in the runner for the given
`SWIN_PHASE`. Prediction JSONs are the runners' standard format (`{"model", "note", "targets_in_order",
"predictions": [{"study_id", "probabilities": [12]}]}`), so `scripts/audit_phase1.py` scores them directly. Their
`model` field reads `d96_r50_exp4b_native_pertarget_v1` for both B1 and B2: the Swin runner reuses the CNN runner's
writer, which stamps the CNN module's version. The run is identified by its folder and by `run_summary.json`
(`cfg.version` `d96n_swin3d_t_b_v1` / `d96n_swin3d_t_b2_v1`).

## 5. Caveats

- One seed per run; between-run per-target noise is ±0.004–0.019 AUC, LB noise about ±0.002.
- B2 changed labels and model together; label-v2 was +0.001 LB on the 1e recipe, so most of B1 → B2 is the model.
- Lateral OA and MCL training labels differ in label-v2 but are scored against the unchanged fold-0 reference.
- The ablation runs fp16 on T4 (Modal runs were bf16); A0 is the control for that.
- Gold-58 predictions are included as development evidence only; nothing was selected on gold.
