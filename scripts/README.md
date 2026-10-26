# Scripts

Flat on purpose: the runners import each other as sibling modules (and the Slurm / Modal / Colab launchers
put this folder on `PYTHONPATH`), so moving files would break them. Grouped index below; the purpose column is the
first docstring line of each file.

Sub-folders:
- [`online_kd/`](online_kd): knowledge distillation (offline / online / ranking), backbone patches (ConvNeXt-S, ResNet-101, BiT-101), fold-0 scoring, the shared inference core with rank-blend and mirror-TTA kernels, Modal lanes and Colab CLI tooling.
- [`meluxina/`](meluxina): Slurm jobs, container definitions and per-lane environment files for the HPC runs.
- [`fm_teacher/`](fm_teacher): teacher experiments with external unlabelled knee MRI, k-fold soups.

## Core training runners

| File | Purpose |
|---|---|
| [`coatnet_d96_runner.py`](coatnet_d96_runner.py) | Experiment 1: the public 0.949 CoAtNet recipe trained on our fold-0 split and HPO-v3.1-D2 labels (H100). |
| [`convnext_v31_runner.py`](convnext_v31_runner.py) | Controlled ConvNeXt-Tiny runner with epoch-boundary recovery (copy of efficientnet_v31_runner.py; only the |
| [`efficientnet_v31_runner.py`](efficientnet_v31_runner.py) | Controlled EfficientNetV2-S runner with epoch-boundary recovery. |
| [`fullband224_runner.py`](fullband224_runner.py) | Phase 1 coverage run: the original 0.925 r50 trainer and recipe, reading the full-band 224px cache. |
| [`highres336_runner.py`](highres336_runner.py) | Resolution-only ResNet50 reader, durable epochs, dual validation references. |
| [`j2x_runner.py`](j2x_runner.py) | Exp J2-X: ResNet-50 (label-v2 / 1e recipe) + Swin3D-T (B2 recipe) trained together, output = the two heads' logit |
| [`joint_r50_swin_runner.py`](joint_r50_swin_runner.py) | Exp J1: one network with a 2D / 2.5D ResNet-50 branch (Exp 1e) and a 3D Swin3D-T branch (Model B2), joined by |
| [`l4c_loc_runner.py`](l4c_loc_runner.py) | Exp L4C-LOC (docs/l4c_loc_spec.md v2, ledger row L4C-LOC): a learned crop locator for the frozen v4 l4copy, distilled |
| [`l4copy_r1_runner.py`](l4copy_r1_runner.py) | Exp L4C-R1 (round 1 on v4 l4copy, ledger row L4C-R1): v4 l4copy trained on crops it places itself (MS1-free). |
| [`l4copy_r2_runner.py`](l4copy_r2_runner.py) | Exp L4C-R2 (round 2 on v4 l4copy, ledger row L4C-R2): warm-start the LB-0.939 l4copy (run_r50_gl4_mlx/v4_l4copy_s2026, |
| [`local_global_runner.py`](local_global_runner.py) | Pinned baseline loop, native data/model adapters, durable epoch recovery. |
| [`ms1_gl_runner.py`](ms1_gl_runner.py) | Exp MS1-GL: MS1 trained on both views, the whole field (global) and per-finding zoomed crops (local), with one shared |
| [`r50_cosine_ema_runner.py`](r50_cosine_ema_runner.py) | ResNet50 v3.1-D2 224px with a fixed schedule: warmup + cosine over 20 epochs, no early stopping, no |
| [`r50_gl3_runner.py`](r50_gl3_runner.py) | Exp r50-GL v3: one ResNet-50 that reads the whole study (global windows) and per-finding zoomed regions (local), with |
| [`r50_gl4_runner.py`](r50_gl4_runner.py) | Exp r50-GL v4: one ResNet-50 that reads the whole study (global windows) and a zoom of each finding (local crops), |
| [`r50_gl_runner.py`](r50_gl_runner.py) | Exp r50-GL: the label-v2 r50 recipe trained on two views of each study, the whole field (global windows, as label-v2 |
| [`swin3d_d96n_runner.py`](swin3d_d96n_runner.py) | Model B (Phase 2, pre-registered in the ledger before launch): a 3D video transformer on whole native-density series. |
| [`synovitis_weight_candidate_runner.py`](synovitis_weight_candidate_runner.py) | Original 224px ResNet50 with persistent epoch recovery and fixed validation. |
| [`synovitis_weight_runner.py`](synovitis_weight_runner.py) | Original 224px ResNet50 with persistent epoch recovery and fixed validation. |

## Cloud / cluster launchers (Modal, Colab, Kaggle)

| File | Purpose |
|---|---|
| [`colab_lane_guard.sh`](colab_lane_guard.sh) | Host-side guard for a detached PUB-KD lane on a Colab CLI session (after scripts/colab_pubkd_lane.sh started it). |
| [`colab_pubkd_lane.sh`](colab_pubkd_lane.sh) | PUB-KD lane on a Colab VM (after scripts/colab_pubkd_setup.py printed SETUP_OK): pubkd.sbatch's run() for one lane, |
| [`colab_pubkd_setup.py`](colab_pubkd_setup.py) | PUB-KD lanes on Colab (Colab CLI): VM-side setup, run with `colab exec -f` after the bundle and token are uploaded. |
| [`colab_ssh.sh`](colab_ssh.sh) | SSH into a Colab CLI session (separate from its kernel, so it works while an exec runs), via the CLI's proxy mode. |
| [`kaggle_audit.sh`](kaggle_audit.sh) | Run inside a Kaggle notebook terminal after cloning/installing this repository. |
| [`kaggle_cli.py`](kaggle_cli.py) | Container entry point: credentials mounted read-only, never copied into the project. |
| [`local_global_data.py`](local_global_data.py) | Native-neighbor cache and selected-window batches shared by train and inference. |
| [`local_global_model.py`](local_global_model.py) | One ResNet pass: native local triplets and study-wide contextual correction. |
| [`local_global_resnet34_kaggle.py`](local_global_resnet34_kaggle.py) | Kaggle native local+global ResNet34, HPO trainval, all 60 epochs. |
| [`local_global_speed.py`](local_global_speed.py) | Short paired timing preflight; retain only faster choices within baseline VRAM. |
| [`modal_cnn_auc_margin.py`](modal_cnn_auc_margin.py) | Detached T4 AUC refinement; persistent caches and full resumable state. |
| [`modal_coatnet_d96.py`](modal_coatnet_d96.py) | Experiment 1 on Modal: copy the d96 cache shards from Kaggle, CPU preflight, train on one H100. |
| [`modal_coatnet_gpu_bench.py`](modal_coatnet_gpu_bench.py) | Experiment 1 GPU choice: CoAtNet-2 training/eval throughput on Modal T4 / L4 / A100-80GB / H100. |
| [`modal_convnext_v31.py`](modal_convnext_v31.py) | Single detached T4 experiment; cached inputs and recoverable state on Volume. |
| [`modal_cost.py`](modal_cost.py) | Modal cost per app (run) from the workspace billing report, split into GPU / CPU / memory. |
| [`modal_d96_eval_crop.py`](modal_d96_eval_crop.py) | Exp 1d (inference only): re-predict fold 0 + gold with the existing d96 SWA weights at eval crops 320 / 352 / 384. |
| [`modal_efficientnet_v31.py`](modal_efficientnet_v31.py) | Single detached T4 experiment; cached inputs and recoverable state on Volume. |
| [`modal_fmcache.py`](modal_fmcache.py) | fastMRI d96 cache on Modal (2026-10-22, for lane C): r7's build (fmb.tar = fastmri_build_cache.py sha 451f3ed8 + |
| [`modal_fullband224.py`](modal_fullband224.py) | Detached T4 run: Phase 1 coverage. r50 v3.1-D2 224px with the original 0.925 recipe on the full-band cache |
| [`modal_highres336.py`](modal_highres336.py) | Detached, single-T4 336px ResNet50 training with persistent recovery. |
| [`modal_highres336_l4.py`](modal_highres336_l4.py) | Detached, single-T4 336px ResNet50 training with persistent recovery. |
| [`modal_hpo_blend_add281_auc_control.py`](modal_hpo_blend_add281_auc_control.py) | Repeat the HPO +281 run on one personal Modal T4 with AUC-aligned control. |
| [`modal_hpo_blend_add281_coatnet.py`](modal_hpo_blend_add281_coatnet.py) | Compare CoAtNet-0 with the 0.918 HPO-v1 +281 ResNet-34 run on one T4. |
| [`modal_hpo_blend_add281_resnet50.py`](modal_hpo_blend_add281_resnet50.py) | Compare ResNet-50 with the 0.918 HPO-v1 +281 ResNet-34 run on one T4. |
| [`modal_hpo_blend_add281_resnet50_radimagenet.py`](modal_hpo_blend_add281_resnet50_radimagenet.py) | Train the HPO-v1 +281 ResNet-50 reader from RadImageNet initialization. |
| [`modal_hpo_blend_add281_trainonly.py`](modal_hpo_blend_add281_trainonly.py) | Add the 281 unassigned HPO studies to training on one personal Modal T4. |
| [`modal_hpo_blend_attention_only.py`](modal_hpo_blend_attention_only.py) | Train the isolated same-series attention reader on one personal Modal T4. |
| [`modal_hpo_blend_fit160_trainonly.py`](modal_hpo_blend_fit160_trainonly.py) | Train the 0.918 HPO-v1 CNN recipe with Fit160 preprocessed cache shards. |
| [`modal_hpo_blend_resnet.py`](modal_hpo_blend_resnet.py) | Run the HPO-v1 label comparison on one personal Modal T4. |
| [`modal_hpo_blend_resnet50_warmstart.py`](modal_hpo_blend_resnet50_warmstart.py) | Warm-start a short ResNet-50 refinement from its best HPO-blend checkpoint. |
| [`modal_hpo_blend_sequence_conditioned.py`](modal_hpo_blend_sequence_conditioned.py) | Train the isolated sequence-conditioned classifier on one personal Modal T4. |
| [`modal_hpo_blend_series_attention.py`](modal_hpo_blend_series_attention.py) | Train the warm-started, same-series attention reader on one personal Modal T4. |
| [`modal_hpo_v2_provenance_candidate.py`](modal_hpo_v2_provenance_candidate.py) | Train the HPO v2 provenance candidate with the 0.924 ResNet-50 recipe. |
| [`modal_local_global.py`](modal_local_global.py) | Explicit CPU transfer or detached single-GPU training; no automatic duplicate run. |
| [`modal_local_global_hpo_validation.py`](modal_local_global_hpo_validation.py) | Detached local+global HPO-validation comparison; reuse all four native shards. |
| [`modal_medgemma_stage0.py`](modal_medgemma_stage0.py) | Exp MG0 (ledger row MG0): MedGemma 1.5 4B as a knee-MRI reader, Stage 0 = what its pretraining gives for free. |
| [`modal_od_probe.py`](modal_od_probe.py) | OrthoDiffusion go/no-go probe on Modal: frozen OrthoDiffusion + ImageNet r50 features for all d96 cache shards |
| [`modal_pillar0_reader.py`](modal_pillar0_reader.py) | Run one report-fold Pillar-0 reader experiment on a personal Modal T4. |
| [`modal_pillar0_regional_reader.py`](modal_pillar0_regional_reader.py) | Five-fold regional Pillar-0 reader experiment on personal Modal T4. |
| [`modal_pubkd.py`](modal_pubkd.py) | PUB-KD / KD-CV lanes on Modal: the MeluXina job scripts/meluxina/pubkd.sbatch, one lane per H100 container. |
| [`modal_r50_cosine_ema.py`](modal_r50_cosine_ema.py) | Detached T4 run: r50 v3.1-D2 224px, fixed warmup+cosine schedule (20 epochs, peak 1.5e-4), EMA 0.999, seed2026. |
| [`modal_report_label_exp4.py`](modal_report_label_exp4.py) | Run a matched label-source screen with the original Experiment 4 trainer on Modal. |
| [`modal_steven_contrastive_exp.py`](modal_steven_contrastive_exp.py) | Run a matched BCE-vs-contrastive Experiment 4 comparison on Modal T4. |
| [`modal_steven_temporal_cnn.py`](modal_steven_temporal_cnn.py) | Run the paired Steven v4 temporal-CNN vs Experiment 4 comparison on Modal. |
| [`modal_synovitis_weight_candidate.py`](modal_synovitis_weight_candidate.py) | Detached candidate-only T4 run; existing 224px cache; original baseline seed2026. |
| [`modal_synovitis_weight_pair.py`](modal_synovitis_weight_pair.py) | One detached coordinator, sequential matched T4 runs, existing 224px cache. |
| [`preflight_r1.py`](preflight_r1.py) | MeluXina preflight for the L4C-R1 job (first step of the allocation; exit != 0 stops the job): offline pretrained |
| [`preflight_v4.py`](preflight_v4.py) | MeluXina preflight for the r50-GL v4 job (first step of the allocation, ~2-5 min on one GPU; exit != 0 stops the job): |
| [`run_modal_personal_attention_only.py`](run_modal_personal_attention_only.py) | Run the AUC-control experiment with the isolated personal Modal profile. |
| [`run_modal_personal_auc_control.py`](run_modal_personal_auc_control.py) | Run the AUC-control experiment with the isolated personal Modal profile. |
| [`run_modal_personal_coatnet.py`](run_modal_personal_coatnet.py) | Launch one detached CoAtNet experiment using only the personal Modal account. |
| [`run_modal_personal_convnext_v31.py`](run_modal_personal_convnext_v31.py) | Launch the ConvNeXt-Tiny v3.1-D2 fold-0 run on personal Modal in detached mode. |
| [`run_modal_personal_efficientnet_v31.py`](run_modal_personal_efficientnet_v31.py) | Launch the HPO v2 provenance comparison on personal Modal in detached mode. |
| [`run_modal_personal_fit160.py`](run_modal_personal_fit160.py) | Launch the Fit160 cache stage and baseline-comparison run on personal Modal. |
| [`run_modal_personal_fullband224.py`](run_modal_personal_fullband224.py) | Deploy and spawn the Phase 1 full-band run on the personal Modal profile, independent of the local CLI. |
| [`run_modal_personal_hpo_v2_provenance.py`](run_modal_personal_hpo_v2_provenance.py) | Launch the HPO v2 provenance comparison on personal Modal in detached mode. |
| [`run_modal_personal_local_global.py`](run_modal_personal_local_global.py) | Deploy and launch the local+global job using personal credentials. |
| [`run_modal_personal_local_global_hpo_validation.py`](run_modal_personal_local_global_hpo_validation.py) | Launch a single detached HPO-validation job using personal Modal credentials. |
| [`run_modal_personal_r50_cosine_ema.py`](run_modal_personal_r50_cosine_ema.py) | Deploy and spawn the r50 cosine+EMA run on the personal Modal profile, independent of the local CLI. |
| [`run_modal_personal_radimagenet.py`](run_modal_personal_radimagenet.py) | Launch the RadImageNet ResNet-50 experiment detached on the personal Modal account. |
| [`run_modal_personal_resnet50_warmstart.py`](run_modal_personal_resnet50_warmstart.py) | Launch the ResNet-50 warm-start run with only the personal Modal profile. |
| [`run_modal_personal_sequence_conditioned.py`](run_modal_personal_sequence_conditioned.py) | Run the AUC-control experiment with the isolated personal Modal profile. |
| [`run_modal_personal_sequence_from_scratch.py`](run_modal_personal_sequence_from_scratch.py) | Launch the from-scratch sequence experiment in a detached personal Modal app. |
| [`run_modal_personal_series_attention.py`](run_modal_personal_series_attention.py) | Run the AUC-control experiment with the isolated personal Modal profile. |
| [`run_modal_personal_synovitis_candidate.py`](run_modal_personal_synovitis_candidate.py) | Resume the saved candidate as a deployed job independent of the local CLI. |

## Data, cache and label builders

| File | Purpose |
|---|---|
| [`build_d2_border_labels.py`](build_d2_border_labels.py) | Write the Exp 3a label set as a file: HPO-v3.1-D2 with borderline cells (0.3 < p < 0.7) set to 0.85 for |
| [`build_d2_state_labels.py`](build_d2_state_labels.py) | Write the Exp 3b label set: HPO-v3.1-D2 with cells re-valued by D2's own rule-extractor state (fixed rule, no model). |
| [`build_e3_train_labelset.py`](build_e3_train_labelset.py) | E3 labelset for the teammate (user request 2026-10-17, choice "E3 in-sample, all studies"). CPU, numpy/pandas only. |
| [`build_fm_targets.py`](build_fm_targets.py) | SURPASS-F joint teacher targets: the PUB-KD-R Raptor teacher (0.65 pct(384) + 0.35 pct(224), order; logit blend, |
| [`build_gold_auc_html.py`](build_gold_auc_html.py) | Build a self-contained HTML report of the gold AUC analysis (no network, no external assets, no images of data). |
| [`build_kaggle_kd_gap_kernel.py`](build_kaggle_kd_gap_kernel.py) | Build the Kaggle kernel for KD-GAP (ledger row KD-GAP): why do the r50 students stay below the Raptor teacher? |
| [`build_kaggle_kd_view_sweep.py`](build_kaggle_kd_view_sweep.py) | Build the Kaggle kernel for KD-VIEW (ledger row KD-VIEW): which student training view keeps the teacher's answer and |
| [`build_kaggle_xai_kdr_kernel.py`](build_kaggle_xai_kdr_kernel.py) | Build the Kaggle kernel for the PUB-KD-R xAI comparison on fold 0 (kaggle/submission_kernels/pubkd_r_ra_xai_fold0/): |
| [`build_kaggle_xai_kernel.py`](build_kaggle_xai_kernel.py) | Build the Kaggle kernel for the v4 l4copy XAI + ablations on fold 0 (kaggle/submission_kernels/v4_l4copy_xai_fold0/). |
| [`build_l4c_quota_prior.py`](build_l4c_quota_prior.py) | Quota prior for L4C-LOC's zero-training baseline Q (docs/l4c_loc_spec.md): from MS1's crop table on the training folds |
| [`build_native_triplet_cache.py`](build_native_triplet_cache.py) | Build deduplicated native-neighbor slices; shared preprocessing for inference. |
| [`build_pubkd_swap_targets.py`](build_pubkd_swap_targets.py) | PUB-KD-E targets: teacher ORDER from one public teacher, VALUE scale from the other (labelsets audit 2026-10-20). |
| [`build_pubkd_targets.py`](build_pubkd_targets.py) | PUB-KD teacher targets: the public 0.958 blend (goodpjw2008 v25/27) re-computed on our studies, as training targets. |
| [`build_rulecap_labels.py`](build_rulecap_labels.py) | Label-v3 (CPU, no LLM, no model): label-v2 with rule-state caps on every target. |
| [`build_severity_aux_targets.py`](build_severity_aux_targets.py) | Exp P1-sev asset (CPU, 2026-10-14): cumulative severity targets for the auxiliary heads, training folds only. |
| [`build_student_labels.py`](build_student_labels.py) | Phase 3 student labels (CPU): mix the per-target training labels with cross-fitted teacher predictions. |
| [`build_student_labels_v2.py`](build_student_labels_v2.py) | P3 v2 student labels (CPU; ledger P3v2-S): label-v2 mixed with cross-fitted CK-A ConvNeXt teacher predictions. |
| [`build_synovitis_weight_experiment.py`](build_synovitis_weight_experiment.py) | Training-only Synovitis weight ablation; never modifies target probabilities. |
| [`build_tag_aux_targets.py`](build_tag_aux_targets.py) | Exp P1-tags asset (CPU, 2026-10-16): extra-finding tag targets for the auxiliary heads, training folds only. |
| [`build_teacher_splits.py`](build_teacher_splits.py) | Phase 3 teacher splits (CPU): 4 cross-fitting splits over the 3,524 training studies (folds 1-5); fold 0 untouched. |
| [`extract_fusion_features_gpu.py`](extract_fusion_features_gpu.py) | Exp F0 (fusion diagnostic): per-finding pooled features of 1e (ResNet-50, d96) or B2 (Swin3D-T, d96n) on fold 0. |
| [`extract_ms1_crop_features_gpu.py`](extract_ms1_crop_features_gpu.py) | Exp MS1-CP (crop probe): does a zoomed crop chosen by MS1's own token maps add signal to MS1's whole-field view? |
| [`extract_multiplane_features_gpu.py`](extract_multiplane_features_gpu.py) | Exp MP-0 (multi-plane probe): does a zoomed view of the same anatomy point in the sagittal, coronal and axial series |
| [`make_ids_eval.py`](make_ids_eval.py) | Kaggle submission kernel -> eval twin on the first N study IDs of tumis001/rsna-fast-eval/eval_ids.txt |
| [`prepare_auc_margin_experiment.py`](prepare_auc_margin_experiment.py) | Prepare fixed report-only evidence and validation inputs without GPU work. |
| [`prepare_blended_label_dataset.py`](prepare_blended_label_dataset.py) | Convert the public HPO blended labels to our CNN reader's JSON format. |
| [`prepare_highres336_datasets.py`](prepare_highres336_datasets.py) | Sequential CPU cache preparation, private Kaggle publication and Modal upload. |
| [`prepare_hpo_add281_split.py`](prepare_hpo_add281_split.py) | Extend the HPO fold-0 training split with the 281 previously unassigned rows. |
| [`prepare_hpo_v2_provenance_candidate.py`](prepare_hpo_v2_provenance_candidate.py) | Build a conservative, provenance-aware candidate from the HPO v1 labels. |
| [`prepare_kaggle_experiment.py`](prepare_kaggle_experiment.py) | Build the readable, cell-by-cell Kaggle milestone notebook. |
| [`prepare_kaggle_notebook.py`](prepare_kaggle_notebook.py) | Bundle only project source into a private, self-contained Kaggle audit notebook. |
| [`prepare_local_global_hpo_validation.py`](prepare_local_global_hpo_validation.py) | Build locked trainval assets from the completed local+global experiment. |
| [`prepare_report_extraction_commit.py`](prepare_report_extraction_commit.py) | Prepare the committed Kaggle report extraction notebook. |

## Training entry points

| File | Purpose |
|---|---|
| [`train_cnn_auc_margin.py`](train_cnn_auc_margin.py) | Fine-tune the pinned HPO-v1 ResNet50 with soft BCE and masked AUC margin. |
| [`train_cnn_coatnet_wrapper.py`](train_cnn_coatnet_wrapper.py) | Load the audited CNN trainer with an ImageNet-pretrained CoAtNet-0 reader. |
| [`train_pillar0_reader.py`](train_pillar0_reader.py) | Train a single fold of the frozen-Pillar-0 multiscale token reader. |
| [`train_pillar0_regional_reader.py`](train_pillar0_regional_reader.py) | Train the scale-aligned regional reader on cached Pillar-0 features. |
| [`train_steven_contrastive.py`](train_steven_contrastive.py) | Train the Steven-v4 Experiment-4 reader with target-aware contrastive loss. |
| [`train_steven_temporal_cnn.py`](train_steven_temporal_cnn.py) | Paired Experiment 4 vs temporal-CNN reader training on Steven v4 labels. |

## Evaluation and scoring

| File | Purpose |
|---|---|
| [`bench_model_b_encoders.py`](bench_model_b_encoders.py) | Model B encoder benchmark (timing + equivalence only; no training run, no scoring). |
| [`bench_v4_a100.py`](bench_v4_a100.py) | r50-GL v4 capacity bench on MeluXina's GPU type (A100-40GB, 16 CPUs): is a run data-bound or GPU-bound, and how |
| [`check_cross_finding_x0.py`](check_cross_finding_x0.py) | Exp X0 (pre-registered in docs/experiment_ledger.md row X0): does a combiner that sees all 12 findings beat averaging? |
| [`check_local_global.py`](check_local_global.py) | CPU mechanism checks; no downloaded weights, GPU or gold labels. |
| [`check_local_global_hpo_validation.py`](check_local_global_hpo_validation.py) | Offline checks for locked labels and audit-only per-epoch Steven scoring. |
| [`check_local_global_illustration.cjs`](check_local_global_illustration.cjs) | const {chromium}=require('playwright-core'); |
| [`check_local_global_resnet34_kaggle.py`](check_local_global_resnet34_kaggle.py) | Offline gradient, disabled-stopping recovery, and notebook-lock checks. |
| [`check_swin3d_tpatch.py`](check_swin3d_tpatch.py) | t_patch 1 inflation check (CPU, random-init stand-in for the Kinetics weights): on a volume whose slices are |
| [`eval_ms1_resolution_gpu.py`](eval_ms1_resolution_gpu.py) | Exp MS1-Q1: is input resolution MS1's thin-structure bottleneck? (inference only; Modal H100) |
| [`eval_ms1_token_maps_gpu.py`](eval_ms1_token_maps_gpu.py) | Exp LOC-0-MS1: per-finding token maps of MS1 on fold 0 (inference only; Modal H100). Does MS1 localise by itself? |
| [`eval_shift_protocol_gpu.py`](eval_shift_protocol_gpu.py) | Exp P1-shift-dx: does P1-shift's fold-0 loss come from the train / eval window mismatch? (inference only; Modal) |
| [`eval_slice_use_gpu.py`](eval_slice_use_gpu.py) | Exp MS1-L1: does a model use the neighbouring slices packed into R / B? (inference only; Modal H100) |
| [`gold_fit160_predictions.py`](gold_fit160_predictions.py) | Predict the 58 gold studies with the Fit160 r34 checkpoint (HPO-v1 +281, epoch 15, LB 0.903), CPU only. |
| [`gold_gl4_predictions.py`](gold_gl4_predictions.py) | Finished r50-GL4 students on the 58 host gold studies (inference only; no training, no selection). |
| [`gold_image_model_predictions.py`](gold_image_model_predictions.py) | Score saved image-model checkpoints on the 58 host gold studies (inference only; no training, no selection). |
| [`lb_predictor.py`](lb_predictor.py) | LB-PRED: public-LB predictor from a fold-0 prediction file (development evidence; CPU, numpy + pandas, read-only). |
| [`score_before_after.py`](score_before_after.py) | Before/after scoring: name=predictions.json ... -> per-study /delta/ vs the first file, gold-58 and fold-0 macro AUC. |
| [`score_swin3d_b3_ablation.py`](score_swin3d_b3_ablation.py) | B3-abl scoring (CPU, fold 0): pre-registered fine-4 / val12 rules, clear6, 10-target, per target, curves. |
| [`screen_ensemble_refresh.py`](screen_ensemble_refresh.py) | Ensemble refresh screen (CPU, fold 0): L4 variants with B2, 1e-label-v2, P1-cnx; equal probability means. |
| [`screen_lean_ensembles.py`](screen_lean_ensembles.py) | Lean-ensemble screen (CPU, fold 0, 2026-10-16): every equal-probability-mean subset of <= 4 members from the existing |

## Audits and diagnostics

| File | Purpose |
|---|---|
| [`analyze_l4copy_r1.py`](analyze_l4copy_r1.py) | Prediction-level analysis of the l4copy family after L4C-R1 / R2 (CPU, fold 0, no training). 2026-10-19. |
| [`analyze_pubkd_r_findings.py`](analyze_pubkd_r_findings.py) | PUB-KD-R students vs v4 l4copy on fold 0, per finding x band x branch, and rank shifts by report state. |
| [`audit_attention_map_breakdown.py`](audit_attention_map_breakdown.py) | AM-0 breakdown (post-hoc, no gate; user 2026-10-17: "how many cases of unsure per target is it getting hurt ... where it |
| [`audit_attention_map_distance.py`](audit_attention_map_distance.py) | AM-0 (ledger, pre-registered 2026-10-17): are report-unsure cells whose per-finding attention maps resemble the |
| [`audit_attention_map_impact.py`](audit_attention_map_impact.py) | AM-1 (ledger, pre-registered 2026-10-17): whole-score impact of the AM-0 evidence score (total positive CAM mass). CPU. |
| [`audit_bbf.py`](audit_bbf.py) | Score BB-CMP and V4-FUSE (ledger rows) with the pre-registered readouts. SWA fused fold-0 predictions, 1,000 paired |
| [`audit_bbf_e3.py`](audit_bbf_e3.py) | BB-CMP / V4-FUSE against the project goal (gates-vs-project-goal): does a new model improve E3 (LB 0.944) on fold 0? |
| [`audit_cache_band_coverage.py`](audit_cache_band_coverage.py) | How much of each cached series does the 20-80% central band (12 slices per slot) leave out? (CPU, manifests only) |
| [`audit_cache_input_coverage.py`](audit_cache_input_coverage.py) | What does the 140 mm / 224 px top-k support cache give the model? (no gold; report-rule states only pick examples) |
| [`audit_cam_localisation.py`](audit_cam_localisation.py) | LOC-0 (ledger, pre-registered 2026-10-17): can Exp 1e's CAM peaks serve as pseudo-coordinates for a localise -> crop |
| [`audit_ck_ab.py`](audit_ck_ab.py) | Exp CK-A / CK-B audit (pre-registered in docs/experiment_ledger.md row CK-A / CK-B): stronger ConvNeXt-T checkpoints |
| [`audit_clear_band_lb.py`](audit_clear_band_lb.py) | Clear-band screen (6 targets with >= 20 report-moderate+ positives) for every fold-0 candidate, its fit to the public |
| [`audit_coatnet_d96.py`](audit_coatnet_d96.py) | Score experiment 1 (CoAtNet d96, fold 0) against the pre-registered rule. |
| [`audit_d96_attention_side.py`](audit_d96_attention_side.py) | Attention-placement audit of Exp 1e on fold 0 (Kaggle output of kaggle/submission_kernels/d96_r50_attention_side_audit). |
| [`audit_d96_border.py`](audit_d96_border.py) | Score Exp 3a (Exp 1c, D2 borderline -> 0.85 on 7 targets) against the pre-registered rule. |
| [`audit_d96_cam_centring.py`](audit_d96_cam_centring.py) | Analyse the Kaggle CAM output of Exp 1c on fold 0 (kaggle/submission_kernels/d96_r50_cam_centring_audit). |
| [`audit_d96_centring.py`](audit_d96_centring.py) | Idea 2 (joint-centred input), step 1: is the knee off-centre in the d96 crops? Geometry only, no images, no GPU. |
| [`audit_d96_ctx.py`](audit_d96_ctx.py) | Score Exp 2a (Exp 1c + context between windows) against the pre-registered rule. |
| [`audit_d96_eval_crop.py`](audit_d96_eval_crop.py) | Score Exp 1d (evaluation crop 320 / 352 / 384, inference only) against the pre-registered rule. |
| [`audit_d96_geometry.py`](audit_d96_geometry.py) | Exp 2a pre-launch check: how many d96 slices get an anatomical position, and whether the orientation is sane. |
| [`audit_d96_inputs.py`](audit_d96_inputs.py) | What the d96 recipe actually feeds the model, per slot: sequence type picked vs available, slice coverage, |
| [`audit_d96_miss_severity.py`](audit_d96_miss_severity.py) | Are Exp 1e's fold-0 misses mostly mild / borderline findings (label + subtlety) or clear findings (seeing)? |
| [`audit_d96_native.py`](audit_d96_native.py) | Exp 4b: native-capped input (d96n) + 1e per-target labels vs Exp 1e (two seeds). Pre-registered primary: |
| [`audit_d96_native_ablation.py`](audit_d96_native_ablation.py) | Score the Exp 4b ablation (kaggle/submission_kernels/d96_r50_native_ablation) next to Exp 1e's (2-abl). |
| [`audit_d96_pertarget.py`](audit_d96_pertarget.py) | Score Exp 1e (d96 r50 on the teammate's per-target labels) against the pre-registered rule. |
| [`audit_d96_plane.py`](audit_d96_plane.py) | Score Exp 2c (Exp 1c + per-plane branches with auxiliary plane losses) against the pre-registered rule. |
| [`audit_d96_plane_pertarget.py`](audit_d96_plane_pertarget.py) | Score Exp 2d (2c per-plane head + 1e per-target labels) against its pre-registered rule. |
| [`audit_d96_routing_stitch.py`](audit_d96_routing_stitch.py) | Zero-training proxy for per-target plane routing: stitch saved fold-0 predictions by anatomy (pre-specified). |
| [`audit_d96_seed.py`](audit_d96_seed.py) | Exp 1e-s2: seed re-run of Exp 1e (seed 2026 -> 2027, nothing else). Measurement of the noise floor. |
| [`audit_d96_series_ablation_tta.py`](audit_d96_series_ablation_tta.py) | Score the Kaggle series-ablation / shift-TTA output (kaggle/submission_kernels/d96_r50_series_ablation_tta) for |
| [`audit_d96_slice_gap.py`](audit_d96_slice_gap.py) | Does through-plane subsampling hurt the targets that depend on that plane? (fold 0, CPU only, correlational) |
| [`audit_d96_state.py`](audit_d96_state.py) | Score Exp 3b (Exp 1c, D2 re-valued by rule state: BORD -> 0.85 effusion/MCL/medial OA; BELOW -> 0.15 |
| [`audit_d96_swa_diversity.py`](audit_d96_swa_diversity.py) | Exp 6a secondary: diversity of SWA members (cyclical tail vs 1e's cosine tail), from saved per-epoch fold-0 predictions. |
| [`audit_d96_window_evidence.py`](audit_d96_window_evidence.py) | Missing-evidence audit (CPU, correlational): do 16-window training draws contain the windows that carry a finding? |
| [`audit_demographics.py`](audit_demographics.py) | Demographics audit (CPU, cross-fitted, 2026-10-13): does patient age / sex (DICOM headers, Kaggle scan |
| [`audit_duplicate_studies.py`](audit_duplicate_studies.py) | Duplicate / near-duplicate studies across folds (CPU, 2026-10-13): gate for the cross-fitted teacher stage. |
| [`audit_ensemble_subsets.py`](audit_ensemble_subsets.py) | Which equal-mean ensemble subsets of our fold-0 models are worth their members? (fold 0, Steven-v4 confident cells) |
| [`audit_ensemble_subsets_v2.py`](audit_ensemble_subsets_v2.py) | Lean ensemble search (<= 5 equal-mean members) over every fold-0 model we have, including the 2026-10-12/13 runs. |
| [`audit_extent_pooling.py`](audit_extent_pooling.py) | Extent audit (CPU, cross-fitted, 2026-10-13): does the number of windows showing a finding carry signal that 1e's softmax |
| [`audit_fold0_checkpoint_zoo.py`](audit_fold0_checkpoint_zoo.py) | Inventory every saved fold-0 prediction file and test what the checkpoint collection can add (fold 0, Steven-v4). |
| [`audit_fold0_combination_rules.py`](audit_fold0_combination_rules.py) | Better ways to combine the 0.937 quad than per-target AUC weights: honest fold-0 estimate by cross-fitting. |
| [`audit_fold0_effnet_member.py`](audit_fold0_effnet_member.py) | Does the existing EfficientNetV2-S (HPO v3.1-D2, fold 0) add to the 0.935 ResNet50 trio? (fold 0, Steven-v4 reference) |
| [`audit_fold0_ensemble.py`](audit_fold0_ensemble.py) | Single models vs prediction ensembles on the fixed 825 fold-0 validation studies (Steven-v4 reference). |
| [`audit_fold0_learned_weights.py`](audit_fold0_learned_weights.py) | Do learned ensemble weights beat a plain probability mean out of sample? (fold 0, Steven-v4 reference) |
| [`audit_fold0_local_global_swaps.py`](audit_fold0_local_global_swaps.py) | Quad ensemble vs quad + local_global and vs swapping one quad member for local_global (fold 0, Steven-v4). |
| [`audit_fold0_member_addition.py`](audit_fold0_member_addition.py) | Does a candidate model add to the current submitted ensemble? (fold 0, Steven-v4 reference) |
| [`audit_fold0_per_target_weights.py`](audit_fold0_per_target_weights.py) | Per-target ensemble weights for the 0.937 quad: honest fold-0 estimate by cross-fitting. |
| [`audit_fold0_submission_candidates.py`](audit_fold0_submission_candidates.py) | Candidate submission ensembles on the fixed 825 fold-0 studies (Steven-v4 reference), including the 336 px run. |
| [`audit_fold0_tta.py`](audit_fold0_tta.py) | Score the pre-registered fold-0 TTA sets (kernel tumis001/rsna-r50-tta-fold0) against Steven-v4. |
| [`audit_gold_ambiguous_image.py`](audit_gold_ambiguous_image.py) | Gold-58 mechanism check (development evidence only, not for selection): within cells whose report is |
| [`audit_gold_fit160.py`](audit_gold_fit160.py) | Gold-58 per-target AUC of the Fit160 r34 (HPO-v1 +281, ep15, LB 0.903) vs 140 mm checkpoints (development evidence). |
| [`audit_gold_image_model.py`](audit_gold_image_model.py) | Image model vs report readers on the 58 host gold studies, per target (evaluation only). |
| [`audit_gold_report_ceiling.py`](audit_gold_report_ceiling.py) | How well do report readers agree with the 58 host gold labels? (evaluation only) |
| [`audit_gold_synovitis_effusion.py`](audit_gold_synovitis_effusion.py) | Diagnostic (evaluation only, aggregate): does gold Synovitis track Effusion, and does each model's Synovitis head carry |
| [`audit_joint_findings.py`](audit_joint_findings.py) | JF-0: do the other 11 findings' logits add signal to each finding (radiologist secondary signs / co-occurrence)? |
| [`audit_joint_j1.py`](audit_joint_j1.py) | Exp J1 scorer (pre-registered in docs/experiment_ledger.md row J1, before the run). |
| [`audit_kd_gap.py`](audit_kd_gap.py) | KD-GAP, CPU part: why no student matched its teacher (read-only; no training, validation labels unchanged). |
| [`audit_kd_gap_xai.py`](audit_kd_gap_xai.py) | KD-GAP, GPU part scored on CPU: the Kaggle kernel tumis001/rsna-kd-gap-xai (scripts/build_kaggle_kd_gap_kernel.py). |
| [`audit_kd_view_sweep.py`](audit_kd_view_sweep.py) | KD-VIEW: score the Kaggle sweep (scripts/build_kaggle_kd_view_sweep.py) on fold 0. |
| [`audit_l4c_nf.py`](audit_l4c_nf.py) | Score L4C-NF (ledger row L4C-NF): OA crops partly on the sagittal non-fluid series, inference only. |
| [`audit_l4copy_breadth.py`](audit_l4copy_breadth.py) | Exp L4C-B (ledger row L4C-B, pre-registered 2026-10-18): is v4 l4copy's fold-0 gain broad or carried by one or two |
| [`audit_l4copy_clear6.py`](audit_l4copy_clear6.py) | clear6 screen (audit_clear_band_lb.py definitions) for v4 l4copy inference variants from the XAI kernel arrays. |
| [`audit_l4copy_diag.py`](audit_l4copy_diag.py) | Exp L4C-D scorer (ledger row L4C-D, pre-registered 2026-10-18). Fold 0, label-v2 confident cells, 2,000 paired study |
| [`audit_l4copy_xai.py`](audit_l4copy_xai.py) | Score + render the v4 l4copy XAI / ablation outputs (scripts/l4copy_xai.py; Kaggle kernel rsna-v4-l4copy-xai-fold0). |
| [`audit_label_ceiling.py`](audit_label_ceiling.py) | Is 1c limited by its training labels? On fold 0 (never trained on), treat each training-label set's own soft |
| [`audit_labelv3_residuals.py`](audit_labelv3_residuals.py) | Residual label mismatches after Exp 7a (label-v3 rule caps) - CPU, no LLM is run, no label is changed. |
| [`audit_lean_ensemble.py`](audit_lean_ensemble.py) | Pre-specified lean ensembles vs octa (fold 0, Steven-v4 confident cells, 2,000 paired bootstraps). |
| [`audit_local_global_hpo_validation_results.py`](audit_local_global_hpo_validation_results.py) | Rescore completed local+global validation-reference experiment on CPU. |
| [`audit_local_global_validation.py`](audit_local_global_validation.py) | CPU rescore of two pinned checkpoints on both fixed fold-0 label references. |
| [`audit_medgemma_stage0.py`](audit_medgemma_stage0.py) | Exp MG0 scorer (CPU, numpy only; ledger row MG0). Reads the Stage 0 extraction (part_*.npz, copied from the Modal |
| [`audit_mentioned_cells.py`](audit_mentioned_cells.py) | Fold-0 Steven-v4 AUC restricted to cells the report actually mentions. |
| [`audit_ms1.py`](audit_ms1.py) | MS1 (ledger): MedSigLIP-448 fine-tuned on label-v2, pre-registered fold-0 readout + learning diagnostics. CPU, read-only. |
| [`audit_ms1_crop_probe.py`](audit_ms1_crop_probe.py) | Exp MS1-CP (ledger, pre-registered 2026-10-17): does a crop chosen by MS1's own token maps add signal to MS1's |
| [`audit_ms1_gl.py`](audit_ms1_gl.py) | Exp MS1-GL scorer (ledger, pre-registered 2026-10-17). Fold 0, label-v2 confident cells (weight > 0, y <= 0.3 / >= 0.7). |
| [`audit_ms1_q1.py`](audit_ms1_q1.py) | MS1-Q1 scorer (ledger, pre-registered 2026-10-17): MS1 re-scored at 448 / 560 / 672 px. CPU, read-only. |
| [`audit_multiplane_probe.py`](audit_multiplane_probe.py) | Exp MP-0 (ledger, pre-registered 2026-10-18): does a zoomed view of each finding's anatomy point in the coronal and |
| [`audit_od_probe.py`](audit_od_probe.py) | OrthoDiffusion go/no-go: frozen-feature linear probe of the public OrthoDiffusion backbones vs ImageNet r50. |
| [`audit_p3v2.py`](audit_p3v2.py) | P3 v2 audit (ledger P3v2-T / P3v2-S): teacher sanity and student fold-0 screen vs label-v2. |
| [`audit_phase1.py`](audit_phase1.py) | Phase 1 scorer (P1-r949, P1-r949b, later steps): pre-registered gate + per-target readout vs both 1e seeds. |
| [`audit_pubkd_labelsets.py`](audit_pubkd_labelsets.py) | PUB-KD label-set audit (CPU, read-only; development evidence, validation labels unchanged). |
| [`audit_pubkd_r.py`](audit_pubkd_r.py) | Score PUB-KD-R (ledger row): v4 l4copy r50 students of the Raptor-only public teacher vs v4 l4copy of the same seed. |
| [`audit_questions_1017.py`](audit_questions_1017.py) | Diagnostic questions of 2026-10-17 (CPU, read-only, no training, no submission). Gold-58 is development evidence only: |
| [`audit_r50_gl.py`](audit_r50_gl.py) | Exp r50-GL scorer (ledger, pre-registered 2026-10-18). Fold 0, label-v2 confident cells (weight > 0, y <= 0.3 / >= 0.7), |
| [`audit_r50_gl3_xai.py`](audit_r50_gl3_xai.py) | Exp r50-GL3-XAI scorer (CPU): v3 XAI / ablations vs v2's XAI, and the pre-registered ceiling estimates. |
| [`audit_r50_gl4.py`](audit_r50_gl4.py) | Exp r50-GL4 (v4) scorer (ledger row r50-GL4, pre-registered 2026-10-18). Fold 0, label-v2 confident cells (weight > 0, |
| [`audit_r50_gl_xai.py`](audit_r50_gl_xai.py) | Exp r50-GL-XAI scorer (ledger, pre-registered 2026-10-18). Reads run_r50_gl_v2/xai/xai_{val,train}.npz from |
| [`audit_selfloc.py`](audit_selfloc.py) | Exp SL-0 scorer: r50-GL v3 SWA on its own regions (self-located) vs MS1's regions, fold 0, label-v2 confident cells |
| [`audit_selfloc_v4.py`](audit_selfloc_v4.py) | Exp SL-0v4 scorer (ledger row SL-0v4, pre-registered 2026-10-18). Fold 0, label-v2 confident cells, 12-target mean AUC, |
| [`audit_ser1.py`](audit_ser1.py) | SER-1 (ledger): CPU follow-ups to the SER-0 series ablation. |
| [`audit_series_ablation.py`](audit_series_ablation.py) | SER-0 (ledger): which series group each v4 finding draws on. Kaggle kernel rsna-v4-series-ablation-fold0 outputs |
| [`audit_shift_protocol.py`](audit_shift_protocol.py) | Exp P1-shift-dx scorer (CPU): clear6 / 10-target for each window-layout variant, scored as scripts/audit_phase1.py. |
| [`audit_site_threshold.py`](audit_site_threshold.py) | Site-threshold audit (CPU, correlational, 2026-10-13): do reporting thresholds differ by site (report language), and does |
| [`audit_slice_skips.py`](audit_slice_skips.py) | SK-0: are v4's misses concentrated where the d96 cache dropped acquired slices? (radiologist: follow a structure |
| [`audit_slice_use.py`](audit_slice_use.py) | MS1-L1 scorer (ledger, pre-registered 2026-10-17): does MS1 use the neighbouring slices? CPU, read-only. |
| [`audit_ssl_probe.py`](audit_ssl_probe.py) | Exp 5a diagnostic: frozen-feature linear probe of r50 backbones (Kaggle kernel kaggle/submission_kernels/d96_ssl_probe). |
| [`audit_synovitis_supervision.py`](audit_synovitis_supervision.py) | Where does Synovitis supervision come from in HPO-v1, HPO-v3.0 and the exact v3.1-D2 file used in training? |
| [`audit_synovitis_weight_candidate.py`](audit_synovitis_weight_candidate.py) | Readouts for the Synovitis supervision-weight candidate vs its paired 0.925 baseline (fold 0, Steven-v4 reference). |
| [`audit_target_evidence_map.py`](audit_target_evidence_map.py) | LOC-1 (ledger, pre-registered 2026-10-17): per-target evidence map for crop design. CPU, read-only, descriptive. |
| [`audit_teacher_mix.py`](audit_teacher_mix.py) | T-mix (ledger, pre-registered 2026-10-17): would a mixed-architecture teacher, mean(ConvNeXt CK-A, CoAtNet 1b), be a better |
| [`audit_v3_vs_1e.py`](audit_v3_vs_1e.py) | Exp v3 vs Exp 1e, CPU part: score the branch ablations and summarise the spatial XAI written by |
| [`audit_v3_vs_1e_gpu.py`](audit_v3_vs_1e_gpu.py) | Exp v3 vs Exp 1e: branch ablation and spatial-attention XAI on fold 0 (inference only; runs on Modal). |
| [`audit_v4imp.py`](audit_v4imp.py) | Score V4-IMP (docs/v4_improve_spec.md v2, ledger V4-IMP) with the pre-registered gates. |
| [`diag_pubkd_r.py`](diag_pubkd_r.py) | from pathlib import Path |
| [`diag_r50_gl_gpu.py`](diag_r50_gl_gpu.py) | Exp r50-GL diagnostic (inference only, no training): why does the reader not use the crops? |
| [`diag_swin3d_b3.py`](diag_swin3d_b3.py) | B3 diagnostics (CPU, fold 0, read-only): clear-band per target, pair with 1e-label-v2, rank correlation, |
| [`render_l4copy_diag.py`](render_l4copy_diag.py) | Exp L4C-D renders: draw render_arrays.npz (scripts/l4copy_diag_gpu.py) as one PNG per (finding, study): top row the |
| [`render_local_global_architecture.py`](render_local_global_architecture.py) | CPU-only architecture illustrations using locally matched real DICOM cache rows. |
| [`render_local_global_cache_notebooks.py`](render_local_global_cache_notebooks.py) | Refresh embedded cache modules/config in the four CPU notebooks. |
| [`render_local_global_resnet34_notebook.py`](render_local_global_resnet34_notebook.py) | Build a fresh Kaggle notebook from the checked local+global modules/assets. |
| [`render_ms1_gl_jitter.py`](render_ms1_gl_jitter.py) | MS1-GL preview: what the +-16 px crop-centre jitter looks like on real studies (host, CPU, no training). |
| [`render_r50_gl_xai_panels.py`](render_r50_gl_xai_panels.py) | Exp r50-GL-XAI panels: one row per (study, finding) from run_r50_gl_v2/xai/panels.npz. Columns: (1) the window the |
| [`selfloc_coat0_cpu.py`](selfloc_coat0_cpu.py) | BB-CMP coat0 self-location cost on fold 0 (descriptive, for reading its LB score): the submission kernel's own |
| [`selfloc_r50_gl3_gpu.py`](selfloc_r50_gl3_gpu.py) | Exp SL-0 (self-location): can r50-GL v3 pick its own regions, so inference needs no MS1? |
| [`selfloc_v4_gpu.py`](selfloc_v4_gpu.py) | Exp SL-0v4 (self-location for v4): can an r50-GL v4 SWA model place its own crops, so Kaggle inference needs no MS1? |
| [`study_input_fidelity.py`](study_input_fidelity.py) | Is the d96 cache an information bottleneck for the KD student? (CPU, read-only; development evidence only.) |
| [`study_pipeline.py`](study_pipeline.py) | Pipeline evolution study (CPU, read-only on existing fold-0 / target files; development evidence only). |
| [`xai_kdr_compare.py`](xai_kdr_compare.py) | PUB-KD-R xAI comparison on fold 0: v4 l4copy s2026 vs the Raptor-target students ra (pure) / rb (mix73), plus the |
| [`xai_r50_gl3_gpu.py`](xai_r50_gl3_gpu.py) | Exp r50-GL v3 XAI + ablations (inference only, no training): why did v3 fail and can any fix beat the best single? |
| [`xai_r50_gl_gpu.py`](xai_r50_gl_gpu.py) | Exp r50-GL XAI (inference only, no training): where does each head look, what does the reader use, and which input |

## Tests (CPU smoke tests of runners)

| File | Purpose |
|---|---|
| [`test_bbfuse_cpu.py`](test_bbfuse_cpu.py) | CPU tests for BB-CMP (V4_BB) and V4-FUSE (V4_IMP=fuse) in scripts/r50_gl4_runner.py, localhost/rsna-mlx-train:v1, |
| [`test_coat0_submission.py`](test_coat0_submission.py) | BB-CMP coat0 submission (kaggle/submission_kernels/v4_coat0_submission) vs the MeluXina fold-0 SWA predictions: on the |
| [`test_kd_cv_cpu.py`](test_kd_cv_cpu.py) | CPU tests for KD-CV (V4_VIEW=consistent) in scripts/r50_gl4_runner.py / r50_gl_runner.prepare_gl, localhost/rsna-mlx-train:v1, |
| [`test_l4c_loc_cpu.py`](test_l4c_loc_cpu.py) | CPU tests for scripts/l4c_loc_runner.py (L4C-LOC spec v2 §4), in localhost/rsna-mlx-train:v1, --network=none. |
| [`test_l4c_loc_submission.py`](test_l4c_loc_submission.py) | Submission A' (kaggle/submission_kernels/v4_l4copy_loc_submission) vs the L4C-LOC scoring arrays: on the first N fold-0 |
| [`test_l4copy_r1_cpu.py`](test_l4copy_r1_cpu.py) | CPU tests for scripts/l4copy_r1_runner.py (real timm, no network). Run in localhost/rsna-mlx-train:v1: |
| [`test_l4copy_r2_cpu.py`](test_l4copy_r2_cpu.py) | CPU tests for scripts/l4copy_r2_runner.py (real timm, real v4 l4copy weights, no network). Run in localhost/rsna-mlx-train:v1 |
| [`test_pubkd_cpu.py`](test_pubkd_cpu.py) | CPU tests for PUB-KD in scripts/r50_gl4_runner.py (V4_BB=rap2 and V4_TARGETS), localhost/rsna-mlx-train:v1, |
| [`test_r50_gl4_cpu.py`](test_r50_gl4_cpu.py) | CPU tests for scripts/r50_gl4_runner.py (real timm, random init, no network). Run in localhost/rsna-mlx-train:v1: |
| [`test_surpass_f_cpu.py`](test_surpass_f_cpu.py) | CPU tests for SURPASS-F in scripts/r50_gl4_runner.py (V4_EXT_CACHE, V4_NOISE, V4_EPOCHS, V4_STAGE / V4_INIT, |
| [`test_v4imp_cpu.py`](test_v4imp_cpu.py) | CPU tests for the V4-IMP lanes of scripts/r50_gl4_runner.py (docs/v4_improve_spec.md v2), localhost/rsna-mlx-train:v1, |

## Other analysis and utilities

| File | Purpose |
|---|---|
| [`benchmark_selected_windows.py`](benchmark_selected_windows.py) | CPU-only assembly timing on real baseline caches, with exact tensor parity. |
| [`compare_hpo_resnet_label_runs.py`](compare_hpo_resnet_label_runs.py) | Paired CPU analysis of pinned HPO-v1 / v3.1-D2 validation predictions. |
| [`d96_positions.py`](d96_positions.py) | Exp 2a: anatomical position of every d96 slice, from the DICOM geometry the cache builder recorded (numpy only). |
| [`fastmri_build_cache.py`](fastmri_build_cache.py) | SURPASS-F: the fastMRI knee DICOM batches (NYU fastMRI, knee_DICOMs_batch{1,2}.tar.xz) as a d96 cache, same layout |
| [`fastmri_dedupe.py`](fastmri_dedupe.py) | SURPASS-F: flag fastMRI studies that duplicate a competition study (same images), so they never enter training targets. |
| [`fastmri_gate.py`](fastmri_gate.py) | SURPASS-F Job 1 gate (revised before any full-build data was seen, coordinator 2026-10-20: the original ">= 90% fill |
| [`fastmri_misslot.py`](fastmri_misslot.py) | SURPASS-F: list fastMRI studies whose d96 slots got the wrong contrast (read-only over manifest.jsonl). |
| [`fusion_diagnostic.py`](fusion_diagnostic.py) | Exp F0 (CPU): does a learned fusion of 1e and B2 per-finding features beat averaging their predictions? |
| [`heads_to_eval.py`](heads_to_eval.py) | head_predictions.json ({study: {"global": [12], "local": [12]}}) -> eval_predictions-style JSON for one head. |
| [`joint_j1_xai.py`](joint_j1_xai.py) | Exp J1-XAI: instrumented fold-0 inference of the trained J1 joint model (SWA 16-20) and the label-v2 ResNet-50 |
| [`kd_att_free_test.py`](kd_att_free_test.py) | KD-ATT-T: free test before window-attention distillation (CPU, read-only; pre-registered in the ledger row KD-ATT-T). |
| [`l4copy_diag_gpu.py`](l4copy_diag_gpu.py) | Exp L4C-D on Modal (ledger row L4C-D): the Kaggle diagnostic notebook's per-study code |
| [`l4copy_xai.py`](l4copy_xai.py) | XAI + ablations of v4 l4copy (the LB-0.939 single) on fold 0, CPU in localhost/rsna-mlx-train:v1 (no training, |
| [`multiplane_points.py`](multiplane_points.py) | Exp MP-0: map each finding's anatomy point into the sagittal, coronal and axial series of the same knee (geometry |
| [`od_probe_features.py`](od_probe_features.py) | OrthoDiffusion go/no-go, feature extraction (Modal H100; replaces the slower Kaggle T4 kernel |
| [`pillar0_support_feature_audit_modal.py`](pillar0_support_feature_audit_modal.py) | Frozen Pillar-0 spatial-support audit using temporary Modal compute/storage. |
| [`pubkd_raptor_teacher.py`](pubkd_raptor_teacher.py) | PUB-KD teacher, Raptor slot: the two public nartaa CoAtNet checkpoints on every study of our d96 cache. |
| [`public_label_agreement.py`](public_label_agreement.py) | How the public CC0 report-label sets relate to our HPO-v3.1-D2 labels and to the Steven-v4 fold-0 ruler. |
| [`slim_ms1_crops.py`](slim_ms1_crops.py) | MS1 crop table -> slim copy for v4 (ids, centres, a_top only; drops MS1's logits and features, 1.3 GB -> ~2 MB). |
| [`soup_fold0_predict.py`](soup_fold0_predict.py) | Weight-averaged r50 checkpoints ("model soups"): fold-0 predictions with the trainer's own evaluation path. |
| [`ssl_barlow_d96.py`](ssl_barlow_d96.py) | Exp 5a step 1: continued self-supervised pretraining (Barlow Twins) of a ResNet-50 on the d96 knee-MRI cache (H100). |
| [`text_alignment_beyond_labels.py`](text_alignment_beyond_labels.py) | Ledger row T-pre2: is there finding content in the report embedding beyond what the labels already say? |
| [`text_alignment_prechecks.py`](text_alignment_prechecks.py) | Ledger row T-pre: CPU pre-checks for a training-only report-text alignment loss. |
| [`text_alignment_site_stripped.py`](text_alignment_site_stripped.py) | Ledger row T-pre3: site-stripped report-embedding target (arm B) and extra-finding tag quality (arm A). |
| [`text_encoder_bakeoff.py`](text_encoder_bakeoff.py) | Ledger row T-enc: score each encoder's report embeddings with the T-pre / T-pre2 / T-pre3 checks. |
