# Kaggle kernels and notebooks

Kaggle runs the competition's hidden-test submissions only from notebooks / scripts, offline, so every
submission and every Kaggle-side experiment lives here as a self-contained kernel folder (code + `kernel-metadata.json`).
Push one with `kaggle kernels push -p <folder>`; the metadata names the datasets it attaches (mostly the team's private
Kaggle datasets, so a fork needs its own copies of the weights).

- `submission_kernels/`: submission and evaluation kernels (single models, ensembles, fast efficiency-track kernels,
  fold-0 / gold evaluation, diagnostics).
- `experiment_kernels/`: training, cache-building and data-preparation kernels.
- `notebooks/`: dataset audit and milestone notebooks (see its README for which ones are authoritative).

## submission_kernels

| Folder | Kaggle kernel | What it does |
|---|---|---|
| [`cnn_pairwise_auc_submission`](submission_kernels/cnn_pairwise_auc_submission) | rsna-cnn-top-k-pairwise-auc-submission | Run the support-aware top-k ResNet on the competition test studies. |
| [`cnn_pairwise_gold_eval`](submission_kernels/cnn_pairwise_gold_eval) | rsna-cnn-pairwise-gold-evaluation | CPU-only gold-holdout evaluation for the pairwise-AUC CNN checkpoint. |
| [`cnn_plateau60_best_submission`](submission_kernels/cnn_plateau60_best_submission) | rsna-cnn-top-k-support-plateau60-submission | Run the support-aware top-k ResNet on the competition test studies. |
| [`cnn_slice_series_reader_submission`](submission_kernels/cnn_slice_series_reader_submission) | rsna-cnn-slice-series-reader-submission | Inference for the selected fold-0 slice-series CNN reader. |
| [`cnn_topk_refine_fold0_submission`](submission_kernels/cnn_topk_refine_fold0_submission) | rsna-cnn-topk-refine-fold0-submission | Submit the completed fold-0 full-coverage Top-k CNN checkpoint. |
| [`cnxs_notorch`](submission_kernels/cnxs_notorch) | rsna-fast-cnxs-notorch | Torch-free ConvNeXt-S (E4-CNXS) TensorRT kernel. The commit-run start of the TensorRT kernel (fast_cnxs_trtlean_cv) is |
| [`cnxs_notorch_eval`](submission_kernels/cnxs_notorch_eval) | rsna-fast-cnxs-notorch-eval | Torch-free ConvNeXt-S (E4-CNXS) TensorRT kernel. The commit-run start of the TensorRT kernel (fast_cnxs_trtlean_cv) is |
| [`cnxs_pack`](submission_kernels/cnxs_pack) | rsna-cnxs-pack | TensorRT pack for the ConvNeXt-S fast kernel (E4-CNXS weights). Builds cnxs_pack/submission.py = a Kaggle T4 kernel with |
| [`cnxs_speed`](submission_kernels/cnxs_speed) | rsna-cnxs-speed | Speed + fidelity A/B of the ConvNeXt-S (E4-CNXS) fast kernel variants on one Kaggle 2 x T4 VM, internet OFF (user |
| [`cnxs_trt`](submission_kernels/cnxs_trt) | rsna-cnxs-trt | TensorRT test of the ConvNeXt-S fast kernel (E4-CNXS weights; same arch as R7-D). User 2026-10-22: "lets see how fast |
| [`coatnet_d96_exp1b_submission`](submission_kernels/coatnet_d96_exp1b_submission) | rsna-coatnet-d96-exp1b-submission | Submit Exp 1b: CoAtNet-rmlp-2 d96, fold-0 training studies, SWA of epochs 20-23 (fold 0 Steven-v4 0.9050). |
| [`contrastive_best_submission_tensorrt`](submission_kernels/contrastive_best_submission_tensorrt) | rsna-steven-contrastive-tensorrt-submission | Offline dual-T4 TensorRT submission for the Steven-v4 contrastive checkpoint. |
| [`d96_cnx_1e_pair_submission`](submission_kernels/d96_cnx_1e_pair_submission) | rsna-d96-cnx-1e-pair-submission | ConvNeXt partner probe (2026-10-14): mean probability of Exp 1e (r50, LB 0.936) and P1-cnx (ConvNeXt-T on the |
| [`d96_cnx_ck_in12k_submission`](submission_kernels/d96_cnx_ck_in12k_submission) | rsna-d96-cnx-ck-in12k-submission | CK-A single-model submission (2026-10-16): ConvNeXt-T from the timm ImageNet-12k checkpoint |
| [`d96_labelv2_fold0_eval`](submission_kernels/d96_labelv2_fold0_eval) | rsna-d96-labelv2-fold0-eval | FOLD-0 EVAL (no submission): Exp 1e (control) and the teammate's 1e retrained on label-v2 (fold 0 held out in |
| [`d96_od_probe`](submission_kernels/d96_od_probe) | rsna-d96-orthodiffusion-probe | OrthoDiffusion go/no-go: frozen-feature probe of the public OrthoDiffusion knee MRI diffusion backbones (T4, inference only). |
| [`d96_plane_parity_fold0`](submission_kernels/d96_plane_parity_fold0) | rsna-d96-plane-parity-fold0 | PARITY CHECK on 120 fold-0 train studies: the submission notebooks' d96 path (2d notebook code) for 1e (control), |
| [`d96_r50_attention_side_audit`](submission_kernels/d96_r50_attention_side_audit) | rsna-d96-r50-attention-side-audit | Attention-placement audit: does Exp 1e (best single d96 r50, SWA 21-25) look on the anatomically right side of the knee? |
| [`d96_r50_cam_centring_audit`](submission_kernels/d96_r50_cam_centring_audit) | rsna-d96-r50-cam-centring-audit | Idea 2 content check: where does Exp 1c look? Exact per-location logit decomposition (class activation maps) |
| [`d96_r50_exp1c_submission`](submission_kernels/d96_r50_exp1c_submission) | rsna-d96-r50-exp1c-submission | Submit Exp 1c: ResNet-50 on the d96 recipe, fold-0 training studies, SWA of epochs 21-25 (fold 0 Steven-v4 0.9074). |
| [`d96_r50_exp1e_mirror_submission`](submission_kernels/d96_r50_exp1e_mirror_submission) | rsna-d96-r50-exp1e-mirror-submission | Submit Exp 1e WITH anatomical mirror TTA (2026-10-13): identical to the Exp 1e submission (LB 0.936) except the |
| [`d96_r50_exp1e_pertarget_submission`](submission_kernels/d96_r50_exp1e_pertarget_submission) | rsna-d96-r50-exp1e-pertarget-submission | Submit Exp 1e: ResNet-50 on the d96 recipe trained on the teammate per-target labels, SWA of epochs 21-25 (fold 0 Steven-v4 0.9037). |
| [`d96_r50_exp1e_seed2027_submission`](submission_kernels/d96_r50_exp1e_seed2027_submission) | rsna-d96-r50-exp1e-seed2027-submission | Exp 1e-s2 standalone: Exp 1e (ResNet50 d96, teammate per-target labels, SWA 21-25) re-trained with seed 2027 |
| [`d96_r50_exp1e_seedavg_submission`](submission_kernels/d96_r50_exp1e_seedavg_submission) | rsna-d96-r50-exp1e-seedavg-submission | Seed-average probe (2026-10-13): mean probability of Exp 1e (seed 2026, LB 0.936) and Exp 1e-s2 (seed 2027, |
| [`d96_r50_exp2d_plane_pertarget_submission`](submission_kernels/d96_r50_exp2d_plane_pertarget_submission) | rsna-d96-r50-exp2d-plane-pertarget-submission | Generate submission.py for Exp 2d standalone from the L4 notebook: d96 section only (no quad), one member |
| [`d96_r50_exp3a_border_submission`](submission_kernels/d96_r50_exp3a_border_submission) | rsna-d96-r50-exp3a-border-submission | Submit Exp 3a: ResNet-50 on the d96 recipe trained on HPO-v3.1-D2 with borderline cells (0.3 < p < 0.7) set to 0.85 for |
| [`d96_r50_exp3b_state_submission`](submission_kernels/d96_r50_exp3b_state_submission) | rsna-d96-r50-exp3b-state-submission | Submit Exp 3b: ResNet-50 on the d96 recipe trained on HPO-v3.1-D2 re-valued by rule state (BORD -> 0.85 effusion / MCL / |
| [`d96_r50_exp7a_rulecap_submission`](submission_kernels/d96_r50_exp7a_rulecap_submission) | rsna-d96-r50-exp7a-rulecap-submission | Submit Exp 7a: ResNet-50 on the d96 recipe (Exp 1e exactly) trained on label-v3 = the teammate's label-v2 with |
| [`d96_r50_native_ablation`](submission_kernels/d96_r50_native_ablation) | rsna-d96-r50-native-ablation | Exp 4b ablation on the 825 fold-0 studies (inference only): what does the native-capped model rely on, and what |
| [`d96_r50_p1_r949_submission`](submission_kernels/d96_r50_p1_r949_submission) | rsna-d96-r50-p1-r949-submission | Submit Exp P1-r949 (2026-10-13): Exp 1e inference exactly (no TTA); only the weights differ (1e recipe + pos_weight, |
| [`d96_r50_p3v2_student_a050_submission`](submission_kernels/d96_r50_p3v2_student_a050_submission) | rsna-d96-r50-p3v2-student-a050-submission | P3 v2 student S-a050 single-model submission (2026-10-16): ResNet-50 (resnet50.tv_in1k) on the exact Exp 1e d96 |
| [`d96_r50_series_ablation_tta`](submission_kernels/d96_r50_series_ablation_tta) | rsna-d96-r50-series-ablation-tta | Two inference-only audits of the d96 r50 SWA models (Exp 1c and Exp 1e) on the 825 fold-0 studies (train_series |
| [`d96_r50_v3_spatial_submission`](submission_kernels/d96_r50_v3_spatial_submission) | rsna-d96-r50-v3-spatial-submission | Submit Exp v3: Exp 1e with r50 at output stride 16 and per-finding spatial pooling over each window's layer-4 cells |
| [`d96_ssl_probe`](submission_kernels/d96_ssl_probe) | rsna-d96-ssl-probe | Exp 5a diagnostic: frozen-feature probe of the starting / self-supervised ResNet-50 backbones (T4, inference only). |
| [`d96n_swin3d_b1_submission`](submission_kernels/d96n_swin3d_b1_submission) | rsna-d96n-swin3d-b1-submission | Submit Model B1 alone (LB baseline for the Swin3D pipeline): Swin3D-T with per-plane stage-4 branches on whole |
| [`d96n_swin3d_b2_submission`](submission_kernels/d96n_swin3d_b2_submission) | rsna-d96n-swin3d-b2-submission | Submit Model B2 alone (user request; below the pre-registered submit gate, LB read only): Swin3D-T with frozen |
| [`dicom_demographics_scan`](submission_kernels/dicom_demographics_scan) | rsna-dicom-demographics-scan | DICOM demographics / acquisition header scan (CPU, headers only, 2026-10-13). |
| [`e3_train_labelset_shard0`](submission_kernels/e3_train_labelset_shard0) | rsna-e3-train-labelset-shard0 | E3 labelset for the teammate (user request 2026-10-17): the lean E3 kernel (LB 0.944, rsna-lean-e3-d96-probability-ensemble) |
| [`e3_train_labelset_shard1`](submission_kernels/e3_train_labelset_shard1) | rsna-e3-train-labelset-shard1 | E3 labelset for the teammate (user request 2026-10-17): the lean E3 kernel (LB 0.944, rsna-lean-e3-d96-probability-ensemble) |
| [`e4_gold_eval`](submission_kernels/e4_gold_eval) | rsna-e4-gold-eval | r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells): |
| [`e4all_gold_eval`](submission_kernels/e4all_gold_eval) | rsna-e4all-gold-eval | r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells): |
| [`e4all_slim`](submission_kernels/e4all_slim) | rsna-e4all-slim | E4-ALL full v4 l4copy checkpoint -> slim fp16 global-head weights for the fast kernel (FAST-G). |
| [`edge_slice_audit`](submission_kernels/edge_slice_audit) | rsna-edge-slice-audit | What lies in the slices the 20-80% central band drops? (CPU only, training studies, no model, no gold) |
| [`efficientnet_v31_d2_submission`](submission_kernels/efficientnet_v31_d2_submission) | rsna-hpo-v31-d2-efficientnet-submission | Submit the epoch-15 HPO v3.1-D2 EfficientNetV2-S checkpoint. |
| [`exp4_best_submission`](submission_kernels/exp4_best_submission) | rsna-experiment-4-best-label-source-submission | Run the frozen MedSigLIP MIL checkpoint on the complete competition test set. |
| [`exp4_best_submission_tensorrt`](submission_kernels/exp4_best_submission_tensorrt) | rsna-experiment-4-steven-tensorrt-submission | Offline dual-T4 TensorRT submission for the Steven v4 label-source checkpoint. |
| [`exp4_threeway_score`](submission_kernels/exp4_threeway_score) | rsna-exp4-threeway-score | Experiment 4: three-way report-label comparison |
| [`exp6_aux_extraction`](submission_kernels/exp6_aux_extraction) | rsna-exp6-aux-findings-extraction | Exp 6 step 1: report-derived auxiliary findings, Qwen3-14B AWQ on vLLM (T4x2). |
| [`fast_cnxs`](submission_kernels/fast_cnxs) | rsna-fast-cnxs-submission | Fast E4-CNXS submission: E4 recipe (mix73 KD targets, folds 1-5, SWA 21-25) on an ImageNet-22k ConvNeXt-S |
| [`fast_cnxs_cv_eval`](submission_kernels/fast_cnxs_cv_eval) |  | Fast E4-CNXS submission: E4 recipe (mix73 KD targets, folds 1-5, SWA 21-25) on an ImageNet-22k ConvNeXt-S |
| [`fast_cnxs_eval`](submission_kernels/fast_cnxs_eval) | rsna-fast-cnxs-eval | Fast E4-CNXS submission: E4 recipe (mix73 KD targets, folds 1-5, SWA 21-25) on an ImageNet-22k ConvNeXt-S |
| [`fast_cnxs_trtfull_cv_eval`](submission_kernels/fast_cnxs_trtfull_cv_eval) |  | Fast E4-CNXS submission: E4 recipe (mix73 KD targets, folds 1-5, SWA 21-25) on an ImageNet-22k ConvNeXt-S |
| [`fast_cnxs_trtlean_cv`](submission_kernels/fast_cnxs_trtlean_cv) | rsna-fast-cnxs-trt-submission | Fast E4-CNXS submission: E4 recipe (mix73 KD targets, folds 1-5, SWA 21-25) on an ImageNet-22k ConvNeXt-S |
| [`fast_cnxs_trtlean_cv_debug`](submission_kernels/fast_cnxs_trtlean_cv_debug) | rsna-fast-cnxs-trt-debug | Fast E4-CNXS submission: E4 recipe (mix73 KD targets, folds 1-5, SWA 21-25) on an ImageNet-22k ConvNeXt-S |
| [`fast_cnxs_trtlean_cv_eval`](submission_kernels/fast_cnxs_trtlean_cv_eval) | rsna-fast-cnxs-trt-eval | Fast E4-CNXS submission: E4 recipe (mix73 KD targets, folds 1-5, SWA 21-25) on an ImageNet-22k ConvNeXt-S |
| [`fast_cnxs_trtlean_cv_evalsub`](submission_kernels/fast_cnxs_trtlean_cv_evalsub) | rsna-fast-cnxs-trt-evalsub | Fast E4-CNXS submission: E4 recipe (mix73 KD targets, folds 1-5, SWA 21-25) on an ImageNet-22k ConvNeXt-S |
| [`fast_cnxs_trtlean_cv_nocudnn_record`](submission_kernels/fast_cnxs_trtlean_cv_nocudnn_record) | rsna-fast-cnxs-trt-nocudnn-record | Fast E4-CNXS submission: E4 recipe (mix73 KD targets, folds 1-5, SWA 21-25) on an ImageNet-22k ConvNeXt-S |
| [`fast_cnxs_trtlean_cv_record`](submission_kernels/fast_cnxs_trtlean_cv_record) | rsna-fast-cnxs-trt-record | Fast E4-CNXS submission: E4 recipe (mix73 KD targets, folds 1-5, SWA 21-25) on an ImageNet-22k ConvNeXt-S |
| [`fast_global`](submission_kernels/fast_global) | rsna-fast-global-submission | Fast global-only submission (efficiency track): r50-GL v4 l4copy global head (F1 / E4-ALL weights), nothing else. |
| [`fast_global_ab`](submission_kernels/fast_global_ab) | rsna-fast-global-ab | Fast global-only submission (efficiency track): r50-GL v4 l4copy global head (F1 / E4-ALL weights), nothing else. |
| [`fast_global_eval`](submission_kernels/fast_global_eval) | rsna-fast-global-eval | Fast global-only submission (efficiency track): r50-GL v4 l4copy global head (F1 / E4-ALL weights), nothing else. |
| [`fast_io_ab`](submission_kernels/fast_io_ab) | rsna-fast-io-ab | Fast global-only submission (efficiency track): r50-GL v4 l4copy global head (F1 / E4-ALL weights), nothing else. |
| [`fast_online_global`](submission_kernels/fast_online_global) | rsna-fast-online-global | Fast global-only submission (efficiency track): ONLINE-KD (teammate, online Raptor distillation of E4) global head, nothing else. |
| [`fast_online_global_eval`](submission_kernels/fast_online_global_eval) | rsna-fast-online-global-eval | Fast global-only submission (efficiency track): ONLINE-KD (teammate, online Raptor distillation of E4) global head, nothing else. |
| [`fast_r7d`](submission_kernels/fast_r7d) | rsna-fast-r7d-submission | fast_global/submission.py -> fast_r7d/submission.py (FB-F R7-D ConvNeXt-S, global head): same io / volume code, |
| [`fast_r7d_eval`](submission_kernels/fast_r7d_eval) | rsna-fast-r7d-eval | Fast R7-D submission: FB-F R7-D (heliosli R7 ConvNeXt-S init, FB-F mix73 targets over competition + fastMRI, |
| [`fast_rap2d`](submission_kernels/fast_rap2d) | rsna-fast-rap2d-submission | fast_global/submission.py -> fast_rap2d/submission.py (PUB-KD rap2-D CoAtNet, global head): same io / volume code, |
| [`fast_rap2d_eval`](submission_kernels/fast_rap2d_eval) | rsna-fast-rap2d-eval | Fast rap2-D submission: PUB-KD rap2-D (Raptor 384 init CoAtNet-rmlp-2, full-blend mix73 targets, SWA 4-6), global |
| [`fast_trt_bench`](submission_kernels/fast_trt_bench) | rsna-fast-trt-bench | CUDA-X test on the E4 fast kernel (user 2026-10-22: "test if they can improve inference on our E4, measure against |
| [`fast_trt_eval`](submission_kernels/fast_trt_eval) | rsna-fast-trt-eval | Offline TensorRT test of the E4-ALL fast kernel (user 2026-10-22: TensorRT gave -18% ms/study with pip-installed |
| [`hepta_quad_d96_submission`](submission_kernels/hepta_quad_d96_submission) | rsna-hepta-quad-d96-probability-ensemble | Generate submission.py: the 0.937 quad notebook code (unchanged, main renamed) + one d96 pass for CoAtNet Exp 1b, |
| [`hexa_quad_d96_submission`](submission_kernels/hexa_quad_d96_submission) | rsna-hexa-quad-d96-probability-ensemble | Generate submission.py: the 0.937 quad notebook code (unchanged, main renamed) + one d96 pass for CoAtNet Exp 1b |
| [`hpo_blend_add281_auc_control_submission`](submission_kernels/hpo_blend_add281_auc_control_submission) | rsna-hpo-blend-add281-auc-control-submission | Submit the best checkpoint from the HPO blend +281 AUC-control run. |
| [`hpo_blend_add281_submission`](submission_kernels/hpo_blend_add281_submission) | rsna-hpo-blend-add281-trainonly-submission | Submit the best checkpoint from the HPO blend +281 train-only run. |
| [`hpo_blend_best_submission`](submission_kernels/hpo_blend_best_submission) | rsna-hpo-blend-resnet34-submission | Run the support-aware top-k ResNet on the competition test studies. |
| [`hpo_blend_fit160_submission`](submission_kernels/hpo_blend_fit160_submission) | rsna-hpo-blend-fit160-preprocessing-parity-check | Inference-time implementation of the Fit160 cache transformation. |
| [`hpo_blend_resnet50_submission`](submission_kernels/hpo_blend_resnet50_submission) | rsna-hpo-blend-resnet50-submission | Submit the best checkpoint from the HPO blend +281 ResNet-50 run. |
| [`hpo_blend_sequence_conditioned_submission`](submission_kernels/hpo_blend_sequence_conditioned_submission) | rsna-hpo-blend-sequence-conditioned-cnn-submission | Fast inference for the sequence-conditioned support-aware ResNet-34. |
| [`hpo_v2_provenance_resnet50_submission`](submission_kernels/hpo_v2_provenance_resnet50_submission) | rsna-hpo-v2-provenance-resnet50-submission | Submit the best checkpoint from the HPO-v2 provenance ResNet-50 run. |
| [`hpo_v31_d2_resnet50_submission`](submission_kernels/hpo_v31_d2_resnet50_submission) | rsna-hpo-v31-d2-resnet50-submission | Submit the best checkpoint from the HPO blend +281 ResNet-50 run. |
| [`hpo_v31_trainval_resnet50_336_submission`](submission_kernels/hpo_v31_trainval_resnet50_336_submission) | rsna-hpo-v31-trainval-resnet50-336-submission | Submit epoch17 from the 336px HPO v3.1-D2 ResNet50. |
| [`hpo_v31_trainval_resnet50_submission`](submission_kernels/hpo_v31_trainval_resnet50_submission) | rsna-hpo-v31-trainval-resnet50-submission | Submit epoch18 from HPO v3.1-D2 training and validation ResNet50. |
| [`joint_j1_submission`](submission_kernels/joint_j1_submission) | rsna-joint-j1-submission | J1-sub (docs/experiment_ledger.md row J1-sub; post-hoc LB read of Exp J1, user decision 2026-10-16): the J1 joint |
| [`kd_gap_xai`](submission_kernels/kd_gap_xai) | rsna-kd-gap-xai | KD-GAP xAI (no training, no submission). Built by scripts/build_kaggle_kd_gap_kernel.py; kd_sub = the E4 submission |
| [`kd_view_sweep`](submission_kernels/kd_view_sweep) | rsna-kd-view-sweep | KD-VIEW sweep (no training, no submission). Built by scripts/build_kaggle_kd_view_sweep.py; kd_sub = the E4 |
| [`label_source_audit`](submission_kernels/label_source_audit) | rsna-report-label-source-audit | RSNA report-label source audit (CPU-only) |
| [`lean4_b2_d96_submission`](submission_kernels/lean4_b2_d96_submission) | rsna-lean4-b2-d96-probability-ensemble | Build E-ref E2 (L4 + Model B2): lean4_d96_submission/submission.py with the B2 Swin3D submission script embedded and |
| [`lean4_cnx_d96_submission`](submission_kernels/lean4_cnx_d96_submission) | rsna-lean4-cnx-d96-probability-ensemble | E-ref E1: L4 (r50 D2 224 + CoAtNet d96 1b + r50 d96 1e + r50 d96 2c) + P1-cnx ConvNeXt-T d96, equal probability mean. |
| [`lean4_d96_submission`](submission_kernels/lean4_d96_submission) | rsna-lean4-d96-probability-ensemble | Generate submission.py for the lean 4-member ensemble (L4) from the octa notebook: quad section reduced to |
| [`lean_e3_d96_submission`](submission_kernels/lean_e3_d96_submission) | rsna-lean-e3-d96-probability-ensemble | E3 (pre-registered in docs/experiment_ledger.md row E3): lean all-d96 equal probability mean of |
| [`octa_quad_d96_submission`](submission_kernels/octa_quad_d96_submission) | rsna-octa-quad-d96-probability-ensemble | Generate submission.py: the 0.937 quad notebook code (unchanged, main renamed) + one d96 pass for CoAtNet Exp 1b, |
| [`pillar0_regional_gold_eval`](submission_kernels/pillar0_regional_gold_eval) | rsna-pillar-0-regional-reader-gold-evaluation | Build a Kaggle GPU script for one-time inference-only gold evaluation. |
| [`pillar0_regional_submission`](submission_kernels/pillar0_regional_submission) | rsna-pillar-0-regional-reader-submission | Build the private Kaggle inference kernel for the selected Pillar-0 reader. |
| [`pillar0_xai_audit`](submission_kernels/pillar0_xai_audit) | rsna-pillar0-reader-xai-audit | Inference-only XAI audit of the saved Pillar-0 target-query reader. |
| [`pubkd_cv1_submission`](submission_kernels/pubkd_cv1_submission) | rsna-pubkd-cv1-submission | r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells): |
| [`pubkd_e4all_submission`](submission_kernels/pubkd_e4all_submission) | rsna-pubkd-e4all-submission | r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells): |
| [`pubkd_kd_b_submission`](submission_kernels/pubkd_kd_b_submission) | rsna-pubkd-kd-b-submission | r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells): |
| [`pubkd_kd_e4_submission`](submission_kernels/pubkd_kd_e4_submission) | rsna-pubkd-kd-e4-submission | r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells): |
| [`pubkd_r6_teacher`](submission_kernels/pubkd_r6_teacher) |  | PUB-KD teacher part (b): build the Kaggle script kernels that score R6 (heliosli's compact ConvNeXt-small bundle |
| [`pubkd_r_ra_submission`](submission_kernels/pubkd_r_ra_submission) | rsna-pubkd-r-ra-submission | r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells): |
| [`pubkd_r_ra_xai_fold0`](submission_kernels/pubkd_r_ra_xai_fold0) | rsna-pubkd-r-xai-fold0 | PUB-KD-R xAI, fold 0 (no submission). Built by scripts/build_kaggle_xai_kdr_kernel.py; the embedded files are the |
| [`pubkd_r_rb_submission`](submission_kernels/pubkd_r_rb_submission) | rsna-pubkd-r-rb-submission | r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells): |
| [`pubkd_rap2d_global_submission`](submission_kernels/pubkd_rap2d_global_submission) | rsna-pubkd-rap2d-global-submission | PUB-KD rap2-D, global head only (accuracy-pick candidate, ledger PUB-KD C / D; teacher dissection 2026-10-21). |
| [`quad_resnet50_effnet_probability_submission`](submission_kernels/quad_resnet50_effnet_probability_submission) | rsna-quad-resnet50-effnet-probability-ensemble | Equal probability mean of three pinned ResNet50 checkpoints and one pinned EfficientNetV2-S checkpoint. |
| [`r50_cosine_ema_submission`](submission_kernels/r50_cosine_ema_submission) | rsna-r50-v31-d2-cosine-ema-submission | CPU check of the pinned EMA checkpoint against the submission kernel's own expectations (no competition data). |
| [`r7d_notorch`](submission_kernels/r7d_notorch) | rsna-fast-r7d-notorch | Torch-free ConvNeXt-S TensorRT kernel, R7-D weights (FB-F R7-D: heliosli R7 init, FB-F mix73 targets over competition + fastMRI, SWA 6-8). The commit-… |
| [`r7d_notorch_eval`](submission_kernels/r7d_notorch_eval) | rsna-fast-r7d-notorch-eval | Torch-free ConvNeXt-S TensorRT kernel, R7-D weights (FB-F R7-D: heliosli R7 init, FB-F mix73 targets over competition + fastMRI, SWA 6-8). The commit-… |
| [`r7d_pack`](submission_kernels/r7d_pack) | rsna-r7d-pack | TensorRT pack for the ConvNeXt-S fast kernel (generated by cnxs_pack/make.py; see its docstring). |
| [`r7dc_notorch_eval`](submission_kernels/r7dc_notorch_eval) | rsna-fast-r7dc-notorch-eval | Torch-free ConvNeXt-S TensorRT kernel, R7-D weights (FB-F R7-D: heliosli R7 init, FB-F mix73 targets over competition + fastMRI, SWA 6-8). The commit-… |
| [`r7dc_pack`](submission_kernels/r7dc_pack) | rsna-r7dc-pack | TensorRT pack for the ConvNeXt-S fast kernel (generated by cnxs_pack/make.py; see its docstring). |
| [`raptor_teacher_ref_submission`](submission_kernels/raptor_teacher_ref_submission) | rsna-raptor-teacher-ref-submission | Teacher reference (SURPASS-R, docs/experiment_ledger.md): the PUB-KD-R teacher itself on the public LB. |
| [`report_label_ab`](submission_kernels/report_label_ab) | rsna-report-label-ab | Matched Confidence-V2 vs Flight hybrid-v4 label A/B |
| [`resnet50_trio_max_confidence_submission`](submission_kernels/resnet50_trio_max_confidence_submission) | rsna-resnet50-trio-max-confidence-ensemble | Most-confident member of three pinned ResNet50 checkpoints (per study and target, the probability furthest from 0.5). |
| [`resnet50_trio_probability_submission`](submission_kernels/resnet50_trio_probability_submission) | rsna-resnet50-trio-probability-ensemble | Equal probability mean of three pinned ResNet50 checkpoints. |
| [`steven_v4_fewshot_calibration`](submission_kernels/steven_v4_fewshot_calibration) | rsna-few-shot-label-calibration | Gold-calibrated report-label audit for Steven v4 |
| [`surpass_f1_global_eval`](submission_kernels/surpass_f1_global_eval) | rsna-surpass-f1-global-eval | r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells): |
| [`surpass_f1_global_submission`](submission_kernels/surpass_f1_global_submission) | rsna-surpass-f1-global-submission | r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells): |
| [`teammate_oa_stage_submission`](submission_kernels/teammate_oa_stage_submission) | rsna-teammate-r34-oa-stage-submission | Submit the r34 AUC-control model with the OA presence x location stage (fold 0, epoch 57). |
| [`teammate_plane_split_balanced_wideband`](submission_kernels/teammate_plane_split_balanced_wideband) | rsna-hpo-plane-split-balanced-wideband-fold-0 | Balanced plane-split ResNet-34 on the wide-band cache, 0.918 recipe otherwise. |
| [`teammate_plane_split_balanced_wideband_submission`](submission_kernels/teammate_plane_split_balanced_wideband_submission) | rsna-plane-split-wideband-submission | Submit the report-fold-best checkpoint from the wideband plane-split run. |
| [`teammate_side_aware_submission`](submission_kernels/teammate_side_aware_submission) | rsna-teammate-r34-side-aware-submission | Submit the teammate's side-aware ResNet-34 (fold 0, epoch 15; Steven-v4 fold-0 0.8813, gold 0.8946). |
| [`text_encoder_bakeoff`](submission_kernels/text_encoder_bakeoff) | rsna-text-encoder-bakeoff | Ledger row T-enc: whole-report embeddings from several multilingual text encoders (Kaggle T4, fp16). |
| [`text_encoder_bakeoff_score`](submission_kernels/text_encoder_bakeoff_score) | rsna-text-encoder-bakeoff-score | Ledger row T-enc (amended): embed reports with five encoders and score them with the T-pre / T-pre3 checks on a T4. |
| [`trt_pack`](submission_kernels/trt_pack) | rsna-trt-pack | TensorRT runtime pack for the E4-ALL fast kernel (generated by trt_pack/make.py). Kaggle T4, internet on: |
| [`v4_coat0_submission`](submission_kernels/v4_coat0_submission) | rsna-v4-coat0-submission | BB-CMP coat0: the v4 l4copy recipe on CoAtNet-0 (coatnet_0_rw_224.sw_in1k at img_size 384), s2026 SWA 21-25, |
| [`v4_l4copy_diag_fold0`](submission_kernels/v4_l4copy_diag_fold0) | rsna-v4-l4copy-diag-fold0 | L4C-D FOLD-0 DIAGNOSTICS (no submission; ledger row L4C-D): where does v4 l4copy lose when it places its own crops, |
| [`v4_l4copy_loc_nopatch_submission`](submission_kernels/v4_l4copy_loc_nopatch_submission) | rsna-v4-l4copy-loc-nopatch-submission | Submission A'-np (ledger L4C-LOC, deconfound after A' LB 0.938): v4 l4copy s2026 + learned crop locator, NO global-only |
| [`v4_l4copy_loc_submission`](submission_kernels/v4_l4copy_loc_submission) | rsna-v4-l4copy-loc-submission | Submission A' (ledger L4C-LOC): v4 l4copy s2026 + learned crop locator + PF OA / fracture global-only. |
| [`v4_l4copy_submission`](submission_kernels/v4_l4copy_submission) | rsna-v4-l4copy-selfloc-submission | r50-GL v4 l4copy s2026 single-model submission (ledger rows r50-GL4, SL-0v4). Fold 0 (label-v2 confident cells): |
| [`v4_l4copy_xai_fold0`](submission_kernels/v4_l4copy_xai_fold0) | rsna-v4-l4copy-xai-fold0 | v4 l4copy XAI + ablations, fold 0 (no submission). Built by scripts/build_kaggle_xai_kernel.py; the embedded files are |
| [`v4_selfloc_fold0_check`](submission_kernels/v4_selfloc_fold0_check) | rsna-v4-selfloc-fold0-check | SL-0v4 FOLD-0 CHECK (no submission; ledger row SL-0v4): the v4 l4copy submission code (v4_submission.py in the |
| [`v4_series_ablation_fold0`](submission_kernels/v4_series_ablation_fold0) | rsna-v4-series-ablation-fold0 | SER-0 (ledger row SER-0; no submission): which series group each v4 finding draws on. The LB v4 l4copy submission |
| [`kernel-metadata.json`](submission_kernels/kernel-metadata.json) | | single file |
| [`report-label-audit.ipynb`](submission_kernels/report-label-audit.ipynb) | | single file |

## experiment_kernels

| Folder | Kaggle kernel | What it does |
|---|---|---|
| [`cnn_full48_window_coverage_fold1`](experiment_kernels/cnn_full48_window_coverage_fold1) | rsna-cnn-full-48-windows-fold-1 | Paired fold-1 test: 24 randomly sampled versus all 48 cached windows/study. |
| [`cnn_slice_series_reader`](experiment_kernels/cnn_slice_series_reader) | rsna-cnn-slice-series-reader-fold-0 | Fold-0 Kaggle run for the slice-preserving, series-aware CNN reader. |
| [`coatnet_bench`](experiment_kernels/coatnet_bench) | rsna-coatnet-throughput-bench | Throughput benchmark for experiment 1 (their CoAtNet recipe on our data), Kaggle T4. |
| [`coatnet_d96_cache`](experiment_kernels/coatnet_d96_cache) |  | Experiment 1 cache: the public 0.949 CoAtNet corpus layout ("d96"), rebuilt on our studies (CPU only). |
| [`coatnet_d96_cache_shard0`](experiment_kernels/coatnet_d96_cache_shard0) | rsna-coatnet-d96-cache-shard-0 | Experiment 1 cache: the public 0.949 CoAtNet corpus layout ("d96"), rebuilt on our studies (CPU only). |
| [`coatnet_d96_cache_shard1`](experiment_kernels/coatnet_d96_cache_shard1) | rsna-coatnet-d96-cache-shard-1 | Experiment 1 cache: the public 0.949 CoAtNet corpus layout ("d96"), rebuilt on our studies (CPU only). |
| [`coatnet_d96_cache_shard2`](experiment_kernels/coatnet_d96_cache_shard2) | rsna-coatnet-d96-cache-shard-2 | Experiment 1 cache: the public 0.949 CoAtNet corpus layout ("d96"), rebuilt on our studies (CPU only). |
| [`coatnet_d96_cache_shard3`](experiment_kernels/coatnet_d96_cache_shard3) | rsna-coatnet-d96-cache-shard-3 | Experiment 1 cache: the public 0.949 CoAtNet corpus layout ("d96"), rebuilt on our studies (CPU only). |
| [`coatnet_d96_cache_smoke`](experiment_kernels/coatnet_d96_cache_smoke) | rsna-coatnet-d96-cache-smoke | Experiment 1 cache: the public 0.949 CoAtNet corpus layout ("d96"), rebuilt on our studies (CPU only). |
| [`d96_native_cache`](experiment_kernels/d96_native_cache) |  | Render the native-capped cache kernels from run_template.py: 6 shard kernels + 1 smoke kernel (shard 0, 20 studies). |
| [`fullband224_cache`](experiment_kernels/fullband224_cache) |  | Build one full-band 224px cache shard on CPU (Phase 1 coverage experiment). |
| [`highres336_cache`](experiment_kernels/highres336_cache) | rsna-cnn-140mm-336px-cache | Build one 336px report-only cache shard on CPU. |
| [`highres336_cache_shard1`](experiment_kernels/highres336_cache_shard1) | rsna-cnn-140mm-336px-cache-shard-1 | Build one 336px report-only cache shard on CPU. |
| [`highres336_cache_shard2`](experiment_kernels/highres336_cache_shard2) | rsna-cnn-140mm-336px-cache-shard-2 | Build one 336px report-only cache shard on CPU. |
| [`hpo_v1_attention_only`](experiment_kernels/hpo_v1_attention_only) |  |  |
| [`hpo_v1_blend_auc_control_tpu`](experiment_kernels/hpo_v1_blend_auc_control_tpu) |  | Generate the single-run Kaggle TPU notebook from the fixed-path runner. |
| [`hpo_v1_blend_coatnet0_modal`](experiment_kernels/hpo_v1_blend_coatnet0_modal) |  |  |
| [`hpo_v1_blend_resnet50_batch16_t4x2`](experiment_kernels/hpo_v1_blend_resnet50_batch16_t4x2) | rsna-hpo-blend-resnet50-batch16-t4x2-fold0 | Train the fixed HPO-v1 +281 ResNet-50 experiment across both Kaggle T4s. |
| [`hpo_v1_blend_resnet50_batch32_memory_check`](experiment_kernels/hpo_v1_blend_resnet50_batch32_memory_check) | rsna-hpo-blend-resnet50-batch32-memory-check | One-step T4x2 memory check for global batch 32; no epoch/training loop. |
| [`hpo_v1_blend_resnet50_radimagenet`](experiment_kernels/hpo_v1_blend_resnet50_radimagenet) | rsna-hpo-blend-resnet50-radimagenet-fold0 | RadImageNet-initialized ResNet-50 on the fixed HPO-v1 fold |
| [`hpo_v1_blend_resnet50_t4x2`](experiment_kernels/hpo_v1_blend_resnet50_t4x2) | rsna-hpo-blend-resnet50-t4x2-fold0 | Train the fixed HPO-v1 +281 ResNet-50 experiment across both Kaggle T4s. |
| [`hpo_v1_blend_resnet_tpu`](experiment_kernels/hpo_v1_blend_resnet_tpu) | rsna-hpo-blend-resnet-tpu-fold0 | One controlled ResNet run: HPO v1 blend training, Steven-v4 fold-0 scoring. |
| [`hpo_v1_sequence_conditioned`](experiment_kernels/hpo_v1_sequence_conditioned) |  |  |
| [`hpo_v1_series_attention`](experiment_kernels/hpo_v1_series_attention) |  |  |
| [`hpo_v2_input_preflight`](experiment_kernels/hpo_v2_input_preflight) | rsna-hpo-v2-label-mount-preflight | CPU-only smoke test for the HPO-v2 training inputs; performs no training. |
| [`hpo_v2_provenance_resnet50_datasets_fold0`](experiment_kernels/hpo_v2_provenance_resnet50_datasets_fold0) | rsna-hpo-v2-provenance-resnet50-datasets-fold0 | Embed the fixed runner in a Kaggle notebook for interactive log visibility. |
| [`hpo_v2_provenance_resnet50_fold0`](experiment_kernels/hpo_v2_provenance_resnet50_fold0) | rsna-hpo-v2-provenance-resnet50-fold0-kaggle | Embed the fixed runner in a Kaggle notebook for interactive log visibility. |
| [`hpo_v31_d2_resnet50_fold0`](experiment_kernels/hpo_v31_d2_resnet50_fold0) | rsna-hpo-v3-1-d2-resnet50-fold0 | Kaggle runner for the HPO-v3.1-D2 label comparison. |
| [`hpo_v31_trainval_resnet50_fold0`](experiment_kernels/hpo_v31_trainval_resnet50_fold0) | rsna-hpo-v31-trainval-resnet50-fold0 | Kaggle runner for the HPO-v3.1-D2 label comparison. |
| [`hpo_v31_trainval_resnet50_fresh`](experiment_kernels/hpo_v31_trainval_resnet50_fresh) | rsna-hpo-v31-trainval-resnet50-fresh | Kaggle runner for the HPO-v3.1-D2 label comparison. |
| [`local_global_native_cache`](experiment_kernels/local_global_native_cache) |  | CPU-only native acquired-neighbor cache. Fixed inputs, bounded output. |
| [`local_global_resnet34_hpo60`](experiment_kernels/local_global_resnet34_hpo60) | rsna-local-global-resnet34-hpo-60-epochs | Local+global ResNet34: 60 epochs |
| [`pillar0_input_fidelity_audit`](experiment_kernels/pillar0_input_fidelity_audit) | rsna-knee-input-fidelity-audit | Build the no-training Kaggle audit of DICOM support and physical scale. |
| [`pillar0_spatial_audit_visual_qa`](experiment_kernels/pillar0_spatial_audit_visual_qa) | rsna-knee-pillar0-spatial-audit-visual-qa | Pillar-0 spatial audit: visual QA |
| [`pillar0_spatial_representation_audit`](experiment_kernels/pillar0_spatial_representation_audit) | rsna-knee-pillar0-spatial-audit | Visual review gate |
| [`pillar0_spatial_representation_audit_modal_transfer`](experiment_kernels/pillar0_spatial_representation_audit_modal_transfer) | rsna-knee-pillar0-spatial-transfer | Pillar-0 spatial audit: prepare transfer artifact |
| [`pillar0_support_gated_reader`](experiment_kernels/pillar0_support_gated_reader) | rsna-knee-pillar-0-spatial-token-cache | Build the self-contained Kaggle notebook for the Pillar-0 support test. |
| [`pillar0_support_path_probe`](experiment_kernels/pillar0_support_path_probe) | rsna-knee-pillar0-support-path-probe | Pillar-0 support-path probe (inference inputs only) |
| [`r50_soups`](experiment_kernels/r50_soups) | rsna-r50-soups-fold0 | Weight-averaged r50 checkpoints ("model soups"): fold-0 predictions with the original trainer's evaluation path. |
| [`r50_tta`](experiment_kernels/r50_tta) | rsna-r50-tta-fold0 | Test-time augmentation for the r50 224 models on fold 0, with the original trainer's evaluation path. |
| [`swin3d_ablation`](experiment_kernels/swin3d_ablation) | rsna-swin3d-b3-ablation | Model B3 ablations on Kaggle (free T4 x2): scripts/swin3d_d96n_runner.py (SWIN_PHASE=swin3d_b2) on the d96n cache |
| [`kernel-metadata.json`](experiment_kernels/kernel-metadata.json) | | single file |
| [`pillar0_small_sample.ipynb`](experiment_kernels/pillar0_small_sample.ipynb) | | single file |
