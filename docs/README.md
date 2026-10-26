# Documentation index

Research record of the project. Every experiment, its pre-registered decision rule and its outcome is in the
[experiment ledger](experiment_ledger.md); the documents below hold the designs, reviews and analyses it refers to.

## Start here

| Document | Topic |
|---|---|
| [COMPETITIVE_NOTEBOOK_NOTES.md](COMPETITIVE_NOTEBOOK_NOTES.md) | Notes from strong public RSNA knee solutions |
| [development.md](development.md) | Development environment and data audit |
| [experiment_ledger.md](experiment_ledger.md) | Experiment ledger |
| [improvement_tracker.md](improvement_tracker.md) | Improvement tracker (Priyanjali, 2026-10-21) |
| [pipeline_evolution_study.md](pipeline_evolution_study.md) | Pipeline evolution study (2026-10-20) |
| [project_learning_synthesis.md](project_learning_synthesis.md) | RSNA Knee project: consolidated evidence and lessons |

## Final distilled students (E4, PUB-KD, teachers)

| Document | Topic |
|---|---|
| [l4c_loc_spec.md](l4c_loc_spec.md) | L4C-LOC: a learned crop locator for v4 l4copy, distilled from MS1 placements (spec for review, 2026-10-19) |
| [pubkd_full_recipe_review.md](pubkd_full_recipe_review.md) | PUB-KD full-target student: recipe for review (2026-10-20) |
| [teacher_dissect_r6_reader_2026-10-21.md](teacher_dissect_r6_reader_2026-10-21.md) | R6 teacher and 2.5D ConvNeXt reader: how they are built, and what E4 can borrow (2026-10-21) |
| [teacher_dissect_raptor_2026-10-21.md](teacher_dissect_raptor_2026-10-21.md) | Raptor teacher dissection vs our E4 student (2026-10-21) |
| [v4_improve_spec.md](v4_improve_spec.md) | V4-IMP: single-change improvements to v4 l4copy (spec v2, 2026-10-19) |
| [window_sampling_first_principles.md](window_sampling_first_principles.md) | Sampling proposal: inspection before another run |

## Global + local reader (r50-GL v3 / v4) designs and reviews

| Document | Topic |
|---|---|
| [XAI_EXPERIMENT_NOTES.md](XAI_EXPERIMENT_NOTES.md) | XAI experiments for the RSNA knee models |
| [exp_v3_review_round1.md](exp_v3_review_round1.md) | Exp v3 peer review, round 1 (2026-10-12) |
| [exp_v3_review_round2.md](exp_v3_review_round2.md) | Exp v3 peer review, round 2 (2026-10-12) |
| [exp_v3_spatial_finding_attention.md](exp_v3_spatial_finding_attention.md) | Exp v3: finding-specific spatial pooling at output stride 16 |
| [exp_v4_crossplane_design.md](exp_v4_crossplane_design.md) | r50-GL v4: one r50 that reads the whole study and each finding's 3D point across planes (design, revision 1, 2026-10-18) |
| [exp_v4_design_rev2.md](exp_v4_design_rev2.md) | r50-GL v4, revision 2: one r50, a global path that local training cannot damage, a local path on its own adapters, and fusion learned against out-of-fold global errors (2026-10-18) |
| [exp_v4_review_round1_impl.md](exp_v4_review_round1_impl.md) | r50-GL v4 (cross-plane) design — implementation / feasibility review, round 1 (2026-10-18) |
| [exp_v4_review_round1_methods.md](exp_v4_review_round1_methods.md) | r50-GL v4 (cross-plane) — methods review, round 1 (2026-10-18) |
| [exp_v4_review_round2_impl.md](exp_v4_review_round2_impl.md) | r50-GL v4 rev 2 — implementation / feasibility review, round 2 (2026-10-18) |
| [exp_v4_review_round2_methods.md](exp_v4_review_round2_methods.md) | r50-GL v4 rev 2: methods review, round 2 (2026-10-18) |
| [local_global_architecture_illustration.md](local_global_architecture_illustration.md) | Local and global reader, with real MRI inputs |
| [local_global_cnn_experiment_plan.md](local_global_cnn_experiment_plan.md) | Combined local and global MRI-context experiment |
| [local_global_hpo_validation_experiment.md](local_global_hpo_validation_experiment.md) | Local+global model with HPO v3.1-D2 validation |
| [local_global_resnet34_hpo60_kaggle.md](local_global_resnet34_hpo60_kaggle.md) | Local+global ResNet34, HPO validation, 60 epochs |
| [xai_failure_diagnosis_research.md](xai_failure_diagnosis_research.md) | xAI and model-debugging techniques: where v4 fails and what to change |

## d96 study-volume track

| Document | Topic |
|---|---|
| [exp1e_best_single_model.md](exp1e_best_single_model.md) | Exp 1e: best single model (public LB 0.936) |
| [handoff_d96_2026-10-12.md](handoff_d96_2026-10-12.md) | Handoff: d96 single-model track (as of 2026-10-12) |
| [swin3d_model_b_handoff.md](swin3d_model_b_handoff.md) | Handoff: Model B (Swin3D-T on d96n), as of 2026-10-15 |

## Label and supervision work

| Document | Topic |
|---|---|
| [gold_auc_audit_checkpoint_handoff.md](gold_auc_audit_checkpoint_handoff.md) | Gold AUC audit checkpoint and dataset handoff |
| [hpo_resnet50_backbone_experiment.md](hpo_resnet50_backbone_experiment.md) | HPO-v1 ResNet-50 backbone experiment |
| [hpo_resnet50_warmstart_refinement.md](hpo_resnet50_warmstart_refinement.md) | ResNet-50 warm-start refinement |
| [hpo_v1_resnet50_auc_margin.md](hpo_v1_resnet50_auc_margin.md) | HPO-v1 ResNet50 AUC-margin refinement |
| [hpo_v1_resnet50_auc_margin_results.md](hpo_v1_resnet50_auc_margin_results.md) | AUC-margin refinement: result and diagnosis |
| [hpo_v1_vs_v31_resnet_analysis.md](hpo_v1_vs_v31_resnet_analysis.md) | HPO-v1 versus HPO-v3.1-D2: paired ResNet50 analysis |
| [hpo_v2_provenance_candidate.md](hpo_v2_provenance_candidate.md) | HPO v2 provenance candidate |
| [hpo_v31_d2_efficientnet_modal.md](hpo_v31_d2_efficientnet_modal.md) | EfficientNetV2-S versus HPO v3.1-D2 ResNet50 |
| [hpo_v31_d2_kaggle_experiment.md](hpo_v31_d2_kaggle_experiment.md) | HPO v3.1 D2 Kaggle label experiment |
| [hpo_v31_trainval_resnet50_experiment.md](hpo_v31_trainval_resnet50_experiment.md) | v3.1-D2 validation-control experiment |
| [hpo_v3_1_review.md](hpo_v3_1_review.md) | Independent HPO-v3.1 review |
| [hpo_v3_review.md](hpo_v3_review.md) | Review of external agent's HPO-v3 candidate |
| [image_detail_and_gold_ceiling_audit.md](image_detail_and_gold_ceiling_audit.md) | Image detail, report ceiling and fold-0 ensemble audit (2026-10-07) |
| [steven_v4_compact_temporal_cnn.md](steven_v4_compact_temporal_cnn.md) | Compact, regularized temporal reader |
| [steven_v4_target_contrastive_experiment.md](steven_v4_target_contrastive_experiment.md) | Steven v4 target-aware contrastive experiment |
| [steven_v4_temporal_cnn_reader.md](steven_v4_temporal_cnn_reader.md) | Steven v4 temporal-CNN reader over MedSigLIP slice embeddings |
| [synovitis_supervision_audit.md](synovitis_supervision_audit.md) | Synovitis supervision in HPO-v1, HPO-v3.0 and v3.1-D2 (2026-10-08) |
| [synovitis_weight_experiment.md](synovitis_weight_experiment.md) | Synovitis supervision weight experiment |

## Earlier tracks: CNN, Pillar-0, MedSigLIP, RadImageNet

| Document | Topic |
|---|---|
| [fit160_fov_experiment.md](fit160_fov_experiment.md) | Fit160 field-of-view comparison |
| [highres336_experiment_preparation.md](highres336_experiment_preparation.md) | ResNet50 336 px experiment preparation |
| [input_fidelity_audit_2026-09-25.md](input_fidelity_audit_2026-09-25.md) | Pillar-0 input-fidelity audit |
| [medsiglip_deepdive.md](medsiglip_deepdive.md) | MedSigLIP-448: reference notes (2026-10-17) |
| [pillar0_reader_xai_and_stopping.md](pillar0_reader_xai_and_stopping.md) | Pillar-0 reader: XAI findings and validation rules |
| [pillar0_spatial_feature_audit_2026-09-25.md](pillar0_spatial_feature_audit_2026-09-25.md) | Pillar-0 support and spatial-feature audit |
| [pillar0_support_path_probe_2026-09-25.md](pillar0_support_path_probe_2026-09-25.md) | Correction to the Pillar-0 support-path probe |
| [pillar0_support_token_gating_test_2026-09-25.md](pillar0_support_token_gating_test_2026-09-25.md) | Pillar-0 support-token gating test |
| [pillar0_target_query_experiment.md](pillar0_target_query_experiment.md) | Pillar-0 target-query reader, fold 0 |
| [radimagenet_resnet50_experiment.md](radimagenet_resnet50_experiment.md) | RadImageNet-initialized ResNet-50 experiment |
| [resnet50_trio_probability_submission.md](resnet50_trio_probability_submission.md) | Three-ResNet50 probability ensemble |
| [teammate_cnn_experiment_handoff.md](teammate_cnn_experiment_handoff.md) | CNN experiment handoff |
| [teammate_hpo_0918_0924_recipe_handoff.md](teammate_hpo_0918_0924_recipe_handoff.md) | HPO-v1 training recipes: 0.918 ResNet-34 and 0.924 ResNet-50 |
| [teammate_pillar0_experiment_handoff.md](teammate_pillar0_experiment_handoff.md) | Pillar‑0 experiment handoff |

## Background research

| Document | Topic |
|---|---|
| [radiologist_perspective_research.md](radiologist_perspective_research.md) | Radiologist perspective on knee MRI findings (research 2026-10-19) |

## Figures and history

- `figures/`: interactive HTML explainers of the reader architectures and pipelines, and illustrations.
- `history/project_plan_initial.md`: the original project plan (milestone record).
