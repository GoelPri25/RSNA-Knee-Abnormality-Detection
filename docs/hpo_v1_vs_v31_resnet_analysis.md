# HPO-v1 versus HPO-v3.1-D2: paired ResNet50 analysis

Exact reference checkpoints: HPO-v1 epoch17 public0.924; v3.1-D2 epoch15 public0.925. Both use the fixed825 Steven-v4 validation studies, identical target order and confidence mask (<=0.3 or >=0.7, positive cell weight). Predictions aligned by study ID; AUCs reproduced within1e-5 export-rounding tolerance. No new training or GPU inference was used.

| Target | HPO v1 AUC | v3.1-D2 AUC | Change |
|---|---:|---:|---:|
| MCL | 0.79257 | 0.83225 | +0.03968 |
| Lateral Meniscus | 0.89674 | 0.92004 | +0.02330 |
| Fracture | 0.90434 | 0.92223 | +0.01789 |
| Lateral OA | 0.84426 | 0.84996 | +0.00571 |
| PF OA | 0.86734 | 0.87298 | +0.00565 |
| ACL | 0.88349 | 0.88769 | +0.00420 |
| Medial Meniscus | 0.96077 | 0.96412 | +0.00335 |
| Effusion | 0.86458 | 0.86300 | -0.00158 |
| Medial OA | 0.89144 | 0.88689 | -0.00455 |
| Contusion | 0.91339 | 0.90611 | -0.00728 |
| Baker's | 0.90847 | 0.89966 | -0.00881 |
| Synovitis | 0.96347 | 0.95099 | -0.01248 |

Macro gain0.005423. Paired study bootstrap1000 draws: 95% interval [-0.000796,+0.010654]. This interval includes zero; it does not estimate training-seed variability and does not prove label causality. Per-target intervals are exploratory, unadjusted for12 comparisons.

## What changed

MCL, lateral meniscus and fracture gains sum to0.08087 AUC points across targets, contributing approximately0.00674 macro points before other targets offset them. Ranking gains are concentrated; the refinement is not uniformly beneficial against the report reference.

Fracture reference-positive mean prediction rises0.328→0.529, with negative mean0.047→0.093. This is consistent with stronger fracture predictions and improved ranking, not proof it detects fractures better on images. MCL mean positives0.243→0.303, negatives0.070→0.116: its improved ranking does not automatically improve classification at0.5 (errors90→98). Do not optimize a fixed threshold when competition metric is AUC.

Effusion positive mean falls0.602→0.529; Baker cyst positive mean0.642→0.520; PF OA0.583→0.503. These directions are compatible with stricter severity labels. Steven-v4 reference BCE/threshold accuracy can penalize those changes even if host severity agreement improves. Gold-image inference is required before calling them harmful. AUC itself falls slightly for Effusion and more for Baker, so it is not purely a calibration shift.

Synovitis AUC falls0.96347→0.95099, while mean positive probability rises0.526→0.541 and negative probability rises0.145→0.192. This indicates less useful ordering against Steven reference, even though soft BCE improves. It cautions against assuming that explicit Synovitis rules necessarily improve the learned discriminator.

## Next experiment implications

Keep the running EfficientNet experiment unchanged. Its backbone test now has specific targets to examine: preserve MCL/meniscus/fracture gains and avoid losing other-target ranking. Do not change its label file mid-run.

For the next label experiment, isolate the hypothesis rather than rebuild every target: e.g. preserve v3.1 for ACL/MCL/menisci/fracture and use HPO-v1 Synovitis, holding all other targets and training settings fixed. This is a candidate motivated by report evidence, not an established clinical improvement. Avoid automatically reverting severity-based Effusion/Baker corrections solely to match Steven.

Gold comparison not performed: existing local gold cache is Fit160; that preprocessing mismatch would confound inference for original140mm models. No ground-truth claims or anatomical XAI conclusions follow from this prediction-only analysis.

Reproduce: `uv run --with numpy --with scipy --with scikit-learn python scripts/compare_hpo_resnet_label_runs.py`. Detailed outputs under data/hpo_v1_vs_v31_resnet_analysis (gitignored).
