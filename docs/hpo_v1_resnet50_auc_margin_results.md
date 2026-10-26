# AUC-margin refinement: result and diagnosis

App `ap-4yZ0V0tsoh4wmMryS4LsID` completed by early stopping after eight additional
epochs and is stopped. No further compute was launched. Epoch work totals
approximately 1.516 hours, excluding setup and initial evaluation.

Original epoch 17: report macro AUC 0.890904, BCE 0.516915. Best refinement
epoch 18: AUC 0.890227, BCE 0.507085. Final epoch 25: AUC 0.889152, BCE
0.521697. All additional epochs had lower AUC than the original; the portable
best checkpoint remains the original. Confident-source validation AUC started
at 0.952319, peaked at 0.954565 in epoch 20, ended at 0.951658.

## Direct CPU evidence

Saved original validation predictions were matched to the exact evidence mask.
Every target already had positive-minus-negative mean probability separation
above the chosen margin of 0.3. These are validation measurements, not a claim
that every training target satisfied the margin throughout every minibatch.

| Finding | Original confident-subset mean separation | Misordered positive/negative pairs |
|---|---:|---:|
| ACL | 0.5700 | 5.52% |
| MCL | 0.5037 | 2.50% |
| Medial Meniscus | 0.6769 | 3.67% |
| Lateral Meniscus | 0.5184 | 8.91% |
| Medial OA | 0.6026 | 2.87% |
| Lateral OA | 0.4085 | 9.67% |
| PF OA | 0.5374 | 4.51% |
| Effusion | 0.6246 | 2.52% |
| Synovitis | 0.4334 | 1.04% |
| Baker's | 0.6717 | 3.44% |
| Contusion | 0.5028 | 5.39% |
| Fracture | 0.3151 | 7.12% |

The margin therefore did not explicitly distinguish the remaining ordering
errors. Small dual means (0.002–0.007 at epoch ends) are consistent with weak
mean-separation pressure, but averages cannot establish every target's dual
behavior. The variance term continues acting even when separation is sufficient.

The AUC saddle objective fell from 0.2030 to 0.03084, while learned center means
moved from positive/negative 0.5824/0.3741 to 0.8071/0.09757. Centers began at
0.5; objective reduction partly reflects centers fitting existing scores.
It cannot be read as a proportional increase in neural ranking performance.
At weight 0.1, its epoch-average scalar contribution fell from 0.02030 to
0.00308; scalar sizes do not establish relative gradient contributions.

Eligibility preferentially retains source-agreeing, easier cases (baseline
subset AUC 0.9523 versus full report AUC 0.8909). Ambiguous/disagreeing cells
still receive BCE but no direct AUC-margin pressure. Agreement is not proof
of correct clinical labeling. Compression within a binary class may conflict
with useful severity variation in soft labels; this is a plausible mechanism,
not a demonstrated causal explanation.

MCL has only 92 eligible positives among 3524 training studies. Under independent
uniform batch-8 sampling, approximately 81% of batches have no eligible MCL
positive. Fixed cohort normalization is unbiased for linear moment estimators,
but estimates can be noisy; projected dual dynamics need not average away that
noise. We did not log per-target dual trajectories or neural gradient ratios.

## Implementation assessment and limitations

Original checkpoint evaluation reproduced AUC 0.890904 exactly and BCE to
rounding precision. Target order, loss algebra, evidence masking and dual
projection checks pass. Losses and learned variables changed, ruling out an
entirely inactive branch. This does not prove the custom optimizer adaptation
was optimal or free of every numerical issue.

This was AdamW plus SGD centers/projected ascent, not the paper's PESG recipe;
the classifier was retained. Centers and duals were not initialized from
checkpoint score statistics. These are substantive design differences.
No matched BCE-only continuation control was run, so lower AUC cannot be
causally attributed to the AUC term rather than ordinary further fitting,
optimizer reset, or stochastic variation. Per-target validation metrics and
predictions were not retained for every epoch; target-specific gains/losses
cannot honestly be reconstructed from the aggregate history alone.

Do not continue unchanged or claim that AUC optimization generally failed.
For a future design, initialize/calibrate statistics on the starting model,
target remaining ranking errors without treating uncertain reports as truth,
record per-target metrics/predictions and gradient contributions, and compare
against a matched BCE-only continuation. Keep architecture and label source
fixed. Existing pairwise experiments must inform any proposal rather than
simply repeating their loss.

Artifacts: `data/hpo_v1_resnet50_auc_margin_fold0/results/`, including
`baseline_auc_margin_diagnosis.json`, saved predictions, history and summary.
Full best/latest state remains on `rsna-cnn-auc-margin-artifacts-v1`.
