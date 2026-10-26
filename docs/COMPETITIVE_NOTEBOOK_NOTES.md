# Notes from strong public RSNA knee solutions

Reviewed 2026-09-22 from the published inference code:

- [0.941 SOTA RSNA Knee — DINOv2 + RadImageNet](https://www.kaggle.com/code/beraterolelk/0-941-sota-rsna-knee-dinov2-radimagenet)
- [Knee MRI: twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)

These notes record useful experimental ideas. Reported scores are notebook-author claims
unless we independently reproduce them.

## Main conclusion

The single-model notebook is the cleaner source of evidence. It reports roughly 0.924
public leaderboard AUC from one CoAtNet study model. The 0.941 notebook is an inference-only
ensemble of DINO, RadImageNet, Raptor and CoAtNet checkpoints, multiple folds and views,
nested rank blends, and per-target weights adjusted through public-leaderboard probes.

The gap suggests that the large gains come from report supervision, MRI-aware preprocessing,
sequence diversity and target-specific aggregation. A large ensemble supplies the final
increment rather than explaining the full performance.

## Single-model recipe

The model converts each variable study into five semantic slots and 64 sampled slices:

| Slot | Slices | Preference |
|---|---:|---|
| Sagittal | 18 | fluid-sensitive |
| Sagittal | 14 | non-fluid-sensitive |
| Coronal | 12 | fluid-sensitive |
| Coronal | 8 | non-fluid-sensitive |
| Axial | 12 | any axial series |

Missing slots are zero-filled and masked. Series are physically ordered using DICOM
orientation and position. Slices are sampled across 6–94% of the stack; the author reports
that central-only sampling lost peripheral anatomy relevant to collateral ligaments and the
lateral meniscus.

Each slice is center-cropped to a 140 mm physical field of view using `PixelSpacing`, then
resized. Intensities are normalized per selected series with the 2nd and 98th percentiles.
Three adjacent slices form the three channels of a 2D input. A CoAtNet backbone processes
42 such windows and a separate attention distribution is learned for every target before
classification.

The weights were trained from report-derived soft probabilities for 4,349 report-only
studies. The notebook says the 58 official-label studies were excluded from training and
used as a validation gate. This is likely a larger source of performance than the precise
backbone choice.

The narrative contains inconsistent historical figures: approximately 0.924 public AUC,
0.9167 gold macro AUC, 0.9054 on a 58-study gate, and a comment mentioning 0.914 live score.
They may refer to different checkpoints or runs and should not be treated as one evaluation.

## 0.941 ensemble recipe

The higher-scoring notebook combines four broad families:

1. DINOv2 models with six plane/sequence slots, partially fine-tuned transformer blocks,
   target queries and, in some variants, hand-defined anatomical slot priors.
2. Additional DINO folds with mean/max, label-attention, cross-attention and slot-token heads.
3. RadImageNet ResNet50 features with several series layouts and attention heads.
4. Raptor/CoAtNet models using several fixed 2.5D study views and slice-selection schemes.

Predictions from each branch are converted to percentile ranks before blending. This matches
ROC-AUC, which depends on ordering rather than calibrated probability values. The notebook
then changes the outer blend per target. Its final Lateral Meniscus prediction uses only the
Raptor/CoAt arm.

The notebook reports that a common final blend scored around 0.939 and that a later public
leaderboard probe raised it to 0.941 using target-specific routing. Therefore 0.941 is evidence
for the complete ensemble and its tuning process, not for DINOv2 alone. Public-score tuning
also creates a meaningful risk of overfitting the public test subset.

## Experiments worth carrying forward

### 1. Rank-based blending

Compare probability averaging with per-target percentile-rank averaging for existing model
predictions. This is cheap and directly aligned with AUC. Fit any blend weights using held-out
predictions, not leaderboard probes.

### 2. Physical field-of-view consistency

Measure how much of the 448-pixel MedSigLIP input the knee occupies across scanners. Compare
the current preprocessing with a documented 140 mm physical crop. Reject the crop if visual
checks show truncated anatomy.

### 3. Target-specific attention audit

For every gold study, export the highest-attended series and slices with plane, sequence
metadata and normalized depth. Check whether learned attention differs anatomically by target.
If all targets emphasize the same inputs, target-specific attention is not providing useful
specialization.

### 4. Peripheral-slice audit

Group slices into peripheral and central depth bands and measure target attention and ablation
effects. Pay particular attention to MCL and lateral meniscus. Do not remove slices as a speed
optimization before this audit.

### 5. Confidence-aware report supervision

Keep `present`, `absent`, `uncertain` and `not_mentioned` distinct. Use high-confidence present
and absent states as supervision, mask `not_mentioned`, and initially mask or down-weight
uncertain cases. Test soft targets only against the locked gold split.

### 6. One simple 2.5D control

As a separate baseline, test adjacent-slice channels plus target-specific attention using the
same report supervision and split. This isolates whether gains come from the encoder or from
supervision and aggregation. It should remain a compact control rather than grow into the
published multi-family ensemble.

### 7. Per-target complementarity

For each candidate model, store out-of-fold predictions and compute per-label AUC, prediction
correlation and disagreement cases. Use specialized per-label blends only where held-out
evidence shows complementary errors.

## Ideas not yet justified

- Reproducing the full 0.941 ensemble.
- Selecting per-label weights through repeated public leaderboard submissions.
- Treating the 0.941 score as a DINOv2 result.
- Converting `not_mentioned` report states into negative labels.
- Removing large parts of each series solely to reduce inference time.
- Assuming the published validation figures are directly comparable without reproducing
  their exact checkpoints, label corpus and split definitions.
