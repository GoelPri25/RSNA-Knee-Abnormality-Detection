# Steven v4 target-aware contrastive experiment

## Question

Does target-aware contrastive supervision improve the existing MedSigLIP
Experiment 4 reader when its frozen slice embeddings, Steven v4 soft labels,
architecture, folds, and optimizer schedule are held fixed?

This experiment does not use Pillar-0 features. The Pillar-0 token-generation
job remains independent and can finish on its own schedule.

## Model and objective

```text
Frozen MedSigLIP slice embeddings (already cached)
                     │
                     ▼
        Existing hierarchical MIL reader
 ordered slice context → target-aware slice pooling
    → target-aware series pooling → study representation
              ┌──────┴─────────┐
              ▼                ▼
       12 classifiers     shared projection MLP
              │          → 12 target-specific vectors
              ▼                ▼
          12 logits       supervised contrastive
              │                │
              └───────┬────────┘
                      ▼
             BCE + λ × contrastive loss
```

The image encoder is not trained. The MIL reader and temporary projection head
are trained. The projection head has a shared MLP trunk and produces one
64-dimensional vector for each finding. For each finding separately, a confidently present study is a
positive key for other confidently present studies, and confidently absent
studies are negative keys. Steven v4 targets at or above 0.70 count as present;
targets at or below 0.20 count as absent. Intermediate targets and missing
labels do not form contrastive pairs, but valid soft labels still participate
in the ordinary weighted BCE. A 512-study detached FIFO queue supplies keys
across mini-batches, so the method is not limited to the few examples in each
batch of eight.

The first run used thresholded labels (`p >= 0.70` present, `p <= 0.20`
absent), and its development AUC did not beat the matched control. Its reported
nonzero anchors and loss show that the objective executed; they do not show
that its pair construction was useful. In particular, the old log-sum-exp
numerator could be satisfied mostly by the easiest positive, while a larger
negative pool could dominate the denominator.

The revised `absolute_mass_v2` objective removes the hard label cutoffs. It
computes the probability that two labels share a state or oppose one another,
reduces pair influence as either label approaches 0.5, and preserves the
absolute amount of positive and negative evidence in the loss. The objective
is the log-sum-exp over all relation mass minus the log-sum-exp over positive
relation mass. This reduces to the old weighted multi-positive loss for hard
0/1 labels, without inventing a fixed negative share when disagreement is
weak. It excludes same-study and stale queue copies and skips anchors without
both positive and negative support. `thresholded_v1` remains selectable for
reproduction. The v2 formulation assumes the soft targets behave like
calibrated probabilities; that assumption still needs validation against
their provenance and score behavior.

The proposed arm uses `lambda=0.05`, temperature `0.10`, and 64-dimensional
target projections. The contrastive projections are auxiliary: the submission
still uses the original 12 classifier logits, and its checkpoint stores the
reader weights separately so the established inference code can load them.

## Matched comparison

The Modal job runs two arms with the same seed, initial reader weights, report
fold, Steven v4 labels, MedSigLIP cache, 60-epoch cosine schedule, batch size,
learning rate, weight decay, slice sampling, and checkpoint selection:

| Arm | Objective |
| --- | --- |
| Control | Experiment 4 target-balanced BCE (`lambda=0`) |
| Treatment | Same BCE + `absolute_mass_v2` contrastive loss (`lambda=0.05`) |

The report-validation BCE selects the checkpoint, matching the established
reader. Both arms save predictions for the same 29 development-gold studies;
those predictions can be scored separately with the development labels. The
29 evaluation-gold labels are never loaded. The prior Steven v4 submission
score (0.849) is a reference, not a training or checkpoint-selection signal.

The job writes epoch logs and plots locally and publishes a private Kaggle
artifact version every 15 epochs. The artifact includes both arms' latest/best
checkpoints, training histories, curves, and development predictions; it does
not include DICOM data or gold labels.

## Decision rule

Do not call a lower report-validation BCE a win by itself. Compare the matched
arms on development macro ROC AUC and per-target AUC, using report-validation
BCE as the checkpoint criterion and calibration/thresholded metrics only as
diagnostics. Keep the locked evaluation untouched until a candidate is chosen.
If the treatment does not improve development ranking performance, retain the
Steven v4 submission model and do not add contrastive complexity to inference.
