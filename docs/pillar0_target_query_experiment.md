# Pillar-0 target-query reader, fold 0

This experiment tests whether the frozen Pillar-0 encoder's multiscale spatial
tokens provide a better study representation than the earlier one-vector-per-
series readers. It trains a small reader over report-derived labels only; the
58 official-label studies are excluded from training, validation, and model
selection.

Each cached series contributes 512 fine 8³ tokens, 512 native 8³ tokens, and 8
coarse 2³ tokens. Every token keeps its 384-dimensional encoder feature,
three-dimensional cell location, scale identity, and acquisition metadata.
Support coverage becomes an additive log prior in attention with a 0.05 floor,
so unsupported cells are downweighted while possibly useful globally
contextualized encoder vectors are not erased.

For each of the 12 targets, a learned query attends across all tokens from all
series in that study. This avoids compressing a series or study into one pooled
feature before the target-specific read. The Pillar-0 encoder remains frozen;
only the 265,313-parameter reader and prediction heads train.

The run uses Steven v4 blended report labels and the same fold-0 split used by
the ResNet comparison. It trains for at most 60 epochs, selects the best
confident report-validation macro AUC with patience 8 and minimum improvement
0.0005, and applies ReduceLROnPlateau to report-validation BCE (factor 0.5,
patience 2). Validation BCE and AUC are computed over report labels; they are
not substitutes for the locked official evaluation. Kaggle stores the source
cache and interval-published checkpoints/logs, while a single Modal T4 performs
the training in temporary scratch storage.
