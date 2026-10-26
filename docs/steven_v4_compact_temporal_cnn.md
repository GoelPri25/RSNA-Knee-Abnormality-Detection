# Compact, regularized temporal reader

## Motivation

The five-fold paired run found the 384-wide, two-block temporal CNN almost tied
with Experiment 4 in macro AUC, while its mean BCE was slightly worse. That says
local slice context is plausible, but this reader has not shown that its extra
capacity improves generalization. The next run keeps its defining idea—predict
from every ordered slice and aggregate over the study—while sharply reducing
the reader size and adding stronger regularization.

## Reader

```text
Each MRI series is processed separately; slice order is preserved.

MedSigLIP slice feature [1152]
       │
       ├── LayerNorm + feature dropout (p = 0.10)
       │
       └── shared per-slice Linear (1152 → 128) + GELU ──┐
                                                         +── one residual block
normalized slice position ── Linear (1 → 128) ──────────┤   depthwise Conv1D(k=3)
plane + fluid metadata ── learned embeddings ────────────┘   pointwise Conv1D
                                                             dropout (p = 0.35)
                                                                  │
                                                 shared per-slice Linear (128 → 12)
                                                                  │
                                       masked mean over real study slices only
                                                                  │
                                                           12 logits
```

The original reader uses a 384-wide temporal stem, a two-layer position MLP,
two residual temporal blocks, and dropout 0.20. The compact profile projects
each slice independently from 1,152 to 128 features before temporal processing,
uses a single linear position projection and one residual block, and applies
feature dropout 0.10 plus hidden dropout 0.35. At the current dimensions, this
reduces reader parameters from 1,785,996 to 169,996 (about 90%) while retaining
all slice predictions and the same study-level masked-mean rule. It does not
merge a series into one vector before the reader sees its slice sequence.

## Controlled comparison

- Same frozen MedSigLIP embeddings, Steven v4 soft labels, report folds, and
  class-balanced masked BCE as the prior paired run.
- Experiment 4 remains the fixed comparator on every fold.
- AdamW learning rate remains `3e-4`; batch size remains 8; early stopping still
  monitors report-validation BCE with patience 10 and minimum improvement 0.001.
- Experiment 4 keeps weight decay `0.05`; the compact reader uses `0.10` and
  dropout `0.35`. No change to labels, folds, loss, or selection criterion.
- All 58 gold studies remain excluded and locked. Save per-epoch logs, curves,
  latest/best checkpoints, and OOF predictions to Kaggle at intervals.

Report paired fold AUC and BCE, out-of-fold macro AUC, per-target AUC, parameter
count, and fold-to-fold variability. This one run tests whether the prior CNN's
small AUC edge survives capacity reduction and stronger regularization; it does
not test a new encoder or prove a causal gain from any one regularizer.
