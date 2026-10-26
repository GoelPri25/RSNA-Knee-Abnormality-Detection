# Steven v4 temporal-CNN reader over MedSigLIP slice embeddings

## Question

Does the teammate CNN's useful recipe—local adjacent-slice context followed by
simple mean pooling—transfer to the existing frozen MedSigLIP slice embeddings?
The teammate's public CNN uses trainable 2D image convolutions over DICOM-derived
2.5D inputs. Those 2D filters cannot be copied onto a pooled 1,152-value vector:
its feature coordinates are not image-plane coordinates. This experiment uses
1D convolutions only along physically ordered slices.

## Reader graph

```text
For every study, process each series independently

MedSigLIP embeddings [B,S,N,1152] ── LayerNorm ── Conv1D(1152→384, k=3) ──┐
                                                                         +── GELU
normalized position [B,S,N,1] ── position MLP(1→384→384) ───────────────┤
plane/fluid metadata [B,S] ── embeddings(→384), broadcast over slices ────┘
                                                                            │
              ┌── residual block 1: LayerNorm → depthwise Conv1D(k=3)
              │                   → pointwise Conv1D(1×1) → GELU/dropout ──┐
              └──────────────────────── identity shortcut ──────────────── +
                                                                            │
              ┌── residual block 2: same operation and identity shortcut ── +
              └────────────────────────────────────────────────────────────┘
                                                                            │
                           shared per-slice linear head: 384→12 logits
                                                                            │
                           masked mean over all real study slices
                                                                            │
                                      12 study logits
```

The temporal stem and residual blocks preserve the slice axis. Each output
slice vector sees approximately seven adjacent slices. Convolution operates on
one series at a time; series do not share a boundary. Metadata and position are
side inputs that join the stem representation. Residual blocks have a genuine
identity path around their convolution branch. The model does not use attention,
contrastive loss, a projection head, or a learned slice-pooling query.

The head emits raw per-slice logits. We mask padded slice logits and average the
remaining logits across all series and slices in a study. This reproduces the
teammate CNN's slice-count-weighted mean pooling. BCE is applied only to the
resulting study logits, so gradients flow from a study target through every
contributing slice and its temporal neighbors.

## Fixed comparison protocol

- **Features:** existing cached MedSigLIP embeddings; image encoder frozen.
- **Labels:** Steven v4, confirmed to match the teammate CNN's label source.
- **Splits:** the existing report-study folds; no study's slices cross folds.
- **Gold:** all 58 official-label studies remain outside training, validation,
  and checkpoint selection.
- **Objective:** same target-wise class-balanced soft BCE as the Experiment 4
  comparator; no auxiliary objective.
- **Optimization:** AdamW, learning rate `3e-4`, batch size `8`, weight decay
  `0.05`, warmup fraction `0.05`, gradient clipping `1.0`, dropout `0.2`.
- **Selection:** up to 60 epochs; stop after 10 epochs without a validation-BCE
  improvement of at least `0.001`; retain the lowest report-validation BCE.
- **Inputs:** all eligible, ordered cached slices. No slice cap, series dropout,
  or cross-series windows. Variable-length padding is excluded from pooling and
  endpoint-replicated before temporal convolutions.
- **Comparison:** train the CNN reader and Experiment 4 reader on the same
  Steven v4 studies and folds. Report paired fold metrics and out-of-fold macro
  AUC; also record validation BCE and per-target AUC. Gold is a later, separate
  check and is never used to pick a model.

Training logs, epoch history, learning curves, best/latest checkpoints, and
out-of-fold predictions are saved as derived artifacts. Modal is compute only;
the durable checkpoint snapshots go to a private Kaggle dataset at intervals.

## Interpretation limits

This reader can model through-plane feature changes but cannot reconstruct
2D anatomy or produce image-space heatmaps from a globally pooled embedding.
Slice-window occlusion can still show which physical ranges affect a prediction.
Any gain tests the combination of temporal convolutions and full-slice access;
it does not show that MedSigLIP's encoder itself improved.
