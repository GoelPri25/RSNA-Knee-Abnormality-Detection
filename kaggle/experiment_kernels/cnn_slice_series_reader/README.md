# Slice-preserving, series-aware CNN reader (fold 0)

This experiment changes the reader while holding its main controls steady: the
ImageNet-initialized ResNet34, Steven v4 report-only labels, fold-0 split,
confidence-weighted BCE, and a 24-slice training budget per study. It is one
training run, not a sweep or ensemble.

```text
one acquired slice + its support mask
                 │ shared ResNet34 (per slice)
                 ▼
          ordered slice tokens
                 │ 1-D residual mixing, within each series only
       position + plane + fluid-sensitive context
                 ▼
       contextual tokens grouped by series
                 │ target-specific top-k evidence
                 ▼
        one finding vector per series
                 │ target-specific cross-series gate
                 ▼
          12 study-level logits
```

The baseline makes a 2.5-D window from three adjacent slices and sends that
stack through the first convolution together. Here the same trunk encodes each
slice separately. A small depthwise 1-D convolution then exchanges information
between nearby ordered slice tokens within a series, with normalized physical
position added to each token. The model pools high-evidence slices separately
for each target and series, then learns which series matter for each target.
Series masks, plane, and fluid-sensitive metadata are retained during this
fusion.

Checkpoint selection uses confident-label report-validation macro AUC. The
ReduceLROnPlateau scheduler watches report-validation BCE; early stopping waits
for eight epochs without a 0.0005 AUC gain. The 58 gold studies are excluded
from the label file, training, and checkpoint selection. After training, gold
evaluation can be run as a separate, locked assessment if desired.

The fold-0 comparison is intended to isolate this reader redesign. The earlier
48-window run used fold 1, so it is not a directly paired comparator. Runtime
will be close to the 24-window ResNet recipe, with modest added work for local
mixing and the series gate. The run writes a latest checkpoint and CSV/JSON
history after every epoch, plus the best-AUC checkpoint and loss/AUC/LR plot.

## Files

- `rsna-cnn-slice-series-reader-fold-0.py`: Kaggle entry point.
- `kernel-metadata.json`: private T4 kernel specification.
- `source/skanee/`: self-contained source snapshot, new model, and run config
  intended for a private Kaggle source dataset.

## Acceptance reading

Compare fold-0 report-validation macro AUC and BCE to the existing 24-window
fold-0 result (AUC 0.8729; BCE 0.4835). A single-fold result is an experiment
signal, not proof of general improvement. Inspect per-target AUC, selected
epoch, learning curves, series-gate distributions, and Grad-CAM before deciding
whether to use the checkpoint for submission.
