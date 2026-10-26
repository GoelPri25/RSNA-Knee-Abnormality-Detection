# Wideband balanced plane-split checkpoint submission

This notebook submits the report-fold-best checkpoint from Priyanjali's `RSNA HPO Plane Split Balanced Wideband Fold 0` run.

## Pinned model and selection

- Checkpoint: epoch 15, selected by fold-0 report-validation macro AUC.
- Report-validation: 0.887776 macro AUC, 0.496993 BCE (825 studies).
- Split: 3,524 training studies and 825 validation studies; the 281 additional studies are training-only.
- Architecture: ImageNet-initialized ResNet-34; shared stem/layer1/layer2, then independent layer3/layer4/heads for sagittal, coronal, and axial series; learned target-specific plane trust; top-k pooling with minimum k=2.
- Checkpoint SHA-256: `0caab4cf3325e0051d4d1ca8844dfec2c22a30fbc352f0db7f79085d23189406`.

## Inference path

The private Kaggle asset dataset `tumis001/tumis-rsna-plane-split-wb-1005` contains the pinned checkpoint, run summary, and exact preprocessing/model source. Inference mirrors the cache builder: choose up to four series, use slices from the central 10–90% of each ordered stack (up to 40 evenly spaced slices), clip each series at its 1st/99th percentiles, crop to 140 mm at 224×224, and feed three neighboring image slices plus their three support-mask channels. Windows are processed in chunks of 24, then pooled with the model's original per-plane top-k and trust logic.

The notebook checks the checkpoint hash/configuration, writes a competition-schema CSV, and validates IDs, order, shape, finite values, and probability bounds. It does not read the gold labels.

## Kaggle run

- Notebook: https://www.kaggle.com/code/tumis001/rsna-plane-split-wideband-submission
- Competition submission ID: `56849489` (accepted; score was pending when recorded).
- Inference: 3/3 test studies; 358 windows; zero decode failures; 38.03 seconds total (22.01 seconds preprocessing, 2.09 seconds model inference).
