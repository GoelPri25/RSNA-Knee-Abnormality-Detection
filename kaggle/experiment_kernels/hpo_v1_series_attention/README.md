# HPO-v1 series-aware attention experiment

This single Modal T4 run warm-starts from the ResNet-34 checkpoint that scored
0.918 publicly. It tests one reader change: the model can exchange context among
windows from the same series, using normalized within-series position. Its
attention mask blocks all cross-study and cross-series communication.

The initial residual gate is zero. The CNN, original per-target head, and
top-10%-mean pooling therefore begin with the baseline function; training can
learn whether contextual messages help. Two metadata planes are appended after
augmentation, and their convolution weights start at zero.

Training retains the same HPO-v1 blend labels, the 281 training-only additions,
the fixed 825-study report validation fold, 24-window budget, batch size,
optimizer, scheduler, and AUC-based early stopping. The 24 training windows are
allocated evenly across available series slots, with one randomly selected
slice from each contiguous position bin in each slot. This covers the series
length every epoch while changing the exact windows sampled. Validation uses up
to all 48 cached windows.

## Run and persistence

From the repository root, launch with:

```bash
python scripts/run_modal_personal_series_attention.py
```

The wrapper reads only `~/.config/rsna/modal-personal.env`. Modal reuses the
existing derived-cache volume; raw DICOMs and gold labels do not leave Kaggle.
Checkpoints, logs, predictions, and curves are published every two epochs to
the private Kaggle dataset `tumis001/rsna-hpo-blend-series-attention-fold0`.

## Main files

- `source/skanee/scripts/train_cnn_series_attention.py`: training loop, sampler,
  same-series attention reader, warm-start checks, and metrics.
- `source/skanee/configs/cnn_hpo_v1_blend_series_attention.json`: run config.
- `../../../scripts/modal_hpo_blend_series_attention.py`: Modal T4 launcher,
  fixed-input checks, and interval publishing.

The report-validation macro AUC selects the checkpoint, matching prior runs.
The 58 gold studies remain locked and are not used for training or selection.
