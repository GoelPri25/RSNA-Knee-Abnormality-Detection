# HPO-v1 attention-only experiment

This single Modal T4 run warm-starts from the ResNet-34 checkpoint that scored
0.918 publicly. It tests one reader change: the model can exchange context among
windows from the same series, using normalized within-series position. Its
attention mask blocks all cross-study and cross-series communication.

The attention residual has a fixed 0.1 scale so its parameters receive gradients
from the first update. The baseline CNN and per-target head warm-start unchanged;
the active reader adds context to the slice representations. Two metadata planes
are appended after augmentation, and their convolution weights start at zero.

Training retains the same HPO-v1 blend labels, the 281 training-only additions,
the fixed 825-study report validation fold, 24-window budget, batch size,
optimizer, scheduler, and AUC-based early stopping. It uses the original
baseline sampler: choose 24 windows uniformly from all windows in each study,
then sort them into acquisition order. Validation uses all available cached
windows, up to 48.

## Run and persistence

From the repository root, launch with:

```bash
python scripts/run_modal_personal_attention_only.py
```

The wrapper reads only `~/.config/rsna/modal-personal.env`. Modal reuses the
existing derived-cache volume; raw DICOMs and gold labels do not leave Kaggle.
Checkpoints, logs, predictions, and curves are published every two epochs to
the private Kaggle dataset `tumis001/rsna-hpo-blend-attention-only-fold0`.

## Main files

- `source/skanee/scripts/train_cnn_attention_only.py`: training loop, sampler,
  same-series attention reader, warm-start checks, and metrics.
- `source/skanee/configs/cnn_hpo_v1_blend_attention_only.json`: run config.
- `../../../scripts/modal_hpo_blend_attention_only.py`: Modal T4 launcher,
  fixed-input checks, and interval publishing.

The report-validation macro AUC selects the checkpoint, matching prior runs.
The 58 gold studies remain locked and are not used for training or selection.
