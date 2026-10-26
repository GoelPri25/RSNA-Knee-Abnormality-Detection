# HPO v1 blended-label ResNet experiment

This is a single controlled label-source comparison against the 0.906 CNN
submission. It keeps the ResNet-34 reader, support-aware 2.5D cache, 24
training windows per study, batch size 8, augmentation, BCE objective,
ReduceLROnPlateau schedule, AUC-based checkpoint selection, and early stopping.
The only intended experimental change is the HPO v1 blended labels used for
training. Fold-0 report validation is scored against Steven v4; the 58 gold
studies stay excluded.

The Kaggle kernel requests TPU v5e-8 but the training process uses one XLA
device to preserve the reference batch behavior. Fixed 48-window evaluation
covers the cache maximum (4 series slots × 12 slices), and repeated padding
rows are masked out of training and validation. TPU bfloat16 arithmetic differs
from GPU mixed precision, so bit-for-bit identity and a speedup are not
guaranteed. Checkpoint and epoch history are saved under `/kaggle/working`
after every epoch.

The Kaggle run is submitted as a notebook at
`notebook_kernel/rsna_hpo_blend_resnet_tpu_fold0.ipynb`. It embeds the same
fixed-path runner and does not scan the filesystem recursively. The earlier
queued script submission was cancelled before this notebook was submitted.
