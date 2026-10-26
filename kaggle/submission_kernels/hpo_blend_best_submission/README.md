# HPO blend ResNet34 submission

This Kaggle code submission runs the epoch-14 best checkpoint from the HPO v1
blended-label fold-0 experiment. It reuses the tested support-aware 2.5D
ResNet34 preprocessing and Top-K Support reader used by the 0.906 CNN model.

The kernel uses explicit Kaggle input paths for the source, checkpoint dataset,
and competition files. It does not recursively search `/kaggle/input`. DICOM
files are read only inside each listed test-series directory. The checkpoint
epoch and SHA-256 are checked before inference; the generated CSV is checked
against the sample-submission schema and probability bounds before it is used.

The model uses PyTorch inference on a Kaggle T4, matching the established CNN
submission path. Training and checkpoint creation happened on Modal; this
kernel performs inference only.
