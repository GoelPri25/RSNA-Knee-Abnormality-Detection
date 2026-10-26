# Top-k refine fold-0 submission

This inference kernel uses the completed fold-0, full-48-window ResNet checkpoint
from `priyanjaliml/rsna-knee-topk-refine-fold-0-to-blend` and the same DICOM
preprocessing/model implementation as the successful Top-k CNN submission.

Before inference, it checks the attached run summary for 4,349 cached studies
(3,479 train / 870 validation), fold 0, and 48 windows per study. It also checks
that `checkpoint_best.pt` is the selected epoch 13 checkpoint and matches the
run summary. If Kaggle attaches a stale or different output version, the kernel
fails before writing a submission.

The source dataset and competition input paths are fixed; there is no recursive
search. No training is performed.
