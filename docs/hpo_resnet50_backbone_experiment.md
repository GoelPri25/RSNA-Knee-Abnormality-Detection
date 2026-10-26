# HPO-v1 ResNet-50 backbone experiment

This is a single-change replication of the public-score **0.918** HPO-v1
ResNet-34 experiment. It trains a pretrained ResNet-50 from scratch on the
same 3,524 training studies, with the same 825 report-only validation studies,
fixed report-group split, labels, 24 sampled windows per study, support-aware
2.5D inputs, augmentation, batch size, optimizer, loss, scheduler, checkpoint
selection, and early stopping. The 58 gold studies remain locked. Only the
ImageNet-pretrained backbone changes from ResNet-34 to ResNet-50.

The baseline public submission scored 0.918; its best report-fold macro AUC was
0.885516 at epoch 15. Training ran for 21 epochs and 8,379.5 seconds on one
Modal T4. ResNet-50 has a larger feature extractor, so expect slower epochs and
higher activation memory. Keep the same batch recipe for a clean comparison;
if the T4 runs out of memory, stop and report that rather than silently changing
batch size or gradient accumulation.

The Modal runner reuses the existing persistent slice-cache volume, downloads
only labels if needed, and publishes checkpoints and metrics to a new private
Kaggle dataset every two epochs. It does not transfer raw DICOMs or read gold
labels. Launch detached with:

```bash
source ~/.config/rsna/modal-personal.env
modal run --detach scripts/modal_hpo_blend_add281_resnet50.py
```

The final comparison should include best report-fold macro AUC and epoch, report
validation BCE, per-target AUC, early-stop point, T4 runtime, and the Kaggle
public score from a correctly formatted submission.
