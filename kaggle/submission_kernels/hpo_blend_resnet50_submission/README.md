# HPO blend ResNet-50 best-checkpoint submission

This submission uses the best checkpoint from the Modal ResNet-50 run:

- Checkpoint: epoch 17, SHA-256 `7c76e8760b6989fb5bb3afa2fe14cf016ac766981ad612c94fc058acb703120a`
- Report-fold macro AUC: 0.8909040738783388
- Public reference: HPO-v1 +281 ResNet-34 submission scored 0.918

Inference follows the proven HPO-blend ResNet submission pipeline: the same
support-aware 2.5D DICOM preprocessing, series and slice selection, Top-K reader,
mixed-precision PyTorch inference, and output checks. DICOM header reads and
selected-slice decoding run concurrently, and T4 convolutions use channels-last
memory layout; slice order, preprocessing, model, and precision stay the same.
Only the checkpoint and backbone differ. Input paths are explicit; the notebook
does no recursive dataset search. It uses the existing source dataset and
private ResNet-50 artifact dataset and runs offline on a Kaggle T4.

Before submission, the script verifies the checkpoint hash, epoch, model
configuration, competition schema, row order, and probability bounds. It does
not load or use training labels at inference.
