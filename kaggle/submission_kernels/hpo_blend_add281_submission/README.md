# HPO blend +281 train-only submission

This kernel submits the best checkpoint from the Modal experiment that added
281 previously unassigned HPO-labeled studies to training while preserving the
original 825-study fold-0 validation set. The selected checkpoint is epoch 15
(report-validation macro AUC 0.885516; checkpoint SHA-256
`f26823f9bf63c759f298cec085dbedc8792ec7a3e893f4a98bf025117f4e0a93`).

Inference reuses the known HPO blend ResNet34 submission pipeline, including
its support-aware 2.5D preprocessing, Top-K reader, and mixed-precision GPU
inference. It reads the checkpoint and run summary from the private artifact
dataset `tumis001/rsna-hpo-blend-add281-trainonly-fold0`, with explicit Kaggle
input paths and no recursive dataset search. It validates the checkpoint hash,
epoch, run-summary metrics, split counts, output schema, row order, and
probability bounds before producing `submission.csv`.

The existing TensorRT notebooks accelerate MedSigLIP feature extraction; they
do not contain an engine for this pixel-input ResNet34 reader. This submission
keeps the already-tested ResNet inference path so the checkpoint's preprocessing
and pooling behavior remain unchanged.
