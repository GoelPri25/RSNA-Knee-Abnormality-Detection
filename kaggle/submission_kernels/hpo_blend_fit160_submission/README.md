# Fit160 checkpoint submission

This submission uses the best checkpoint from the first HPO v1 blend +281 train-only Fit160 run.

- Fixed checkpoint: private dataset `tumis001/rsna-hpo-blend-fit160-best-epoch15`
- Epoch: 15
- Report-fold macro AUC: 0.8818230427951721
- Validation BCE: 0.5142054332386363
- SHA-256: `427dd72baa09185055b5b013e3b6a8e54c6f148e6a7dbbb630f2a9ed2a11a512`
- Input: ResNet-34, 2.5D adjacent slices and support masks, target-specific top-k pooling
- Inference preprocessing: Fit160 fit-to-canvas, not 140 mm center cropping

`verify_fit160_preprocess.py` checks the submission preprocessing against the teammate's gold cache using only the 58 gold image studies. It does not read labels. All Kaggle paths in the verifier are explicit; it does not recursively search `/kaggle/input`.

The serialized training checkpoint config has an inherited 140 mm cache value. The actual cache summary and training manifest pin the preprocessing used in this experiment to Fit160 (`fov_mode=fit`, 224 px, canonicalized series direction); the submission code follows that recorded cache configuration.
