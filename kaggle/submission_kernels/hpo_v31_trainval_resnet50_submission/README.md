# HPO v3.1-D2 train-and-validation ResNet50 submission

Selected epoch 18: v3.1-D2 validation AUC 0.94652391294286, BCE 0.44249457465277775; same checkpoint Steven-v4 AUC 0.8910988855695257. Run stopped early after 27 epochs. Train 3524, fixed validation 825, gold excluded. Both LR scheduling and checkpoint/early-stop controls use v3.1-D2 validation.

Checkpoint SHA-256: `05ea2934e00bde6e40e74186447f1a2682379ebb744a23cbf896988412c23e19`.

Reuses the successful 0.925 ResNet50 submission preprocessing, target pooling and PyTorch AMP/channels-last inference. Changes artifact pins and descriptive validation metadata only. Fixed input paths; no recursive search, TensorRT conversion, or online weights. Strict CPU weight load and finite 12-target forward passed.

Private artifact: https://www.kaggle.com/datasets/tumis001/rsna-hpo-v31-trainval-resnet50-best

Notebook: https://www.kaggle.com/code/tumis001/rsna-hpo-v31-trainval-resnet50-submission

Kaggle version 1 inference passed: 3 studies, 144 windows, zero decode failures/empty studies; script runtime 14.5 seconds.
