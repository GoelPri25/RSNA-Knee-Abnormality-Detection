# EfficientNetV2-S HPO v3.1-D2 submission

Selected epoch 15 from the Modal fold-0 run: fixed Steven-v4 validation macro AUC 0.8973555395389408, BCE 0.4895744061710859. Training stopped after 24 epochs through early stopping.

Uses the established 140 mm / 224 px six-channel preprocessing and target-specific top-k pooling from the successful ResNet50 submission. Only the model adapter and artifact pins change. Inference uses PyTorch AMP and channels-last; no TensorRT, recursive input search, or online weight download. Activation checkpointing is unnecessary in evaluation; the regular EfficientNet features have the same state keys.

CPU verification: strict state loading matched all keys; finite 12-target forward; checkpoint hash and epoch matched published history. Kaggle version 1 completed all three sample studies (144 windows) with zero decode failures or empty studies; script runtime 21.7 seconds. Submitted version 1 through the Kaggle CLI.

Notebook: https://www.kaggle.com/code/tumis001/rsna-hpo-v31-d2-efficientnet-submission

Artifacts: https://www.kaggle.com/datasets/tumis001/rsna-hpo-v31-d2-efficientnet-v2-s-fold0

Checkpoint SHA-256: `c037722ef37a348b8d1c52853513a5b4bd72d9568b2efcb791981ecb2f36e60a`.

Local outputs: `data/efficientnet_v31_d2/submission_output/`.
