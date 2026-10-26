# ResNet-50 global-batch-32 T4x2 memory check

This fixed-input Kaggle script loads one shuffled batch of 32 studies from the
same HPO-v1 +281 training split and prepares up to 24 slice windows per study.
It performs one mixed-precision forward/backward pass with the same two-T4
DataParallel wrapper as the batch-8 run, then reports peak memory for each GPU.
It does not step the optimizer, enter an epoch loop, or save a model checkpoint.

The check is intentionally separate from training. A successful result means
batch 32 fits memory; it does not establish that batch 32 improves validation
macro AUC.
