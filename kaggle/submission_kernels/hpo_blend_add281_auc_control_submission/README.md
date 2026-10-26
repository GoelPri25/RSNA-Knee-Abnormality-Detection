# HPO blend +281 AUC-control submission

This inference kernel uses the best checkpoint from the Modal HPO blend +281 run. It pins checkpoint epoch 22 (report-fold macro AUC 0.8858800135; SHA-256 `c1e822bd797ff0a89cd1905276c0fd4c9ad3b5bdfd51cebf95507a597bfd3e4e`) and verifies it against the training history and progress metadata in the private artifact dataset `tumis001/rsna-hpo-blend-add281-auc-control-fold0`.

Kaggle public score: **0.910** (submission ref `56749022`).

Inference follows the established support-aware 2.5D ResNet34 Top-K pipeline on a Kaggle T4. It uses explicit input paths, validates the checkpoint/config/split provenance, checks the generated CSV against the competition schema and probability bounds, and performs no training. TensorRT is not used because this model consumes preprocessed image windows through the ResNet reader; the proven PyTorch CNN path is retained.
