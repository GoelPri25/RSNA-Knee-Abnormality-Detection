# HPO v1 +281 AUC-controlled ResNet on Kaggle TPU

This is a single, controlled follow-up to `hpo_v1_blend_add281_trainonly_fold0` (Modal). It retains the 3,524 report-only training studies, exact 825 fold-0 validation studies, HPO v1 blend training labels, Steven v4 validation labels, ResNet-34 support-aware 2.5D input, 24 training windows per study, batch size 8, and full 48-window validation. The locked 58-study gold set is not read.

Only the training controller changes. `ReduceLROnPlateau` monitors report-validation macro AUC (`mode=max`, factor 0.5, patience 5, threshold 0.0003 absolute, cooldown 2). Early stopping also monitors report-validation macro AUC, with patience 12 and `min_delta=0`: every new raw AUC best resets the counter, including a small genuine gain. The selected checkpoint is the highest raw AUC. Maximum training is 60 epochs.

Before allocating TPU time, the notebook checks Kaggle API credentials and creates or verifies the private artifact dataset `tumis001/rsna-hpo-blend-auc-control-tpu-fold0`. It publishes best/latest weights, per-epoch history, curve, progress JSON, and run manifest every two completed epochs and at the end. Add Kaggle Secret `KAGGLE_API_TOKEN` (or legacy `KAGGLE_USERNAME` and `KAGGLE_KEY`) to the notebook.

Run files: `run.py` is the fixed-input orchestrator; `build_notebook.py` creates the one-cell training notebook; the source package is in `source/`; Kaggle metadata is in `notebook_kernel/kernel-metadata.json`. There is no recursive dataset discovery.
