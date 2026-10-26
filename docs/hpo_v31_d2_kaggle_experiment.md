# HPO v3.1 D2 Kaggle label experiment

Notebook: https://www.kaggle.com/code/tumis001/rsna-hpo-v3-1-d2-resnet50-fold0 (version 1).

Training labels: private dataset `tumis001/rsna-knee-hpo-v31-d2-report-labels`, file `soft_labels_hpo_v3_1_d2_structured.json`, SHA256 `38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d`.
This derived candidate preserves the other agent's v3.1 originals. Five reviewed definite minimal/trace joint-synovitis rule states were promoted from borderline to positive before rebuilding source fusion and training-only co-occurrence calibration. All training weights are 1; the silence-downweighted variant is not used.

Reference is the original 0.924 ImageNet ResNet50 checkpoint, not Fit160, RadImageNet, or batch-16. Single GPU, batch 8, 24 randomly sampled cached windows for training, all 48 for validation, original 140mm/224px support-aware cache. Train 3524 studies including the 281 training-only additions; validation remains the fixed 825 Steven-v4 studies. Gold 58 excluded.

BCE objective, AdamW lr 0.0003, weight decay 0.03, seed 2026, maximum 60 epochs. BCE ReduceLROnPlateau factor 0.5, patience 2, cooldown 1, absolute threshold 0.0005, minimum lr 1e-6. AUC checkpoint selection and early stopping patience 8, minimum improvement 0.0005.

All three `tumis001/rsna-cnn-topk-support-cache-shard-{0,1,2}-rehost` datasets are attached; their files are at dataset root. Paths are fixed alternatives, with no recursive searching. Source dataset is `tumis001/rsna-cnn-hpo-blend-xla-source-v1`.

The unchanged trainer saves latest/best weights and history each epoch into Kaggle working outputs. An epoch-boundary time guard finishes normally before a 10.5-hour budget when the estimated next epoch would exceed it. This is not remote per-epoch backup or a guarantee against abrupt infrastructure failure. No additional model or control run was launched.

Launcher and notebook are in `kaggle/experiment_kernels/hpo_v31_d2_resnet50_fold0/`.
