# Three-ResNet50 probability ensemble

Authorized 2026-10-08. Equal arithmetic probability mean of:

- HPO v3.1-D2 224px, epoch15, SHA256 `4091948e1ce9000e8f4ab7dbd7d859de9370a31d3fe09bd09bf99eaf45d8e48c`, public .925.
- HPO v3.1-D2 trainval 336px, epoch17, SHA256 `51aa5e5b907bfce0f0ce770dd70f546b0189ff3d0b8687358459d63958fbe9d7`, public .925.
- HPO-v1 224px, epoch17, SHA256 `7c76e8760b6989fb5bb3afa2fe14cf016ac766981ad612c94fc058acb703120a`, public .924.

Selection evidence: agent's fixed Steven-v4 fold0 probability mean AUC .9077, reported paired bootstrap delta +.011 [.008,.015] versus v31D2. This is selection on the same validation cohort used for checkpoint selection; bootstrap intervals do not account for candidate-search optimism or training variance. Exact trio has not been evaluated on gold at 336px. No cohort percentile ranks, target-specific ensemble weights or fitted calibrations.

Source: `kaggle/submission_kernels/resnet50_trio_probability_submission/`. One fixed input source directory and three fixed checkpoint directories; no recursive search. Exact hash, config version, epoch and full preprocessing config assertions. Same raw selected slices/header ordering/clipping as successful standalone submissions; physical crop computed independently at224 and336. Both224 models share the same prepared tensor. One forward per model/study with at most48 windows preserves exact global top10% pooling. AMP and channels-last, no TensorRT. Individual predictions saved alongside CSV and timing summary for standalone parity checking.

CPU verification: all three checkpoints strictly loaded into original six-channel ResNet50; finite [1,12] outputs at224/336. Saved Kaggle version2 completed on GPU: 3 studies,144windows,0decode failures,0empty studies,19.7sec internalruntime. Individual outputs match all three standalone submission CSVs within5e-9; ensemble CSV equals arithmetic probability mean within5e-9. Version2 was submitted via Kaggle CLI on2026-10-08; scoring pending.

## Public scoring result

2026-10-08: user reports public macro AUC **0.935** for saved version 2, submission 56961865. This is +0.010 over the best constituent score of 0.925. It supports complementary errors across checkpoints; it does not prove that a single local+global model will reproduce the ensemble gain.
