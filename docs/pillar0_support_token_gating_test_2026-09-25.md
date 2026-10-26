# Pillar-0 support-token gating test

The pinned, frozen Pillar-0 checkpoint was run on the same three canonical MRI series twice: once normally and once with patch-token vectors zeroed where the source-support fraction was at most 0.05. Partially supported boundary tokens were retained. The intervention was applied to patch-embedding outputs immediately before the Atlas stages. No labels or reports were read, and no parameters were trained.

| Plane | Tokens gated | Cosine to baseline embedding | Relative L2 change |
|---|---:|---:|---:|
| Axial | 85.6% | 0.8701 | 0.5097 |
| Coronal | 87.6% | 0.5827 | 0.9136 |
| Sagittal | 88.2% | 0.8263 | 0.5894 |

All outputs were finite 1,152-dimensional unit embeddings. Mean inference time was about 1.0 second per series on a Tesla T4. The exact checkpoint revision and weight hash are in [`support_token_gating_summary.json`](../reports/pillar0_support_token_gating_2026-09-25/support_token_gating_summary.json).

This is evidence that unsupported patch tokens materially affect the frozen representation. It does not show whether that influence is useful or harmful for knee-abnormality prediction: the audit covered one report-only study, had no labels, and did not evaluate AUC. It also gates token vectors before Atlas rather than changing the Atlas attention masks or its final multi-scale pooling, so it is a deliberately narrow diagnostic.

The next and only planned supervised comparison is a paired reader evaluation: produce baseline and support-gated frozen features from the same eligible Kaggle studies, train the same small reader on each, and compare on the existing held-out report folds. Keep the splits, labels, architecture, seed, and stopping rule fixed; use learning rate `1e-4` (the earlier reader used `3e-4`), and leave the 58-study gold evaluation locked. If the paired held-out result does not improve consistently, reject support gating and stop this branch. The current three-series artifact is insufficient for this training comparison; the feature cache must be generated on Kaggle, with only compact derived embeddings transferred for Modal compute if needed.

## Scope and provenance

- Input: three identifier-free, preprocessed report-only series from a single study; source and support tensors remained in Kaggle until transferred to an ephemeral Modal volume.
- Encoder: frozen revision `be5050ba482607f90e607e044f03a2ebc38f98c3`, weights SHA256 `d658fa8aad744c66d1abf43c2bf6bf594e75e834186dce9aadd1b7a9e4381bf0`.
- Compute: one Tesla T4 inference call. The Modal volume and app were ephemeral; no persistent Modal storage was used.
- Runner: [`pillar0_support_feature_audit_modal.py`](../scripts/pillar0_support_feature_audit_modal.py).
