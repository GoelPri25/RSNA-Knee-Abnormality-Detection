# Pillar-0 support and spatial-feature audit

The frozen audit used the exact checkpoint already packaged by the private Kaggle kernel `tumis001/pillar0-breastmri-offline-bundle`: revision `be5050ba482607f90e607e044f03a2ebc38f98c3`, weights SHA256 `d658fa8aad744c66d1abf43c2bf6bf594e75e834186dce9aadd1b7a9e4381bf0`. Three canonical, support-aware volumes came from the completed private Kaggle kernel `tumis001/rsna-knee-pillar0-spatial-transfer`. The checkpoint and the identifier-free volumes were read from a read-only anonymous Modal volume for one T4 function call; the Modal App and volume were ephemeral. No training, labels, report text, or identifiers were involved.

## What we measured

For each series, a forward hook recorded the output of Pillar-0's `breast_mr` 3D convolutional patch embed, before Atlas attention. Its grid was `(64, 32, 16)`, matching the support map. We grouped cells as fully supported (`support >= 0.95`), mostly unsupported centre padding (`support <= 0.05`), or mixed boundary cells. For each group we measured mean squared activation per cell and its share of total activation energy. The model stayed frozen and in evaluation mode.

| Plane | Canvas support | Supported cells | Padding cells | Padding/core mean energy | Total feature energy from padding |
|---|---:|---:|---:|---:|---:|
| Axial | 12.0% | 3,273 (10.0%) | 28,049 (85.6%) | 0.558 | 78.4% |
| Coronal | 10.3% | 2,921 (8.9%) | 28,707 (87.6%) | 0.549 | 81.0% |
| Sagittal | 9.8% | 2,645 (8.1%) | 28,894 (88.2%) | 0.540 | 81.1% |

The padding cells have a substantial nonzero feature response: their average per-cell squared activation is about 54–56% of the fully supported cells. Since padding occupies 86–88% of the grid, it accounts for 78–81% of the sum of squared patch activations. This is a concrete signal that centre padding is represented inside the frozen encoder, even though those input voxels are zero.

## What this does and does not mean

This audit supports the architectural question the team raised: Pillar-0 receives only the normalized volume, not the support map, so its attention and max-pooling stages cannot explicitly distinguish anatomy from centre padding. It does **not** show that the final embedding is mostly “about padding.” Squared activation energy is measured before Atlas attention and global pooling; it is not a causal attribution, and the spatial cell counts differ sharply. Nor does one study establish any AUC or leaderboard gain.

The verified Pillar-0 preprocessing already handles DICOM validity correctly before encoding: declared padding becomes `NaN`, support is propagated through canonical LPS resampling, partial-support interpolation is corrected, and the p20–p99 window is computed over supported voxels while retaining legitimate zeros. The prior paired input probe compared against an older failed notebook's preprocessing, so it is not a valid baseline comparison; its corrected status is recorded in [`pillar0_support_path_probe_2026-09-25.md`](pillar0_support_path_probe_2026-09-25.md).

The follow-up frozen counterfactual has now been run: patch-token vectors with support at most 0.05 were zeroed before the Atlas stages, and embeddings changed substantially across the three planes. See [`pillar0_support_token_gating_test_2026-09-25.md`](pillar0_support_token_gating_test_2026-09-25.md). This establishes sensitivity, not predictive benefit. The only planned next step is a paired held-out reader comparison using matched baseline and support-gated features, a learning rate of `1e-4`, and the 58-study gold evaluation locked; stop this branch if the held-out improvement is not consistent.

## Reproducibility and scope

- The runnable Modal client is [`pillar0_support_feature_audit_modal.py`](../scripts/pillar0_support_feature_audit_modal.py). It uses the pinned Kaggle bundle, checks the checkpoint hash, mounts inputs read-only, and places all uploaded artifacts in an anonymous Modal volume that is deleted when the run exits.
- Aggregate results are in [`feature_audit_summary.json`](../reports/pillar0_spatial_feature_audit_2026-09-25/feature_audit_summary.json). Individual series embeddings and MRI tensors are not committed.
- Inference took about 1.1 seconds per series on a Tesla T4 after model load. No training run occurred.
