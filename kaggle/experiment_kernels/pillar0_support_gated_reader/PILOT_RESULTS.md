# Pillar-0 spatial-token probe and pilot history

Kaggle notebook: [RSNA Knee Pillar-0 Support-Gated Features](https://www.kaggle.com/code/tumis001/rsna-knee-pillar-0-support-gated-features)

## Corrected geometry check — version 6

Version 6 ran one series on a Tesla T4 after correcting the volume axis order. The
canonical volume is `[Z,Y,X]`; passing it directly to Conv3D gives `[D,H,W]`, which
matches Pillar-0's configured `[H,W,D]=[384,384,192]` and its anisotropic patch stride.
This produces the expected `32×32×32` patch lattice. The earlier pipeline permuted the
axes and produced `64×32×16`; although the token count was unchanged, Atlas was given a
different spatial factorization than its configuration expects. Treat the earlier
feature pilot below as historical only; do not use those embeddings for training.

The version-6 probe captured the three post-Atlas, pre-global-pooling scales in memory.
It read no labels, trained no parameters, and saved no token tensors.

| Check | Result |
| --- | ---: |
| Patch lattice | `32×32×32` |
| Contextual scale 0 | 32,768 tokens × 384 channels |
| Contextual scale 1 | 512 tokens × 384 channels |
| Contextual scale 2 | 8 tokens × 384 channels |
| Total contextual tokens | 33,288 per series |
| FP16 storage for full pyramid | 25,565,184 bytes (about 24.4 MiB) per series |
| Max error reproducing standard 1,152-D embedding by scale-wise max-pooling | 0.0 |
| Series inference timing (baseline + support-gated passes) | 5.87 s |
| Token tensors persisted | No |

The full-resolution pyramid would require about 623 GB for 24,371 series, so it is not
a viable cache as-is. A practical candidate for training is to pool the fine `32³` scale
to an `8³` spatial grid, while retaining the native `8³` and `2³` contextual scales.
Version 7 tested that representation on eight series. The fine scale was average-pooled
in `4×4×4` spatial blocks; native `8³` and `2³` scales were retained. Each scale was
quantized per channel to int8, with scale factors saved alongside it. The chunks were
atomically written, reloaded, shape/finite-checked, and detected by the resume scan.

| Check | Result |
| --- | ---: |
| Series in compression probe | 8 |
| Mean compressed chunk size | 289,971 bytes per series |
| Mean cosine after int8 round trip — fine pooled scale | 0.999997 |
| Mean cosine after int8 round trip — native 8³ scale | 0.999989 |
| Mean cosine after int8 round trip — native 2³ scale | 0.999996 |
| Maximum relative RMSE across scales and series | 0.00331 |
| Spatial pooling global-mean check | Passed |
| Chunk save/reload and resume detection | Passed |

This indicates the proposed storage conversion itself adds little numerical error on
these eight series. It does not establish that pooling `32³` to `8³` preserves the
diagnostic signal; only report-supervised training and held-out report validation can
answer that. At the measured compressed file size, all 24,371 listed series would take
about 7.1 GB, excluding metadata and filesystem overhead. The probe took about 5.05
seconds per series including both ordinary and support-gated encoder passes; the full
cache should use one ordinary pass, so that timing is not a direct estimate of the
single-pass extraction duration.

## Earlier pooled-feature pilot — versions 3–5 (superseded)

Version 3 processed 12 series sampled across the competition table, reconstructed each
volume with its support mask, and generated the ordinary frozen Pillar-0 embedding and
the support-gated embedding. No labels were read and no reader was trained. The later
geometry review showed that this pipeline used the axis permutation described above.
Its measured timing and feature differences are retained as historical observations,
not as evidence for the corrected Pillar-0 representation.

| Check | Result |
| --- | ---: |
| Series processed | 12 |
| Mean end-to-end time per series | 4.63 s |
| Pilot processing time | 0.99 min |
| Embedding shape per view | 1,152 values |
| Stored precision | FP16 |
| Feature chunk size | 55,140 bytes |
| Finite-value check | Passed for both views |
| Embedding norm range after storage | 0.99996–1.00003 |
| Paired baseline/gated cosine similarity | Mean 0.7975; minimum 0.5391 |
| Native finite fraction in sampled series | 1.000 for all 12 |
| Resampled canvas support fraction | 0.075–0.173 |

The progress logs reported that 79.3%, 88.9%, and 90.7% of patch tokens were gated
out at the 4th, 8th, and 12th series checkpoints. Those three values are checkpoint
observations, not a full per-series distribution. The embedding changes are measurable,
but this inference-only pilot cannot tell us whether they improve label prediction.

At the measured rate, processing all 24,371 listed series would take about 31.3 T4
hours for the paired baseline/gated extraction. This is a straight-line estimate from
only 12 samples; series complexity varies. The estimate is large enough that we should
not start full extraction without first narrowing which series types are relevant to
the intended reader and confirming a durable checkpoint route. Full mode still
requires Kaggle API credentials for interval uploads. Pilot outputs remain available
as the completed Kaggle notebook version and were downloaded temporarily for validation.

The support-gated features are a representation ablation, not yet a new classifier or
competition submission. The next meaningful comparison needs report-derived training
labels, a fixed patient/study split, and the same downstream reader evaluated on the
baseline and support-gated features. The 58-study gold evaluation remains untouched.
