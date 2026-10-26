# Sampling proposal: inspection before another run

No GPU experiment was launched for this analysis. A fresh task training run,
initialized with ImageNet ResNet50 weights, would isolate a sampler change
against the 0.924 recipe. Fine-tuning the 0.924 model would instead answer
whether changed sampling helps its existing representation; these are different
experiments and should not be conflated.

Inspected the pinned epoch-17 checkpoint config, baseline `prepare_batch`,
`StudyCache.study`, `stack_channels`, cache builder `select_series` and
`select_slice_indices`, all three saved shard manifests, and teammate's
`balanced_plane_choice` in the plane-split training/submission source.
The remote Git branch copy contains the same balanced sampling function.

## Actual pipeline

Cache: select up to four series, cover distinct planes first, prefer fluid
sensitivity within each plane, then seek distinct plane/fluid combinations.
Keep up to 12 physically ordered slices per series from the 20–80% central
band, falling back to the full stack if the band has too few slices. Crop
140 mm to 224 pixels; retain acquired-support masks. This cache is already
deliberately organized, not a random collection of MRI slices.

Training: construct three-channel intensity windows from neighboring cached
slices within each series, pair with three support channels, then choose up
to 24 centers uniformly without replacement from the study's concatenated
windows. The model gets no plane identity in the original baseline. Validation
and submission process all available cached windows. Neighboring cached slices
can skip original DICOM slices; they are not necessarily consecutive raw slices.

## Measured and calculated coverage

Manifests: 4349 studies; all include Sagittal, Coronal and Axial; 4348 have four
series, one has three; 4324 have 48 windows, 24 have 47 and one has 36.
The most common pattern is two sagittal series plus one coronal and one axial
(3957 studies). These are metadata categories, not proof of diagnostic value.

For 24 centers drawn from a 48-window study containing four 12-window series:
each series contributes six centers in expectation. A given 12-window series
has probability 0.0000388146 of contributing none. Probability of at most two
centers is 0.00866845; at most three is 0.0466539. Across the actual 3524-study
training cohort, expected absent available planes total about 0.273 per epoch.
Thus a claim that random sampling routinely omits planes is unsupported.

Common plane pattern: random sampling gives 12 sagittal, six coronal and six
axial centers in expectation. Equal-plane sampling gives eight per plane;
that is a change in evidence weighting, not just a cure for missing coverage.
Nothing measured establishes eight per plane as clinically optimal.

Neighbor channels further reduce information loss. In the common four-by-12
case, random 24-center selection includes about 41.36 distinct cached slices
among its intensity channels in expectation. This is a combinatorial input
coverage calculation, not proof that the CNN uses neighboring channels equally.

## Existing experiment and distinct alternative

The teammate's plane-balanced function already apportions the 24 centers
equally among available planes with capacity redistribution. Its reader also
uses plane-specific later ResNet34 layers, heads, pooling and learned plane
trust. The wide-band run modifies the cache too. Those runs do not isolate
sampling alone, but the sampling idea itself is not new.

A different single-change hypothesis is series-and-depth stratification:
allocate six centers to each of four 12-slice series, divide each ordered series
into six adjacent pairs and randomly select one center per pair. Keep stochastic
choice every epoch. For this common case, all 12 cached slices necessarily
occur somewhere in the three-slice input channels, while only six centers are
processed. Rare shorter-series studies require explicit capacity redistribution;
never duplicate windows silently. This can be done with existing slot metadata.

This preserves average per-series weighting while reducing spatial gaps and
slot-count variability. It does not ensure a finding is present in a selected
center, nor that seeing it as a neighboring channel is equivalent. The expected
benefit is more consistent training exposure, not new MRI information.

Risks: loss of useful sampling randomness, persistent training/inference top-k
differences (three pooled logits from 24 training windows versus five from 48),
forcing redundant series allocation, and no improvement if crop/label quality
is the actual bottleneck. Cannot recover omitted source series, peripheral
slices outside the cached central band, or anatomy outside the crop. It does
not make the network understand MRI contrast or fat suppression.

Conclusion: reject the strong rationale for plane-balanced sampling as a major
coverage fix. Series-and-depth stratification is a distinct, implementable
experiment with unchanged inference cost, but current evidence supports only
a modest, unproven coverage-consistency hypothesis. No guaranteed AUC gain.
If pursued, use a fresh ImageNet-initialized ResNet50, unchanged HPO-v1 labels,
3524/825 split, augmentation, batch, scheduler and BCE; sampler alone changes.
One seeded comparison still cannot establish a general causal effect beyond
run-to-run variation. CPU statistics are in `data/window_sampling_analysis/coverage.json`.
