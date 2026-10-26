# XAI experiments for the RSNA knee models

Source reviewed: *Catalogue of Explainability Techniques for AI Applications in Space*,
ESA public release, version 1.0, 2025-01-29. The catalogue is not medical guidance; this note
translates its explanation taxonomy into model-development tests for this project.

## Principle

An explanation is useful here only when it can reveal a data defect, leakage, shortcut,
unstable prediction, or missing model capability. A visually attractive heatmap is not enough.

The catalogue separates explanations along three axes:

- purpose: attribution, sensitivity, counterfactual, or interpretable model;
- scope: local, semi-global, or global;
- access: model-specific or model-agnostic.

We should cover more than one cell of this taxonomy. A single local SHAP or saliency example
cannot establish what the model learned across the dataset.

## Explanation stack for our hierarchical model

The natural explanation units follow the model hierarchy:

```text
study prediction
  -> contribution of each MRI series
  -> contribution of slices/windows inside a series
  -> contribution of image regions inside a slice
  -> similarity to representative studies or concepts
```

Begin at series and slice level. These units have stable clinical meaning and are less
sensitive to pixel-level noise than raw saliency maps.

## Priority 1: structured occlusion and ablation

This is the most actionable model-agnostic attribution method for the project.

For each target and study, recompute the logit after masking:

- one complete series;
- one anatomical plane;
- fluid-sensitive or structural series;
- peripheral, intermediate, or central slice-depth bands;
- selected contiguous slice windows.

Record `original_logit - ablated_logit`. Aggregate results globally and within true-positive,
true-negative, false-positive and false-negative groups.

This can answer whether ACL uses sagittal series, effusion uses fluid-sensitive sequences,
and MCL or lateral meniscus depends on peripheral slices. It can also expose dependence on
padding, image borders or one scanner protocol.

## Priority 2: attention validation

Attention weights are internal model signals, not automatically faithful explanations.
Validate them by checking whether removing highly attended series/slices changes the target
logit more than removing low-attention inputs.

Report:

- rank correlation between attention and ablation effect;
- top series and depth bands by target;
- stability across folds/seeds;
- agreement for augmented or near-duplicate inputs;
- distributions by correct and incorrect prediction groups.

Attention that is attractive but fails the ablation test should not be interpreted clinically.

## Priority 3: metadata SHAP

SHAP is appropriate for the small study aggregator's explicit metadata, including plane,
fluid sensitivity, fat suppression, spacing, slice count and scanner fields. Compute it for
metadata-only and metadata-plus-image models.

The goal is shortcut detection:

- excessive scanner/manufacturer importance;
- dependence on slice count or spacing;
- protocol variables overwhelming the image embedding;
- different metadata effects between validation groups.

Do not apply generic pixel SHAP to complete 3D volumes as the default. It is expensive,
baseline-dependent and difficult to interpret at individual-pixel resolution.

## Priority 4: Integrated Gradients on selected cases

Use Integrated Gradients only after series/slice attribution identifies a small number of
important slices. Generate class-specific maps for representative true positives, false
positives and false negatives.

Baseline choice must be tested. Compare at least a black/zero-normalized baseline, a blurred
slice, and a low-information slice from the same acquisition. Record whether the highlighted
regions remain stable. An attribution that changes radically with a plausible baseline is weak
evidence.

Run deletion tests: progressively remove the most attributed regions and compare the logit
drop against random regions. This tests faithfulness rather than visual appeal.

## Priority 5: prototype and retrieval explanations

The catalogue's prototype methods map directly to the preserved `z_study` representation.
For each prediction, retrieve:

- nearest gold-labelled studies with the same label;
- nearest studies with a different label;
- nearest high-confidence report-labelled studies;
- the most similar study that contradicts the prediction.

Display similarity, labels, prediction vectors, protocol metadata and representative series.
Aggregate neighbourhood label enrichment. This supports statements such as “this study is
represented similarly to these known ACL-positive studies,” while contradictions identify
hard cases and possible label noise.

Do not initially modify the classifier into ProtoPNet. Retrieval from the existing embedding
tests the idea without introducing another architecture.

## Priority 6: concept tests inspired by TCAV

Define concepts using existing metadata or curated examples:

- sagittal versus coronal versus axial;
- fluid-sensitive versus structural;
- fat-suppressed versus non-fat-suppressed;
- peripheral versus central anatomy;
- high versus low field strength;
- scanner families;
- padding or truncated-field-of-view artifacts.

Fit linear concept directions in series or study embedding space and measure each target's
sensitivity to them. Repeat with several random control sets. A concept is credible only when
the result is stable across repeats and splits.

Metadata-derived concepts are useful for auditing representation structure, but they are not
equivalent to clinical pathology concepts. Clinical concepts require reviewed examples.

## Counterfactuals to use and avoid

Generated medical-image counterfactuals are risky because a visually plausible change can be
anatomically or diagnostically invalid. Do not begin with diffusion-generated “add a tear” or
“remove a fracture” images.

Use constrained counterfactual tests instead:

- remove or replace a series;
- perturb intensity within realistic bounds;
- vary crop or padding while preserving anatomy;
- reorder slices as a deliberate negative control;
- change explicit metadata while holding embeddings fixed;
- find an observed nearest-neighbour study across the decision boundary.

These tests measure robustness without pretending that synthetic pathology is clinically real.

## Evaluation requirements

For every explanation method, measure at least one relevant property:

- faithfulness: does removing important input reduce the score more than random removal?
- stability: do similar inputs and repeated model fits produce similar explanations?
- sparsity: can a small number of series/slices explain most of the effect?
- plausibility: does the explanation use valid anatomy and acquisition context?
- global consistency: do aggregate explanations agree with per-target performance?
- utility: does the finding lead to a concrete preprocessing, supervision or model test?

Keep explanation results separated by target and by outcome group. Global averages can hide
different mechanisms for true and false predictions.

## Minimal implementation order

1. Export per-target series attention and normalized slice positions.
2. Add leave-one-series and leave-one-depth-band ablations.
3. Compare attention ranks with ablation effects.
4. Add nearest-neighbour and contradiction retrieval from `z_study`.
5. Run metadata SHAP to identify protocol/scanner shortcuts.
6. Apply Integrated Gradients to a small, deliberately selected error set.
7. Add concept-direction tests only after the preceding outputs are reliable.

The first four steps require no new model architecture and should provide the best immediate
training feedback.
