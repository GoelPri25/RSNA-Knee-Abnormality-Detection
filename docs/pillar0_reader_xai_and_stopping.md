# Pillar-0 reader: XAI findings and validation rules

## What was evaluated

This was an inference-only inspection of the saved fold-0 target-query reader,
checkpoint epoch 26. The audit used 825 report-validation studies and 4,543
series from the existing Pillar-0 cache. It did not train, load official gold
labels, or access raw DICOM. The checkpoint reproduced its stored validation
result: macro AUC 0.746529 and weighted soft-label BCE 0.545956.

The reader has 265,313 trainable parameters. Pillar-0 itself is frozen and its
features are precomputed. Each series contributes 512 fine 8³ tokens, 512
native 8³ tokens, and 8 coarse 2³ tokens, each 384-dimensional. Twelve learned
target queries pool tokens from all series in a study. The comparisons below
intervene on that reader’s inputs; they cannot establish whether the frozen
Pillar-0 encoder represented every relevant image feature.

## Counterfactual feature tests

| Validation condition | Macro AUC | Change from baseline | Weighted BCE | Change from baseline |
|---|---:|---:|---:|---:|
| Baseline | 0.74653 | — | 0.54596 | — |
| Fine 8³ visual features set to zero | 0.74069 | −0.00584 | 0.54899 | +0.00304 |
| Native 8³ visual features set to zero | 0.71617 | −0.03036 | 0.63570 | +0.08974 |
| Coarse 2³ visual features set to zero | 0.69721 | −0.04932 | 0.63199 | +0.08603 |
| All visual features set to zero | 0.55879 | −0.18774 | 0.63044 | +0.08449 |
| Protocol metadata replaced with unknown IDs | 0.73404 | −0.01249 | 0.55782 | +0.01187 |
| Zero-support tokens hard-masked | 0.74471 | −0.00182 | 0.54812 | +0.00216 |
| Visual features shuffled within each series, coordinates kept fixed | 0.73102 | −0.01551 | 0.56736 | +0.02140 |

The largest finding is that the reader depends on both native 8³ and coarse 2³
visual content. Fine 8³ contributes much less on this fold. Removing all visual
content leaves AUC only modestly above chance, so the result is not explained
by scan protocol and coordinates alone. Shuffling visual tokens while leaving
their coordinates in place hurts performance, which is evidence that
feature-to-position correspondence helps. These interventions are deliberately
strong and can create inputs outside the training distribution; their metric
changes are diagnostic, not estimates of the gain from retraining without a
scale.

Coarse 2³ has only 8 tokens per series, about 0.8% of all tokens, but receives
14.3% of attention mass. Native 8³ receives 57.6% of mass and about 76% of the
absolute Grad×Input attribution on the 96-study explanation subset. Fine 8³
receives 27.8% of attention but about 8% of Grad×Input attribution. The
attention enrichment for coarse tokens is roughly 18× their token-count share.
This suggests the reader is using the coarse scale as a compact, high-value
global summary, while relying most on native 8³ features for prediction.

Protocol metadata also matters, but less than visual features. Replacing plane,
fluid-sensitive, and fat-suppression IDs with unknown IDs lowers AUC by about
0.0125. The audit combines those fields; it does not separate their individual
effects.

## Support and spatial evidence

In the validation tokens, 78.4% have support exactly zero. They still receive
21.0% of attention mass. The reader adds `log(clamp(support, 0.05, 1))` to its
attention logits, so zero support is only a finite penalty of about −3 nats;
it is not an exclusion. Tokens with support above 0.25 are 13.1% of tokens but
receive 60.3% of attention. Tokens in the (0.05, 0.25] band are 5.7% of tokens
and receive 17.6% of attention.

Hard-masking zero-support tokens slightly worsened both metrics on this fold
(AUC −0.00182; BCE +0.00216). That does not show the unsupported image regions
are intrinsically useful: Pillar-0 contextualizes tokens before caching, so a
token whose own cell has no valid voxels may still contain context from nearby
valid voxels. The mask is applied only in the downstream reader. We should not
replace the current soft prior with hard masking based on this one-fold result.

The target-wise native 8³ Grad×Input maps concentrate around a similar central
region, with some label-specific shifts. This is not evidence of lesion
localization: the maps are pooled across studies, projected over one spatial
axis, and may reflect the common crop/support geometry. The audit did not
compare these maps to an anatomical atlas or per-study support occupancy map.
The first kernel export had a target-axis broadcasting error in the saved grid;
the normalized maps were recovered by matching numerator and denominator
target axes, and each target’s recovered map sums to 1. The local corrected
visualization is
`data/pillar0_xai/results/pillar0_xai/gradient_input_native8_by_target.png`.

## Are we using Pillar-0 fully?

The reader does use all three cached scales, and the interventions show they
are not interchangeable. It is not using the cache to its full spatial
resolution, though. The encoder’s 32³ fine map has already been averaged to
8³ before caching; the reader then combines all series tokens into one
target-specific context vector with one cross-attention operation. It cannot
learn new interactions between neighboring tokens or between a local region in
one series and a matching region in another series, except for relationships
already encoded by frozen Pillar-0 features. It also cannot adapt the encoder
to this task. The XAI evidence favors preserving the coarse and native scales;
it does not support simply adding more fine tokens as the next change.

The highest-value next representation change is a small, controlled reader
comparison: keep the same cached inputs and label split, but let target queries
first pool local neighborhoods in the native 8³ map and then perform the
current cross-series pooling over those regional summaries. Preserve the
coarse 2³ tokens as a separate global context path. Keep the baseline reader
as a paired control and compare on the same report-validation folds. This
directly tests whether the current global token competition is discarding
useful regional structure; it avoids changing the encoder, labels, and loss at
the same time.

## More robust checkpoint, scheduler, and stopping rules

The goal metric is AUC, so checkpoint selection and early stopping should
continue to use report-validation macro AUC. BCE remains useful for monitoring
probability fit, but changing BCE should not silently determine which model is
kept when competition score is AUC. Keep two notions separate: save the raw
highest-AUC checkpoint, and use a predeclared minimum improvement only to
decide whether patience resets. That makes small raw wins visible without
letting a single noisy fluctuation control stopping.

In the saved run, the scheduler monitored BCE with factor 0.5, patience 2,
cooldown 1; early stopping monitored macro AUC with patience 8 and minimum
improvement 0.0005. The factor 0.5 by itself is not unusually large—PyTorch
describes reductions by factors of 2–10 as common—but the combination caused
repeated halvings: the learning rate fell from 3e-4 to 9.375e-6 by epoch 33.
That is a 32× reduction. Epoch 34’s raw AUC was 0.746553, slightly above
epoch 26’s 0.746529, but by far less than the 0.0005 checkpoint threshold.
The selected epoch remains 26. The tiny difference should be treated as
unresolved measurement noise, not proof that epoch 34 generalizes better.

For the next run, use a single declared primary metric for selection; save the
raw best AUC independently of the patience threshold; allow at least four
epochs after each LR reduction before early stopping can fire; and slow the
combined decay schedule (for example, scheduler patience 4 and cooldown 2,
with factor 0.5). Calibrate the AUC improvement threshold from paired
study-level bootstrap differences on report validation rather than choosing
0.0005 by convention. Because repeatedly consulting the same fold can itself
overfit model selection, use all five report folds for comparing architectures
and report a paired interval for out-of-fold AUC differences. Keep the official
gold set locked until a final model decision.

This follows the scheduler’s documented meaning of patience, threshold, and
cooldown; early-stopping research also emphasizes that the stopping criterion
trades computation against generalization rather than having a universal
optimum. The paired bootstrap is appropriate for comparing two models on the
same studies because their AUC estimates are correlated. Attention weights
are only routing weights, so they were cross-checked against Grad×Input and
input interventions rather than treated as explanations alone.

### Reading

- PyTorch, [`ReduceLROnPlateau`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html): patience, threshold, cooldown, and common factor range.
- Lutz Prechelt, [“Early Stopping—but when?”](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf): stopping criteria trade training time against generalization; the empirical setting differs from this MRI task, so it is guidance, not a recipe.
- Sarthak Jain and Byron Wallace, [“Attention is not Explanation”](https://aclanthology.org/N19-1357/): attention distributions alone can fail as faithful explanations.
- Mukund Sundararajan et al., [“Axiomatic Attribution for Deep Networks”](https://proceedings.mlr.press/v70/sundararajan17a.html): Integrated Gradients and attribution axioms; motivates using gradient-based evidence alongside interventions.
- Elizabeth DeLong et al., [“Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves”](https://pure.uva.nl/ws/files/1631221/112984_06.pdf): paired comparison of correlated ROC-AUCs.

## Limits of this audit

All performance interventions used one report fold and are subject to its
sampling noise. The running audit version did not calculate bootstrap
confidence intervals for intervention deltas, so small effects—especially
zero-support hard masking—are inconclusive. Grad×Input describes local model
sensitivity, not causal evidence about anatomy. The model input cache is
already context-encoded and int8-quantized; this audit does not measure
information lost in the original encoder, the pooling used to form the fine
scale, or the quantization step.
