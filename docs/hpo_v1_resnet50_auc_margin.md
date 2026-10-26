# HPO-v1 ResNet50 AUC-margin refinement

The experiment starts from the epoch-17 ResNet50 that scored 0.924 publicly
(checkpoint SHA256 `7c76e8760b6989fb5bb3afa2fe14cf016ac766981ad612c94fc058acb703120a`).
It keeps the six-channel reader, target-specific top-10% pooling, original
slice caches, 24 training windows, augmentation, HPO-v1 probabilities, 3524
training studies and fixed 825-study Steven-v4 validation. Gold is excluded.

The objective is soft BCE plus 0.1 times a macro AUC-margin saddle objective.
Only cells with two available, agreeing sources both above 0.7 or below 0.3
enter its positive/negative moments. Steven's 0.25 sentinel is excluded.
Source masks indicate availability, not verified clinical correctness.
All cells retain their existing BCE targets. Labels are not rewritten.

For each target, sigmoid scores estimate positive/negative conditional means
and within-class squared deviations from learned centers. A nonnegative dual
variable penalizes inadequate mean separation with margin 0.3. Cohort support
rates normalize uniform minibatches; both classes need not occur in each batch.
This adapts Yuan et al., ICCV 2021; it is **not** exact LibAUC/PESG reproduction.
The classifier is retained, AdamW trains network weights, SGD trains centers,
and projected ascent updates the dual variables. No inference layers change.

Settings: batch 8; neural LR 3.75e-5; center/dual LR 0.005; max 12 additional
epochs. Validation BCE controls ReduceLROnPlateau (factor 0.5, patience 2,
cooldown 1, minimum LR 1e-6). AUC chooses checkpoints; early stopping allows
8 epochs without cumulative improvement of 0.0005. The starting checkpoint
remains the fallback if no additional epoch beats its reevaluated AUC.

Run `scripts/prepare_auc_margin_experiment.py` on CPU first. Inputs live under
`data/hpo_v1_resnet50_auc_margin_fold0/inputs`. Launch
`modal run --detach scripts/modal_cnn_auc_margin.py` with personal credentials.
The app uses one T4 and the existing `rsna-cnn-topk-support-cache-v1` volume;
it does not download raw DICOMs. Results persist under `fold0/run` in
`rsna-cnn-auc-margin-artifacts-v1`. Checkpoints commit every 50 batches and
each epoch, containing neural/center/dual state, optimizers, scheduler, scaler,
shuffle position and RNG states. Preemption retries resume the last committed
snapshot; at most 49 completed batches may need replay. Timeout is 12 hours.
Kaggle artifact publication is a separate follow-up after training.

Loss equivalence, evidence filtering and masked-gradient checks are in
`tests/test_auc_margin.py`. This run tests AUC refinement potential, not whether
HPO-v1 is intrinsically better than HPO-v2, and guarantees no score improvement.
