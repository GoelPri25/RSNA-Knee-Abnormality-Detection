# Synovitis supervision weight experiment

Matched from-ImageNet ResNet50 runs at 224px, seed 2027. Based on the 0.925 HPO-v3.1-D2 / Steven-v4 validation recipe. Baseline then candidate run sequentially on one T4; no GPU ensemble work is launched here.

Only candidate training weights change: 2,930 Synovitis cells multiply their existing weight by 0.25. Preserve 594 reports with multilingual synovial-root text or non-silent Synovitis rule evidence, conservatively including uncertain scope. All probabilities, other eleven target weights, and all validation rows are unchanged. No gold labels or predictions are used. Build with `python scripts/build_synovitis_weight_experiment.py`.

Loss is the existing sum(weight * BCE) / sum(weight). This reduces the relative contribution of inferred Synovitis; the shared denominator slightly rescales other targets' gradients. No per-target normalization or loss-code change is introduced. The loader reads explicit JSON weights directly, not a distance-from-0.5 rule.

Fixed Steven-v4 validation controls: BCE plateau factor .5/patience2/absolute threshold .0005/cooldown1/minLR1e-6; AUC checkpoint and stop patience8/minDelta.0005, max60epochs, batch8,24training windows,48evaluation,140mm crop,AdamW LR3e-4/weight decay.03,original augmentation/dropout.4. Seed2027 is shared between arms; original historical run seed2026 provides additional context, not the paired baseline.

Recovery commits every epoch (weights, optimizer, scheduler, scaler, RNG, counters and predictions) to existing `rsna-cnn-topk-support-cache-v1` under `experiments/synovitis_weight_pair_seed2027`. Incomplete epochs replay on retry. GPU attempts have 12h timeout and 3 retries; coordinator is detached and invokes one GPU arm at a time. Batch progress printed every50; datasets published independently after each arm. A publication failure never starts training again.

Readouts: paired fixed-reference macro/per-target AUC, other-target shifts, Synovitis/Effusion correlation and disagreement. Correlation is descriptive, not an objective to minimize. Report silence is never a diagnostic negative. Existing gold is development evidence only; public scoring assesses overall utility but cannot attribute gains to Synovitis. One paired seed cannot establish reproducibility; repeat if results justify compute.

## Candidate-only continuation (2026-10-08)

At user request, stopped app ap-uDti0YYcvX3ztbRAXrABzV during its baseline arm. No baseline rerun will launch. Candidate-only uses seed2026, matching the original .925 baseline, in a new output path `experiments/synovitis_weight_candidate_seed2026/candidate/run`; it never loads the partial seed2027 baseline. Assets in `data/synovitis_weight_candidate_seed2026/assets`. Launch `scripts/modal_synovitis_weight_candidate.py` detached. Same weighting, fixed validation and recovery; seed change occurs before candidate training starts.

## Cancellation and deployed recovery (2026-10-08)

App `ap-jVx0enTJ09Qz5iaHsO1Nqa` received cancellation signals during epoch 18. The available logs identify cancellation and a client shutdown, not early stopping or a training timeout. The recorded launch used `--detach`, but the local entrypoint waited synchronously for the coordinator. The trigger for cancellation is not established.

The volume recovery was inspected remotely on CPU: saved epoch 18, 8,379 optimizer steps, three stale epochs, best epoch 15 with fixed Steven-v4 macro AUC 0.8894333842143732. The candidate resumed at zero-based epoch 19, restoring Adam, scheduler, scaler and RNG. No training recipe or labels changed.

Use `python scripts/run_modal_personal_synovitis_candidate.py`. This guards against an active duplicate, deploys the app, and invokes a detached local entrypoint that submits the deployed coordinator with `.spawn()` and exits. There is no local result waiter. The deployed job keeps running after that launcher completes, following [Modal's background job pattern](https://modal.com/docs/guide/job-queue). A deployed app can remain listed after completion without an active GPU container.

Verified deployed app: `ap-MaVtipJq2UvGx0ze9RYTWX`; coordinator call `fc-01M4EK2N9M2SY86BD1ERYNJ5XH`. Launch receipt and captured resume logs are in `data/synovitis_weight_candidate_seed2026/`. The small stopped app `ap-G6sQi0NcGaD1Hz5wWOUIoM` is the completed submission launcher, not another training run.

## Result (2026-10-08)

Candidate seed 2026, best epoch 15 (same as the paired 0.925 baseline). Readouts by `scripts/audit_synovitis_weight_candidate.py`,
outputs in `data/synovitis_weight_candidate_seed2026/readouts/`. Label check passed: probabilities identical, other eleven
weights identical, 2,930 Synovitis cells at 0.25x.

- Fold-0 Steven-v4 macro 0.8894 vs 0.8963 baseline: -0.0069 [-0.0126, -0.0014]. Synovitis +0.0105 [-0.0013, +0.0230].
  Largest drops are on unchanged targets: Fracture -0.038 [-0.067, -0.011], PF OA -0.023 [-0.038, -0.008], MCL -0.025.
  The 224px rerun with identical labels differs from the baseline by up to 0.036 per target (MCL), so these are
  consistent with run-to-run variance; one pair cannot separate the two.
- Coupling unchanged: Synovitis-vs-Effusion head Spearman 0.889 vs 0.887; studies with Steven Effusion >= 0.7 and
  uncertain Synovitis rank at the same mean percentile (0.583 vs 0.581). Within-effusion-positive AUC (0.918 vs 0.857)
  has only 8 negatives and is not interpretable.
- Ensemble: replacing the v3.1-D2 224 trio member -0.0020 [-0.0047, +0.0005]; adding as a 4th member -0.0001.
- Note: 2,930 x 0.25 = 733 effective inferred cells still outweigh the 594 direct-evidence cells.

Conclusion: no evidence that 0.25x weighting reduced the Effusion coupling, and no fold-0 or ensemble gain. Not submitted.
