# Exp v3 peer review, round 2 (2026-10-12)

Reviewer: an independent agent with read-only access to the runner, cache builder, ledger and audit scripts.
Reviews revision 2 of `docs/exp_v3_spatial_finding_attention.md`.

**Verdict: minor revision.** All edits are applied in revision 3.

## Factual errors (all accepted)

1. **"Studies whose field is under 140 mm" are not padded.** The cache builder crops `min(140 mm, h, w)` and
   resizes it to 384 (`kaggle/experiment_kernels/coatnet_d96_cache/run.py:95-97`). The only padding comes from the
   augmentation's `grid_sample` (`runner:323-324`).
2. **Line references.** The window softmax is at `runner:355`; the 2-evid filter is at
   `audit_d96_window_evidence.py:40`.
3. **p ≠ g at initialisation when padding is present.** α is uniform over valid cells, so p is the masked mean,
   while g is v2's unmasked mean.
4. **H0 wording.**
   - 4b, 6a and 6b failed the clear-band floor (0.9617, 0.9652, 0.9626).
   - 6a lost to seed 2027 by 0.0038.
   - So "traded targets or failed the clear band" is the accurate description, not "ties".
5. **Cost.** The estimate was about 2× too high. 4b ran 36 min; with about +50–60% FLOPs, a v3 run takes roughly
   50–65 min, about $5–6.

## Round-1 concerns

| Status | Concerns |
|---|---|
| **Resolved** | Reframing; deferring the anatomical prior; keeping the GAP path; LN placement; replication; diagnostics |
| **Partly resolved** (finished in revision 3) | The same-stride control was only a wiring check; the gate optimised the wrong band; bundling stride with pooling; dropping the frozen-trunk test without a stated reason |
| **Turned into decisions** | The 576 vs 400 cell domain gap; peaky pooling with 16-window MIL sampling |

## Implementation notes (adopted)

- Use `forward_features` to get the feature map, with C0 and v3 in one class.
- An exact fp32 equality preflight, plus check (a′): `p == g` with no padding.
- `U = 0` initialisation.
- Compute the pooled features with an einsum, without materialising `α·z`.
- The mask is warped from a ones-image and used in training only.
- Memory is estimated at about 12–15 GB.
- v3 − C0 carries about one seed of noise: the new parameters consume CPU RNG, which changes the window draws, and
  dropout uses CUDA RNG.

## Answers to the open questions (adopted)

- **C0:** run it only if v3 passes, and require it to pass the gate by itself before promoting it.
- **GAP path:** keep it, with a pre-registered "branch used" rule.
- **384 evaluation:** a secondary readout, not a change to training.
- **L4 swap:** use it as the submission selector.
- **Temperature floor:** none.

## Decision-rule notes (adopted)

- Condition 1 in effect needs about +0.004–0.005.
- C0's attribution comparison is not a gate.
- There is a multiplicity guard on C0.
- "Falling clear band" is replaced by the floor alone.
- Two-seed replication bootstraps the averaged predictions.
