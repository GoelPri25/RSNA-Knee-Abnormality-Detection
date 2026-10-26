# L4C-LOC: a learned crop locator for v4 l4copy, distilled from MS1 placements (spec for review, 2026-10-19)

Status: **v2, revised after an independent review (2026-10-19, verdict "launch after fixes"; all blocking items fixed, see §7); nothing launched.** Ledger context: L4C-D, L4C-R1, L4C-R2, L4C-X (points 8–10), L4C-NF.
Constraints: MS1 placements may be used at training time; **MS1 must not run at inference** (user 2026-10-19: inference
speed). The submission must stay about as fast as the current v4 path (20 s on the 3 visible studies).

## 1. Why (evidence)

| fold 0, s2026 (LB 0.939) | 12-target label-v2 confident | clear band (LB tracker) |
|---|---|---|
| global only | 0.9443 | 0.9726 |
| self-placed crops (deployed) | 0.9453 | 0.9725 |
| self + PF OA / fracture global-only ("patch") | 0.9468 | 0.9730 |
| MS1-table crops + patch (**ceiling**, MS1 at inference) | 0.9492 | 0.9749 |
| s2027: self + patch → MS1 + patch | 0.9441 → 0.9471 | 0.9718 → 0.9753 |

Clear-band fit: LB = −0.687 + 1.671 × clear6 (v4 s2026 fit 0.938 vs LB 0.939; E3 0.945 vs 0.944).

- The crop branch was trained on MS1's placements; at inference it gets the self-located table (top-4 windows by the
  global attention, gradient × activation centroid). Self costs −0.0022 (12-target) vs MS1 (L4C-D).
- The difference is the **series / plane mix**, not the slice within a series (L4C-X point 10). Self and MS1 sagittal slices
  agree within 1 slice. Share of crops by series:

  | finding | MS1 | self |
  |---|---|---|
  | Medial / lateral / PF OA | 24–29% sagittal non-fluid | 2–3% |
  | Baker's | 85% axial | 0% |
  | MCL | 84% coronal | 69% |
  | Contusion / fracture | 42–48% coronal | 21–42% |

  The global attention that drives self placement reads SagF almost only (axial 1–2%; dropping axial changes no finding
  by > 0.002), so it cannot propose MS1's mix.
- L4C-D decomposition: MS1 windows + self points = −0.0034 vs MS1 (12-target), so **both** window choice and in-plane point
  matter. On the OA findings, MS1 windows + self points are the best placement measured (clear band medial OA 0.977, lateral
  OA 0.944 vs self 0.970 / 0.939).
- What failed and why this differs:
  - L4C-R1: MS1-free training from scratch, −0.007 to −0.010; the trunk lost coronal localisation.
  - L4C-R2: crop branch fine-tuned on its own placement, +0.0013, gate fail.
  - This spec changes **no existing weight**. It only learns to reproduce the placement the crop branch was trained on.
- L4C-NF (Lead 1, inference-only SagNF substitution for OA) is running in parallel; its result informs but does not gate this.
- Comparator that needs no training: the s2026 + s2027 self-placed mean + patch already fits LB 0.942 (clear6 0.9746) at 2×
  inference cost. The full MS1 ceiling for one model also fits 0.942. So the value of this run is reaching ~0.942 with
  **one** model at today's speed, not exceeding the 2-seed mean.

## 2. Design

**Frozen:** every weight of v4 l4copy (s2026, and s2027 as the replication seed). Global logits and the crop branch are
bit-identical to the deployed model; only the crop table changes.

**New module `Locator` (about 0.55 M parameters per model):**

1. **Window head.** Per cropped finding j (10), the score of evaluation window k is
   `s_kj = att_j'(h_k) + e[slot(k), j]`, where:
   - `h_k = model.norm(mean-pooled fm_k)`, v4's own pooled window feature (already computed in the global pass);
   - `att'` is a trainable copy of v4's attention MLP (2048 → 256 tanh → 12, CF rows used);
   - `e` is a 5 × 10 series-slot bias, initialised to 0.

   At initialisation `s` equals v4's attention logits, so top-4 = the self table (equivalence test).
   Loss: listwise KL(q_j ‖ softmax_k s_kj) over the study's 94 windows, where q_j is MS1's 4 window weights `a_top`
   (normalised) on its 4 chosen windows.
2. **Point head.** Per finding, a 1 × 1 conv (2048 → 10) on the frozen layer4 map fm_k (10 × 10 cells at 320 px, 32 px per
   cell) gives spatial logits, softmax over the 100 cells.
   - Target: a Gaussian (σ = 1 cell) at MS1's (row, col) **after `predict_study`'s clip to [hb, 384 − hb]** (Baker's 6.2%
     and fracture 2.9% of MS1 points sit at that edge); cell coordinate = (r − 32) / 32 − 0.5. All MS1 points lie inside the
     320 crop (rows / cols in [64, 320]).
   - Secondary to the constant points of baseline (i) below; the point head is kept only if it beats them.
   - Trained on MS1's 4 windows per finding only.
   - Inference point = soft-argmax → pixel (row, col) in the 384 px frame.
3. **Crop weights at inference:** softmax of the window scores over the chosen 4 (mirrors MS1's `a_top`, which the crop
   branch's training draw used).

**Inference table:** windows = top-4 of `s_·j`; points from the point head; weights from (3).

- Cost vs current: one extra small MLP on 94 vectors + one 1 × 1 conv on ≤ 40 maps, and the gradient × activation backward
  pass of `locate` disappears. Net ≤ current.
- Measured on Kaggle before any submission: must be ≤ current + 5%.

**Training data:**
- All folds except 0 (folds 1–5, **3,524 studies**, the same set v4 trained on, `l4copy_r1_runner.py`:196); asserted.
- These placement targets are crop positions, not classifier labels, and the crop branch already trained on them, so the
  out-of-fold label rule does not apply. MS1 placements on folds 1–5 are in-sample for MS1. The reviewer found no measurable
  shift vs fold 0 (slot mix ≤ 0.02 apart; mean `a_top` max 0.298 vs 0.299).
- MS1 table from `cp_*.npz`; MS1 window indices map to centres by `cs[cen[..., 0]]`, as in `l4copy_xai.py` (asserted on
  load).
- Features exactly as `predict_study`:
  - 94 windows, 320 centre crop, **fp16 autocast with fp32 `h`** (the deployed precision), BN running statistics;
  - `att'` in eval mode (v4's attention MLP contains Dropout 0.2).
- The trunk is frozen and there is no augmentation, so features are **extracted once**: `h` [94, 2048] fp16 for every study
  (≈ 1.4 GB), and layer4 maps only at the MS1 windows if the point head is trained. Training then takes seconds per epoch.

**Optimisation:**
- AdamW, LR 1e-3 for `e` and the point head, 1e-4 for the `att'` copy, wd 0.01.
- 10 epochs (raised from 4 before any result, since features are cached and an epoch takes seconds), batch 8 studies, cosine to 1%, seed 2026.
- Fixed epoch count; nothing selected on fold 0.

**Variants scored on fold 0** (all with the PF OA / fracture patch):

| variant | windows | points | weights | training |
|---|---|---|---|---|
| **W+P (primary, the only one that can be promoted directly)** | learned | learned | learned | yes |
| W+M | learned | per-(finding, slot) median MS1 point from folds 1–5 | learned | yes |
| (i) Q+M quota rule | self attention ranking inside a per-finding series quota = MS1's folds 1–5 slot mix (blank slices excluded) | median MS1 point | uniform | **none** |
| W | learned | self (gradient × activation) | learned | yes |
| self (reference) | self | self | v4 attention | none |
| MS1 (ceiling) | MS1 | MS1 | MS1 | none |

- Why the zero-training baselines: they separate this run from the failed `self_plane` / plane routing.
  - `self_plane` (L4C-D) lost through its points: self-located axial Baker's boxes missed the popliteal fossa, local 0.609.
  - MS1's points are regular: the median distance to the folds 1–5 per-(finding, slot) median is 10–21 mm.
  - Shifts up to 20 mm cost nothing (L4C-D, −0.0001).
- If the zero-training Q+M gets within 0.0005 clear6 of W+P, Q+M is the submission: it is leaner, with no new weights.
- Any variant other than W+P or Q+M needs an extra +0.0005 margin over the gate (best-of-n, as in L4C-NF).

## 3. Pre-registered gates (fold 0; label-v2 confident cells; 1,000 paired study bootstraps)

- **PASS (submit as submission A′ = s2026 + locator + patch, after your OK)** if, for **W+P vs self (both patched)**, on
  **both s2026 and s2027**:
  - clear6 Δ ≥ +0.0010 with ≥ 80% of bootstraps > 0 (clear band = the LB tracker; ceiling +0.0019 / +0.0035);
  - 12-target Δ ≥ −0.0005 (guard).

  The same rule as L4C-NF. The earlier 12-target +0.0015 / 90% gate was near-unpassable: the MS1 ceiling itself was
  +0.0022 [+0.0001, +0.0041].
- Expected LB if PASS: clear-band fit of the variant, stated before submission.
- **FAIL:** the frozen features cannot reproduce MS1's placement. Record the diagnostics; the next step would be fine-tuning
  the trunk with the locator loss (separate spec).
- **Diagnostics (not gates):**
  - top-4 window overlap with MS1 per finding and **per series** (Baker's on axial especially: frozen v4 features are weak
    there; axial-only Baker's 0.36), and the slot mix vs MS1's;
  - point distance to MS1 in mm;
  - per-finding Δ (OA, Baker's, MCL);
  - train/val locator loss per epoch (logged like val BCE).

## 4. Implementation and tests

- `scripts/l4c_loc_runner.py`: reuses `l4copy_r1_runner.Vols` / cache reader and the `v4_submission.py` model and
  windowing (the deployed code, not the training-side model class, so the deployed path is what is trained against).
- Outputs:
  - `locator_s{seed}.pt` (locator only, about 2 MB);
  - fold-0 tables and crop logits for every variant;
  - `log.json` (losses per epoch, diagnostics, timing).
- `scripts/test_l4c_loc_cpu.py` (CPU, `localhost/rsna-mlx-train:v1`, `--network=none`), on the local fold-0 shard:
  1. At init the window ranking equals `sub.locate`'s top-4 on 3 studies.
  2. The MS1 mapping `cs[cen]` reproduces `l4copy_xai`'s `tms1`.
  3. Loss decreases over 20 steps on 4 studies.
  4. Frozen weights unchanged after steps (hash).
  5. The fold-0 exclusion assert fires on a fold-0 id.
  6. Soft-argmax maps cell centres back to pixel coordinates correctly.
  7. `predict_study(table=locator_table)` equals the runner's fused output.
  8. Eval mode: dropout off in `att'`. fp16 feature parity with `predict_study` (≤ 1e-5).
  9. Point targets use the clipped centre.
  10. The training-ID set is all folds except 0 (3,524 on the full manifest).
  11. Timing: locator table vs `locate` on the same study (locator must not be slower).
  12. Q+M quota rule: quotas sum to 4 per finding; ranking inside a slot follows v4 attention.
- Submission integration (only after PASS): `v4_submission.py` gains an optional `LOCATOR` file; with it absent the output
  is unchanged (equivalence test on the 3 visible studies).

## 5. Compute (MeluXina; needs your go)

- **Spec:** 1 node (4 × A100), 1 node-hour allocation (the minimum).
- **Lanes:** 2 in parallel: s2026 on GPU 0, s2027 on GPU 1.
- **Container:** the existing `rsna-mlx-train.sif` (the same image as R1/R2); no new dependencies.
- **Data:** the d96 cache staged to `$LOCALSCRATCH` as in `r1.sbatch`; network not needed.
- **Estimate:**
  - one feature-extraction pass: 4,349 studies × 94 windows ≈ 410 k r50 forwards at 320 px, ~1,200–1,500 img/s fp16 on one
    A100 ≈ 5 min;
  - training on cached features: seconds per epoch;
  - fold-0 scoring of 6 variants ≈ 10 min;
  - staging ≈ 5 min;
  - total ≈ 25 min per lane, inside 1 node-hour.
- Locally: CPU tests only; no host GPU.

## 6. Risks

- **Leakage:**
  - MS1 was trained on folds 1–4 labels. Its placements on those studies may be sharper than on unseen studies; the
    locator imitates placements only, never MS1 logits.
  - Fold-0 MS1 placements are used only for the ceiling and diagnostics, never for training.
  - The crop branch was already trained on these placements.
- **Capacity:** if v4's frozen features do not separate SagNF / axial windows for OA / Baker's, the slot bias `e` can still
  shift the mix (a learned per-finding series prior). That is the smallest change that can express MS1's 24–29% / 85% mix.
- **LB transfer:** every placement variant so far moved fold 0 by ≤ 0.003; with the ±0.002 LB noise, one submission
  cannot confirm a +0.0015 gain. The decision rests on the clear-band fit plus the LB direction.
- **Time:** deadline 22 Oct; this runs in < 1 h once approved, and the code + tests take about half a day.

## 7. Review (independent agent, 2026-10-19) and changes

Verdict on v1: launch after fixes. Verified correct by the reviewer:
- the window mapping `cs[cen[..., 0]]`;
- init equivalence (softmax is monotone per finding; the softmax over the chosen 4 = the deployed weights);
- no MS1 point lies outside the 320 crop;
- `a_top` is normalised and near-uniform.

Fixed in v2:
1. Precision fp16, not bf16.
2. Training folds 1–5, 3,524 studies (not 1–4).
3. Gate moved to clear6 ≥ +0.0010 / 80% on both seeds with a 12-target guard, matching L4C-NF.
4. Zero-training baselines Q+M and W+M added, so the result can be told apart from `self_plane` / plane routing.

Also taken: cached features, clipped point targets, per-series overlap diagnostic, extra CPU tests, the 2-seed comparator.

## 8. Implementation status (2026-10-19)

- Files:
  - runner `scripts/l4c_loc_runner.py`;
  - tests `scripts/test_l4c_loc_cpu.py`;
  - quota prior `scripts/build_l4c_quota_prior.py` → `data/coatnet_d96_fold0/l4copy_analysis/quota_prior.json`;
  - launch script `scripts/meluxina/loc.sbatch` (1 node, 1 h, 2 lanes; weight sha256 checked on the node).
- CPU tests pass (`data/coatnet_d96_fold0/l4copy_analysis/loc_test/test.log`):
  - init table = `sub.locate`;
  - MS1 mapping;
  - clipped point targets;
  - cell round trip (interior < 2 px; border bias ≤ 9 px = 3.3 mm from the truncated Gaussian, under the 5 mm no-loss shift);
  - loss decreases (31.9 → 22.4 in 20 steps);
  - v4 weights unchanged;
  - no fold-0 training study.
- End-to-end smoke (3 + 3 studies, 2 epochs):
  - self fused = `predict_study` (max |Δp| 1.5e-8);
  - locator table 5.6 ms vs `locate` 38.5 ms on CPU (faster: no backward pass).
