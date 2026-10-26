# V4-IMP: single-change improvements to v4 l4copy (spec v2, 2026-10-19)

v1 (same day) proposed four lanes: ELR, silent ×0.25, MC-PAIR, and mixup. Two independent reviews changed it:

- **Methods / literature review:**
  - ELR in sigmoid form sharpens predictions instead of pulling them to the memory: gradient (1−2t)·p(1−p)/(p+t−2pt), about 25× BCE on easy negatives.
  - The ELR memory is the model's own in-fold output, which raises the out-of-fold rule question.
  - v4 shows no memorisation phase to fix. In v4 s2026, val BCE plateaus at 0.387 from epoch 17 and the train/val gap stays at 0.03–0.04.
  - Mixup suits symmetric noise and long schedules. Here the noise is systematic (report omission) and the schedule is 25 epochs.
  - Recommended: a confidence-aware loss on silent cells (Hill / ASL), and replicate the input-change lane.
- **Implementation review:**
  - Map partner channels row by row, not as a neighbouring triplet: in 42% of sagittal pairs the partner's ±1 neighbours are more than 1.5 mm away.
  - Pairing must require opposite fluid flags.
  - Zero unpaired channels after normalisation.
  - Mirror each triplet separately.
  - Use a fixed loss denominator in the silent lanes (otherwise the effective LR changes).
  - Don't down-weight silent cells that label-v2 already scores positive.
  - Inflating conv1 must not consume RNG.

ELR and mixup are dropped.

## Why

- Every single model sits at LB 0.933–0.939, whatever the architecture. Only ensembles help.
- A′ (learned crop locator) gained about +0.003 on fold 0 but scored LB 0.938.
- The working diagnosis is that the limit is the training signal and the input, not the head.
  - **Training signal:** report-silent cells are 49% of training cells at full weight.
  - **Input:** FS and NF images are never seen together.
- Gate for the next step (teacher-student): a single model above LB 0.940.

## Common recipe

`scripts/r50_gl4_runner.py` with `V4_ARM=l4copy` and one change selected by `V4_IMP`. With `V4_IMP` unset, the runner is v4 exactly.

Recipe:

- label-v2;
- 25 epochs, SWA 21–25, BN refresh;
- MS1 crop table (training only);
- fold 0 = validation (825 studies); 3,524 training studies.

References (no rerun): v4 l4copy s2026 (LB 0.939) and s2027.

## Lanes

### silw: silent assumed-negative cells ×0.25

- **Cells:** rule state SIL (`rule_states.csv`, sha ce4473c7) with label-v2 y < 0.5. That is 46.9% of training cells.
- SIL cells that label-v2 scores ≥ 0.5 keep full weight; 571 of them have y ≥ 0.7.
- **Change:** BCE weight 0.25 on these cells, in both heads.
- **Denominator:** the cell count (12·b global, 10·b local), so the step size matches v4.
- **Optimum:** still p = y; only the influence of these cells drops.
- **Mechanism:** reports omit minor and secondary findings. Full-weight assumed negatives teach the model to suppress image evidence the report didn't mention.
- **Per-target share of negatives that are SIL:** synovitis 99%, fracture 85%, Baker's 77%, contusion 71%, ACL / MCL / OA 64–67%, menisci 19–21%.

### hill: silent assumed-negative cells, Hill loss

- **Change:** on the same cells, BCE's negative term becomes the Hill loss (Zhang et al. 2021, missing-label multi-label): loss = y·softplus(−z) + (1−y)·(1.5 − p)·p². Full weight, cell-count denominator.
- **Gradient on the negative side:** 3p²(1−p)².
  - It still pushes uncertain predictions down.
  - It fades as p → 1, so confident image evidence on a silent cell is not punished.
- **Equilibrium:** at y = 0.06, p = 0.159; BCE's is 0.06.
- **Silent-cell ranking:** silent cells rank above explicit report negatives (NEG). That is the intended prior.
- **Loss shape only:** no model-made targets, so the out-of-fold rule question does not arise.

### mcpair: FS + NF co-registered input (seeds 2026 and 2027)

**Geometry checks.**

- The cache is a centred 140 mm crop at 384 px.
- Pairing uses DICOM IPP along the slice normal.
- On 984 fold-0 mid-slice pairs, 86% align in-plane within 1.5 mm (NCC of gradient images).
- CPU test: paired slices have NCC 0.55, against 0.14 for a random row of the partner series.

**Pairing table, per cache row of SagF / SagNF / CorF / CorNF.** The partner is the complementary series' row nearest along this series' normal, if all of these hold:

- distance ≤ 1.5 mm;
- normals within 5°;
- in-plane corner offset ≤ 2 mm;
- the two series' fluid flags differ.

Otherwise the row has no partner (−1). Axial rows are always −1. 41.6% of valid rows are paired.

**Input.**

- 6 channels: the window's own triplet [c−1, c, c+1] (unchanged) plus the partner rows P[c−1], P[c], P[c+1], channel by channel.
- Unpaired channels are exactly 0 after normalisation.
- Crops get the same: the partner rows at the same centre, through the same affine.
- Sagittal mirror reverses each triplet separately, and the partner flags follow.
- Partner dropout 0.2 per window and per crop (training), so missingness can't become a shortcut.

**Model.**

- conv1 is inflated to 6 channels; the partner half is zero-init. At init the model equals v4 (CPU test: outputs match within 1e-4, no extra RNG draws).
- Sequence-type bias b[own fluid × centre paired, 12] on the attention logits, zero-init. This tells the model which contrast it is reading.

**Replication.** Seeds 2026 and 2027, because this lane is judged on fold 0, where 1-seed verdicts have flipped before (L4C).

## Run

- **Placement:** MeluXina, one node (4×A100-40GB), `scripts/meluxina/imp.sbatch`, one lane per GPU.
- **Smokes:** each lane first runs a 32-step smoke (GL_SMOKE=1); the job stops unless all four pass.
- **Duration:** v4 takes about 1.6 h; mcpair is expected around 1.8–2.2 h because reads double.
- **Wall limit:** 3 h 30 min, so at most 3.5 node-hours. Expected use is about 2.3 node-hours.
- **Resume:** `recovery.pt` per lane.

## Gates (pre-registered; `scripts/audit_v4imp.py` in the container)

Primary readout: SWA fused fold-0 predictions vs v4 l4copy of the same seed, 1,000 paired study bootstraps.

| Lane | Pass condition | Then |
|---|---|---|
| mcpair | Both seeds clear6 Δ ≥ +0.0015, mean Δ ≥ +0.0025, each P(Δ>0) ≥ 80%, and 12-target confident Δ ≥ −0.0020 | Submission pairing code (equivalence-tested against the training table), then LB |
| silw / hill | Fold 0 scores SIL cells as confident negatives, so it is biased against these lanes. Only a catastrophe floor: clear6 Δ ≥ −0.0050 | LB |

Reported for every lane:

- clear6, the mild band, per-target clear AUC;
- val BCE trajectory and train gap;
- silent lanes only: AUC split by fold-0 cell state (SIL vs non-SIL).

Reading the LB against v4's 0.939 (LB noise is about 0.003):

| LB | Reading | Next |
|---|---|---|
| ≥ 0.942 | Gain | Adopt the lane |
| 0.940–0.941 | Inconclusive | Combine with another lane, don't adopt alone |
| ≤ 0.939 | No gain | Drop the lane |

## Improve plan (whatever the outcome)

| Result | Next |
|---|---|
| silw or hill ≥ 0.942 | Silent handling is the lever. Dose-response (weight 0 / 0.1, Hill λ), per-target treatment, combine with mcpair. Then teacher-student (L5-TS) with the better loss, since the gate is met. |
| Both silent lanes ≤ 0.939 | Silent cells are not the bottleneck at v4's level. The label lever is then image-side soft targets (L5-TS), with the user deciding on the gate. |
| mcpair passes | Submission pairing; combine with the better silent lane; extend pairing to coronal through slice interpolation (only 28–42% of coronal rows are paired). |
| mcpair fails on both seeds | Cross-contrast fusion at window level is not the missing information. Log the per-target split (meniscus / contusion / fracture should move first); try late fusion (paired attention tokens) only if those targets moved. |
