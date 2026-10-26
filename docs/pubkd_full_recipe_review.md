# PUB-KD full-target student: recipe for review (2026-10-20)

**What we want from you.** A review of the training recipe below **before** we spend a MeluXina node on it. Nothing has been launched. The questions are in section 6. The most important one is Q1, about how the target values are built: the full-blend targets turned out much sharper than the Raptor-only targets that gave our 0.943.

**Deadline.** Competition ends Thu 10-23 01:59 CEST. We have about 2.5 days, about 100 node-hours, and 5 Kaggle submissions per day.

---

## 1. Where we are (public LB)

| Model | What it is | LB |
|---|---|---|
| v4 l4copy r50 | Our best recipe on our own labels (label-v2), fold 0 held out | 0.939 |
| **PUB-KD-R ra** | Same recipe and seed, trained on **Raptor-only teacher** targets (pure) | **0.943** |
| PUB-KD-R rb | Same, targets 0.7 teacher + 0.3 label-v2 (mix73) | 0.943 |
| E3 | Our best ensemble (4 members) | 0.944 |
| Raptor teacher reference | nartaa Raptor 384 + 224, mirror TTA, rank blend 0.65 / 0.35 (the ra/rb teacher, re-run by us) | **0.949** (submission 57421071; matches the public number) |
| Public 0.958 blend | goodpjw2008 v25/27: Raptor 384 + 224, R6, 2.5D reader | 0.958 |

Changing only the training targets moved the LB +0.004 (40% of the v4 → teacher gap; the students sit 0.006 below their 0.949 teacher), while fold-0 AUC on label-v2 did not move. The fold-0 ruler cannot see this gain. Rank-based LB prediction (½ clear6 + ½ rapc_12, paired against v4) does track it (ra predicted 0.9432, actual 0.943).

**Diagnosis of ra** (`data/coatnet_d96_fold0/pubkd/students_r/diag_pubkd_r.txt`, `docs/pipeline_evolution_study.md`):
- **Fit is not the limit.** Train BCE is 0.549 against the targets' entropy floor of 0.538. v4's excess over its own floor is 0.107.
- **The student copies the teacher about as closely as the teacher's own two members agree.** Rank distance student~teacher is 0.060; Raptor 384~224 is 0.059.
- **The gain comes from the teacher's label convention.** It compresses severity, puts report-silent cells above report-absent ones, and ranks mild/uncertain positives higher. The gain is in the mild band: MCL +0.021, Lateral OA +0.007, Effusion +0.006, Baker's +0.006.
- **The crop branch adds about nothing under KD.** Fused − global is about +0.0007, vs +0.0022 for v4.
- **The full blend is a materially different target from Raptor alone.** Mean rank move is 0.064, about as far as ra moved from v4.

**xAI, ra vs v4 vs teacher on fold 0** (`students_r/xai_ra_vs_v4.txt`; descriptive only, teacher in-sample):
- **Attention moved toward the teacher's split across series.** Distance from the teacher: v4 0.315 → ra 0.217 (rb 0.257). Attention is flatter: entropy 0.785 → 0.837.
- **The teacher reads the axial series much more than the students do.** Baker's: teacher 37% axial vs ra 3.5% and v4 1%. Synovitis 27% vs 10%; Effusion 24% vs 13%; MCL 21% vs 11%; PF OA 18% vs 8%. This is a candidate for the remaining 0.006 gap to the teacher, not a fix: plane-routing ideas were null before.
- **Students depend less on any one series.** Dropping the sagittal fat-sat series costs ra −0.015 vs v4 −0.036. Using the axial series alone costs ra −0.166 vs v4 −0.369.
- **The crop branch got better but redundant.** Crop-only AUC is +0.023 vs v4, but global~crop correlation rose 0.79 → 0.89, so fused ≈ global. MS1 crop placements help v4 (+0.005) but not ra (−0.001).

---

## 2. Teacher and targets

### 2.1 Members

All members were re-run by us on all 4,407 training studies.

| Member | Source | TTA | Run on |
|---|---|---|---|
| Raptor 384 | nartaa CoAtNet-rmlp-2, all-data SWA, sha 7e5315da… | mirror | MeluXina, our d96 cache |
| Raptor 224 | same lineage at 224, sha 3394fdd7… | mirror | MeluXina, our d96 cache |
| R6 | rostigor | flip | Kaggle T4, from DICOMs, 3 shards (one on the teammate account) |
| 2.5D reader | goodpjw, 3 folds | — | Kaggle T4, from DICOMs |

**All members are in-sample on our training studies.** The public models were trained on all competition data, and some on OAI as well. Using public teachers this way is an explicit exception to our out-of-fold label rule. Validation labels are unchanged.

### 2.2 Blend

This is their exact blend, per finding, with ranks taken over all studies (`scripts/build_pubkd_targets.py`):

```
raptor = 0.65·pct(r384) + 0.35·pct(r224)
stack  = w·raptor + (1−w)·pct(R6)          w = 0.5, except Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35
final  = pct((1−v)·pct(stack) + v·pct(reader))   v = 0.003–0.035 per finding
```

### 2.3 Turning ranks into target values

A rank is not a probability. Each finding's target therefore keeps the blend's **order** exactly. Its **values** come from the same-weight blend of member **logits**: the k-th lowest final rank gets the k-th lowest logit-blend probability.

The pure and mix73 files differ only in how much of our own labels is mixed in:

| File | Contents | sha256 |
|---|---|---|
| `pubkd_targets_pure.json` | teacher only | `e29eeffd…fa86a` |
| `pubkd_targets_mix73.json` | 0.7 teacher + 0.3 label-v2 (every training cell has label-v2 weight > 0, so the mix applies everywhere) | `1aca523c…099f0` |

All 12 cells carry weight 1.

### 2.4 QC (in-sample, sanity only)

| | r384 | r224 | R6 | reader | **full target** |
|---|---|---|---|---|---|
| gold-58 macro AUC | 0.923 | 0.916 | 0.937 | 0.915 | **0.938** |
| fold-0 label-v2 confident 12t | 0.969 | 0.973 | 0.975 | 0.973 | **0.979** |

- R6 and the reader agree with r384 at rank correlation 0.83 and 0.88.
- The full target agrees with the Raptor-only target at per-finding Spearman 0.89 (Baker's) to 0.995 (Synovitis).

### 2.5 ⚠ Calibration finding (new today; this is Q1)

Measured over the 3,524 training studies:

| | mean p | mean entropy | cells < 0.05 or > 0.95 | mean \|logit\| |
|---|---|---|---|---|
| label-v2 (our labels) | 0.261 | 0.260 | 54% | 3.1 |
| Raptor 384 / 224 | 0.45 | 0.54 | 5% | 1.2 |
| **R6** | 0.263 | **0.070** | **88%** | **8.3** |
| reader | 0.306 | 0.392 | 15% | 1.9 |
| Raptor-only target (ra/rb, LB 0.943) | 0.450 | 0.538 | 5% | 1.15 |
| **Full target (pure)** | 0.279 | **0.155** | **73%** | 4.6 |

R6 is close to memorised on its own training studies. Because target **values** come from a logit blend, R6's large logits dominate them. The full target is therefore near-hard, sharper even than label-v2. The blend's **order** is preserved, but with values near 0/1 the BCE barely rewards ordering inside the confident bands. The soft, convention-carrying values that ra learned from would mostly be gone.

Raptor's own probabilities are soft and badly calibrated: mean 0.45 against a 0.26 prevalence. The ra result says this did not hurt AUC, and may have helped.

So the planned run would change **two** things relative to ra: the order (intended) and the value scale (not intended).

Candidate value mappings (Q1):
1. **As built.** Logit-blend values; R6-dominated and near-hard.
2. **Full-blend order + ra's value distribution.** Per finding, quantile-map the full blend's order onto the sorted values of the Raptor-only target. This is a one-variable change from ra (only the order changes) and keeps the softness that worked. *(My recommendation.)*
3. **Standardised logit blend.** z-score each member's logits per finding before blending, so no member dominates by scale.
4. **Temperature.** Soften option 1 with a per-finding temperature to match ra's entropy.

Each is a CPU rebuild of a few minutes. Options 2–4 would get new hashes and a ledger note before launch.

---

## 3. Student recipe

### 3.1 Lanes A/B: r50 v4 l4copy

This is the recipe of ra/rb unchanged, only `V4_TARGETS` differs.

| Item | Setting |
|---|---|
| Runner | `scripts/r50_gl4_runner.py` (`V4_ARM=l4copy`, `D96_PHASE=r50_gl`), base config `scripts/coatnet_d96_runner.py`, `scripts/r50_gl_runner.py` |
| Data | d96 cache: 5 slots (SagF, CorF, Ax, SagNF, CorNF), up to 96 slices, 384 px. Train folds 1–5 = 3,524 studies; fold 0 (825) held out |
| Backbone | ResNet-50 `tv_in1k` |
| Global path | 16 random windows per study (training) → window-attention head per finding (LayerNorm → attention → linear) |
| Crop path | 10 findings (all but Effusion and Synovitis). Per finding, one of MS1's top-4 attended windows, ±16 px jitter, box crop resized to 384. Shared stem–layer3 with crop BN statistics, then the crop branch's **own copy of layer4** (ImageNet init, backbone LR) and its own head |
| Loss | `BCE(g, y) + 0.5·BCE(l, y)` on the 10 cropped findings. Soft-target BCE (= KL to the teacher + constant), no temperature, cell weights 1 |
| Optimiser | AdamW, LR 1e-4 backbone + layer4 copy, 1e-3 heads, weight decay 0.05, 1 warm-up epoch, cosine to 0.01×, grad clip 1.0, bf16 autocast |
| Batch | 2 studies × 4 accumulation = 8 studies/step |
| Augmentation | mirror 0.5, rotation ±10°, scale ±10%, shift ±6%, gamma 0.25, gain 0.10; one draw shared by both views |
| Schedule | 25 epochs, SWA of epochs 21–25, BN refresh (global stats on 16 eval windows; crop stats on all 4 crops) |
| Readout | fused = sigmoid((g + l)/2) on the 10 cropped findings, g on Effusion / Synovitis |
| Evaluation | 94 windows, 320 centre crop. Fold 0 uses MS1-placed crops; the **submission self-locates crops** (top-4 windows by the global attention; no MS1 at inference) |
| Lanes | A = pure targets, B = mix73; seed 2026 |
| Cost | ra–rd (4 lanes on one 4×A100 node) took 105 min |

### 3.2 Lanes C/D: rap2 (Raptor CoAtNet init)

Same pipeline with these changes:
- `V4_BB=rap2`: backbone + global head initialised from the Raptor 384 checkpoint.
- Crop branch = copy of its `stages.3` + norm, crop head = copy of its head.
- LR 5e-5 / 5e-4 (warm-up 1 epoch, cosine to 0.01×).
- 6 epochs, SWA 4–6.
- 1 study × 8 accumulation, gradient checkpointing.
- Globals trained and evaluated at 384 (no 320 crop).
- C = pure, D = mix73.

The init saw fold 0 (all-data checkpoint), so **fold 0 is in-sample for C/D: LB only**. The bar is the teacher reference. Wall time at full scale has not been measured; each lane runs its 32-step smoke first.

### 3.3 Job

`scripts/meluxina/pubkd.sbatch kd_a kd_b kd_c kd_d`: one node, one lane per GPU, 9 h limit.
- Inputs are hash-checked: assets, Raptor init, target files.
- Each lane runs a 32-step smoke first; a lane that fails its smoke is dropped.
- CPU tests: `scripts/test_pubkd_cpu.py`.

---

## 4. Evaluation and submission rules (pre-registered)

- **Fold 0** (r50 lanes only): clear6 / 12t vs label-v2 is descriptive only. Fold 0 is known to under-rate teacher-label training (ra: fold-0 Δ −0.002, LB +0.004).
- **Fidelity** (descriptive): Spearman vs the teacher on fold 0.
- **Submission ranking for r50:** paired LB predictor vs v4 on the **global** file (`scripts/lb_predictor.py predict … --parent … --parent-lb 0.939`). Submit if predicted ≥ ra's − 0.001.
- **Submission for rap2:** the LB only. "Surpasses the teacher" = LB ≥ teacher reference + 0.003. Within ±0.002 = tie.
- Every submission is asked for first. Two are kept for final selection.

---

## 5. Known risks and caveats

1. **Value scale** (section 2.5): the full targets are near-hard.
2. **In-sample teachers.** On our training studies the members partly reproduce their own training labels (R6 most of all). Their "knowledge" (OAI, other labels) transfers only through these predictions.
3. **Reader decode gap.** The reader's JPEG/gdcm decoders did not install on Kaggle's Python 3.13, so any JPEG-compressed series is skipped silently. The reader's weight is 0.3–3.5% and its gold-58 AUC (0.915) looks normal, so this is judged minor. R6 is unaffected.
4. **The 0.958 is partly ensemble diversity**, which one student cannot fully absorb.
5. **rap2 has no validation signal** and costs 1–2 submissions per variant. CoAtNet-2 at 384 × 94 windows is also slower at inference (Kaggle runtime not yet measured for rap2).
6. **Crop branch.** It adds about 0 under KD (section 1), but removing it is untested at full scale. L4C-B showed crop co-training helped the global branch on label-v2.

---

## 6. Questions for the reviewer

**Q1. Target values (section 2.5).** As built (R6-dominated, near-hard), or option 2 (full-blend order + ra's value distribution), 3 (standardised logits) or 4 (temperature)?
- My view: option 2 for the main lanes. It isolates the order change, and ra's soft values are the only thing we know works on the LB.

**Q2. pure vs mix73.** ra = rb on the LB (0.943 both).
- Keep both lanes, or replace mix73 with a different lane (e.g. option-1 values as a contrast, or r50 global-only)?

**Q3. Crop branch.** Keep l4copy for the full-target student, or add a global-only lane (no MS1 at train time, half the crop compute) to decide the all-data final?

**Q4. rap2 settings.** LR 5e-5 / 5e-4, 6 epochs, SWA 4–6, warm-up 1 epoch, no layer decay, the same augmentation as r50. Reasonable for fine-tuning an already-converged all-data checkpoint on soft targets, or should the LR be lower / the epochs fewer?

**Q5. Loss.** Plain soft BCE with no temperature, and local loss weight 0.5 as in v4.
- Any reason to prefer a rank / pairwise term, given that the metric is AUC and the targets are rank-defined?

**Q6. All-data final.** Plan: retrain the winning recipe on folds 0–5 (4,349 studies; teacher targets exist for all) as the last submission, with no validation.
- Acceptable risk?

**Q7. Anything in the teacher reconstruction that looks off?**
- Raptor parity on Kaggle vs MeluXina is 0.0003.
- R6 / reader shards: 4,407/4,407 studies, no fallbacks, consistent across shards.

---

## 7. Files

All code is in git on branch `d96-handoff-2026-10-12`. Paths under `data/` are gitignored and stay local; the Raptor teacher predictions are in the Kaggle dataset below.

| What | Path |
|---|---|
| Target builder (+ QC log) | `scripts/build_pubkd_targets.py`, `data/coatnet_d96_fold0/pubkd/build_pubkd_targets.txt` |
| Student runner | `scripts/r50_gl4_runner.py` (+ `r50_gl_runner.py`, `coatnet_d96_runner.py`) |
| Lane configs | `scripts/meluxina/env/pubkd_kd_{a,b,c,d}.env`, smoke `pubkd_smoke_kd_*.env`; job `scripts/meluxina/pubkd.sbatch` |
| CPU tests | `scripts/test_pubkd_cpu.py` |
| Submission kernel (ra) | `kaggle/submission_kernels/pubkd_r_ra_submission/submission.py` |
| ra/rb diagnosis | `data/coatnet_d96_fold0/pubkd/students_r/{audit_pubkd_r,diag_pubkd_r,findings_by_band_state}.txt` |
| Pipeline study | `docs/pipeline_evolution_study.md` |
| LB predictor | `scripts/lb_predictor.py`, `data/lb_predictor/report.txt` |
| Ledger rows | `docs/experiment_ledger.md`: PUB-KD, PUB-KD-R, SURPASS-R, LB-PRED |
| Teacher predictions (already shared) | Kaggle private dataset `tumis001/pubkd-raptor-teacher-preds` |
