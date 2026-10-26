# Raptor teacher dissection vs our E4 student (2026-10-21)

Scope: the public nartaa "Raptor" CoAtNet checkpoints (`raptor_ft_alldata_t16_blendjev_oai_d96_{r384,c08_r224}_swa.pt`,
LB 0.949 / 0.945; their 0.65 / 0.35 blend with mirror TTA = our teacher-ref submission 57421071, LB 0.949), compared with
our E4 / E4-ALL student (LB 0.946). Read-only analysis; no training, no submission.

Evidence tags: **[V]** verified in a file or the checkpoint; **[A]** stated by the author (forum / README), not checkable
(OAI part is private); **[I]** inferred by me.

**New primary source found:** nartaa published the exact training code on 2026-10-17 (Apache-2.0) in
`nartaa/rsna-knee-s6fall-weights/training_code/` (forum 749359 and 750180). The ledger's line "their training script is
not public" (Exp 1, `scripts/coatnet_d96_runner.py:9`) is out of date. Pulled to
`data/forum_research/teacher_dissect_1021/nartaa__rsna-knee-s6fall-weights/` (`train_knee.py` sha fffd79ab…,
`ext_trainer.py` a0899b97…, `RUNS.md` 174c567b…; the first two match the hashes in their RUNS.md). The full training
write-up is forum 746792, message 3537858 (`data/forum_research/sweep_1019/t_746792.json`).

## 0. Summary

- The Raptor architecture and its d96 input are already reproduced exactly in our code (parity |dp| 0.0003, LB 0.949). E4
  uses the same input, the same 94-window / 320-crop evaluation and the same head. Only the backbone differs.
- With the training code available, almost every recipe difference turns out to be something **we have already tested
  on r50**: their augmentation-free view (cv1, LB 0.945), pos_weight with their learning rates (P1-r949 / r949b, fold 0
  −0.014), backbone swaps (Exp 1b vs 1c, BB-CMP), labels (P1-e / PUB-KD), all-data training (E4-ALL, LB 0.946).
- Two differences remain untested:
  - **(a) direct OAI image-truth supervision.** 2,399 knees with hard MOAKS labels on PF OA, Lateral OA and Synovitis
    only. Their own ladder credits it with +0.005 LB, though that step also added 55% more optimizer updates.
  - **(b) teacher-size capacity under distillation.** CoAtNet-rmlp-2, 73 M parameters, ImageNet-12k pretrained.
- (a) cannot be done in 27 h. (b) can be probed cheaply for the accuracy pick, because the rap2-D student weights
  already exist.
- On gold-58, the only image-truth set, E4-ALL (0.927) and cv1 (0.928) already match or beat Raptor 384 (0.922). That
  Raptor number is optimistic, because its SWA epochs were chosen on gold-58. The 0.003 LB gap is about one LB SD.

## 1. Architecture

| Item | Raptor (nartaa) | Tag / source |
|---|---|---|
| Backbone | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (384 model), `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k` (224 model). timm weights pretrained on ImageNet-12k and fine-tuned on 1k, `num_classes=0`, global avg pool | [V] checkpoint keys `arch`, `src='timm-pretrained'`; `train_knee.py:400-408` |
| Size | 73,166,201 parameters in **both** models: the rel-pos **MLP** (`rmlp`) has no size-dependent tables, so the 384 weights run at 320 unchanged. Stages are 2 MBConv (128 ch) / 6 MBConv (256) / 14 transformer (512) / 2 transformer (1024), feature dimension 1024. 25 BatchNorm layers, all in the stem and the MBConv stages 0–1 | [V] `scratchpad` inspection of the state dict |
| "ft" | Not a fine-tune of dreaddevelopment's SSL Raptor: the default `--ckpt timm` keeps the timm ImageNet weights. `raptor_ft_<tag>` is just the trainer's output prefix | [V] `train_knee.py:474, 411-413, 884`; checkpoint `src` |
| Input | The d96 corpus: 96 uint8 slices at 384 px over 140 mm, in 5 slots: sag-fluid 26, sag-non-fluid 22, cor-fluid 18, cor-non-fluid 12, axial 18. Per-series 2/98 percentile window. Our cache is built from their geometry | [V] `scripts/fastmri_build_cache.py:42-43,134`; `train_knee.py:209`; RUNS.md "built with the Raptor corpus geometry" |
| Window | A 2.5D triplet: slices c−1, c, c+1 in the three RGB channels, centres anywhere in the study's valid range, slot boundaries ignored. ImageNet normalisation | [V] `train_knee.py:256-268, 343-345, 380-381`; argv `--norm imagenet` |
| "t16" / "d96" / "c08" / "r384" | 16 epochs / 96-slice corpus / centre crop 0.8 of the field before resizing / training resolution | [V] RUNS.md argv `--epochs 16`, `RAPTOR_CROP_FRAC=0.8`, `--res`; `train_knee.py:207, 315-324` |
| 384 vs 224 | **384:** trained on the full 384 px field (0.365 mm/px); scored on the native centre crop [32:352] = 320 px, no resize. **224:** 0.8 crop (307 px, 112 mm) resized to 224 (0.50 mm/px) in training and evaluation. Same architecture family and recipe | [V] RUNS.md; `pubkd_raptor_teacher.py:62-73`; forum 746792 |
| Aggregation | Attention MIL per finding. For each window: LayerNorm(1024) → Linear 256 → tanh → Dropout 0.2 → Linear 12. Softmax over the bag, separately for each of the 12 findings. Pooled feature · clsW[12, 1024] + clsb gives the logits. No position or slot embedding; one shared bag over all series | [V] `train_knee.py:427-465` (drop 0.2 at 429/433); `--pos_emb` not in argv |
| Inference | 94 evenly spaced windows in chunks of 16, fp16 autocast. Anatomical mirror TTA on the 384 model (reverse the slice channels on sagittal, flip width otherwise). The public notebook blends them as 0.65 percentile(384) + 0.35 percentile(224) | [V] `pubkd_raptor_teacher.py`; ledger SURPASS-R |

## 2. Training pipeline

| Item | Raptor | Tag / source |
|---|---|---|
| Data | All 4,349 report-labelled competition studies (no holdout = "alldata") + **2,399 OAI baseline knees** = 6,748 rows. Gold-58 never trained, but used to select epochs | [V] RUNS.md, `ext_trainer.py`; [A] 3537858 |
| RSNA labels "blendjev" | Cell-wise mean of `labels_v1_blend` and riadmohamed42's JEV HYBRID (blend kept where JEV is NaN). `labels_v1_blend` = mean of nartaa's Gemini-3-Flash report labels and Steven `llm_labels_v4_blend`; where v4 = 0.25 ("not addressed", ~19% of cells) only their own value is kept. JEV = pilkwang (ACL, MCL, fracture) + Steven v4 (rest) + JEV's text model. All soft, all 12 cells weight 1 | [V] `build_blendjev.py`; [A] 3537858 |
| OAI labels | Hard 0/1 from MOAKS semi-quantitative reads. PF OA / Lateral OA = full-thickness cartilage loss ≥ 1 in any PF / lateral TF subregion; Synovitis = Hoffa ≥ 1. All other findings NaN (masked; Effusion, Contusion, Fracture, MCL can never be enabled). Counts 2,396 / 2,397 / 1,659 | [V] `ext_trainer.py:55-56`; [A] 3537858 / 746792 c40 |
| Loss | BCE-with-logits on soft targets with `pos_weight = clip((1−p)/p, 1, 10)`, p = RSNA blendjev prevalence clipped to [0.03, 0.7] (pinned to the RSNA rows). Sum of weighted cell losses / (B × 12), masked cells included in the denominator, so OAI rows weigh 3/12 | [V] `train_knee.py:638-650, 759-787`; `ext_trainer.py:289` |
| Sampling | **12 windows per study, uniformly at random over the valid range** (shuffled pool, no slot stratification). Evaluation during training: 24 evenly spaced windows on gold-58 | [V] `train_knee.py:256-268`; argv `--k 12 --k_eval 24`, no `--clean_windows/--strat` |
| Augmentation | **None geometric. No mirror, no affine** (`--aug_affine` default 0, "NO flips (laterality is signal)"). Only a per-study intensity gain of ±10% with a clamp. Training field = full 384 | [V] `train_knee.py:374-379, 514` |
| Optimiser | AdamW, backbone LR 4.01e-5, head 1.95e-3, wd 0.0588. **OneCycle** (warm-up 15%, cosine to ~0, div 25 / final 1e4). Grad clip 3.0. bf16 autocast | [V] `train_knee.py:539, 744-750, 842, 564`; argv |
| Batch / steps | 1 study × 16 accumulation = 16 studies per step; 421 steps/epoch × 16 = 6,736 updates (4,336 without OAI). Backbone runs in checkpointed chunks of 4 windows, so CoAtNet BN batches are 4 windows of one study | [V] argv `--bs 1 --accum 16 --win_chunk 4`, `train_knee.py:444-454, 748` |
| SWA | Uniform average of the **top-3 epochs ranked by gold-58 macro AUC**. BN buffers averaged, not recomputed. 384: epochs 14 / 11 / 12; 224: 15 / 13 / 12 (0-indexed of 16) | [V] checkpoint `swa_over`; `train_knee.py:895-902, 980-1001` |
| Compute | ~22 h on an 8 GB laptop GPU for 16 epochs | [A] 3537858 |
| Gold-58 | 384 SWA 0.9190, 224 SWA 0.9147, both with gold-selected epochs. Our port without TTA: 0.9219 | [V] checkpoint `gold_auc`; ledger KD-CV gold row |

Their own LB ladder ([A] forum 746792; steps are confounded, as the author says): 80 → 96 slices + native 384 cache
**+0.001**; blend + JEV labels **+0.002**; 224/c08 crop + SWA **−0.002**; **+ OAI masked labels +0.005**; 384 full-field
recipe **+0.003**; 320 crop 0; mirror TTA **+0.001**. Their negatives: 94 → 24 windows −0.006; all-plane width flip −0.001;
crop 304 / 288 −0.001 / −0.002; epoch 14 alone vs SWA −0.001; fresh retrain with BN replay −0.004; native-320 crop
training −0.003. dreaddevelopment's original Raptor (44×336 corpus, SSL init) was 0.924.

## 3. Side by side with E4 / E4-ALL (`scripts/r50_gl4_runner.py`, env `pubkd_kd_e4.env`, review doc §3.1)

| Item | Raptor 384 | E4 / E4-ALL | Already tested by us? |
|---|---|---|---|
| Backbone | CoAtNet-rmlp-2, 73 M, IN-12k | ResNet-50 `tv_in1k`, 25 M + layer-4 copy for crops (fast kernel: global only) | CoAtNet-2 vs r50 on our labels: 1b 0.933 = 1c 0.933 (LB). CoAtNet-0 in v4: LB 0.933 vs 0.939. ConvNeXt-T IN-12k in v4: clear6 −0.0006 |
| Input / eval | d96, 384 train, 94 windows, 320 crop | identical | parity verified |
| Head | per-finding attention MIL | identical (+ crop branch, fused ½/½) | crop branch adds ≤ +0.001 under KD (KD-GAP S4) |
| Labels | blendjev soft + OAI hard (3 findings) | full public blend order + Raptor values, 0.7 / 0.3 with label-v2 | blendjev is inside the teacher. pure vs mix: LB equal (ra = rb 0.943); scale: E4 = B 0.946 |
| Training view | 12 uniform windows, full field, no geometry / mirror, gain ±10% | 16 stratified windows, mirror 0.5, rot ±10°, scale ±10%, shift ±6%, gamma, gain | **cv1** (48 windows, 320 crop, intensity only): LB 0.945. P1-r949 (no-aug r50, report labels): plateau / memorisation |
| Loss | BCE + pos_weight | plain soft BCE (+0.5 crop BCE) | P1-r949 / r949b: pos_weight + LR 4e-5 / 1.95e-3 → clear6 −0.014, graded findings −0.01 to −0.024 (fold 0 only) |
| Schedule | OneCycle, 16 epochs, batch 16, clip 3 | 1 warm-up + cosine to 1%, 25 epochs, batch 8, clip 1 | not isolated (bundled in P1-r949) |
| SWA | top-3 by gold-58, BN averaged | fixed 21–25, BN refreshed | 6a (cyclic SWA) null; gold-selection not tried |
| Data | 4,349 + 2,399 OAI | 3,524 (E4) / 4,349 (E4-ALL) + teacher KD; F1 adds 9.2k fastMRI with teacher labels | E4-ALL 0.946 = E4; F1 0.946 |
| TTA | mirror (+0.001 for them) | none (fast kernel) | 1e mirror TTA: LB tie |

## 4. Differences ranked by likely AUC impact, and what we could adopt (~27 h left)

1. **OAI image-truth supervision for PF OA / Lateral OA / Synovitis** (largest; their +0.005, confounded). This is
   information the student only gets second-hand, through teacher probabilities on our studies. KD-GAP S6: E4 closes 0%
   of the v4 → teacher gap on PF OA, and the public blend down-weights Raptor on PF OA / Lateral OA (0.6 / 0.2).
   - **Adopt in 27 h: no.** It needs an NDA DUC, ~140 GB of MRI, MOAKS-to-series linkage (3DReasonKnee) and the team's
     external-data decision (final_days_plan §6). Recording it as the main reason the teacher is hard to beat.
   - Evidence: medium. A first-party ablation, but bundled with more updates.
2. **Teacher-size student under distillation.** The backbone made no difference on report labels. But under KD the
   student underfits its own targets (KD-GAP S1: train KL = fold-0 KL). cv1 removed the view-mismatch part of that floor
   and the LB did not move. So capacity and representation are the main untested explanation of the remaining
   0.003 [I].
   - **Adopt for the accuracy pick only: LB-probe rap2-D.** This is the Raptor 384 init fine-tuned on the full-blend
     mix73 targets for 6 epochs, SWA 4–6. The weights already exist at
     `data/meluxina_mirror/home_rsna/out/pubkd_kd_d/weights_swa4_6.pt`; fold 0 is in-sample, so only the LB can judge it.
   - Cost: a global-only kernel built from `kaggle/submission_kernels/raptor_teacher_ref_submission/` plus one
     submission. Score at a 320 crop or at 384; their ladder says it makes no difference.
   - Adopt if it beats 0.949 + 0.003 (SURPASS rule).
   - Evidence: weak to medium. rap2 trained on the near-hard full targets; its fidelity to them is 0.933, below Raptor's
     own 0.947.
   - Not for the efficiency entry (73 M CoAtNet at 94 windows).
3. **Gold-ranked SWA epoch selection with BN buffers averaged** (small; their SWA vs best epoch +0.001).
   - Cheap only on a rerun that keeps every epoch's weights. For example E4-ALL with all epochs saved, ~1 h on an A100,
     then pick the top-3 by gold-58 on the global head.
   - **Not recommended:** gold-58 moved +0.009 for cv1 and the LB did not follow (n = 58, 9–35 positives per finding).
     Selecting on it is selecting on noise.
   - Evidence: weak.
4. **Training view without geometric augmentation.** Already tested as cv1 (LB 0.945 vs 0.946). Closed. Their ±10% gain
   and no-flip policy is safe for their CoAtNet with 6.7k rows. On r50 with report labels it memorised (P1-r949).
5. **pos_weight BCE + their learning rates + OneCycle.** Already tested on r50 (P1-r949b: fold 0 −0.014, never
   LB-probed). With KD soft targets, pos_weight would also skew the teacher's calibrated values. Not recommended
   (ruled-out list).
6. **The 224 / c08 sibling and mirror TTA.**
   - Both are within noise on the LB: the 384 + 224 blend with mirror = 0.949 = 384 alone; mirror on 1e = tie.
   - Learning for efficiency: in their lineage the 384 → 224 resolution drop costs about 0.004. At ~69 s per 0.001 AUC,
     **do not trade resolution or windows for speed** (also their 94 → 24 windows −0.006). The fast kernel already keeps
     94 windows at 320.
7. **Labels (blendjev +0.002 in their ladder).** Already inside our targets through the teacher. Label-source swaps are
   closed (public-label-sources, PUB-KD-E).

## 5. Side notes

- The same author's **S6 ConvNeXt-S scores 0.950 all-data without OAI** (forum 749359; fold-0-only model 0.948).
  - It is Heliosli's pipeline: 64 windows per study, full field plus 100 mm local crops placed by SKM-TEA anatomy
    anchors, 320 → 288 px, blendjev labels, 16 epochs, no SWA. Its training code is in the same dataset under
    `training_code/s6/`.
  - So a ~0.95 single is reachable without OAI using local crops on the right anatomy. That is the direction our r50-GL
    family already took (E4 = global + MS1 crops).
  - The difference is anatomy-anchored crop placement plus 64 windows. We have not pulled `s6/train_s6f0_all.py`; it is
    worth a read before any further crop work.
- Leakage in the teacher's own numbers: gold-58 picked the SWA epochs, and JEV was tuned on gold. So the teacher's
  gold-58 (0.919–0.922) is optimistic, and our students matching it on gold is a stronger result than it looks.
- KD-W48 (16 → 48 windows on E4) was still running at the time of writing; it is not covered here.

## 6. Files

- Pulled sources (untrusted data, not executed): `data/forum_research/teacher_dissect_1021/nartaa__rsna-knee-s6fall-weights/`
  (`README.md`, `training_code/{README.md,SHA256SUMS,coatnet/{train_knee.py,ext_trainer.py,RUNS.md},labels/build_blendjev.py}`).
- Checkpoints inspected on CPU (weights_only): `data/public_ckpt/nartaa_raptor/*.pt`.
- Forum: `data/forum_research/sweep_1019/{t,m}_746792.json` (write-up + training details msg 3537858),
  `t_750180.json`, `t_749359.json`.
- Ours: `scripts/pubkd_raptor_teacher.py`, `scripts/coatnet_d96_runner.py` (Raptor class, `centres`, `prepare`,
  P1-r949 configs at 216-234), `scripts/r50_gl4_runner.py`, `scripts/r50_gl_runner.py`, `scripts/fastmri_build_cache.py`,
  `docs/pubkd_full_recipe_review.md` §3, `docs/experiment_ledger.md` (Exp 1/1b/1c, P1-r949/b, BB-CMP, PUB-KD*, SURPASS-*,
  KD-GAP/VIEW/CV, E4-ALL, FAST-G).
