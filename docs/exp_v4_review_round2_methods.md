# r50-GL v4 rev 2: methods review, round 2 (2026-10-18)

Reviewed: `docs/exp_v4_design_rev2.md` (below "D:<line>"). Same role as round 1: adversarial methods review. Implementation
and compute are out of scope. New CPU checks for this round are read-only scripts in the session scratchpad
(`anchor_check.py`, `corr_check.py`). They use the existing fold-0 prediction files, the label-v2 confident cells and
2,000 paired study bootstraps (seed 2026), and they reproduce the anchor at 0.9469 (audit 0.9468).

**Verdict: proceed with changes.** Rev 2 is a much better design than rev 1: the anchor is real, it targets fusion,
and its decision rule is mostly fixed. But my new checks contradict two of its three premises.

- **Shared features do not correlate the heads' errors.** Within-run g–l correlation equals cross-run correlation.
- **The detached-LoRA local is the riskiest way to get a v2-quality local.** Those features are where MP-0 found zero
  complementarity.

The OOF batch serves only the residual readout. The evidence says a learned combiner does not beat the fixed mean.
**P(clear 0.9455): ~20% as written; ~30–35% with the changes below.**

---

## New evidence (CPU, fold 0, label-v2 confident cells, 12-target)

### E1. About half of the anchor is generic two-model ensembling

| Fixed 50/50 probability mean on the 10 cropped findings | 12-target | − lv2 r50 [95% CI], share > 0 |
|---|---|---|
| lv2 + v2 **local** (the anchor) | **0.9469** | **+0.0044** [+0.0016, +0.0075], 100% |
| lv2 + v2 **global** (a second lv2-recipe r50 with no local view) | 0.9445 | +0.0020 [−0.0003, +0.0044], 95% |
| lv2 + v2 fused | 0.9446 | +0.0021 |
| lv2 + v3 local (the memorised local, 0.9198 alone) | 0.9455 | +0.0030 [−0.0006, +0.0070], 94% |
| lv2 + P3v2 student | 0.9447 | +0.0023 [+0.0004, +0.0043], 99% |
| lv2 + 1e s2027 (other labels) | 0.9429 | +0.0004 |
| v2 global + v2 local (within one run) | 0.9453 | +0.0028 [−0.0019, +0.0076], 88% |
| anchor − (lv2 + v2 global) | | **+0.0024** [−0.0006, +0.0058], 93% |

Pairing lv2 with a second global r50 gives +0.0020. The part specific to the local view is +0.0024, and its CI
includes 0.

Per finding, the anchor's gain sits in MCL (+0.012 on 16 positives, i.e. +0.001 on the mean), contusion (+0.012),
lateral OA (+0.010), lateral meniscus (+0.0075) and Baker's (+0.007). PF OA loses −0.009.

### E2. Within-run pairing does not lose decorrelation

Mean over the 8 region findings, confident cells:

| Pair | Logit correlation | Within-class correlation |
|---|---|---|
| lv2 – v2 local (cross-run) | 0.868 | 0.772 |
| v2 global – v2 local (same run) | 0.875 | 0.783 |
| lv2 – v3 local | 0.805 | 0.671 |
| v3 global – v3 local | 0.806 | 0.677 |
| lv2 – v2 global (two globals) | 0.960 | 0.928 |

Two conclusions follow.

- **D:17's "third mechanism" (shared features correlate the errors) is refuted.** The local *view* decorrelates the
  heads; the run does not matter.
- **The anchor minus within-run gap is entirely global quality.** 0.9469 − 0.9453 = 0.0016 ≈ lv2 − v2 global
  0.0018.

That global gap (−0.0018 in v2, −0.0027 [−0.0069, +0.0014] in v3) is about one seed spread: 1e's two seeds differ
by 0.0016 on 12 targets (ledger l.207). lv2 is also the *selected* best single, so some regression to the mean is
expected for any fresh draw. In addition, r50-GL v1's epoch-5/10 globals sat inside 1e's seed range ("crops do not
hurt the backbone", ledger l.276).

**So "local training damages the global" is plausible but not established.** Rev 2's central intervention, detaching
the local from the shared weights, buys at most about +0.002 of global. It risks the local quality that delivers
+0.0024 to +0.0044.

### E3. Training-study OOF predictions: none exist for r50, but ConvNeXt OOF does

- **r50:** none. Every r50 `oof_predictions.json` under `data/*/results/` holds 825 rows, i.e. fold-0 validation
  predictions, not training-study OOF. `kaggle_fold0_labelv2/` holds only `val_swa.json`.
- **ConvNeXt:** `run_cnxa_teacher{0..3}/oof_swa.json` are **label-v2 CK-A ConvNeXt-T** predictions. They are
  cross-fitted on `teacher_splits.csv` (sha 7cb597a7, report-duplicate groups kept together), and cover
  882 + 880 + 886 + 876 = all 3,524 training studies exactly once, with fold 0 unseen (ledger l.258).
- **Fit as an offset for r50:** CK-A vs lv2 within-class correlation is 0.864 (r50 cross-run 0.928). It is a usable
  stand-in, mis-targeted by about the cross-architecture gap. It can replace Batch A only for an OOF *variant*, not
  for an r50-matched primary (see F3).

---

## Findings, by impact

### F1 (critical). The detached shared weights + LoRA local is the weakest route to v2-local quality (question 1)

- **What detachment removes.** The anchor's local (v2: 0.9289 12-target; 10 cropped 0.9232) came from a backbone
  *co-trained on zoomed crops* (local BCE ×0.5 into shared weights). Rev 2 removes exactly that.
- **The frozen end of the spectrum is measured, and it is bad.** MP-0 used frozen lv2 r50 features on MS1-placed
  zooms, PCA-256 logistic probes. Per finding, s_src vs v2 local:

  | Finding | Frozen probe | v2 local |
  |---|---|---|
  | ACL | 0.919 | 0.964 |
  | MCL | 0.753 | 0.930 |
  | Medial meniscus | 0.874 | 0.958 |
  | Lateral meniscus | 0.864 | 0.930 |
  | Medial OA | 0.881 | 0.957 |
  | Lateral OA | 0.801 | 0.902 |
  | Contusion | 0.843 | 0.906 |

  More importantly, **B_src − A = −0.0023** (all 10: −0.0028 [−0.0059, −0.0001]). Zoom features from the
  global-trained encoder carry *nothing the global logit lacks*.
- **What the adapters must close.** Part of the ~0.07 per-finding gap is top-1 vs 4-crop (−0.021, v2) and global vs
  crop BN (−0.008, v3 local_gbn). That leaves roughly 0.03–0.04 for rank-32 adapters.
  - The adapters are smaller than D:29 states. Rank 32 on the layer3/4 1×1 convs is 32·(c_in + c_out) per conv,
    about 1 M parameters in total, not 3–4 M (the implementation reviewer should confirm).
- **My expectation: local 0.915–0.925 and *more* correlated with g than v2's.** It is built on the global's features,
  and decorrelation is the thing E1/E2 show produces the gain. That is the expensive failure mode: local looks
  healthy alone, but the fused gain is about +0.001–0.002.
- **The design has a leak path.** In v3, dual BN shares the affine (γ, β) between views. If rev 2 keeps the shared
  affine and only detaches the conv weights, the local loss still moves the global path's 53 BN affines. The local BN
  needs **its own affine**, initialised from the shared one. That is also cheap extra capacity.
- **"No frozen backbone".** From the local view's side, a detached backbone *is* a frozen backbone that drifts. The
  MS1-CP and MP-0 lessons ("the zoom gain needs the encoder trained on zoomed inputs") apply directly.

**Capacity I would use instead, all one network:**

1. **(Recommended primary) v2's sharing, exactly.** Local BCE ×0.5 into the shared weights, global BCE ×1, dual BN
   with separate local affine, and no reader or fused gradient into anything.
   - This is the configuration that produced the anchor's local.
   - Its expected cost to the global (≤ 0.002, E2) is about what detaching could win back.
   - If the global damage is a concern, scale the local gradient into the shared conv weights by λ = 0.25 (a gradient
     multiplier, not a detach). The backbone still sees crops.
2. **(Run as the comparison arm, not the primary) Branch-specific top stage.** Stem–layer3 shared and trained by both
   losses; the crops get their **own copy of layer4** (ImageNet init, local loss only); the global's layer4 sees only
   the global loss.
   - This is the standard single-network multi-branch pattern: Multiple Granularity Network for person re-ID
     (Wang et al., 2018) branches a ResNet-50 after the shared trunk with branch-specific conv5 stages.
   - Honest caveat: layer4 is ~15 M of r50's ~23.5 M parameters, so by parameter count it sits close to the user's
     "no two backbones" line. The user decides.
3. Avoid "own layer3–4": that is two backbones in all but name.

### F2 (high). Rev 2's own premises 1 and 3 should be rewritten (D:14-17)

Premise 1, "global loses 0.002–0.003", is n = 2 draws vs a selected best, within one seed spread (E2). Premise 3,
"shared features correlate errors", is refuted (E2).

The design should say: "the within-run gap is global quality, possibly seed or selection, and is measured by lv2
s2027 and the 4 inner-fold runs". This matters because it changes which intervention the run should test.

### F3 (high). The OOF-residual objective is sound in principle but has the wrong place in the design (question 2)

**Soundness.** BCE(σ(g_oof + r), y) is stacked generalisation with an offset. It is the right fix for the
"in-sample residual is empty" failure (v2's reader captured +0.0009 of the +0.0034 training complementarity), and it
satisfies the out-of-fold label rule. But:

- **(a) The evidence says a learned combiner does not beat the fixed mean here.**
  - Fold-0 cross-fitted stacking with 2 weights per finding "never beats the plain mean" (ledger l.276).
  - r50-GL3-XAI: the per-study gate equals the constant blend (0.9414 vs 0.9415).
  - g + r is a far higher-capacity learned combiner (r is a full local path), fit on 3,524 studies.
  - Prior: g + r ≤ mean(g, l), with more variance.
- **(b) r is itself fit in-sample.** OOF protects the *global* side only. The residuals of g_oof on training studies
  concentrate on hard cases and **label noise**: mislabelled cells are exactly where an honest g_oof disagrees with y.
  Residual fitting is the boosting regime known to chase label noise. Regularise r:
  - a bounded residual, r = c·tanh(r̃/c) with c ≈ 2;
  - an L2 penalty on r;
  - use the label-v2 confidence weights, which the BCE already takes.
- **(c) Specify r vs l.** D:38-39 does not say whether r and l are two linear heads on the same local feature or one
  logit with two losses. One logit is self-contradictory (a residual target and a standalone target). Use two heads.
- **(d) Train/test mismatch.** g_oof comes from models on about 2,640 studies (75% of 3,524), while the test-time g
  uses all 3,524. g_oof's errors are larger, so r over-corrects; the bias is small but systematic.
  - A second mismatch is calibration. Each training study's offset comes from one of 4 different models, while the
    test offset comes from a fifth. Standardise each inner model's logits per finding on its *own* OOF split, match
    their SD to each other, and apply the same transform family to g. No fold-0 fitting.
  - Do **not** mix in the in-sample g. It is over-confident, so it pushes r toward v2's dormancy; its bias goes the
    opposite way only by accident, and the mix weight would be one more untestable knob.
  - Pre-register r × 1. Report r × {0.5, 0.75} as descriptive only.
- **(e) Inner splits.** Use `teacher_splits.csv` (duplicate groups kept together; ledger l.232 2-dup), not "3 of
  folds 1–4" (D:35). Report-duplicate studies split across inner folds make g_oof optimistic on exactly those studies.

**Placement.** Make the **fixed mean(g, l) the PRIMARY readout** (the anchor's form, nothing learned) and **g + r a
pre-registered secondary**. Then **Batch A is off the critical path.**

- Either drop it and start Batch B now (saves an allocation and about 3 h of a 4-day schedule), or keep it
  for its byproduct (F5).
- If the user wants OOF tested without Batch A, run it with the CK-A OOF offsets (E3), labelled as a
  cross-architecture offset.

### F4 (high). Is 0.9468 a fair target for one network? No. The fair reference is 0.9453 (question 3)

- **The anchor pairs a selected-best global with a local from another run.** About half of its gain (+0.0020) is
  available from *any* second lv2-recipe model (E1). A single network gets the decorrelation (E2) but not the second
  global, and its global is a fresh, unselected draw.
- **Fair single-network reference: v2's within-run mean(g, l), 0.9453.** That is 0.0002 *below* the bar. Expected
  fused ≈ E[global] + ~0.0045.
  - With E[global] ≈ 0.9415 (lv2 recipe, adjusted for selection), the expectation is ≈ 0.946.
  - That requires a v2-quality, v2-decorrelated local.
- **The bar (0.9455 = lv2 + 0.003) is the right decision bar.** It is what the submission competes with. Keep
  0.9468 as a motivating reference, not a target; D:8-11 should say so.

### F5 (medium). Decision rule and early stop (question 4)

- **Primary readout.** Pre-register *one* (F3: mean(g, l)). Reporting fused and mean(g, l) and passing on either is
  selection.
- **Bootstrap.** "≥ 90% of 2,000 paired bootstraps (per seed, both reported)" (D:57-58) is ambiguous. Specify: resample
  studies, compute each seed's AUC on the same resample, average over seeds, and require ≥ 90% of resamples with
  mean − 0.9425 > 0 (and point ≥ +0.003). Per-seed shares are descriptive.
- **Protection check.** "|Δ| ≤ seed noise" needs a number: global ≥ 0.9425 − 0.002, or within the range of lv2
  s2027 and the inner-fold runs.
- **Early stop, "global < label-v2 r50's own epoch 10–12 mean by > 0.004": not computable.** label-v2 r50 has no
  per-epoch fold-0 history in the repo; `kaggle_fold0_labelv2/` contains only `val_swa.json`. Use one of:
  - v2's per-epoch global (`run_r50_gl_v2/history.json`, scored every epoch);
  - lv2 s2027 run in the same batch, made to score fold 0 at epochs 10–12.
- **Early stop, "fused − global < 0": too lenient.**
  - v2's mean(g, l) − g was +0.011 at epoch 10, +0.006 at 16 and +0.0045 at SWA (ledger l.276). The gain halves by
    the end.
  - A run below about +0.005 at epochs 10–12 is unlikely to finish ≥ +0.003 over a global that is itself around the
    0.9425 line.
  - Suggested rule: stop if the epoch 10–12 mean of mean(g, l) − g < +0.005, or local train − val AUC gap > 0.06.
- **Batch B's 4th slot (v4-noOOF) has the lowest decision value.** It attributes a secondary readout. Use it for the
  capacity question that decides this design (F1): **v4-shared** (v2 sharing, λ = 0.5 or 0.25) vs **v4-detach**,
  same seed.
- **If Batch A runs, have the 4 inner-fold models also predict fold 0.** That gives 4 more lv2-recipe draws (at
  75% data), which calibrate the seed noise and the selection bias of 0.9425 at no extra cost.
- **SL-0 swap rule (D:43-44).** Re-run SL-0 on v4's SWA, not v3's; the global differs. A −0.002 threshold on the
  *local* is fine. Note that "MS1 for placement at inference" is a second (SoViT-400M) model at inference: the user
  should rule on it before it becomes the submission path.

---

## Probability and the single most important change (question 5)

- **As written (detach + LoRA, primary g + r): ~20%.**
  - P(LoRA local reaches v2 quality *and* decorrelation) ≈ 0.3, giving P(pass) ≈ 0.55–0.6.
  - Otherwise the local gain is about +0.002, giving P(pass) ≈ 0.1–0.15.
  - The g + r primary costs about a fifth on top.
- **With the changes: ~30–35%.** The changes are v2's crop-trained sharing, fixed mean(g, l) primary, separate local
  BN affine, the stricter early stop, and the 4th slot used for the detach comparison. The ceiling is set by
  E[global] + 0.0045 ≈ 0.946 against a 0.9455 bar with about ±0.0015 seed noise, i.e. near a coin flip, minus the
  ≥ 90% bootstrap requirement.
- **Single most important change: do not detach the local path from the shared backbone.** Train it as v2 did, with
  local BCE ×0.5 into the shared weights (optionally λ-scaled), so the local keeps the crop-trained, decorrelated
  features that produced the anchor. Every other part of rev 2 is secondary to whether the local reaches v2 quality,
  and detachment is the only choice that predictably lowers it.
