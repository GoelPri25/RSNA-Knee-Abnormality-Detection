# r50-GL v4 (cross-plane) — methods review, round 1 (2026-10-18)

Reviewer role: adversarial methods review (hypothesis, design choices, decision rule). This review does not cover
implementation or compute. Reviewed: `docs/exp_v4_crossplane_design.md` (rev 1). Evidence used: ledger rows 1e-s2 (l.207),
2a (l.197), v3 / C0 / v3-OS32 / v3.1 (l.220-224), MS1-GL (l.275), r50-GL (l.276), r50-GL-XAI (l.277), MP-0 (l.278),
r50-GL v3 (l.279), r50-GL3-XAI (l.280); `run_sig_lv2/crop_probe/multiplane/audit_multiplane_probe.txt` (below
"MP0:<line>"); `run_r50_gl3/xai/audit_xai3.txt` ("XAI3:<line>"); `scripts/r50_gl3_runner.py`,
`scripts/r50_gl_runner.py`, `scripts/multiplane_points.py`, `scripts/selfloc_r50_gl3_gpu.py`.

**Verdict: proceed with changes, and only after two cheap pre-checks (~$1–2, < 1 h).** As written, v4 has about a
**10–15 %** chance of clearing 0.9455. Its central claim rests on the one MP-0 contrast that does not test it. The
logit-sum fusion recreates v3's in-sample trust problem through the logit scale. Inference uses top-1 placement,
which the ledger has twice measured at about −0.026. The self-location it relies on has never been validated
(SL-0 is written but has not been run).

---

## Findings, ordered by impact

### F1 (critical). MP-0 does not support the hypothesis. The design quotes the wrong contrast.

The design (l.25, l.31-32) cites MP-0 as "+0.0028 [−0.0015, +0.0080], 89 %" with gains on lateral OA, MCL and Baker's.
That number is **B_all − B_src**: all three planes compared with the *source-plane zoom*. v4 has to beat the **global
model**, and the matching contrast is **B_all − A**:

| MP-0 contrast (MP0:19-27) | G5 | all 10 |
|---|---|---|
| B_all − A (3-plane zoom added to the global logit) | **+0.0005** [−0.0033, +0.0046], 61 % | **−0.0011** |
| B_src − A (same-plane zoom added to the global) | −0.0023 | −0.0028 [−0.0059, −0.0001] |
| B_all − B_src (the number the design quotes) | +0.0028, 89 % | +0.0017 |
| B_slice − A (the 3 mapped windows, no zoom) | +0.0017 | −0.0001 |
| B_all − B_slice (zoom vs. right slice) | −0.0011 | −0.0010 |

- The "+0.0028" mostly offsets the −0.0023 penalty from the same-plane zoom. Measured against the global, the
  other planes add **+0.0005 on the five findings picked for being favourable, and −0.0011 on all ten**.
- G5 was chosen *because* lv2 + r50-GL local gained on those findings (ledger l.278, "check 1"), so even the +0.0005 is
  biased upward.
- The per-finding "MCL +0.006" is MCL's B_all − B_src. MCL's **B_all − A is −0.0101 [−0.0207, −0.0018]** (MP0:8),
  a significant *loss*. Baker's B_all − A (+0.0084) matches B_slice − A (+0.0086). Baker's gain therefore comes from
  the mapped *slices*, which the global already reads among its 94 windows. It is not zoom and it is not binding.
- MP-0's probe concatenates plane features into a **linear** logistic, so it measures *additive* evidence from other
  planes. That additive kind of evidence is exactly what the global's attention pooling already sums across windows.
  **Binding (a conjunction such as "this sagittal signal and this coronal signal are the same structure") was never
  measured anywhere.** The hypothesis therefore has no direct support. It does have one indirect negative: additive
  cross-plane evidence beyond the global is about 0.
- The design also says (l.26-28) that every earlier variant reused the same pixels in the same plane, and that v4 is
  different. v4's local tokens are **also crops of cached slices that the global already sees** (94 eval windows
  cover the d96 picks; the native-res check found 1.10× detail). v4 adds no new pixels. Its only new ingredient is the
  operator that groups 5 tokens by 3D point.

**Is the hypothesis a rationalisation?** In part, yes. MP-0 was finished *before* v3. Its pre-registered outcome was
"Q2 FAIL → same-plane regions only" (l.278). The r50-GL3-XAI rule then said **LEAVE** (l.280). v4 reopens a closed
family with no new evidence; it re-reads a number that was already on file. Under the "revisit verdicts" rule this is
allowed only if the new instrument changes the reading, and here it does not. The honest framing: v4 tests an
*untested* mechanism (point binding) whose closest available proxy reads null.

### F2 (critical). The logit-sum fusion is still a fusion weight learned in-sample, through the logit scale.

The design says "no learned per-study gate" (l.60). But in `fused = g + l` with BCE(fused), the relative weight of
the two views is **the ratio of their logit scales**, and the local classifier sets its own scale freely. Every
in-sample fusion so far has moved weight to the head that fits the training studies best:
- v2 reader: fit training residuals from the global tokens (l.276).
- v3 gate: β moved to 0.09–0.23 toward local; local training AUC 0.9998 vs fold 0 0.906 (XAI3:36-43).

If local memorises again, its logits grow and the sum becomes about the local head at test time: v3's failure in a
new form. The 0.5 auxiliaries do not stop this. They train each head to explain the label *on its own*, so both heads
can become large-scale and the fused loss sets the balance.

The cross-fitted evidence says the right balance is **not** learned in-sample. XAI3:94-95 gives constant β (weight on
global) of 0.59–0.80 for v3's heads and 0.17–0.63 for v3-global + v2-local. Even so, (a) and (b) are ≤ 0.9415 and (c)
is 0.9446. Fusion is the binding constraint, not features.

The fused BCE also sends the global head the gradient σ(g+l) − y. As local learns, the global gets less signal on
cases local explains, and its standalone strength drops. The global aux weight is halved to 0.5 (label-v2 uses 1.0),
so "global path: unchanged label-v2 r50" (l.41) is false as specified. This is the most plausible cause of v3's
global −0.003 apart from seed noise.

**Change:** do not backpropagate the fused loss into either head. Use `fused = g + l` (or 0.5/0.5) as a **fixed**
architectural readout, give each head a full-weight own BCE (global ×1, as label-v2), and treat fused as scoring only.
v2's equivalent, mean(g, l), is the only global+local readout that has ever beaten its own global held-out: +0.0045
[+0.0014, +0.0084], and 0.9449 (l.276). If the user counts a fixed sum as "averaging", the honest alternative is a
per-finding scalar fit on a **held-out** slice of the training folds (design open question 3). Those studies must be
held out of training; otherwise it is in-sample again. Never fit it in the joint loss.

### F3 (high). The inference placement contradicts measured evidence: top-1 costs about 0.026.

The design specifies top-1 self-located point at inference (l.52). On record:
- v2 local, top-1 vs 4-crop: 0.9076 vs 0.9289 (−0.021, XAI3:29 vs :10).
- v3 local, top-1 vs 4-region: 0.9016 vs 0.9197 (−0.018, XAI3:18 vs :14).
- r50-GL-XAI: "Top-1 crop −0.0255 vs 4-crop average" (l.277).

Training draws 1 of top-4 ∝ a, which is v2's setup, and v2's best inference was the attention-weighted average over 4.
**Change:** at inference, average the local logit over the top-4 points weighted by a_kn (each point gives up to 5
plane tokens). This is pre-registered and costs nothing in training.

### F4 (high). Self-location is unvalidated and couples local placement to the global's beliefs.

- **SL-0 has never run** (`scripts/selfloc_r50_gl3_gpu.py` is ready; ledger l.279: "SL-0 not run"). Nothing on
  record shows that r50's self-located points are as good as MS1's for a crop-trained local head. Every positive
  local result (MS1-GL crops, v2 local, MP-0) used **MS1 points**.
- **Correlated errors.** v2's gain came from *decorrelated* held-out errors: g–l correlation 0.80–0.93 (XAI3:3), and
  stacking only recovered complementarity (l.276). MS1 placement is decorrelated by construction because it comes
  from a different model. Self-location places local crops **where the global already looks**. When the global
  misses a lesion (its false negatives, where local value would be greatest), it usually attends elsewhere, so the
  local looks elsewhere too and cannot rescue it. That limits v4 to re-scoring what the global already found, which is
  the multi-scale effect the design says caps at +0.004.
- **Label-correlated placement.** `locate` takes the centroid of the *positive part* of a class-signed CAM
  (`selfloc_r50_gl3_gpu.py:61-64`) and falls back to the image centre when that part is empty. Positives get
  lesion-centred points; negatives get diffuse or centre points. The local head can learn "where the crop sits" as a
  proxy for the global's opinion: a shortcut that is redundant with g and easy to memorise.
- **Train/test placement shift.** Training self-locates among 16 random windows on studies the global is partly fit
  to (global training AUC 0.973 vs fold 0 0.936, XAI3:38). Eval self-locates among 94 windows on held-out studies.
  Training points are worse in window choice and better in in-sample confidence: two shifts that do not cancel.
- **Coarse points weaken the binding the design depends on.** The r50 CAM is a 10 × 10 grid of 32 px cells
  (11.7 mm) inside the 320 crop. In-plane error in the sagittal anterior–posterior direction becomes through-plane
  error in coronal (6.6 mm pick spacing), so a one-cell error is about 2 coronal slices off. Add the ±1 slice jitter
  (±6.6 mm coronal/axial, ±3.5 mm sagittal; MCL is about 3 mm thick in sagittal) and "same 3D location" is loose for
  exactly the thin structures the hypothesis targets.

**Pre-check P1 (SL-0, inference only, ~$1, already written):** run SL-0 on v3 SWA (or adapt it to the v2 SWA, whose
local head is healthier). Read local AUC with self-located regions vs MS1 regions, plus agreement with MS1.
Decision rule: if self-located local AUC is more than 0.005 below the MS1-region AUC, "no MS1 at inference" is not
viable, and v4 as designed cannot work.

**Training-placement recommendation (the design's open question):** keep the **training** and **inference**
procedures the same. Use the same self-location procedure in both, but in training mix it with MS1 points
(p = 0.5 throughout, not a 3-epoch warm-up). The warm-up teaches the local head on MS1 placement, then switches
distribution while the local is still in its fast-learning phase (v2 local plateaued at about epoch 11). The p = 0.5
mix makes the local robust to both. MS1 points are label-free and showed no in-sample placement leakage (−0.119 vs
−0.109, l.278). "MS1 throughout training, self at inference only" is the worst option, a guaranteed train/test shift.

### F5 (high). The decision rule has a moving bar, ambiguous aggregation, and roughly coin-flip power at the bar.

- **Aggregation is unspecified.** "Mean of v4 seeds − mean of label-v2 seeds" (l.83) could mean the mean of per-seed
  AUCs or the AUC of averaged predictions. The second is a 2-member ensemble. On record, a 2-seed r50 average gives
  **+0.0044 [+0.0022, +0.0068]** over one seed (l.207), more than the +0.003 bar. v4's seeds will be *more diverse*
  (stochastic placement), so prediction-averaging favours v4 for reasons that will not carry over to a single
  submitted model. **Pre-register the mean of per-seed AUCs**, with a bootstrap of the per-seed-averaged difference.
- **The bar moves.** l.9 says "≥ 0.9455", but the primary is relative to the *mean of label-v2 seeds*, so the absolute
  target depends on seed 2027's draw. Pick one. I recommend the relative bar against the 2-seed lv2 mean, with 0.9455
  reported.
- **Power.** The 1e seed spread is 0.0016 (12-target) and 0.0024 (10-target) (l.207), so σ_seed ≈ 0.0011–0.0017. With
  2 v. 2 seeds the difference has SD ≈ σ_seed ≈ 0.0015 from seeds alone; the study bootstrap adds about 0.002. If the
  true effect is +0.003, P(pass) ≈ 50 %; at +0.002, ≈ 25 %; at +0.001, ≈ 9 %. The best-case ceiling on file is
  (c) +0.0022 (XAI3:92), so the rule is fair but the prior is weak. A pass also carries winner's-curse risk, so it
  needs the LB read the design already plans.
- **The submitted seed is not pre-registered.** Choosing the better v4 seed on fold 0 after the fact is selection.
  Pre-register seed 2026, or submit both and report both.
- **The secondary E3 rule "≥ E3" is a ≥ 0 test**, which is noise. Use ≥ +0.002 with ≥ 90 % (as r50-GL, l.276).

### F6 (high). The early-stop rule is mis-specified and checks the wrong things.

- "Local val BCE below v2's at epoch 12 (0.414)" compares different cell sets: v2 local covers 10 cropped findings,
  v4 local 8 region findings. Recompute v2's epoch-12 BCE on v4's 8 findings, same cells.
- Local health at epoch 12 does not predict v3's failure. v3's local *led* v2 by +0.045 at epoch 4 and lost it by
  epoch 20; memorisation built as the learning rate decayed (l.279). Gate instead on the **local train − val AUC gap**
  (already logged, l.67-68). Example: stop if the gap is > 0.06 at epoch 12 or growing faster than the global's.
- "Fused trails global by > 0.002" from a single epoch is inside v2's per-epoch ±0.003 wobble (l.276). Use the mean of
  epochs 10–12.
- Missing: a **global-health** gate. Global at epochs 10–12 vs label-v2 r50's own history at the same epochs (on
  file). The design lists global damage as risk 2 (l.92-93) but nothing gates it.

### F7 (medium). The cross-plane transformer is excess capacity at the wrong place. Binding happens in geometry.

- All tokens for a finding already come from the same 3D point; the geometry does the binding. Each token is a GAP
  over a 47–70 mm box, so any spatial relation inside it is already pooled away. A 2-layer transformer (d 256, 4
  heads, about 1.5–2 M parameters including the 2048→256 projection) over ≤ 5 tokens can only learn *which plane to
  trust* and *pairwise conjunctions*. The conjunctions would have to be learned from about 70 (MCL), about 280
  (lateral OA) and about 380–400 (ACL, lateral meniscus) training positives (4.3 × the fold-0 counts, MP0:6-15).
  v3's local memorised with less (a 2048-256-12 attention pool).
- **Change:** a per-finding attention pool over plane tokens plus a plane/contrast embedding (1e head shape, as v3's
  region pool), or at most 1 layer with d 128. If binding matters, a pairwise product of the source token with each
  other-plane token, after a shared 256-d projection, is a cheaper and more interpretable conjunction than attention.
- **Pre-register a binding check** (as v2-XAI's shuffle): at eval, replace each non-source plane token with a crop
  from a wrong point in the *same* study (for example the point mirrored medial↔lateral). If fused AUC moves by less
  than 0.001, the model does not use binding, whatever the headline.

### F8 (medium). Plane dropout, jitter, and which findings get local paths.

- **Plane dropout (p_keep 0.7)** is good for forcing cross-plane use. Add an explicit **source-plane drop** (for
  example 30 % of steps without the source token); otherwise the per-finding query locks onto sagittal for 7 of 8
  findings (MP0:3) and the other planes behave like noise. Train and eval token counts differ (all 5 kept at eval);
  softmax pooling tolerates this, but log the local AUC with all-plane vs the training-like random subset at the end.
- **Through-plane jitter ±1 slice** should depend on plane: 0 for the plane where the finding is thin through-plane
  (MCL in sagittal, ACL in coronal), ±1 elsewhere.
- **Findings.** All 8 are defensible only with a *fixed* fusion (F2). MCL is the canary for in-sample fusion: 16
  fold-0 positives; v3 local 0.877 and fused 0.910 vs global 0.970 (XAI3:14-15); MP-0 B_all − A −0.010; yet v2
  mean(g,l) +0.010 over global. MCL is safe with fixed weights and disastrous with learned ones, and alone it is worth
  ±0.004 on the 12-target mean. Medial meniscus and medial OA (global ~0.965) gain nothing in any measurement. If
  findings are cut, do it on a mechanism prior or training-fold CV, **not** fold-0 per-finding numbers (selection
  bias).
- **Baker's.** The design cites "sagittal zoom probe 0.889 vs axial 0.851" (l.70-71) as support. v3 **refuted** that:
  sagittal local 0.881 vs v2 axial 0.917; "the frozen-zoom plane CV did not transfer" (l.279; XAI3:33). v4 sees all
  planes, so this is harmless, but the citation should go.
- **Dual BN** is justified: crops through global BN cost −0.008 (XAI3: local_gbn). Specify that the post-SWA crop-BN
  refresh uses **self-located** crops from the SWA model, since its attention differs from any epoch's.

### F9 (medium). 2a is weaker evidence either way than the design says.

2a (l.197) did add per-plane anatomical position (sagittal medial→lateral, coronal anterior→posterior, axial
superior→inferior) and mixed windows with a transformer. That is a coarse form of location-aware cross-plane mixing,
so "binding … untested" (l.98) overstates the novelty. But 2a's gate stayed dormant (|g| 0.002) and −0.0026 is inside
seed noise (l.207), so it tested little. Fair statement: neither 2a nor MP-0 tested point-level conjunction; both read
null on the coarser forms.

### F10 (low–medium). The v4-1P ablation has low decision value under the deadline.

1P matters only if v4 passes: attribution would not change the submit decision. If v4 fails, the family is closed
regardless. With one seed, the 1P − v4 difference is inside seed noise (about ±0.002) unless the effect is large.
**Better use of slot 3 (same cost):** **v4-F**, v4 with the F2 fusion (fixed sum, no fused gradient, global BCE ×1).
That answers the design's own open question about fusion, and on the evidence it is the variant most likely to pass.
Run 1P after a pass.

---

## Will v4 fix v3's memorisation?

v3's memorisation had three causes, ordered by evidence (XAI3:36-43, l.279-280):
1. **No sampling diversity on the local side.** All 32 regions every step, jitter only. v2 (1 of 4 drawn per step)
   had a train−val gap of 0.038, equal to the global's; v3 (all 4) had 0.094. v4's stochastic point (1 of top-4, ±16
   px, ±1 slice, plane dropout) **fixes this.** It is the design's strongest change.
2. **In-sample fusion pushing the loss through local** (β → 0.1). v4's logit sum **does not fix this** (F2).
3. **Full-weight local loss.** Halved in v4, which helps a little.

Other possible causes the design does not address: label-correlated placement (F4), and the extra capacity of the
plane transformer (F7). My expectation for the design as written is a local train−val gap of 0.05–0.07: better than
v3, worse than v2. Fused will then trail mean(g, l), as in v3.

**Will joint training hurt the global again?** Probably, by 0–0.003, for two reasons: the fused gradient and the
halved global weight (F2). Run 4 (lv2 seed 2027) is necessary to tell this from seed noise; keep it. With F2's change
(global BCE ×1, no fused gradient), the only remaining coupling is the shared backbone. v2's global (0.9407, shared
backbone, g.detach()) suggests that coupling costs at most about the seed spread.

## Probability and the single most valuable change

- **P(v4 as written clears the pre-registered bar) ≈ 10–15 %.** Prior on the true Δ vs lv2: about 40 % ≤ 0 (a
  v3-like collapse through the logit scale), about 35 % in (0, +0.002], about 20 % in (+0.002, +0.004], about 5 %
  above. Combined with the power figures in F5, that gives about 12–19 % before the extra stringency of the bootstrap
  and the moving bar.
- **Single highest-value change: fixed fusion, no fused gradient** (F2: `fused = g + l` as a readout only, global BCE
  ×1, local own BCE). It removes the one mechanism that sank both previous fusions, and it gives v4 the v2 mean(g, l)
  floor (0.9449, held-out) plus whatever the cross-plane tokens add. With F2 + F3 (top-4 inference), I would put P(pass)
  at about 20 %. That is still a long shot, because the fixed-sum floor is 0.9449 < 0.9455 and MP-0 suggests the
  cross-plane increment is small.

## Pre-checks before any paid training (both inference/CPU, < 1 h total)

1. **SL-0** (F4): self-located vs MS1 regions through an existing trained local head. Kill rule: a drop of more than
   0.005 means "no MS1 at inference" does not work.
2. **MP-1: MP-0 with a crop-*trained* encoder.** Re-extract MP-0's three-plane zoom features through **r50-GL v2's
   local path** (crop BN, a backbone trained on zoomed crops in all three planes: MCL coronal, Baker's axial,
   contusion ~42 % coronal). This removes MP-0's stated excuse ("frozen encoder never saw zoomed inputs"; also the
   "frozen probe is not a gate" concern, since this encoder *was* trained on the new input type). Add an interaction
   probe (source ⊙ other-plane after PCA, cross-fitted) next to the additive concat. Read B_all − A and
   B_interact − B_all on **all 10**, not G5. The result shifts the expected value of v4; it is not a hard stop. If
   B_all − A < +0.001 and the interaction term adds < +0.001, the cross-plane increment is unlikely to reach the
   +0.0006 that the fixed-sum floor (0.9449) needs, and the GPU slot is better spent elsewhere.
3. A geometry unit check with **augmentation on**: 3 studies, render the bound sagittal/coronal/axial crops after
   mirror and affine. MP-0 checked by eye without augmentation only. Silent mis-binding would turn v4 into a noisy
   zoom and the run would still "work".

## Alternatives within the constraints

(Single model, one backbone, real global + local synergy, no post-hoc averaging, no frozen backbone, no external raw
data, open-weight models allowed.)

- **A (recommended v4 form).** v4 with F2 + F3 + F7 (small plane pool) + F4 (mixed placement) + F8 (source-plane drop).
  Same compute. This is the design with its measured failure modes removed.
- **B. Held-out residual training for local (the in-sample-residual mechanism, fixed directly).** Train the local head
  on BCE(stopgrad(t_oof) + l), where t_oof is a **cross-fitted** global logit for each training study. The 4
  cross-fitted CK-A teachers exist (P3v2-T, l.258) and are allowed under the out-of-fold label rule. The local head
  then learns the global's *held-out* errors, not its in-sample residuals, which is exactly what v2 (+0.0009 of
  +0.0034 captured) and v3 got wrong. At inference, fused = g + l from one network. Caveat: the teacher is
  ConvNeXt, not r50, so the residual is relative to a different global. Run it as a variant, not as the primary.
- **C. Placement from an open-weight anatomy model, to decorrelate placement from the global.** A public knee-MRI
  segmentation model (bone, cartilage, menisci; for example nnU-Net-style models trained on OAI-ZIB or SKI10, if
  weights and licence check out; OAI-trained open weights are allowed per the OAI rule) would give label-free, per-structure 3D points
  (meniscal horns, ACL footprint, MCL line, popliteal fossa). It is independent of both r50 and MS1, which is what
  made MS1 placement useful, and it removes F4's coupling. It adds a placement model at inference: this is a
  preprocessing step, not a second classifier, but the user should decide whether it violates "one backbone". Given
  the 4-day deadline, this is realistic only if a model can be found and run in about a day.

## Smaller notes

- l.25: "+0.0042 [+0.0015, +0.0072]" for lv2 + r50-GL local is a G5-style cross-fitted figure. On the 12-target mean,
  mean(g, l) − lv2 was +0.0028, 89 % (l.276). Quote the 12-target figure next to the bar.
- The variance-reduction confound for the whole family: mean(g, l) − own global +0.0045 (v2) is the same size as a
  2-seed average (+0.0044, l.207). On the evidence so far, "global + local synergy" cannot be told apart from "two
  decorrelated heads". The binding check (F7) and 1P are the only parts of v4 that could tell them apart. That is a
  reason to pre-register the binding check as a mechanism readout.
- The "fused trails global" early stop and the logged local train/val gap should be computed on the **same 825
  training sample** across v4 seeds (as XAI3 did), so the gaps are comparable.
