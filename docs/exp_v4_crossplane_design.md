# r50-GL v4: one r50 that reads the whole study and each finding's 3D point across planes (design, revision 1, 2026-10-18)

Status: **design for review, nothing run.** Deadline 2026-10-22 23:59 UTC.

## Goal and the bar

- Best single model: label-v2 r50 (`kaggle_fold0_labelv2/val_swa.json`), fold 0 (825 studies), label-v2 confident cells
  (weight > 0, y <= 0.3 or >= 0.7), 12-target mean AUC **0.9425**; public LB 0.937. Best submission E3 LB 0.944.
- Breakthrough bar (as r50-GL3-XAI): **>= 0.9455** (+0.003), >= 90% of 2,000 paired study bootstraps better, AND a gain
  when it joins / replaces label-v2 r50 in E3.
- User constraints: single model with real global + local synergy; fix the diagnosed mechanism (no post-hoc averaging,
  no frozen backbone, no second backbone / ensemble-in-disguise). Labels: label-v2 training labels, rulers untouched.

## What failed so far, and why (evidence)

| Run | Result (fold 0, lv2 confident, 12-target) | Mechanism diagnosed |
|---|---|---|
| MS1-GL (CoAtNet, from-scratch reader) | fused 0.9269 vs MS1 0.9301 | reader never learned MCL; crops helped inside the reader (+0.004 on 11 findings) |
| r50-GL v1 / v2 (residual reader, `g.detach()`) | v2 fused 0.9409, global 0.9407, local 0.9289; mean(g, l) 0.9449 (not cross-fitted) | residual path dormant (fused − global +0.0002); local carries signal only as a second opinion |
| r50-GL v3 (dual BN, 4 MS1 regions/finding, learned per-study gate) | fused 0.9262, global 0.9397, local 0.9198 | **local memorised** (training-sample AUC 0.9998 vs fold 0 0.906; v2 local 0.971 / 0.933): all 4 fixed regions every step + full local loss + most of the fused loss through local (gate β ≈ 0.1); gate fit in-sample |
| r50-GL3-XAI ceilings (cross-fitted inside fold 0) | (a) v3 g + v3 l constant blend 0.9415; (b) per-study logistic gate 0.9414; (c) v3 global + v2 local 0.9446 | per-study gating adds nothing over a constant; even a v2-level local only reaches +0.002 |
| Native-res check | median native/cache detail gain 1.10× | zoom crops are upsampled cache pixels: no new information |
| 2a (context between windows, pooled features) | −0.0026 vs 1c | mixing windows without location bought nothing |
| Spotlights (v3 spatial attention family, C0, OS32, v3.1) | all within ±0.005 of 1e | per-finding spatial attention collapses (cross-finding overlap 0.988) |
| **MP-0 (frozen label-v2 r50, linear probe)** | zoom of each finding's 3D point in sag + cor + ax vs source plane only: **+0.0028 [−0.0015, +0.0080], 89%** (bar +0.003 / 90%: borderline FAIL); lateral OA +0.006, MCL +0.006, Baker's +0.004 | frozen encoder reads best at its trained scale; the crop-TRAINED local head (r50-GL) gave lv2 + local +0.0042 [+0.0015, +0.0072] |

Reading: every global + local variant reused the **same pixels in the same plane**; its ceiling is a multi-scale effect
(≤ +0.004). The one untried source of information that the global path cannot use structurally is **binding the same
anatomical location across series** (sagittal fat-sat / non-fat-sat, coronal, axial). The global head pools 94 windows
independently (attention over pooled window vectors), so it cannot check that a sagittal and a coronal abnormality are the
same structure. MP-0 measured this only with a frozen encoder and a linear concatenation (no cross-plane interaction, no
training on co-registered inputs) and still saw a borderline +0.003. Host labelling rubric (forum sweep 2): a meniscal tear
needs >= 2 slices; radiologists confirm tears / ligament injuries across planes.

## v4 design

One ResNet-50 (label-v2 recipe: same init, labels, schedule, SWA 21–25, 1e augmentation, 16 training / 94 eval windows,
320 eval centre crop), dual BatchNorm statistics (global / local inputs, shared affine — v3, keeps the global path's
statistics clean).

**Global path:** unchanged label-v2 r50 (per-finding window attention head, 12 targets).

**Where to look (no MS1):** for each of the 8 region findings (ACL, MCL, medial / lateral meniscus, medial / lateral OA,
Baker's, contusion), the global head's own top windows by its per-finding window attention a_kn and the gradient ×
activation centroid inside each (code exists: `scripts/selfloc_r50_gl3_gpu.py::locate`), computed under `no_grad`
on the current step's global pass. Cache px → patient (LPS) coordinate → every series via `scripts/multiplane_points.py`
geometry (verified in MP-0: 99–100% of points land inside all 3 planes; median 0–1.7 mm from the nearest kept slice).
- Training: the point is drawn from the top-k (k = 4) windows with probability ∝ a_kn, ±16 px jitter in-plane and ±1
  slice through-plane; during epochs 1–3 (attention not yet meaningful) fall back to MS1's training-fold table
  (label-free MS1 points; only for training studies, never at inference). **Open question for review:** self-location
  from epoch 1 vs MS1 warm-up vs MS1 points throughout training with self-location only at inference.
- Inference: top-1 self-located point per finding (optionally top-2 averaged), no MS1.

**Local path (the new part):** at the 3D point, one 3-slice window per available series (up to 5: sag fat-sat, sag
non-fat-sat, cor fluid, cor other, axial), nearest slice to the point, box = r50-GL's per-finding box (128 / 192 px)
resized to the crop resolution (r50-GL eval crop path). Shared r50 (local BN) → pooled 2048-d token per series → + plane /
contrast embedding → **2-layer cross-plane transformer per study** (d = 256, 4 heads, a per-finding query token reading
that finding's series tokens) → local logit per region finding. Missing series masked.

**Fusion (no learned per-study gate):** fused logit = global logit + local logit for the 8 region findings (other 4 =
global). Loss = BCE(fused) + 0.5 BCE(global) + 0.5 BCE(local). Local logit layer zero-initialised (fused = global at
step 0); the 0.5 auxiliary on the local head keeps it learning even when fused is already right (v2's dormancy).
**Open question for review:** logit sum vs product-of-experts weighting vs a per-finding scalar learned on a held-out
slice of the training folds after training.

**Anti-memorisation (v3's defect):** stochastic points (above); **plane dropout** (each step keeps a random non-empty
subset of the series tokens, p_keep 0.7); local auxiliary weight 0.5; dropout 0.2 in the transformer; per epoch log
train vs val BCE per head and the local train/val AUC gap on a fixed 825-study training sample.

**Baker's:** no remapping; the 3D point is seen in every plane (MP-0 follow-up: sagittal zoom probe best, 0.889 vs axial
0.851).

## Runs (MeluXina, one GPU node = 4 × A100-40GB, 4 independent runs in parallel)

1. v4 seed 2026; 2. v4 seed 2027; 3. **v4-1P ablation**: identical but the local path sees only the series of the
global's chosen window (attribution: cross-plane binding vs plain zoom); 4. label-v2 r50 seed 2027 (baseline seed noise;
1e's two seeds differed by 0.0024 on 10 targets).
Cost estimate: v3 took 75 min on an H100 at 12.2 GB; v4 ≈ same crop count (≤ 5 series × 8 findings vs 4 × 8) +
self-location pass → ~2–2.5 h on an A100 → one node ≈ 2.5–3 node-hours, plus a 30-min `test`-QOS smoke.

## Pre-registered decision (proposal)

- PRIMARY: mean of v4 seeds (fused SWA) − mean of label-v2 r50 seeds ≥ +0.003 with ≥ 90% of 2,000 paired bootstraps;
  per-seed also reported. Secondary: E3 with v4 replacing label-v2 r50 ≥ E3; v4-1P vs v4 (attribution); clear-band 6;
  per-finding; local train/val gap.
- EARLY STOP at epoch 12 (both v4 seeds): local val BCE not below v2's at epoch 12 (0.414) OR fused trails global by
  > 0.002 → stop, report, leave the family.
- PASS → submission candidate (single + E3 swap), LB read; FAIL → leave global + local for good.

## Known risks (for the reviewers to attack)

1. MP-0's cross-plane signal was only +0.0028 (frozen); a trained reader may not grow it.
2. Joint training cost the global head before (v3 global 0.9397 vs label-v2 r50 0.9425, −0.003): the local loss may
   pull shared features.
3. Self-location early in training is noise; MS1 warm-up may leak MS1 dependence into the local path's placement.
4. Through-plane spacing: coronal / axial picks 6.6 mm apart (sagittal 3.5 mm) — thin structures may fall between kept
   slices (~4% of points > 0.6 × spacing).
5. 2a's failure: cross-window mixing added nothing; v4 differs by binding windows by location, untested.
6. Seed noise ≈ the bar; two seeds each is the minimum.
