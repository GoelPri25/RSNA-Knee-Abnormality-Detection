# r50-GL v4, revision 2: one r50, a global path that local training cannot damage, a local path on its own adapters, and fusion learned against out-of-fold global errors (2026-10-18)

Status: **design for review round 2, nothing run.** Replaces the cross-plane revision 1
(`docs/exp_v4_crossplane_design.md`): methods review round 1 (`docs/exp_v4_review_round1_methods.md`) showed MP-0 gives
cross-plane binding no support against the global (B_all − A: G5 +0.0005, all 10 −0.0011); I had quoted the
cross-plane-vs-same-plane-zoom contrast.

## The anchor (the only global + local number above the bar)

- label-v2 r50 (best single, 0.9425) + r50-GL **v2 local** head, fixed 50/50 probability mean on the 10 cropped findings,
  nothing fit on fold 0: **0.9468 (+0.0043)** (`run_r50_gl3/audit_r50_gl_swa.txt:30`; pre-registered v3 PRIMARY bar).
- The same thing inside one model loses it: v2's own mean(g, l) 0.9449 (global 0.9407 = −0.0018 vs label-v2 r50); v3
  0.9409 halfgate / 0.9262 learned gate (global 0.9397 = −0.0027); cross-fitted ceilings with v3's global ≤ 0.9446.
- Two mechanisms explain the loss (r50-GL3-XAI): (1) **local training pulls the shared backbone** and the global head
  loses 0.002–0.003; (2) **fusion is fit on training studies** where every head is over-confident (v3 local training AUC
  0.9998 → gate trusted local; v2's residual reader stayed dormant because the global is already near-perfect in-sample).
  A third, smaller one: shared features correlate the two heads' errors (g–l logit corr v2 0.80–0.93).

## v4 design

One ResNet-50, label-v2 recipe for everything global (resnet50.tv_in1k init, label-v2 labels, seed, 25-epoch cosine,
SWA 21–25, 16 / 94 windows, 320 eval crop, 1e augmentation).

1. **Global path protected by construction.** The shared conv weights receive gradients **only from the global loss**
   (×1, as label-v2 r50). The crop forward pass uses the shared weights detached (functional call with `w.detach()`),
   plus its own parameters (item 2). So the global path is the label-v2 r50 recipe; any change in its AUC is seed-like
   noise, measured by a label-v2 r50 seed-2027 run.
2. **Local path on its own capacity.** Crop-only parameters on top of the detached shared weights: local BatchNorm
   (dual BN, as v3) + LoRA-style low-rank residual convs (rank 32) on the 1×1 convs of layer3 and layer4 (≈ 3–4 M
   parameters) + v2's local head. Crops exactly as r50-GL **v2** (the local that produced 0.9468): MS1 crop table, 10
   cropped findings, training draws 1 of the 4 windows per finding (∝ MS1 attention) with ±16 px jitter; evaluation
   averages all 4 (top-1 costs −0.02 to −0.026: r50-GL-XAI, r50-GL3-XAI). Effusion / synovitis: global only.
   The local head's errors decorrelate from the global's because the local features carry their own trained capacity.
3. **Fusion learned against out-of-sample global errors (stacking, cross-fitted).** Before v4, four label-v2 r50
   inner-fold runs on the training folds (each trains on 3 of training folds 1–4, predicts the 4th; fold 0 never seen)
   give **out-of-fold global logits g_oof** for every training study (label rule: model-shaped training targets only
   cross-fitted — satisfied). v4's local head is trained on two losses per cropped finding:
   - residual: BCE(σ(g_oof + r), y) — r learns what the global gets wrong **on studies it has not seen**;
   - standalone: 0.5 × BCE(σ(l), y) — keeps a calibrated local logit (secondary readout).
   Inference: fused = g + r (own global, trained on all training folds; r from the local path). No learned per-study
   gate, no gradient from the fused output into the global.
   Readouts: fused (PRIMARY), global, standalone local, fixed mean(g, l) (secondary; = the anchor's form).
4. **Self-location (no MS1 at inference)** is out of v4's critical path: SL-0 (written, ~$1 on Modal) runs in parallel;
   if self − MS1 ≥ −0.002 the submission swaps to self-located crops without retraining.

## Runs

- **Batch A (MeluXina node, 4 × A100, parallel):** 4 inner-fold label-v2 r50 runs → g_oof (≈ 2–2.5 h).
- **Batch B (MeluXina node):** v4 seed 2026, v4 seed 2027, label-v2 r50 seed 2027 (baseline seed noise), and
  v4-noOOF (identical, residual trained against v4's own in-sample global logit, detached: isolates the stacking fix).
- Checks before paid compute: CPU tests in the podman container (`--network=none`, stub timm); detach check (shared
  weights' grad from the local loss == 0); inner-fold split leak check (no fold-0 study, each training study predicted
  by exactly one model that did not train on it). MeluXina: no sub-hour smokes (1 node-hour minimum per allocation).

## Pre-registered decision (proposal, round-1 fixes folded in)

- PRIMARY: mean over seeds of per-seed fused AUC (not AUC of averaged predictions) − 0.9425 ≥ +0.003, with ≥ 90% of
  2,000 paired study bootstraps (per seed, both reported). Fixed bar 0.9425 (label-v2 r50 seed 2026, as all prior gates);
  label-v2 r50 seed 2027 is descriptive (noise).
- Secondary: E3 with v4 replacing label-v2 r50 ≥ +0.002 with ≥ 90%; v4 vs v4-noOOF (attribution of the stacking);
  global − 0.9425 (protection check: |Δ| ≤ seed noise); local train−val AUC gap per epoch; clear-band 6.
- EARLY STOP (seed 2026, mean of epochs 10–12): fused − global < 0 or global < label-v2 r50's own epoch 10–12 mean by
  > 0.004 → stop, report, leave.
- Submission seed pre-registered: seed 2026.

## Risks for review

1. LoRA-capacity local may be weaker than v2's fully fine-tuned local (0.9289): the anchor gain may need the full
   backbone. Alternative: a second copy of layer3–4 for crops only (shared stem/layer1–2) — more capacity, closer to two
   backbones; user constraint "no two backbones" to be respected.
2. g_oof comes from models trained on 3/4 of the training folds: slightly weaker than the full model's global at test →
   r may over-correct a little.
3. Detached-shared-weights local sees a moving backbone during training; adapters must track it.
4. The anchor 0.9468 is a fixed 50/50 mean of two independently trained models; a single network may keep less of the
   decorrelation.
