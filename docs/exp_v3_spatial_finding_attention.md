# Exp v3: finding-specific spatial pooling at output stride 16

**Status:** revision 3, 2026-10-12. Nothing has been trained or launched.

**Review history:**
- **Revision 1** was reviewed by the user's reviewer and by an independent agent (`docs/exp_v3_review_round1.md`).
- **Revision 2:**
  - reframed the question;
  - deferred the anatomical prior;
  - dropped Step 0;
  - fixed the normalisation;
  - added a stride-16 GAP control;
  - made the decision rule provisional.
- **Revision 3** applies the round-2 review (minor revision):
  - removes a mask case that does not exist;
  - initialises `U = 0` so v3 starts exactly at C0;
  - runs C0 only if needed;
  - uses the L4 swap to choose the submission;
  - pre-registers a "branch used" rule and a 384 secondary readout;
  - tightens the decision rule.

**Base:** v2 = Exp 1e (`scripts/coatnet_d96_runner.py`, `D96_PHASE=r50_pertarget`).
- Public LB 0.936; the same recipe with seed 2027 scored 0.934.
- Fold-0 10-target 0.9073 / 0.9097; clear-band 6-target 0.9719 / 0.9690 (seeds 2026 / 2027).

---

## 1. Lineage

| | v1 (d2_224 r50) | v2 (d96 r50, 1c / 1e) | v3 (this proposal) |
|---|---|---|---|
| Coverage | ~24 slices | 96 slices over 5 series slots, 3-slice windows | unchanged |
| Input | 224 px | 384 px over a 140 mm field (smaller fields are magnified to 384, not padded); evaluation = centre 320 crop (≈ 117 mm) | unchanged |
| Layer-4 grid per window | 7 × 7 | 12 × 12 in training, **10 × 10 at evaluation** | 24 × 24 in training, 20 × 20 at evaluation (output stride 16) |
| Finding-specific selection | none | per-finding softmax over **windows**, on each window's average | per-finding selection over **locations within each window**, alongside the average; then v2's per-finding window attention |
| Public LB | 0.925 | 0.924–0.936 (1e 0.936) | ? |

Grid spacing is not receptive field. A layer-4 cell sees a much larger region. "Finer grid" is about where a
selection can be made, not about what each cell sees.

## 2. Question and hypotheses

**Question.** Does letting each finding select spatial locations within a window, before averaging, add useful
information to v2? Here "useful" means fold-0 ranking, the clear band, and contribution to L4.

- **H1.** Finding-specific spatial pooling at output stride 16 improves single-model ranking beyond v2 and beyond
  the same backbone with plain averaging. Possible mechanisms, none established:
  - small or low-contrast evidence averaged with irrelevant tissue;
  - per-finding feature selection inside a shared map.
- **H0.** Like 2a, 2c, 4b, 6a and 6b, it either trades targets or fails the clear band. 4b, 6a and 6b failed the
  clear-band floor at 0.9617, 0.9652 and 0.9626, and 6a lost to seed 2027 by 0.0038. The remaining deficit is
  label-convention or feature limited (2-conv; 4b-abl places the trade-offs in the backbone features).

## 3. Motivation, with what it does and does not show

| Observation | What it suggests | What it does not show |
|---|---|---|
| **2-sev:** moderate+ positives AUC 0.93–0.99; mild / hedged classes 0.73–0.95. The share of misses that are mild or hedged is 0.53 (medial meniscus) to 1.00 (MCL) | The headroom is in less conspicuous cases | Report-"mild" is a grade or extent, not a lesion area. The headroom targets (MCL, effusion, PF OA, medial OA) are mostly diffuse |
| **2-evid:** among detected positives, evidence spreads over 4–10 windows | Evidence is distributed | Nothing about misses: positives below the median negative are excluded (`audit_d96_window_evidence.py:40`) |
| **2-side:** misses attend to the correct side, like negatives | Attention is not grossly misplaced | Correct side ≠ correct lesion |
| **Gold-58 check (gold-amb):** in report-ambiguous cells (261, 151 host positives), image predictions separate host labels at AUC 1e 0.775 [0.72, 0.82], 4b 0.807 | The image carries host-label information where reports are a coin flip | Development evidence only; nothing about pooling |
| **Prior spatial attempts:** teammate target-query r34 (labels changed at the same time; LB 0.916); Pillar-0 target-query readers; local_global top-k | Tried, on weaker bases | None on d96 with fixed labels and a GAP path kept |

**Correction to revision 1.** v2 already has a separate softmax over windows for each of the 12 findings
(`coatnet_d96_runner.py:355`). The trade-offs in 2c / 4b / 6b more likely come from shared backbone features or
sampling (4b-abl). v3 keeps a shared backbone, so it does not remove them by construction.

## 4. Design

Everything is Exp 1e except the backbone's output stride and the head:
- labels (sha d40803d0…), 16 stratified training windows, 94 evaluation windows;
- 384 training / 320 centre-crop evaluation, augmentation;
- AdamW, 25-epoch cosine with a 1% floor, seed 2026, SWA 21–25, BN refresh.

1. **Backbone.** `timm.create_model("resnet50.tv_in1k", num_classes=0, output_stride=16)`, which dilates layer 4
   (stride 1, dilation 2) and loads the same pretrained tensors. Feature maps come from `backbone.forward_features(x)`
   → [N, 2048, H, W] → `flatten(2)`. C0 and v3 share one class: a flag switches the spatial path on, and both
   compute `g` from the same `z.mean(cells)`.

2. **Head.** For window *k*, with raw layer-4 cells `z[k,c]` and validity mask `v[k,c]`:
   - **Score branch:** `s[k,c,n] = MLP(LN_s(z[k,c]))`, with MLP = Linear 2048→256, Tanh, Dropout 0.2, Linear 256→12
     (the same form as v2's `att`). The last layer is zero-initialised. `α[k,·,n] = softmax over valid cells`.
   - **Pooled features:**
     - `g[k] = LN(mean over all cells of z)`: unmasked, exactly v2's path;
     - `p[k,n] = LN_p(Σ_c α z)`: finding-specific, computed with `einsum("bcn,bcf->bnf")`, never materialising
       `α·z` (that would be [32, 576, 12, 2048]).
   - **Window attention:** v2's per-finding window attention on `g[k]`, unchanged.
   - **Classifier:** `logit_n = Σ_k a[k,n] · (W_n·g[k] + U_n·p[k,n]) + b_n`.
     - The GAP path keeps broad context.
     - **`U` is initialised to 0**, so v3 starts exactly at C0.
     - `U` gets gradient at step 1. The score MLP's last layer first gets gradient at step 2, out of about 11,000
       steps. Its earlier layers get gradient once that layer is non-zero. This is measured (§6).
   - **At initialisation** α is uniform over valid cells, so `p` is LN_p of the *masked* mean. With no padding
     (evaluation, BN refresh) and LN_p sharing LN's parameters, `p == g` exactly (check a′).
   - The new parameters fall into the head group (lr 1e-3, weight decay 0.05) automatically (`runner:517`).

3. **Validity mask (training only).** The only padding is the zero fill from the augmentation's `grid_sample`
   (`runner:323-324`).
   - `prepare` also warps a ones-image with the same grid (`padding_mode="zeros"`), average-pools it with kernel 16
     and stride 16 to 24 × 24, and thresholds at 0.5.
   - Evaluation and BN refresh use an all-ones mask.
   - Padding is 0 before normalisation, like background air, so the mask is mainly hygiene. Attention mass on
     masked cells is still logged.

4. **Preflight checks** (CPU fp32, then one GPU batch):
   - (a) with the flag on and `U = 0`, logits equal C0's exactly (max |Δ| = 0 in fp32, as in 2a's preflight). This
     proves the wiring only.
   - (a′) with no padding, uniform α and LN_p = LN, `p == g` exactly.
   - (b) shapes and masks for 16 and 94 windows, at 384 and 320.
   - (c) after 2 steps, every new parameter group has a non-zero gradient.
   - (d) throughput and memory. The estimate is about 12–15 GB, and z is about 75 MB in bf16.

5. **Not in v3:** anatomical coordinates (§7), cross-slice mixing, label changes.

**Why there is no frozen-trunk OS32 test first.** The round-1 reviewer proposed it (~$2–3).
- A head retrained on a frozen 1e trunk reads features that were trained under plain averaging. A null result
  would not tell "selection adds nothing" apart from "these features were never shaped for selection".
- The full v3 run costs only about $5–6, so the cheap test buys little.

## 5. Runs and decision rule (provisional)

| Run | What | When | Cost |
|---|---|---|---|
| **v3** | the design above, seed 2026 | now | ~1 h H100, ~$5–6 |
| **v3 @ 384** | v3's SWA weights evaluated on the 384 input (secondary readout) | with v3 | inference, ~$1 |
| **C0** | same class with the spatial path off (OS16 GAP), seed 2026 | only if v3 passes conditions 1–2 | ~$5–6 |
| **v3-s2** | v3, seed 2027 | only if v3 passes conditions 1–2 | ~$5–6 |

**Primary readout:** SWA 21–25, 320 crop, no TTA.

**Conditions for v3 (seed 2026):**
1. **10-target Δ against the mean of 1e's two seeds ≥ +0.003,** with ≥ 90% of 2,000 paired study bootstraps
   better against each seed. In practice this needs about +0.004–0.005: 4b's +0.0018 against seed 2027 reached
   only 76.5%.
2. **Clear-band 6-target ≥ 0.9695** (1e seed mean 0.9705 − 0.001).
3. **Branch used:**
   - α entropy (eval mode) below 0.95 · log(valid cells) for at least 4 findings;
   - the `U·p` term is at least 10% of the logit magnitude (mean |U·p| / mean |logit| on fold 0).

   If this fails, the run is "**not tested**", whatever conditions 1–2 show (the 2a precedent: its zero-init gate
   ended at 0.002).

**Submission selector.** If conditions 1–3 pass:
- score L4 with 1e swapped for v3 on fold 0;
- submit that L4 if the swap is ≥ 0 on 10-target and on clear-band 6;
- otherwise submit v3 alone.

The best LB is the ensemble, so a standalone v3 matters through its contribution to L4.

**Attribution (C0), not a gate.** If v3 passes, run C0.
- v3 − C0 differs by about one seed's noise: the new parameters consume CPU RNG and so change the window draws, and
  dropout uses CUDA RNG.
- v3 − C0 ≥ +0.003 is read as support for spatial selection; a smaller gap means stride and selection are not
  separated.
- **C0 is promoted only if it passes conditions 1–2 by itself.** The ledger then records that two arms were tried
  against the gate.

**New base.** v3 becomes the base only if v3-s2 also passes. On the two-seed mean, condition 1 is evaluated by
bootstrapping the **averaged predictions** of v3-s1 and v3-s2 against the averaged predictions of 1e's two seeds
(2,000 paired study resamples), and condition 2 by the clear-band AUC of the averaged predictions.

**Caveats on the rule:**
- The headline 10-target has about zero LB correlation across our singles (2-band). It stays as condition 1 only
  together with the clear-band floor and the L4 swap.
- Two baseline seeds are not a noise bound, and study bootstraps do not capture training randomness.
- An LB of 0.939 is an aspiration, not an expected consequence.

## 6. Diagnostics (reported; only condition 3 gates)

- **α entropy** per finding, in train mode and eval mode. A gap between them flags collapse onto outlier cells.
- **Branch gradient norms** per epoch: each score-MLP layer, `U` vs `W`.
- **Attention mass on masked cells** (training).
- **Per-target results,** with focal (menisci, ACL, contusion, fracture) and diffuse (MCL, effusion, OA, synovitis)
  findings reported separately.
- **Band split** (clear / mild) and mirror TTA.
- **Ensemble:** rank correlation vs 1e; L4 swap; L4 + v3.
- **α maps** for 10 fold-0 studies per finding, chosen by the lowest SHA-256 of the study id (rule fixed now).

## 7. Deferred: anatomical prior

A shared knee frame needs:
- full IOP and IPP vectors and both pixel spacings;
- the original Rows/Columns and the crop/resize offsets (not in the d96 manifest);
- a knee-relative origin (e.g. the 2-pre consensus centre);
- coordinate maps transformed consistently with every augmentation and with the evaluation crop.

`d96_positions.py` normalises within each series: it gives ordering, not alignment. The prior is revisited only if
v3 passes, the geometry passes a landmark test, and the coordinate maps follow the augmentation.

## 8. Settled in review

| Question | Decision |
|---|---|
| C0 in parallel or conditional | Conditional on v3 passing conditions 1–2. Under the null, v3 − C0 is a coin flip |
| GAP path plus spatial vs spatial only | Keep the GAP path. Spatial-only resembles the 0.916 teammate run and risks the diffuse targets. Condition 3 guards against the path going unused |
| 384 vs 320 domain gap | Pre-registered 384 secondary readout. No change to the training crop (1d: 384 eval was neutral for r50, −0.0001) |
| L4 swap | Used as the submission selector |
| Temperature floor | None. Masking, dropout and the GAP path suffice; train- and eval-mode entropy are logged to catch collapse |

## 9. Outcomes

- **Conditions 1–3 pass:** submit (L4 swap or v3, by the selector). Run C0 and v3-s2.
  - v3-s2 passes → new base.
  - v3 − C0 ≥ +0.003 → selection credited.
- **Conditions 1–2 fail, condition 3 passes:** spatial selection does not add information on this base. Close it;
  move to labels (Exp 7) or cross-slice context.
- **Condition 3 fails:** not tested. Inspect gradients and α. One rerun is allowed, only with a change pre-registered
  before seeing fold-0 metrics: a higher head learning rate for the score MLP, or a non-zero init of its last
  layer.
