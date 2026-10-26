# Exp v3 peer review, round 1 (2026-10-12)

Reviewer: an independent agent with read-only access to the ledger, runner code and audit kernels.
Reviews: `docs/exp_v3_spatial_finding_attention.md`.

**Verdict: major revision. Do not launch Step 1 as written.**

## Factual corrections (all accepted; to be fixed in the proposal)

- **2-sev ranges.** Mild/hedged AUCs are 0.73–0.95, not 0.75–0.88. The share of misses that are mild ranges
  0.53–1.00: medial meniscus 0.53, lateral meniscus 0.73, Baker's 0.76. That band also includes positives the
  report reader missed (reader-absent).
- **"Zero-sum shared pool" is wrong.** v2's window softmax is already per finding (12 attention logits); only the
  256-unit hidden layer is shared. 4b-abl places the trade-off in the backbone features ("no slot mask recovers
  them"). That answers threat #8 against v3, since v3 keeps the shared trunk.
- **2c / 4b do not uniformly hurt contusion and menisci** (2c contusion +0.005; 4b menisci and contusion +0.003).
- **Over-read audits:**
  - 2-side measured only the side share of sagittal attention. "Misses attend like negatives" does not
    discriminate H0 from H1.
  - 2-evid covers detected positives only and says nothing about misses. Mild evidence was more spread out
    (n50 8 vs 6), which argues mildly against the "1–4 cells" picture.
- **Grid size.** Scoring uses the 320 crop, a 10 × 10 grid (100 cells), not 12 × 12.
- **Missing provenance.** The gold-58 ambiguous-cell check (AUC 0.775 / 0.807) has no ledger row or committed
  script yet.
- **Prior spatial-attention attempts** exist beyond the one cited: teammate target-query r34, Pillar-0
  target-query readers, local_global top-k.
- **Identity at initialisation is not exact:** LN-then-pool ≠ pool-then-LN. A zero-initialised last layer has the
  same gradient dynamics as 2a's zero gate. The comparison is moot anyway, because v3 trains from ImageNet with a
  dilated layer 4.
- **Compute.** OS16 is about +60% FLOPs and 4× layer-4 activation memory, not +30–40%.
- **Coordinates are not physical.** `d96_positions.py` rescales each slot to [0, 1] over its own picks; they are
  not millimetres.

## Major concerns

1. **H1 is mismatched to its targets.** The mild-band headroom is in diffuse, extent-graded findings (MCL,
   effusion, PF OA, medial OA). Focal findings (menisci) have the lowest share of mild misses. That is H0's home
   ground.
2. **Dilution is mostly a red herring for a fine-tuned trunk.** The GAP logit equals the mean of the per-cell CAM.
   The backbone can learn to make lesion cells large, and layer-4 receptive fields span far more than one cell.
   Medical-imaging literature: peaky pooling helps when the lesion/FOV ratio is far below 1% (mammography, GMIC),
   but changes classification AUC little on chest X-ray (it mainly helps localisation).
3. **The peaky pool interacts badly with 16-window MIL sampling.** A top-3 evidence window is in the draw only
   46–48% of the time. When it is missing, a per-cell softmax learns to latch onto outlier cells to fit the
   positive label.
4. **The gate optimises the wrong band.**
   - The headline and mild-band AUCs do not track the LB; only the clear band does.
   - v3 targets the mild band, which is 4b's profile (all of its gain was mild; it had the lowest clear-6).
   - The clear-6 floor of 0.9690 equals the weaker 1e seed, which is a weak guard.
5. **The coordinate frame is not feasible as specified.**
   - The manifest stores `ipp`, `iop` and `ps`, and pixels are in DICOM order.
   - But Rows/Columns are not stored, so the crop offset cannot be recovered when the FOV is under 140 mm.
   - LPS coordinates have no anatomical origin.
   - The augmentation (mirror, affine) moves pixels but not the coordinate map.
6. **The train/eval softmax domain differs:** 576 cells including the periphery in training vs 400 at evaluation.
   Effusion and Baker's carry 8–13% of their evidence outside the 320 crop.
7. **Bundling** OS16, attention and coordinates leaves a pass unattributable. Run the minimal version first.

## Step 0 (zero-training top-k / LSE test) is not valid as written

- **It cannot be computed from saved outputs.** The audit kernel saved only clamped row/column marginals, at 384
  px. A new pass that saves the signed per-cell CAM at 320 is needed.
- **A GAP-trained CAM is indifferent to how its mass is spread across cells,** so the test is open to a false
  positive (dropping negative-context cells, artefacts) and a false negative (the CAM was never trained to be
  peaky).
- **The reading is not well-posed:**
  - multiplicity (4 pools × 4 targets);
  - the "mild band" in `audit_d96_native.py` is the else-branch (mild, mild-to-moderate, uncertain,
    reader-absent);
  - "clear band moves < 0.003" is not defined.

## Required changes (ranked)

1. **Replace Step 0 with a frozen-trunk learned-pool test.**
   - Use 1e SWA weights, freeze the backbone, keep OS32.
   - Retrain two heads for 3–4 epochs on the train folds:
     - (A) the GAP head, as the control;
     - (B) per-finding cell attention plus window attention.
   - Score both on fold 0 with a paired bootstrap. About $2–3.
   - This isolates pooling from the backbone.
2. **First training run:** location attention at OS32, no coordinates. Add OS16 only if (1) is positive.
3. **Fix the initialisation description** and add an α-entropy diagnostic: a run whose α stays near-uniform is
   "not tested", not a fail.
4. **Coordinates only after:**
   - a header pass for Rows/Columns;
   - a shared frame from the 2-pre consensus centre;
   - warping the coordinate map with the augmentation grid;
   - a landmark unit test.
5. **New decision rule.**
   - Two seeds, seed-matched to 1e (2026 and 2027), with an early kill at epoch 12.
   - PASS requires all of:
     - two-seed-mean 10-target Δ ≥ +0.003;
     - clear-6 two-seed mean ≥ 0.9705 − 0.001;
     - a mechanistic signature: focal targets gain at least as much as diffuse ones, and α is not uniform;
     - L4 swap Δ ≥ 0.
   - A gain only in the mild band is treated as convention-chasing, not a PASS.

## Reviewer's alternative v3

**Aligned within-series context before pooling:**
- Sample 4 contiguous slabs of 4 windows from the fluid series instead of 16 scattered windows (same image count).
- Add a zero-init residual depthwise slice-axis conv (k = 3) on the layer-3 and layer-4 maps within each slab.
  These are spatially aligned maps from the same series, so this is true 2.5D mixing before GAP.
- Then v2's per-finding window attention.

**Supporting evidence:**
- evidence spans 4–10 windows (2-evid);
- context added after GAP failed (2a);
- slice density helped through training (4b);
- MCL improves at a low slice gap (2-gap, suggestive).

**Distinguish it from the slice-series r34** (LB 0.903), which mixed pooled vectors and changed the input at the
same time.

If the frozen-trunk pool test (change 1) is positive, add per-finding cell attention at OS32 to this design.
If it is null, close H1 and move effort to labels (Exp 7).
