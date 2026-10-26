# R6 teacher and 2.5D ConvNeXt reader: how they are built, and what E4 can borrow (2026-10-21)

Read-only analysis. Nothing was trained, submitted, downloaded or installed. Checkpoints were opened on CPU with `python3 -I`, as key and shape listings only.
Marks: **V** = verified in source, manifest or checkpoint. **I** = inferred.

Path shorthands:
- `R6B` = `data/public_ckpt/rostigor_r6/` (Kaggle dataset rostigor/rsna-knee-r6-fe055406)
- `R6S` = `data/_session_tmp/fm_sample/r6src/`. Its `compact_train.py` has sha256 04df7e2e…, which equals `R6B/manifest.json` src_sha256, and md5 8622009e3db0414124d5614ed252d7a2, which equals the manifest trainer_md5. So this is the exact trainer (v9.8). **V**
- `R7M` = `data/forum_research/r6_copy_compare/heliosli/manifest.json`: R7 (cnx_r7) from the same trainer md5, with its full run_config.
- `RD` = `data/public_ckpt/goodpjw_reader/` (goodpjw2008/rsna-knee-2-5d-convnext-reader)
- `NB` = `data/forum_research/nb_1019/goodpjw2008_rsna-knee-coatnet-r6-2-5d-mil-lb-0-958/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958.ipynb` (the 0.958 blend)

## 0. Where each model matters in the 0.958 blend

| Fact | Source | Mark |
|---|---|---|
| Raptor slot = 0.65·rank(384) + 0.35·rank(224). Raptor weight w per finding: Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35, every other finding 0.5. R6 gets 1 − w. | NB blend cell | V |
| R6 carries most weight on **Lateral OA (0.8)** and Baker's (0.65), and half on most other findings. | NB | V |
| Reader weight is 1/10 of the slot (0.003–0.035 per finding). goodpjw: at full weight the reader no longer adds (0.956 base + reader gave 0.954). | NB markdown + metadata | V |
| Reader alone LB 0.929; 0.944 as a 30% blend member in an earlier blend. | RD dataset metadata | V |

So R6 is the teacher worth dissecting. The reader is nearly decorative in the 0.958 blend.

## 1. R6 architecture (`cnx_r6fm_c352`, arm `E_lrprior`)

| Part | What it is | Source | Mark |
|---|---|---|---|
| Backbone | convnext_small, init `convnext_small_in22k_ft_in1k`; 50.07 M params | `R6B/manifest.json`; checkpoint `E_lrprior_s0_f0.pt` keys `enc.*` | V |
| Input unit | 2.5D window: 3 adjacent slices as channels. Physical slice step 3.5 mm: stacks thinner than 1.75 mm are averaged into 3.5 mm slabs. Intensity `series_companded`. | manifest `slices 3, phys_mm 3.5`; `R6S/px_cache.py:19-21` | V |
| Cache and crop | Cache 384 px; random crop 352 in training (`augment`, `compact_train.py:2428-2432`) | manifest size 384 / crop 352 | V |
| Views (v4) | Per series: full field (view 0) plus 100 mm **knee-anchored** locals. Sagittal/axial: centre (1), patellofemoral at the patella anchor (4), posterior / popliteal (5). Coronal: medial and lateral locals at ±30 mm (2/3). The anchor is the joint centre from an SKM-TEA segmentation on the sagittal series (`anatomy.pt`), with a slab prior as fallback. | `px_cache.py:32-42, 83-100`; `knee_anchor_bank_v2.py` header | V |
| Per-window tokens | enc → proj 768→384 + LN, plus slot Emb(10), view Emb(6), geom Linear(4) (rel_pos, log_spacing, extent, fluid), then win_proj and out_norm | ckpt keys; `Studies` `compact_train.py:179-264` | V |
| Spatial attention inside each window | Per class, a query `sp_q`(12,128) attends over the feature map, with keys `sp_key`(128,768) and a fixed sin/cos 2-D position `sp_pos`(128,16). | `spatial_pool` `:541-563`; ckpt | V |
| MIL over windows | Per class, attention `q_win`(12,384) over every window of every series and view. Then `cls_w`·study + `bias`. | `head` `:603-642` | V |
| Extent head (OA) | z += ext_gamma·log1p(area/100). Area = sagittal × coronal extent (medial/lateral OA) or axial × sagittal (PF OA). Learned ext_gamma 0.37 / 0.41 / 0.51. | `extent_area` `:565-601`; ckpt | V |
| Report-state head | `state_w`(12,8,384) predicts each finding's report state (5–8 states, e.g. ACL complete_tear, Effusion moderate). `bin_w`(13,384) predicts binaries (osteophytes, subchondral marrow, bone-on-bone, acute). Evidence term `logit(Σ p(s)·prior_s) − base_logit` is added through a learned `state_gate` (Effusion 0.51). | `state_evidence` `:462-466`; `R6B/state_spec.json`; ckpt | V |
| Synovitis decomposition | Synovitis logit partly built from Effusion etc.; learned `syn_beta` 0.27 | `_decompose_syn` `:644-655`; ckpt | V |
| fastMRI+ box auxiliary | v9.8 fm-aux: dense + presence BCE on fastMRI+ lesion boxes, λ 0.3, 96 windows/step, on the shared encoder. The tag "r6fm" points to it, but the R6 manifest has no fm_aux key, and forum post m_749858 notes the same. R7 has `fm_aux fm384_v1, weight 0.3`. | `compact_train.py:1-33, 1686`; `R7M` | **I** for R6 |
| Folds | 4 folds by `sha1(uid)[:8] % 4`; pool 4344; gold 58 never in training (`official_labels_read_by_trainer false`) | `:2015`; manifest | V |
| Side handling | (a) Canonical frame: lateral at +column. heliosli's geometry code flips coronal/axial images for right knees, `flip_for_plane(plane, side): return plane != "Sagittal" and side == "R"`, with side from the Laterality tag or sagittal IPP x. (b) `side_cls.pt` (timm resnet34, 1-channel input) flags mirrored or wrong-side studies at threshold 2.0. | `data/public_ckpt/heliosli_orthofoundation/cloud_code/training/dicom_geometry.py:630`; `native_cache.py` ~75-100; `R6S/side_common.py`; `R6B/side_cls.pt` | V for the code, I that R6's cache used the same path (R7 has `px_override sidefix`) |
| Inference | Mean logit over the 4 folds. Every window at eval_stride 1. Flip TTA with the medial↔lateral output permutation (MCL from the plain pass). The fluid and state stackers then replace the logit of the class each one adopts. | `R6S/compact_infer.py:1-50`; our wrapper `kaggle/submission_kernels/pubkd_r6_teacher/s0/teacher.py` (FLIP_TTA = True) | V |
| Fluid stacker (Effusion) | z_out = −3.77 + 0.33·z + 0.64·log1p(mL) + 2.30·miss. mL comes from a sagittal fluid-sensitive p99 threshold with a 1.5 mm opening. Cross-fitted gain only **+0.0008 / +0.0009**; adopted by user override against its own +0.003 rule. | `R6B/fluid_stacker.json` | V |
| State stacker (Baker's only) | Logistic regression on [z, state probs]: coefficients z 0.14, none −2.31, small −0.95, present_unspec 1.29, moderate 5.08, large 2.58, ruptured 5.33; b0 −0.96. Gain **+0.038 on the explicit ruler / +0.0077 on the corrected ruler**. Every other class was dropped (e.g. Synovitis −0.014). | `R6B/state_stacker.json` | V |

## 2. R6 training pipeline

| Item | Value | Source | Mark |
|---|---|---|---|
| Labels | `probe_targets_opusv11b_rr6_plr6_cdr.csv`: LLM report-state extraction (spec `opusv11b_rr6`, n 4407), with prior = P(official positive \| state) (e.g. ACL complete_tear 0.934, Effusion moderate 0.85). Certainty weights: definite 1.0, probable 0.7, possible 0.4. Many cells are unknown (no_state: Synovitis 2830, Baker's 2261). | manifest; `state_spec.json` | V |
| Main loss: LR loss | `per = -logaddexp(logsigmoid(z) + log(lr), logsigmoid(-z))`, i.e. the likelihood of a report likelihood ratio, not soft BCE. Arm `E_lrprior = lr:lr_prior`. bias0 initialised from the prior. | `compact_train.py:679-685, 2034-2054` | V |
| State loss | Masked CE over states + BCE on binaries, weight 0.5 | `state_loss` `:688-703`; R7M `state_weight 0.5` | V code / I weight |
| Matched-pair ranking | Families: Lateral/Medial meniscus, the 3 OA findings, Synovitis, MCL, ACL; v9.x adds Effusion, Baker's, Contusion and Fracture grade boundaries. Pairs match on co-finding signature. Loss w·softplus(q_neg − q_pos) with w = (prior_i − prior_j)·min(cert). pair_rank 0.5, pair_frac 0.33. | `:738-850, 882-907, 2424-2427`; R7M | V code / I R6 weights |
| Rare boost | WeightedRandomSampler, weight 1 + 2·MCL(acute) + 2·Fracture + PF OA high + Baker's + Contusion, cap 4 | `:965-1001`; R7M `rare_boost true` | V code / I flag |
| Mirror | p 0.5. In the canonical frame, swap Medial↔Lateral meniscus/OA labels and outputs. MCL is not trained on mirrored batches. | `mirror_perm/mirror_batch/mirror_outputs :1003-1050`; R7M `mirror 0.5` | V |
| Sampling | `cover`: windows at stride 3 × phys step with a random phase; **176 training windows per study** (all series × all views); slice_jitter 0.25 | `covering_centers/window_rows :134-176`; manifest `train_windows 176`; R7M | V |
| Augmentation | Random crop 352/384, rotation ±10°, scale 0.9–1.1, gamma 0.8–1.25, contrast 0.9–1.1. No flips except the mirror. | `augment :2428-2468`; `:1570` | V |
| Optimiser | AdamW, lr 1e-4 (encoder) / 1e-3 (head), wd 0.05, 5% warm-up + cosine, clip 5, **16 epochs, batch 2**, EMA (warm (1+s)/(10+s), 0.999 in R7) | manifest; `:2706-2733`; R7M | V / I (EMA value) |
| Data | Competition pool only (4344, review holdout included); fastMRI enters only through the box auxiliary, if R6 has it | manifest; R7M | V / I |

Reference point: nartaa retrained heliosli's S6 compact pipeline on their own labels and reached **LB 0.950 single** (ConvNeXt-S, spatial attention + window MIL, full field + 100 mm SKM-TEA-anchored locals, 320 cache cropped to 288, 64 windows, 16 epochs, all 4263 studies). Forum m_749359 (V, forum claim). The trainer family is therefore worth ~0.950 as a single model even without the R6 labels.

## 3. 2.5D ConvNeXt reader (`cnxt_v0_fold0.pt`)

| Item | Value | Source | Mark |
|---|---|---|---|
| Series | 6 slots = plane × fat-sat (FS first); largest series per slot | `RD/knee.py:13-30` | V |
| Preprocess | Canonical LPS in-plane orientation from IOP, 0.4 mm/px, 384 px (153.6 mm), max 64 slices, 1–99.5 percentile window. **No laterality flip.** | `RD/preprocess.py` | V |
| Windows | 3-slice windows. k stratified-random in training (k 8), linspace at eval (k 12), over the 4–96% slice range | `knee.py:39-49, 73-74`; ckpt args | V |
| Views | Centred 0.92 FOV crop → **256 px**. Augmentation: rotation ±12°, scale ±12%, shift ±6%, gamma/gain shared per series; no flip | `make_views knee.py:84-109` | V |
| Network | ConvNeXt-T `fb_in22k_ft_in1k` (drop_path 0.1) → Dropout + Linear 384 + LN, plus slot emb + MLP slice-position emb → **2-layer TransformerEncoder** (8 heads, norm_first) over all windows → per-finding tanh attention (256) → per-class weights; 30.7 M params | `knee.py:119-155`; ckpt args | V |
| Loss and schedule | Soft BCE; AdamW 1e-4 / head 5e-4, wd 0.02; 1-epoch warm-up + cosine; clip 3; EMA 0.998 (weights saved as EMA); bf16; 14 epochs, bs 8 | `RD/train.py:50-164` | V |
| Labels | v0 = mean of 4 public LLM tables (steven v4 blend, pilkwang v2, lixin73 gpt56sol, laymond qwen3-8b), clipped 0.01–0.99. 5-fold GroupKFold grouped by normalised report text. Gold used for monitoring only. | `RD/make_labels.py:15-67`; ckpt `labels_v0` | V |
| Release | 3 of 5 folds | RD file list | V |

## 4. Side-by-side with E4

| Axis | R6 | Reader | E4 student (efficiency entry) |
|---|---|---|---|
| Backbone | ConvNeXt-S in22k (50 M) | ConvNeXt-T in22k (31 M) | ResNet-50 tv_in1k + copied layer4 for crops (`scripts/r50_gl4_runner.py`) |
| Field / px | full field + **100 mm locals at 384→352 px (~3.5 px/mm)** | 0.92 × 153.6 mm at 256 px (~1.8 px/mm) | 140 mm at 384 px (2.74 px/mm); eval centre 320 (`fast_global/submission.py:39`) |
| Anatomy anchoring | SKM-TEA joint centre, patella and posterior locals | none | MS1-placed crops at training only; global head only at inference |
| Laterality | canonical flip (lateral = +col) + side classifier | none | none: the d96 cache records the tag but does not canonicalise; tag present in only 50% of train studies (`data/dicom_demographics/header_summary.json`) |
| Train windows | **176** (cover, all views) | 8 | **16** stratified random (KD-W48 testing 48) |
| Eval windows | every slice (stride 1), all views | 12 | 94 |
| In-window pooling | per-class spatial attention | GAP | GAP |
| Across windows | per-class attention | transformer + per-class attention | per-class attention |
| Main loss | LR loss on report-state priors | soft BCE | soft BCE on KD targets (mix73 of the full public blend) + 0.5 local BCE |
| Auxiliary losses | state CE + binaries, matched-pair ranking, extent, fastMRI+ boxes (I) | none | none |
| Sampler | rare boost (cap 4) | uniform | uniform |
| Mirror | canonical frame + M↔L label swap | none | anatomical mirror p 0.5, no label swap (`coatnet_d96_runner.py:614-631`) |
| Schedule | 16 ep, EMA | 14 ep, EMA 0.998 | 25 ep, SWA 21–25 |
| Ensemble / TTA | 4 folds, flip TTA | 3 folds | 1 model, no TTA |
| Post-hoc | Effusion fluid stacker, Baker's state stacker | none | none |

E4's per-finding gap to its 384 teacher, on Steven-confident fold-0 cells (`data/coatnet_d96_fold0/kd_gap_analysis/kd_gap_cpu.txt:69-82`, V): E4 closes **0%** of the v4→teacher gap on PF OA, **−6%** on ACL and **4%** on Fracture, against 37–54% on Medial OA, Lateral OA and Baker's. PF OA and ACL are exactly where R6 adds dedicated anchored locals (the patella local, the sagittal centre local). The PF OA motivation is spelled out in `px_cache.py:32-35` ("PF OA is the class with the largest report-vs-image gap… 0.92 vs 0.82").

## 5. Differences ranked by likely AUC impact, with a ≤ 27 h action each

Deadline: 2026-10-22 23:59 UTC = 2026-10-23 01:59 CEST.

| # | Difference | Evidence | Already tried? | Concrete ≤ 27 h action |
|---|---|---|---|---|
| 1 | **Training-view coverage**: R6 sees 176 windows/step, E4 sees 16 | KD-VIEW: E4's r16_full view has teacher KL 0.029 and costs E4 −0.0155 AUC; r48 has KL 0.0017 and covers 0.98 of teacher attention (`experiment_ledger.md:343`). R6 and nartaa (64 windows, LB 0.950) both train dense. Strong. | **RE-RUN: KD-W48 is running** (`ledger:346`, pre-registered +0.002 rule); KD-CV implemented (`:344`) | No new run. Wait for KD-W48. If it passes, the next run is E4-ALL with 48 windows (Colab A100, ~3× E4 compute). The fast kernel is unchanged: it already evaluates 94 windows. |
| 2 | **High-res anchored locals** (100 mm at ~3.5 px/mm; PF / patella and posterior views) | R6 carries 0.8 of Lateral OA and 0.4 of PF OA in the blend (NB); E4 closes 0% of the PF OA gap and −6% of ACL (`kd_gap_cpu.txt:71,77`); R6's own note on PF OA (`px_cache.py:32-35`). Medium. | Partly: L4C-LOC learned locator LB 0.938 (`ledger:293`); r50-GL v3 crops from the d96 cache = no new pixels (ruled out). **Fixed-fraction locals cut from native DICOM were never tried.** | **Accuracy pick only.** Build a 100 mm local cache at 384 px from DICOM with R6's *fixed* centres (`PF_CENTER` sag (.38,.60), ax (.38,.50); `POST_CENTER` sag (.55,.35)), with no locator and no MS1 at inference. Train it as an extra slot in the l4copy local branch on mix73 targets. About 4–6 h cache on Colab + 1 E4-length run. Too slow for the 13 s kernel (a second pass per view). |
| 3 | **Laterality canonicalisation** (lateral = +col) + M↔L-swap mirror | R6 and heliosli use it (`dicom_geometry.py:630`). E4 must infer side from anatomy, and our anatomical mirror (no swap) doubles the orientations it must handle. Gain unknown. Weak. | **New, never tried.** Mirror TTA on r50 left the LB unchanged (`ledger:38`), but that tests something else. | Run a cheap diagnostic first (CPU, 0 GPU): split E4's fold-0 OOF AUC for Medial/Lateral Meniscus, Medial/Lateral OA and MCL by left vs right knee. Side from the Laterality tag, else the sign of the sagittal IPP x. If one side trails by ≥ 0.01 on the M/L findings, retrain E4 with canonical flips. In the kernel this costs one header read and a flip (~0 s). If the sides are equal, drop it. |
| 4 | **Matched-pair ranking + rare-boost sampler** | R6 trains with both (I for R6, V for R7). Our only test was macro confident-pair ranking on r34 (LB 0.906 = BCE, `ledger:63,93`), which is weaker than R6's signature-matched, prior-weighted pairs. Rare boost is untested. Weak to medium; it helps MCL and Fracture, where E4's gaps are largest in absolute terms. | Pair rank: **partial re-run** (different design). Rare boost: new. | Stack only after KD-W48. One E4 run with a WeightedRandomSampler (1 + 2·MCL + 2·Fracture + Baker's + Contusion, cap 4) on KD target bands (e.g. teacher prob > 0.5). The pair loss needs report states we lack in the KD targets, so skip it. |
| 5 | Per-class **spatial attention** inside each window (vs GAP) | R6 ckpt `sp_q/sp_key/sp_pos`; reader and E4 use GAP. Unknown in isolation. | Not tried as such (2a was context *between* windows) | Too large a change for 27 h, given the risk to the 13 s kernel. Note it for later. |
| 6 | Report-state head + Baker's state stacker | +0.0077 Baker's on the corrected ruler (`state_stacker.json`). The KD targets already carry the blend's Baker's ordering. | P1-sev severity heads **FAIL** (`ledger:241`) | None. A re-run of P1-sev in another form. |
| 7 | Effusion fluid stacker | +0.0008 cross-fitted (`fluid_stacker.json`) | new but negligible | None (≈ +0.0001 macro). |
| 8 | LR loss on report priors | A label-side device. KD targets are already calibrated teacher probabilities. | n/a | None. |
| 9 | Flip TTA | R6 uses it | **Tried**: mirror TTA on r50, LB unchanged (`ledger:38`) | None. It would also cost kernel time. |
| 10 | Cross-window transformer (reader) | Reader is ≤ 0.035 weight in the blend; LB 0.929 alone | **Tried**: 2a, no gain (`ledger:200`) | None. |
| 11 | 4-fold ensemble / EMA | Standard | EMA/schedule ruled out; lean-ensemble preference | None for the efficiency entry. |

## 6. Bottom line

The levers R6 has and E4 lacks are (i) dense training-view coverage and (ii) anatomy-anchored high-resolution locals. KD-W48 is already testing (i). (ii) is the only untried idea with per-finding evidence (PF OA and ACL gaps), but it fits the accuracy pick, not the 13 s kernel. Laterality canonicalisation is the one new, nearly free kernel-side change. Run its left/right OOF diagnostic before spending GPU on it. R6's report-state machinery (state head, LR loss, stackers) is already folded into the KD targets, or was tried and failed (P1-sev). Copying it is not worth the deadline time.
