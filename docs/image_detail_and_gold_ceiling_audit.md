# Image detail, report ceiling and fold-0 ensemble audit (2026-10-07)

CPU-only analysis. No training or GPU inference, and no files outside the output folder were modified. Gold labels
were used **only to score** report readers: no gold clause text was printed or saved, and nothing here feeds a label
build or a model. Everything ran in the offline container (`--network=none`, project mounted read-only, `python -I`).

**Conclusion in one line:** the evidence supports prioritising **image detail** (in-plane resolution) as the next
single-factor experiment. It does **not** justify stopping all label work, and it does not justify relaxing the OA
labels at the same time. Label questions stay open and should be tested separately, so each effect stays attributable.

## Artifacts

| What | Where |
| --- | --- |
| Scripts (tracked) | `scripts/audit_fold0_ensemble.py`, `scripts/audit_gold_report_ceiling.py`, `scripts/audit_cache_input_coverage.py` |
| Exact outputs, logs, montages (gitignored data dir) | `data/image_detail_audit_2026-10-07/` (with `SHA256SUMS`) |
| Montage row → study / series / slot | `data/image_detail_audit_2026-10-07/montage_index.csv` |

Reproduce from the project root (`P` = the project root, `OUT=$P/repository/data/image_detail_audit_2026-10-07`):

```
for s in audit_fold0_ensemble audit_gold_report_ceiling audit_cache_input_coverage; do
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/$s.py /proj /out > "$OUT/$s.log" 2>&1
done
```

The rerun produced the same numbers as the first exploratory run, and the montages are byte-identical. Input hashes are
in `fold0_ensemble_summary.json`, `gold_ceiling_meta.json` and `cache_input_coverage.json`.

**Note on the label file.** The 0.925 ResNet50 run trained on `soft_labels_hpo_v3_1_d2_structured.json` (sha256
`38367d62…`). That is v3.1 plus five minimal/trace synovitis cells promoted from BORD to POS
(`docs/hpo_v31_d2_kaggle_experiment.md`). It is called "v31_d2" below.

## 1. Fold-0 ensemble of existing models (`fold0_auc_single_and_ensemble.csv`)

- Data: saved predictions of 7 models on the fixed 825 fold-0 studies.
- Reference: Steven-v4, with the trainer's confidence mask (≤0.3 or ≥0.7, positive weight).

**Macro AUC:**

| Model | Macro AUC |
| --- | --- |
| resnet_hpo_v31_d2 | 0.8963 |
| resnet_hpo_v1 | 0.8909 |
| resnet_hpo_v2 | 0.8897 |
| plane_split_v1 | 0.8878 |
| topk_pairwise | 0.8772 |
| topk_plateau60 | 0.8729 |
| topk_v1 | 0.8696 |
| **rank average, v1 + v31_d2** | **0.9022** |
| logit average, v1 + v31_d2 | 0.9022 |
| rank average, top 3 | 0.9033 |
| rank average, all 7 | 0.9030 |

**Two-model ensemble vs v31_d2 alone:** +0.00591. The paired study bootstrap (500 draws) gives a 95% interval of
[+0.00298, +0.00851].

**Per target:**

| Target | Reference positives | v1 | v31_d2 | Ensemble v1 + v31_d2 (rank) |
| --- | ---: | ---: | ---: | ---: |
| ACL | 137 | 0.8835 | 0.8877 | 0.8968 |
| MCL | 103 | 0.7926 | 0.8322 | 0.8252 |
| Medial Meniscus | 313 | 0.9608 | 0.9641 | 0.9664 |
| Lateral Meniscus | 97 | 0.8967 | 0.9200 | 0.9187 |
| Medial OA | 300 | 0.8914 | 0.8869 | 0.8959 |
| Lateral OA | 220 | 0.8443 | 0.8500 | 0.8581 |
| PF OA | 383 | 0.8673 | 0.8730 | 0.8791 |
| Effusion | 487 | 0.8646 | 0.8630 | 0.8715 |
| Synovitis | 115 | 0.9635 | 0.9510 | 0.9625 |
| Baker's | 218 | 0.9085 | 0.8997 | 0.9110 |
| Contusion | 124 | 0.9134 | 0.9061 | 0.9180 |
| Fracture | 53 | 0.9043 | 0.9222 | 0.9236 |

**Pattern across all 7 models.** MCL (0.75–0.83), Lateral OA (0.81–0.85), Effusion (0.85–0.87) and PF OA (0.84–0.88)
stay low for every architecture and every label set. Prediction rank correlation between the two ResNets averages 0.85
over targets (`fold0_prediction_rank_corr_vs_v31d2.csv`).

**Caveats:**
- Every checkpoint was selected on these same 825 studies, so absolute values are optimistic.
- Single seed per model, so part of the ensemble gain may be seed averaging rather than label diversity.
- The reference is report-derived, not gold.

## 2. Report readers vs the 58 gold labels (`gold_auc_by_reader.csv`)

**Method.** Readers were run on the gold studies' own reports from `train.csv`:
- the rule reader at each v3.1 correction step;
- the three LLM-derived readers that cover the gold IDs (vmohitrao covers none);
- a v3-style mean blend of those readers. HPO-v1 has no gold rows, so it can't be included.

A silent reader scores 0.04. The gold set is enriched with positives: 9–35 per target, and only 9 for MCL.

| Target | Gold positives | rules v3.0 | rules v3.1 | teammate | pilkwang | laymond | blend (v3.0 rules) | blend (v3.1 rules) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 24 | 0.938 | 0.938 | 0.922 | 0.847 | 0.897 | 0.955 | 0.955 |
| MCL | 9 | 0.985 | 0.985 | 0.976 | 0.908 | 0.908 | 0.978 | 0.978 |
| Medial Meniscus | 26 | 0.877 | 0.877 | 0.879 | 0.871 | 0.840 | 0.920 | 0.920 |
| Lateral Meniscus | 23 | 0.852 | 0.852 | 0.789 | 0.792 | 0.829 | 0.858 | 0.858 |
| Medial OA | 15 | 0.805 | 0.802 | 0.913 | 0.874 | 0.862 | 0.949 | 0.943 |
| Lateral OA | 11 | 0.821 | 0.807 | 0.889 | 0.835 | 0.831 | 0.882 | 0.866 |
| PF OA | 21 | 0.759 | 0.744 | 0.843 | 0.773 | 0.723 | 0.876 | 0.868 |
| Effusion | 35 | 0.796 | 0.794 | 0.750 | 0.638 | 0.708 | 0.839 | 0.839 |
| Synovitis | 27 | 0.659 | 0.660 | 0.722 | 0.692 | 0.696 | 0.758 | 0.758 |
| Baker's | 12 | 0.934 | 0.928 | 0.890 | 0.904 | 0.892 | 0.913 | 0.912 |
| Contusion | 19 | 0.769 | 0.769 | 0.789 | 0.665 | 0.728 | 0.759 | 0.759 |
| Fracture | 18 | 0.829 | 0.829 | 0.831 | 0.786 | 0.811 | 0.822 | 0.822 |
| **Macro** | | 0.835 | 0.832 | 0.849 | 0.799 | 0.810 | **0.876** | **0.873** |

**v3.1 vs v3.0 rules in the blend** (2,000-draw bootstrap, `gold_blend_v31_vs_v30_bootstrap.csv`):

| Target | Δ (v3.1 − v3.0) | 95% interval |
| --- | --- | --- |
| Medial OA | −0.0054 | [−0.0232, 0.0000] |
| Lateral OA | −0.0155 | [−0.0488, +0.0032] |
| PF OA | −0.0084 | [−0.0301, +0.0066] |
| Baker's | −0.0009 | [−0.0054, 0.0000] |
| Synovitis | +0.0006 | [−0.0190, +0.0190] |
| Macro | −0.0025 | [−0.0058, +0.0002] |

All other targets are unchanged.

**Rule state by gold label** (`gold_rule_state_by_gold_label.csv`), selected rows:

| Target | Gold label | Rule-state counts |
| --- | --- | --- |
| MCL | positive (9) | BORD 5, POS 3, BELOW 1 |
| MCL | negative | BELOW 6, BORD 1, NEG 16, SIL 26 |
| Effusion | positive (35) | BORD 18, BELOW 9, POS 5, HEDGE 1, SIL 2 |
| Effusion | negative | BELOW 13, NEG 5, BORD 2, POS 1, SIL 2 |

**What this supports:**
- The report-reader blend reaches about 0.87–0.88 macro AUC against gold. The image model's public score is 0.925, on
  a different, hidden set. Report-derived references, including Steven-v4 validation, have a limited ceiling against
  gold. This helps explain why validation gains move the public score only weakly.
- MCL reports separate gold almost perfectly (0.98), yet image models reach only about 0.83 against report labels on
  fold 0. For MCL, the image input is the more likely bottleneck, not the label wording.

**What this does not establish:**
- **That label work should stop.**
  - Synovitis (0.66–0.76), Effusion (0.84), Contusion (0.76) and Lateral Meniscus (0.86) agree poorly with gold, but
    this audit doesn't show whether better label construction could close that gap.
  - The gold set is 58 enriched studies with wide intervals, and it is not the public test distribution.
- **That the OA labels should be relaxed now.**
  - The v3.1 OA deltas are negative on all three compartments, consistent in direction with the host's "focal
    high-grade loss = positive" example.
  - But every interval includes zero.
  - Changing OA labels together with an image change would confound both. It belongs in a separate, later experiment.
- **That severity labelling for Effusion is wrong.**
  - Nine of 35 gold positives were reported as "small" (rule BELOW), which argues against stricter effusion severity
    labels.
  - The counts are small, and this is one enriched set. The question stays open.

## 3. What the 140 mm / 224 px cache gives the model (`cache_input_coverage.json`, montages)

**Measured** (all 4,349 report-only studies, 17,395 series slots):

| Measure | Result |
| --- | --- |
| Resolution | 140 mm field of view at 224 px = **0.625 mm per pixel**. Median native in-plane spacing is **0.324 mm**, so the input is downsampled about 2×. |
| Slots per study | 4 (one study has 3) |
| Studies with a coronal fluid-sensitive slot | 96.4% |
| Studies with a sagittal fluid-sensitive slot | 94.1% |
| Studies with an axial fluid-sensitive slot | 100% |
| Slice positions | Central 20–80% of each series by design; 3.1% of positions fall more than 0.02 outside that band |
| Crop that is real image (400-slot sample) | 0.99–1.00 for every plane |

**Montages.** Slices shown are slot positions 2, 5, 8 and 10 of 12. Example studies were picked by v3.1 report-rule
state, not by gold. The observations are qualitative and by a non-radiologist reader:

- **`mcl_pos_coronal_fs.png`** (6 studies whose report the rule reader marks MCL POS):
  - The medial soft tissue is generally inside the crop, sometimes with little margin.
  - At 0.625 mm per pixel, the MCL is only a few pixels thick.
  - **Row 4** (study `…41698150…`) is tagged coronal fluid-sensitive, but its slices look sagittal: a likely metadata
    error.
- **`baker_Sag-FS.png`** (4 studies, Baker's LIKELY or POS):
  - The cyst lies at the posterior edge of the crop. In rows 1, 2 and 4 the collection reaches, or nearly reaches, the
    crop boundary.
  - **Row 3** (study `…71744523…`) is tagged fluid-sensitive but looks non-fat-suppressed: a likely metadata error.
- **`baker_Cor-FS.png`:** the cyst is not evident in the central-band coronal slices shown, as expected, since it lies
  in posterior slices outside the band.
- **`baker_Axi-FS.png`:** in rows 1 and 2, a posteromedial fluid collection appears whole inside the crop.
  - Every study has an axial fluid-sensitive slot, so this **partly offsets** the sagittal clipping.
  - Clipping is therefore a secondary concern for Baker's, not a primary one.

The sample sizes (6 and 4 slots) can't estimate how often slots are mislabelled. They only show that mislabelling
happens.

**What this supports:**
- In-plane resolution is a plausible common cause for the two thinnest-structure target groups, MCL and the OA
  compartments, being weak across every model and label set.
- The 288 px config (`cnn_topk_experiment/configs/cnn_highres.json`) exists, but no result for it was found in the
  docs. Its note says competitors report 336 px with 64 slices.

## Recommended next steps

1. **Single-factor image-detail experiment.** Raise in-plane resolution, for example 140 mm at 336–384 px, about
   0.36–0.42 mm per pixel. Hold the v31_d2 labels, split, recipe and seed fixed.
   - Primary readouts: MCL, Lateral OA, PF OA and Medial OA.
   - Check memory and time first.
2. **Before or alongside it, on CPU on Kaggle:** measure the slot plane and fat-suppression misclassification rate from
   DICOM headers (ImageOrientationPatient, sequence and fat-suppression tags), not series descriptions.
3. **Label questions stay open,** each to be tested as its own experiment after the image-detail result:
   - OA definition relaxation, following the host examples;
   - Synovitis supervision. Section 4 makes this the top label question.

   Effusion severity treatment is lower priority after section 4. Gold is now development evidence only.
4. The two-model prediction ensemble remains a separate, cheap submission candidate. Check Kaggle inference time.

## 4. Image models vs report readers on the 58 gold studies (`gold_image_model/`)

Added later the same day. This is the per-target question "label-limited or image-limited?".

**Setup**
- **Inference only.** Run on the local GPU in `docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` with
  `--network=none` and a read-only project mount.
- **Gold cache.** `data/kaggle_cli/cnn_gold_cache_artifacts/slice_cache` has the same cache config as training
  (`cnn_topk_support_v1`, 140 mm / 224 px).
- **No gold in training.** None of the 58 gold studies is in the training cache or folds.
- **Excluded.** Gold-tuned runs (`cnn_topk_gold_dpo`, `cnn_topk_gold_finetune`).
- **Code.** Model and batch code are imported unchanged from the hpo_v31_d2 training script.
- **Fidelity.** Re-scoring 32 fold-0 studies reproduces each saved out-of-fold file. Rank correlation is
  0.9999–1.0 and the maximum absolute difference is ≤ 0.00055. This also identifies
  `hpo_blend_resnet50_submission` as the HPO-v1 ResNet50 run.
- **Test reports.** The public `test.csv` has `StudyInstanceUID` only, so reports cannot be used at test time. The
  combined report-plus-model column below is diagnostic only.

**Scripts**
- `scripts/gold_image_model_predictions.py`: GPU inference. It writes predictions and never reads gold labels.
- `scripts/audit_gold_image_model.py`: scoring and bootstrap, 2,000 draws, seed 0.
- `scripts/audit_gold_synovitis_effusion.py`: Synovitis diagnostic.

Primary model is the v31_d2 ResNet50 that scored 0.925 public.

| Target | Gold + | Report blend | v31_d2 | HPO-v1 r50 | HPO-v2 r50 | 3×r50 rank ens | Model − report [95% CI] | Report adds to model [95% CI] | v31_d2 fold-0 (Steven-v4) |
|---|---|---|---|---|---|---|---|---|---|
| ACL | 24 | 0.955 | 0.960 | 0.951 | 0.946 | 0.968 | +0.004 [−0.067, +0.072] | +0.031 [−0.001, +0.077] | 0.888 |
| MCL | 9 | **0.978** | 0.896 | 0.930 | 0.918 | 0.933 | −0.083 [−0.221, +0.022] | **+0.079 [+0.011, +0.176]** | 0.832 |
| Medial Meniscus | 26 | 0.920 | 0.933 | 0.927 | 0.927 | 0.944 | +0.013 [−0.062, +0.090] | +0.027 [−0.007, +0.071] | 0.964 |
| Lateral Meniscus | 23 | 0.858 | 0.870 | 0.894 | 0.867 | 0.901 | +0.012 [−0.108, +0.126] | +0.040 [−0.018, +0.108] | 0.920 |
| Medial OA | 15 | 0.943 | 0.960 | 0.926 | 0.969 | 0.948 | +0.016 [−0.030, +0.068] | +0.012 [−0.009, +0.045] | 0.887 |
| Lateral OA | 11 | **0.867** | 0.791 | 0.797 | 0.756 | 0.808 | −0.075 [−0.221, +0.051] | +0.043 [−0.020, +0.125] | 0.850 |
| PF OA | 21 | 0.868 | 0.829 | 0.839 | 0.871 | 0.841 | −0.039 [−0.149, +0.075] | +0.047 [−0.006, +0.108] | 0.873 |
| Effusion | 35 | 0.839 | **0.945** | 0.950 | 0.955 | 0.959 | +0.106 [−0.001, +0.224] | −0.010 [−0.082, +0.059] | 0.863 |
| Synovitis | 27 | 0.758 | 0.723 | **0.799** | 0.786 | 0.763 | −0.035 [−0.201, +0.129] | **+0.082 [0.000, +0.170]** | **0.951** |
| Baker's | 12 | 0.912 | **0.958** | 0.980 | 0.971 | 0.970 | **+0.046 [+0.006, +0.101]** | −0.005 [−0.021, +0.007] | 0.900 |
| Contusion | 19 | 0.759 | **0.908** | 0.908 | 0.930 | 0.924 | **+0.149 [+0.043, +0.263]** | −0.026 [−0.081, +0.021] | 0.906 |
| Fracture | 18 | 0.822 | 0.853 | 0.878 | 0.907 | 0.869 | +0.031 [−0.122, +0.167] | +0.043 [−0.030, +0.140] | 0.922 |
| **Macro** | | 0.873 | 0.885 | 0.898 | 0.900 | 0.902 | +0.012 [−0.023, +0.046] | **+0.030 [+0.010, +0.052]** | 0.896 |

The other checkpoints are in `gold_auc_image_vs_report.csv`. On gold, the ResNet34 runs score 0.868–0.899 macro.

**Recovery of report errors by the primary model** (`gold_model_recovery_of_report_errors.csv`)
- **Gold negatives the reports over-call** (blend ≥ 0.5) are ranked below gold positives at a pooled AUC of 0.80.
  - Effusion: model 0.92 vs report 0.72.
  - Contusion: model 0.82 vs report 0.50.
  - Baker's: model 0.91 vs report 0.67.
  - ACL: model 0.89 vs report 0.78.
- **Gold positives the reports miss** (blend < 0.5) are ranked above gold negatives at a pooled AUC of only 0.66. The
  main case is Synovitis, with 10 missed positives at 0.67.

**Synovitis diagnostic** (`audit_gold_synovitis_effusion.log`)

| Measure | v31_d2 | HPO-v1 |
|---|---|---|
| Synovitis head on gold Synovitis | 0.723 | 0.799 |
| Model's Effusion score on gold Synovitis | 0.722 | 0.718 |
| Synovitis head within gold-Effusion-positive studies (n = 35) | 0.549 | 0.661 |

- Gold Synovitis × Effusion: 22 of 27 synovitis-positive studies are also effusion-positive.
- Prevalence on gold: Synovitis 47%, Effusion 60%. In the v3.1 training labels, Synovitis ≥ 0.5 is 15%.

**Reading** (58 studies, wide intervals, so read the direction rather than the size):

- **Main finding: reports and images make complementary errors.**
  - Combining them improves gold macro AUC (+0.030 [+0.010, +0.052]).
  - So the image model has not recovered all the information the reports express.
  - This does not show that the labels are generally sufficient, or that any one cause, such as resolution, explains
    the gap.
- **MCL, Lateral OA, PF OA: resolution is a plausible hypothesis, not a conclusion.**
  - Report readers outperform the model, and report evidence adds to it (significantly for MCL only).
  - These findings involve thin structures, which fits the 0.625 mm-per-pixel concern in section 3.
  - The running 336 px experiment tests this with FOV and training settings fixed. Read these targets individually
    when it finishes, alongside macro AUC.
- **Effusion, Contusion, Baker's: the model outperforms its report labels.**
  - It ranks report over-calls below gold positives better than the reports do.
  - This shows the model generalises beyond some report errors. It does not show that those errors no longer
    constrain it.
  - Severity work for these targets is lower priority, not ruled out.
- **ACL, Medial Meniscus, Medial OA, Lateral Meniscus.** Model and reports are both high and roughly equal.
- **Synovitis is the strongest candidate for the next label experiment.**
  - The v31_d2 model predicts report-derived Synovitis well on fold-0 (0.951 vs Steven-v4) but gold Synovitis poorly
    (0.723).
  - Its Synovitis head ranks gold synovitis no better than its Effusion score does, and barely separates synovitis
    within effusion-positive knees (0.549 vs HPO-v1 0.661).
  - This is consistent with a fluid-related shortcut. It is equally consistent with:
    - correlated disease (22 of 27 gold synovitis cases also have effusion);
    - limited image detail;
    - noisy report supervision.

    It does not by itself prove a label-definition mismatch.
  - The earlier fold-0 Synovitis drop (−0.0125) after the v3.1 change points the same way, but is not significant.
  - Reverting to an older version is not yet justified. First identify which specific changes removed useful
    supervision or introduced an effusion shortcut:
    - how HPO-v1, v3.0 and v3.1 assign Synovitis, especially when only effusion is mentioned;
    - then a Synovitis-only change backed by report evidence and the host's definition, with the other 11 targets
      unchanged.
- **Ensemble.** The 3×ResNet50 rank ensemble is the best macro on gold (0.902 vs 0.885), consistent with the fold-0
  ensemble gain.
- **Gold vs public leaderboard.** Gold macro (0.885) is below the public LB (0.925) for the same model. Treat gold as
  a different and small sample, not as a calibrated estimate of LB.

**Caveat on independence.** These 58 gold studies have now been inspected per target repeatedly. They are
development evidence, not an independent final evaluation. Any label change motivated here must be justified from
report evidence and the host's definitions, and judged on public LB. Report its gold score as in-sample.
