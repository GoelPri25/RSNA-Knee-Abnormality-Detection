# xAI and model-debugging techniques: where v4 fails and what to change

Research note, 2026-10-19. Web research only. Nothing was run, and no project data was sent anywhere.
Scope: techniques whose result points to a **training or data change**, not only a heatmap. Each reference was checked to exist (arXiv/proceedings page found). "Unverified" marks details I could not confirm.

**What we already know (from the ledger), which decides which tools are worth running:**
- Q-1017: moderate-or-worse (MOD+) positives are ranked almost perfectly. Of the AUC loss, 34% is in report-MILD cells and 23% in report-silent (SIL) cells. So most failures sit where the label itself is uncertain. Every diagnostic below must **separate label noise from genuine model misses**. Otherwise it will just find the report state again.
- Noise floor: fold-0 bootstrap SD is 0.006–0.007 for one model and 0.002–0.003 for a paired difference. The public LB has about 390 studies. Clear6 (clear-band AUC) is the yardstick that tracks the LB.
- Done already: Grad-CAM marks anatomy, not the label. Series ablation (SER-0/1). Site offsets per report language (2-site: the offsets are real, but they barely change ranking). A joint-findings combiner (JF-0: fails, but there is an **MCL-fires-on-effusion/meniscus** hint). Slice skip (SK-0, underpowered). Down-weighting silent cells (silw/hill). Teacher–student blending (P3 v2, LB 0.936).

---

## 1. Catalogue

Cost for us: **CPU** = existing fold-0 logits or labels only. **Feat** = needs pooled 2048-d per-finding features (about 0.5–1 GPU-h to extract for fold 0 + training folds). **Inf** = ≤ 1 GPU-h of inference. **Train** = needs retraining or many models.

### 1a. Which cases fail (error slices, label-noise ranking)

| Technique | One-line idea | Key reference | Reveals | Cost | Points to |
|---|---|---|---|---|---|
| Domino | Error-aware Gaussian mixture on cross-modal (image–text) embeddings; captions each slice | Eyuboglu et al., ICLR 2022, [arXiv 2203.14960](https://arxiv.org/abs/2203.14960) | Coherent subgroups with high error | Feat (needs an image–text model for captions; we have none for knee MRI) | Curate / oversample the slice; add a view or crop |
| Spotlight | Searches the final-layer embedding for a contiguous region with high loss | d'Eon et al., FAccT 2022, [arXiv 2107.00758](https://arxiv.org/abs/2107.00758) | Unlabelled weak regions in the representation | Feat, CPU | Same as Domino |
| Failure directions | Per class, a linear SVM on embeddings predicts "model wrong"; the SVM normal is the failure direction | Jain et al., ICLR 2023, [arXiv 2206.14754](https://arxiv.org/abs/2206.14754) | Whether errors are *systematic* (predictable) or idiosyncratic | Feat, CPU | Upweight or add data along the direction; stop if not predictable |
| Hidden stratification | Measure performance on clinically meaningful subsets nobody labelled (schema completion, clustering) | Oakden-Rayner et al., CHIL 2020, [arXiv 1909.12475](https://arxiv.org/abs/1909.12475) | Subsets with >20% relative performance gaps (low prevalence, low label quality, subtle features) | CPU (metadata strata) | Subgroup-targeted data or augmentation |
| Data maps (dataset cartography) | Per-sample mean confidence and variability across epochs → easy / ambiguous / hard-to-learn | Swayamdipta et al., EMNLP 2020, [arXiv 2009.10795](https://arxiv.org/abs/2009.10795) | Hard-to-learn ≈ likely mislabelled; ambiguous ≈ most useful | Free if per-epoch training logits are logged | Relabel/mask the hard-to-learn cells; keep the ambiguous ones |
| AUM (area under the margin) | Average margin over training; compare with deliberately mislabelled "threshold" samples | Pleiss et al., NeurIPS 2020, [arXiv 2001.10528](https://arxiv.org/abs/2001.10528) | Mislabelled training samples | Free with logging (+1 run for the threshold class) | Drop/mask the flagged samples |
| Forgetting events | Count correct→incorrect flips per sample during training | Toneva et al., ICLR 2019, [arXiv 1812.05159](https://arxiv.org/abs/1812.05159) | Unforgettable (redundant) vs often-forgotten (hard or noisy) samples | Free with logging | Prune redundant samples; inspect forgotten ones |
| Confident learning / Cleanlab | Out-of-sample probabilities + per-class thresholds → estimate the joint of noisy and true labels → rank label issues | Northcutt et al., JAIR 2021, [arXiv 1911.00068](https://arxiv.org/abs/1911.00068); test-set audit [arXiv 2103.14749](https://arxiv.org/abs/2103.14749) | Likely label errors, ranked | CPU (we have cross-fitted OOF teachers) | Mask/relabel the top-ranked cells; report noise per finding |
| Active label cleaning | Rank by (estimated label correctness × difficulty) to spend the relabel budget | Bernhardt et al., Nat. Commun. 2022, [arXiv 2109.00574](https://arxiv.org/abs/2109.00574) | Which cells are worth a human look | CPU | Relabel order (for the user/teammates reading reports) |

### 1b. Why it fails (concepts, prototypes, counterfactuals, attribution checks)

| Technique | Idea | Reference | Reveals | Cost | Points to |
|---|---|---|---|---|---|
| TCAV | Linear "concept" direction in activations from example sets; directional-derivative sensitivity of the class to it | Kim et al., ICML 2018, [arXiv 1711.11279](https://arxiv.org/abs/1711.11279) | Whether a human concept (e.g. effusion, a site, a protocol) drives a finding's score | Feat + concept example sets (report states / metadata could serve as concepts) | Decorrelate the concept, add an auxiliary head, or rebalance |
| Concept bottleneck | Predict concepts first, then the label; test-time intervention on concepts | Koh et al., ICML 2020, [arXiv 2007.04612](https://arxiv.org/abs/2007.04612) | Which concept errors cause label errors (their example is knee X-ray OA grading) | Train | A new head (e.g. severity/extent concepts); too slow for 3 days |
| Prototypes (ProtoPNet) | Classify by similarity to learned prototypical parts | Chen et al., NeurIPS 2019, [arXiv 1806.10574](https://arxiv.org/abs/1806.10574) | "This looks like that" evidence | Train (an architecture change) | Not for now |
| Deep kNN | Explain a test case by its nearest training points in each layer's representation; nonconformity = label disagreement among neighbours | Papernot & McDaniel 2018, [arXiv 1803.04765](https://arxiv.org/abs/1803.04765) | Whether an error's neighbours carry conflicting labels (label noise) or none are close (coverage gap) | Feat, CPU | Mask noisy neighbours, or add data/augmentation for uncovered regions |
| Counterfactual GAN (StylEx) | Train a classifier-specific StyleGAN; the attributes that flip the decision are the explanation | Lang et al., ICCV 2021, [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Lang_Explaining_in_Style_Training_a_GAN_To_Explain_a_Classifier_ICCV_2021_paper.html) | The visual attributes the classifier uses | Train (a GAN on MRI: days) | Not for now |
| Saliency sanity checks | Randomise model weights or labels; a faithful map must change | Adebayo et al., NeurIPS 2018, [arXiv 1810.03292](https://arxiv.org/abs/1810.03292) | Maps that are just edge/anatomy detectors | Inf | Stop trusting a map for crop placement if it fails |
| ROAR | Remove the top-attributed features, *retrain*, measure the drop (avoids the OOD artefact of plain deletion) | Hooker et al., NeurIPS 2019, [arXiv 1806.10758](https://arxiv.org/abs/1806.10758) | Whether the attribution ranks importance better than random (most methods do not; SmoothGrad²/VarGrad do) | Train | Choose an attribution method for crop placement |
| Deletion / insertion (RISE) | Score drop as top pixels are deleted, or rise as they are inserted; no retraining | Petsiuk et al., BMVC 2018, [arXiv 1806.07421](https://arxiv.org/abs/1806.07421) | Faithfulness of a map or a window ranking | Inf | Same; cheap but OOD-biased |
| SpRAy (Clever-Hans detection) | Cluster relevance maps across many samples; odd clusters = shortcut strategies | Lapuschkin et al., Nat. Commun. 2019, [arXiv 1902.10178](https://arxiv.org/abs/1902.10178) | Shared spurious strategies (their example: a source watermark) | Inf + CPU | Remove the artefact, or augment against it |

### 1c. Which training data drives the errors

| Technique | Idea | Reference | Cost | Verdict for us |
|---|---|---|---|---|
| Influence functions | Hessian-based leave-one-out approximation | Fragility in deep nets: Basu et al., ICLR 2021, [arXiv 2006.14651](https://arxiv.org/abs/2006.14651) | Train-scale | Unreliable for a deep r50; skip |
| TracIn | Sum over checkpoints of grad(train)·grad(test) | Pruthi et al., NeurIPS 2020, [arXiv 2002.08484](https://arxiv.org/abs/2002.08484) | Inf, if SWA epochs 21–25 are kept; last-layer gradients only | Feasible on the head only; checkpoint-dependent |
| TRAK | Random-projected after-kernel; a few models match datamodels | Park et al., ICML 2023, [arXiv 2303.14186](https://arxiv.org/abs/2303.14186) | Several models + per-sample gradients | Too heavy for 3 days |
| Datamodels | Linear model from training-subset membership to output; needs thousands of retrains | Ilyas et al., ICML 2022, [arXiv 2202.00622](https://arxiv.org/abs/2202.00622) | Infeasible | Skip |
| Data Shapley | Monte-Carlo Shapley value of each training point | Ghorbani & Zou, ICML 2019, [arXiv 1904.02868](https://arxiv.org/abs/1904.02868) | Infeasible for CNNs at our scale | Skip |

The cheap stand-in is **feature-space kNN attribution** (Deep kNN above). It answers "do the training studies that look like this error carry a contradicting label?"

### 1d. What the representation encodes (nuisance, shortcuts)

| Technique | Idea | Reference | Reveals | Cost | Points to |
|---|---|---|---|---|---|
| Linear probes for nuisance | Predict site/scanner/protocol from pooled features | Zech et al., PLOS Med 2018, [PMC6219764](https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764/) (CNNs identified the hospital system with high accuracy) | A decodable confounder | Feat, CPU | Matters only if per-subgroup ranking also differs (2-site: small) |
| Shortcut audit | Test on conditions where the shortcut is broken | Geirhos et al., Nat. Mach. Intell. 2020, [arXiv 2004.07780](https://arxiv.org/abs/2004.07780); DeGrave et al., Nat. Mach. Intell. 2021, [medRxiv](https://www.medrxiv.org/content/10.1101/2020.09.13.20193565v2) | Decision rules that pass the benchmark but use the wrong signal (laterality markers, positioning) | CPU (stratified AUC) | Group-balanced refit / hard-pair weighting |
| CKA | Similarity of representations across layers or models | Kornblith et al., ICML 2019, [arXiv 1905.00414](https://arxiv.org/abs/1905.00414) | Whether two models or branches learned the same features (ensemble diversity; global vs crop branch) | Feat | Ensemble choice; little direct training lever |
| Last-layer refit (DFR) as a probe | Refit only the head on a group-balanced set; if robustness jumps, the features were fine and the head was biased | Kirichenko et al., ICLR 2023, [arXiv 2204.02937](https://arxiv.org/abs/2204.02937) | Head-level vs feature-level failure | Feat, minutes of CPU/GPU | A head refit (minutes) vs a backbone change (hours) |

### 1e. MIL-specific: is the window attention faithful?

| Technique | Idea | Reference | Relevance |
|---|---|---|---|
| Attention MIL | Attention weights as instance importance | Ilse et al., ICML 2018, [arXiv 1802.04712](https://arxiv.org/abs/1802.04712) | Our global head is this pooling |
| Attention ≠ explanation | Attention correlates poorly with gradient importance; different attention can give the same output | Jain & Wallace, NAACL 2019, [arXiv 1902.10186](https://arxiv.org/abs/1902.10186) | Picking crop windows by attention alone is suspect |
| Additive MIL | Make the bag logit a sum of per-instance contributions, so credit is exact and signed | Javed et al., NeurIPS 2022, [arXiv 2206.01794](https://arxiv.org/abs/2206.01794) | v4's global head is *already* additive: logit = Σᵢ aᵢ·(w·LN(hᵢ)) + b, so the exact window contribution cᵢ = aᵢ·(w·LN(hᵢ)) is free |
| xMIL (LRP for MIL) | Explanations that respect instance interactions; faithfulness via patch-dropping curves | Hense et al., NeurIPS 2024, [arXiv 2406.04280](https://arxiv.org/abs/2406.04280) | Supplies the window-deletion faithfulness protocol |

### 1f. Calibration and reliability by subgroup

| Technique | Reference | Note for a mean-of-per-finding-AUC metric |
|---|---|---|
| Reliability diagrams, ECE, temperature scaling | Guo et al., ICML 2017, [arXiv 1706.04599](https://arxiv.org/abs/1706.04599) | A global monotone rescale per finding cannot change AUC. What can matter: (i) **offsets that differ by subgroup** (studies missing a series, a protocol, a site) mis-rank studies across subgroups in the pooled AUC; (ii) members on different scales in a probability-mean ensemble |

---

## 2. Evidence that a diagnostic led to a measured gain

| Source | Diagnostic → change | Result | How matched | Strength |
|---|---|---|---|---|
| Pleiss 2020 (AUM) | Flag by AUM → remove | WebVision50: remove 17% of data, −1.6% absolute error; CIFAR-100: remove 13%, −1.2% | Same model and test set | Medium (natural images, single-label, hard labels) |
| Jain 2023 (failure directions) | SVM direction → choose added data / upweight | CIFAR-100: +100 images per superclass chosen along the direction 0.696→0.720, vs random 0.712 and confidence-ranked 0.716. CelebA minority groups +0.022 to +0.026 | Same budget, same test set | Medium; their best gains are on built-in subpopulation shifts |
| Swayamdipta 2020 (data maps) | Train on the 33% ambiguous | WinoGrande OOD 86.0→87.6 but **ID 79.7→78.7**; 67% of hard-to-learn flags were verified as mislabelled or ambiguous | Same model | Medium for noise detection; weak for in-distribution gain (ours is in-distribution) |
| Northcutt 2021 (test-set audit) | Confident learning → human check | 51% of flagged candidates were real errors; a small rise in test-label noise flips model ranking | Crowd-verified | Strong as a *precision* benchmark; it shows the flags are about half right |
| Liu 2021 (JTT) | Errors of a first model → upweight them in a second model | Closes 85% of the worst-group gap between ERM and group DRO | Same data | Worst-group, not average; average accuracy often drops slightly |
| Kirichenko 2023 (DFR) | Head refit on a group-balanced held-out set | Worst-group accuracy Waterbirds 74.9→92.9%, CelebA 46.9→88.3% | Same features | Worst-group; needs a balanced held-out set |
| Wu 2023 (DISC) | Concept-instability discovery → data intervention | +9.6% on a skin-lesion task ([arXiv 2305.00650](https://arxiv.org/abs/2305.00650)) | Unverified: metric, dataset, worst-group or average | Weak until checked |
| Irvin 2019 (CheXpert) | Per-finding policy for uncertain labels | Atelectasis U-Ones 0.858 vs U-Zeros 0.811; Consolidation U-Zeros 0.932 vs U-Ones 0.899 ([arXiv 1901.07031](https://arxiv.org/abs/1901.07031), Table 3) | Same model and validation set (n≈200, so noisy) | Medium: **the best uncertain-label policy is finding-specific**; this maps to our HEDGE/MILD cells |
| Pham 2020 | Label smoothing on uncertain cells | Mean AUC 0.940 (top of the CheXpert leaderboard at the time) ([arXiv 1911.06475](https://arxiv.org/abs/1911.06475)) | Ablations in the paper; not re-checked | Medium–weak |
| Jain 2021 (labeler quality) | A better report labeler → image model | Weighted AUROC 0.83 (CheXpert/CheXbert labels, U-Zeros) → 0.87 (VisualCheXbert), significant ([arXiv 2104.00793](https://arxiv.org/abs/2104.00793)) | Same image model and test set | Strong for "labels are the lever"; not an xAI step |
| Guendel 2019 | Multi-reader relabel, drop uncertain | 0.883 → 0.945 ([arXiv 1905.06362](https://arxiv.org/abs/1905.06362)) | **Not matched**: 0.945 is on a high-confidence subset | Weak |
| Freesound 2019, 1st place | Low-score curated samples hand-relabelled | Part of the winning recipe ([repo](https://github.com/lRomul/argus-freesound)); no isolated number | Unknown | Anecdotal Kaggle evidence |
| Oakden-Rayner 2020; Zech 2018; DeGrave 2021; Lapuschkin 2019 | Subgroup / shortcut discovery | Found the failures (>20% relative subgroup gaps, hospital confounding, positioning shortcuts, watermark) | — | Diagnostic only; no fix quantified |
| Our ledger | 2-site (site offsets); JF-0; P3 v2 | Offsets real, ranking ±0.002; combiner −0.004; blend LB 0.936 | Matched fold 0 | Our own evidence that *broad* label or reweighting fixes do not pay; targeted ones are untested |

Bottom line: the strongest external evidence is for **noise-flag → remove/mask** (AUM, confident learning; about half of flags are right) and for **errors predictable from embeddings → targeted data** (failure directions). Most gains are reported on worst-group or synthetic-noise settings. In-distribution average gains are small (around 1 point of accuracy). For a 0.944 → 0.958 gap, expect diagnostics to find +0.002–0.005 levers, not the whole gap.

---

## 3. Ranked shortlist for v4 (3 days; each outcome maps to a ≤ 3 h run or to "stop")

Rules for all of them: label-v2 confident cells. Judge on **clear6 plus the 12-target mean**. Both seeds (s2026 and s2027) must agree. 2,000 paired study bootstraps. Pre-register before scoring. To avoid fishing across 12 findings × many subgroups, run discovery on a random half of fold 0 and confirm on the other half.

**D1. Exact window credit vs attention: are crops placed on the windows that actually carry the evidence? (Inf, ≤ 1 GPU-h; Kaggle T4.)**
- Question: v4 picks crop windows by attention aᵢ. The head is additive, so the true signed contribution is cᵢ = aᵢ·(w·LN(hᵢ)). Do the two rankings agree? Is either one faithful?
- Test: per finding, overlap@4 between the top-4 windows by a and by c. Deletion curves: drop the top-k windows (k = 1, 2, 4) by a, by c and at random, then re-normalise the attention. Report ΔAUC. Then re-run the crop branch on the top-4 windows by c instead of by a (zero training, the self-location code path).
- Decision: if c-placement beats a-placement by ≥ +0.0015 on clear6 in both seeds, switch crop window choice to c (free at inference; retrain the crop side as in L4C-R2 if the gain needs it). If deleting the top-k by a hurts no more than random deletion for a finding, that finding's attention is not evidence: use global-only for it, or c-placement. If overlap@4 ≥ 0.75 everywhere, stop: attention is fine.
- Caveat: deletion without retraining is out of distribution (ROAR). Use the random-deletion control at the same k.

**D2. Consensus-error / confident-learning audit on the cross-fitted OOF set: are the AUC-losing cells label errors or model misses? (CPU.)**
- Inputs: P3-T cross-fitted r50 and cnx teachers (OOF on all training studies), v4 fold-0 logits, report state, Steven-family ruler.
- Test: per finding, Cleanlab-style flags (per-finding thresholds = mean OOF p among label-positive and label-negative cells), keeping only cells where **both architectures agree** against the label. Split by report state. On non-silent cells, flag precision = ruler agreement with the models among cells the ruler calls confidently (gold-58 as a tie-break; never edited).
- Decision: if precision ≥ 0.5 (the Northcutt benchmark) and the flags are ≤ 3% of a finding's cells, retrain v4 with those cells **masked (weight 0)**: not blended (P3 v2 failed) and not all silent cells (silw failed). One 2–3 h run on the weak findings first (lateral OA, Baker's, contusion, lateral meniscus). If precision < 0.3, the labels are not the fixable part: stop label work.
- Caveat: test labels are image reads on the host rubric. A report-vs-image "error" may be a true test positive. That is the reason to mask rather than flip.

**D3. Correlated-finding (shortcut) audit with stratified AUC. (CPU, minutes.)**
- Question: JF-0 hints that MCL fires on effusion or medial-meniscus signal. Which findings rank mainly by a correlated finding's signal?
- Test: for each finding f and its top-3 label-correlated findings g: AUC_f within g+ and within g− studies; proxy AUC = g's score against f's label; the share of f's mis-ordered pairs that lie inside g+. Candidates: MCL|effusion, MCL|medial meniscus, contusion|ACL, fracture|contusion, lateral OA|lateral meniscus, Baker's|effusion (SER-1: already specific, so it serves as a control).
- Decision (SER-1 rule plus a mass rule): call it a shortcut if proxy AUC ≥ own AUC − 0.03, or within-g+ AUC ≤ pooled − 0.05, **and** ≥ 40% of mis-ordered pairs are inside g+, in both seeds. Shortcut → hard-pair weighting: upweight f-negatives that are g-positive (JTT-style, 2–3 h); or refit f's head only on a g-balanced sample (DFR, minutes, needs Feat). Do **not** remove the correlation outright: on an i.i.d. test it is real signal.

**D4. Natural-subgroup offsets: series availability and protocol. (CPU.)**
- Question: v4 relies almost only on fat-suppressed sagittal and coronal series. Studies that *naturally* lack them, or have unusual n_files / slice counts, may sit on a shifted logit scale and be mis-ranked in the pooled AUC.
- Test: subgroups = SagF missing, CorF missing, fat-sat flag absent, n_files tertiles, laterality. Per finding: subgroup size, within-subgroup AUC, and the mean-logit offset on clear negatives. Then pooled AUC after cross-fitted subgroup offsets (the 2-site method).
- Decision: offset-corrected clear6 ≥ +0.002 in both seeds → series-dropout augmentation (drop SagF/CorF with p ≈ 0.15, 2–3 h), or an inference offset if the subgroup is large enough to estimate. Otherwise no lane.

**D5. Failure-direction SVM on pooled features + metadata alignment. (Feat + CPU.)**
- Test: per weak finding, 5-fold CV linear SVM on per-finding pooled features predicting "mis-ranked" (FN among clear positives plus MILD positives, FP among clear negatives). Run it separately within the MILD and SIL states, so it cannot rediscover the report state. Correlate the SVM score with metadata (D4 strata, site language, FOV/pixel spacing) and with report severity words.
- Decision: CV AUC ≥ 0.65 in both seeds **and** |r| ≥ 0.3 with a nameable variable → targeted oversampling or augmentation of that subgroup (2–3 h). CV AUC < 0.6 → errors are idiosyncratic (likely label noise or rare positives): no slice fix.

**D6. Feature-space kNN attribution of the top errors (TracIn stand-in). (Feat + CPU.)**
- Test: for the 30 worst fold-0 errors per weak finding, and 30 correctly ranked controls matched on report state, take the 10 nearest training studies (cosine, per-finding pooled feature). Tabulate their labels and report states.
- Decision: if error neighbours carry the contradicting label (for example, an FN's neighbours are mostly SIL-negative) at ≥ 2× the control rate → training label noise drives the error; feed those neighbour cells into D2's mask list. If errors have no close neighbours (distance above the 90th percentile of the controls) → a coverage problem: augmentation or more windows, not labels.

**D7. Log training dynamics in every remaining run. (Free.)**
- Per (study, finding) training-cell loss and margin each epoch (val BCE is already logged). That gives data maps, AUM and forgetting counts at no extra cost. It also checks the V4-IMP observation that "v4 has no memorisation phase": if so, margins separate noisy from clean cells less well.

Suggested order: D1, D3 and D4 now (inference and CPU only, results today). Then D2 (CPU). Then D5/D6 once features are extracted in the same Kaggle kernel as D1. At most two training runs follow: the winner of D1/D3/D4 and the D2 mask.

---

## 4. Pitfalls

| Pitfall | Source | Guard |
|---|---|---|
| Saliency maps can be independent of the model and the labels | Adebayo 2018 | Run a weight-randomisation test before using grad×act for placement; our "maps mark anatomy" result fits this |
| Attention weights are not explanations | Jain & Wallace 2019; Additive MIL | Use the exact signed contribution aᵢ·(w·hᵢ) (D1) |
| Deletion without retraining measures OOD behaviour | Hooker 2019 (ROAR) | Random-deletion control at the same k; treat as a ranking signal only |
| Influence estimates are fragile in deep nets and vary across test points | Basu 2021 | Skip influence functions; use kNN or last-layer TracIn only as a hint |
| Slice discovery rediscovers label noise or report state | Our Q-1017 numbers; Johnson et al. 2023 [arXiv 2306.08167](https://arxiv.org/abs/2306.08167) (user study: slices help hypotheses only modestly) | Condition on report state; confirm on a held-out half; a slice is a hypothesis, not a result |
| Small n and many comparisons (fracture/MCL positives ≈ tens) | Q-1017 (SD 0.006 single, 0.002–0.003 paired) | Discovery/confirm split; both seeds; bars ≥ +0.0015–0.002 on clear6 |
| Label-noise flags are only about 50% precise | Northcutt 2021 | Mask instead of flip; cap at ≤ 3% of cells per finding; never touch rulers/gold |
| A report-vs-image disagreement can be a true positive on the image-read test | Host rubric (Q-1017) | Judge on clear6/gold, not on report agreement |
| Removing a "shortcut" can lower i.i.d. AUC | JF-0; Geirhos 2020 | Reweight hard pairs; never decorrelate outright |
| Calibration fixes do not change per-finding AUC | Metric definition | Only subgroup offsets or ensemble scale matter (D4) |
| Training-dynamics measures assume clean cells are learned early | Swayamdipta 2020, Pleiss 2020 | With soft labels and no memorisation phase, use per-finding thresholds and validate against ruler precision |
| Data Shapley, datamodels and TRAK need many retrains | Ghorbani 2019; Ilyas 2022; Park 2023 | Out of budget for 3 days |
