# Synovitis supervision in HPO-v1, HPO-v3.0 and v3.1-D2 (2026-10-08)

**Question.** Where does each label version's Synovitis value come from? Does any mechanism teach Synovitis to copy
Effusion, or does it supply supervision independent of Effusion? Is there a minimal Synovitis-only label change that
the evidence supports?

**Scope and constraints**
- Only the fixed 3,524 training studies are analysed.
- No gold labels, no model scores and no GPU are used.
- No label file is written or changed; all four label files were re-hashed afterwards and are unchanged.
- The running 336 px experiment is untouched.

**Artifacts**
- Code: `scripts/audit_synovitis_supervision.py`, run in container `localhost/rsna-label-ds` with
  `--network=none` and a read-only project mount.
- Outputs: `data/synovitis_supervision_audit_2026-10-07/`, which holds tables t1–t11, the meta JSON, the run log and
  SHA256SUMS. It is gitignored.

## Labels compared and reproduction checks

| Label | File | sha256 |
|---|---|---|
| HPO-v1 | `data/label_datasets/hpo_v1_blend/soft_labels_hpo_v1_blend.json` | `ffdffcb2…` |
| v3.0 | `label_rules/hpo_v3/soft_labels_hpo_v3_structured.json` | `843014ed…` |
| v3.1-D2 | `data/label_datasets/hpo_v3_1_d2/kaggle/soft_labels_hpo_v3_1_d2_structured.json` | `38367d62…` |

The v3.1-D2 sha is the one recorded in the 0.925 run's `run_summary.json`.

**HPO-v1 is fully reconstructed from its two parents** (maximum error 0.0):
- Steven-v4: `labels_v0raw.parquet`, where 0.25 marks "not mentioned".
- Gemini-low: `labels_v1_gemlow.parquet`.

HPO-v1 is the mean of the two parents where Steven-v4 addressed the finding, and Gemini-low alone otherwise.

**Both v3 builds are reproduced cell for cell** by re-running the unchanged v3.1 builder with the D2 rule states:
- step c0 reproduces v3.0;
- step c6 reproduces D2.

This makes step-by-step attribution exact.

**D2 vs the v3.1 build.** D2 differs from `label_rules/hpo_v3_1/build/…v3_1_structured.json` in cells of every
target, by at most 0.053. Outside Synovitis, every differing cell is an evidence-free co-occurrence cell. Inside
Synovitis, the only non-evidence-free differences are the 5 D2 cells. Changing 5 Synovitis rule states moves the
co-occurrence fit for all targets slightly, so this is expected coupling, not an error.

## Report categories (training studies)

Categories are assigned from report text and the D2 rule-reader states, never from labels.

| Category | Definition | Studies |
|---|---|---|
| SYN_POS | rule Synovitis POS / LIKELY / BORD / HEDGE | 401 |
| SYN_NEG | rule Synovitis NEG / BELOW | 19 |
| SYN_OTHER | a synov- term appears but the rule reader is silent (synovial cyst, plica, synovial fluid, or a missed language) | 174 |
| EFF_ONLY | no synov- term; Effusion mentioned and not negated | 1,552 |
| EFF_NEG | no synov- term; Effusion negated | 933 |
| SILENT | no synov- term; Effusion silent | 445 |

So **2,930 of 3,524 training reports (83%) contain no synovitis wording at all.**

Known boundary errors:
- The rule Effusion extractor reads some negations as positive, for example Spanish "No hay significativo derrame
  articular" → POS.
- It misses some Turkish and Croatian fluid wording, which then lands in SILENT.
- The synov- regex covers Latin, Cyrillic and Greek spellings, including the Greek word for synovitis (υμενίτιδα).

A few studies therefore sit in a neighbouring category.

## Findings by mechanism

### 1. The HPO-v1 parents: the main source of Effusion copying

Table: `t1_category_level_and_effusion_coupling.csv`. Each cell shows the mean Synovitis label / the rank correlation
between the Synovitis and Effusion labels within that category.

| Category | Steven-v4 | Gemini-low | HPO-v1 | v3.0 | D2 |
|---|---|---|---|---|---|
| SYN_POS | 0.938 / 0.47 | 0.781 / 0.65 | 0.860 / 0.64 | 0.880 / 0.60 | 0.890 / 0.55 |
| SYN_NEG | 0.020 / 0.26 | 0.012 / 0.14 | 0.016 / 0.14 | 0.036 / 0.15 | 0.036 / 0.24 |
| SYN_OTHER | 0.373 / 0.70 | 0.267 / 0.74 | 0.314 / 0.70 | 0.370 / 0.44 | 0.370 / 0.44 |
| EFF_ONLY | 0.330 / **0.97** | 0.160 / 0.80 | 0.245 / **0.89** | 0.281 / 0.53 | 0.281 / 0.54 |
| EFF_NEG | 0.130 / 0.88 | 0.069 / 0.47 | 0.100 / 0.66 | 0.133 / 0.31 | 0.133 / 0.31 |
| SILENT | 0.249 / 0.99 | 0.098 / 0.81 | 0.109 / 0.84 | 0.197 / 0.58 | 0.197 / 0.58 |
| **All training** | 0.336 / **0.92** | 0.203 / 0.78 | 0.261 / **0.87** | 0.302 / 0.70 | 0.303 / 0.70 |

**Steven-v4: a fixed value set by whether effusion is mentioned.**
- It gives about 0.33 to every effusion-mentioned report without synovitis wording, about 0.13 when effusion is
  negated, and its 0.25 marker otherwise.
- Within the 2,930 no-synovitis-text reports, the rank correlation between its Synovitis and Effusion values is
  0.978.
- It does not scale with effusion severity: 0.357 for POS effusion vs 0.322 for BELOW (`t3_…csv`).
- Its "mentioned" flag is set for 3,936 of 4,349 studies, yet only 594 of the 3,524 training reports contain a synov-
  term.

**Gemini-low: Synovitis graded by effusion severity.**
- 0.302 for POS effusion, 0.210 for BORD, 0.098 for BELOW.
- Within no-synovitis-text reports, the rank correlation with Effusion is 0.739.

**HPO-v1: inherits both.**
- Within no-synovitis-text reports, its Synovitis ranking is effectively an effusion ranking: rank correlation 0.894.
- Its effusion-only reports rank above effusion-negative or silent ones at AUC 0.952 (`t2_ranking_sources.csv`).

**Verdict: this mechanism encourages copying.** For 83% of training studies, the HPO-v1 Synovitis value carries no
synovitis evidence. It is inferred from effusion, which the host defines as a separate finding.

### 2. How the v3 blend carries the parents forward

Formula: L = (2·HPO-v1 + Σ reader probabilities) / (2 + n_voting). Decomposition is in `t5_…csv`.

**In no-synovitis-text reports, most of each label still comes from HPO-v1:**
- EFF_ONLY: 61% from HPO-v1, mean 1.03 voting readers.
- EFF_NEG: 59% from HPO-v1, mean 0.72 voting readers.
- SILENT: 35% from HPO-v1.

**827 training cells are HPO-v1 copied exactly** (362 EFF_ONLY, 405 EFF_NEG, 60 SILENT):
- No reader votes on them.
- They are not treated as evidence-free, because a parent "mentioned" flag is set (`t7_…csv`).
- For these cells, v3.0 and D2 are HPO-v1, and so Effusion-derived.

**In SYN_POS, HPO-v1 contributes 28%.** The rest comes from about 5 readers that agree with the explicit synovitis
statement.

### 3. Reader votes on reports without synovitis wording: mixed

Vote counts are in `t4_reader_synovitis_votes_by_category.csv`.

| Reader | EFF_ONLY (1,552) | EFF_NEG (933) | SILENT (445) | Behaviour |
|---|---|---|---|---|
| teammate | HEDGE 714 (46%), POS 27 | HEDGE 137 (15%) | HEDGE 45 (10%) | **Follows effusion:** an uncertain synovitis vote, three times as often when effusion is present |
| vmohitrao | HEDGE 160 (10%) | HEDGE 59 (6%) | HEDGE 43 (10%) | Weakly follows effusion |
| laymond | POS 221 (14%) | POS 86 (9%) | POS 133 (30%) | **Does not follow effusion.** Over-represented terms: Hoffa's fat-pad impingement ("pinzamiento de la almohadilla grasa de Hoffa"), "ödemli", bursitis, prepatellar, impingement (`t10_…csv`). Some votes are unrelated, e.g. a PCL avulsion fracture. |
| pilkwang | rarely votes | | | Silent on almost all of these |

**Verdict: these votes are not independent synovitis supervision.**
- `teammate`'s HEDGE adds effusion-conditioned signal.
- `laymond` adds a fat-pad, bursitis and impingement signal. Hoffa's fat-pad signal is a recognised synovitis
  surrogate on non-contrast MRI, but the host defines Synovitis as "Inflammation and thickening of the synovial lining
  of the joint". Fat pad and bursae are not mentioned.

Together these votes are why v3 has 59 no-synovitis-text training cells at ≥ 0.5, against 8 for HPO-v1.

They also lower v3's effusion correlation compared with HPO-v1 (0.70 vs 0.87 over all training). That is only because
they add a different kind of non-synovitis signal; it does not make the label more about synovitis.

### 4. Co-occurrence for evidence-free cells: negligible

- Only 133 training Synovitis cells are evidence-free (129 SILENT, 4 SYN_OTHER).
- In the fitted model, Effusion has by far the largest weight: standardised coefficient 0.615 in D2 and 0.627 in
  v3.0. The next largest are Medial OA at 0.15 and PF OA at 0.15.
- Its mean contribution in SILENT is 0.014.

**Verdict:** this mechanism follows effusion in direction, but its effect is too small to matter.

### 5. The v3.0 → D2 changes: independent supervision, not a shortcut

Step-by-step attribution is in `t6_v30_to_d2_synovitis_by_step.csv`; rule-state transitions are in `t6b_…csv`.

| Step | Training cells changed | Changed > 0.05 | Categories |
|---|---|---|---|
| c1 scope | 196 | 65 | 65 SYN_POS; the remainder are co-occurrence cells moving ≤ 0.13 |
| c2 synovitis (any-severity definite synovitis → POS) | 186 | 55 | 55 SYN_POS; the remainder are co-occurrence cells |
| c3–c4 OA / Baker's | 133 / 127 | 0 | co-occurrence cells only, ≤ 0.002 |
| c5 fracture site (with the 5 D2 cells) | 99 | 4 | 4 SYN_POS |
| c6 training-only co-occurrence fit | 132 | 0 | ≤ 0.005 |
| **Total v3.0 → D2** | **222** | **86** | **89 SYN_POS raised (mean +0.019 across changed cells); 133 co-occurrence cells moved slightly** |

- **Rule transitions on training studies:** BORD → POS 79, NEG → POS 2, POS → HEDGE 3, BORD → HEDGE 3. All other
  states are unchanged.
- **Every substantive change is on a report that explicitly mentions synovitis.** Examples: "Knee effusion combined
  minimal synovitis", "Leve derrame articular con sinovitis difusa", "verdickter Synovia und geringen synovialen
  Proliferationen".
- **No change touches an effusion-only report.** Within SYN_POS, the Synovitis–Effusion correlation falls from 0.60
  to 0.55.

**Verdict:** the v3.1 synovitis correction adds independent supervision. The audit finds no v3.1 or D2 change that
introduces an effusion shortcut or removes useful supervision.

### 6. Why the fold-0 validation reference cannot detect copying

Table: `t9_steven_v4_mask_by_category_train.csv`. This is mechanism only, computed on training rows.

The Steven-v4 confidence mask treats ≤ 0.3 as negative and ≥ 0.7 as positive. Applied to Steven-v4's Synovitis:
- **Excluded:** 1,493 of 1,552 EFF_ONLY studies, because they sit at about 0.33.
- **Positives:** almost only SYN_POS (391 of 401).
- **Negatives:** EFF_NEG (931) and SILENT (388).

An effusion detector would therefore score highly on this reference. On validation, the 0.951 Synovitis AUC is
consistent with effusion copying and is not evidence of synovitis-specific learning. This does not touch fold-0 data;
it follows from how Steven-v4 assigns values.

## Examples and counterexamples

Each line shows the categories plus each reader's vote and each source's value. More are in `t8_examples.csv`.

- **Copying (EFF_ONLY, high HPO-v1).** "There is a large knee joint effusion. There is complete tear of the anterior
  cruciate ligament…" No reader votes on Synovitis. Steven 0.365, Gemini 0.40 → HPO-v1, v3.0 and D2 all 0.383.
- **Copying via a teammate HEDGE.** "Knee effusion is present…" teammate HEDGE. Steven 0.35, Gemini 0.30 → D2 0.350.
- **Counterexample: effusion-only but low.** A Croatian bucket-handle meniscal tear report with effusion. Every reader
  votes NEG or BELOW; Steven 0.025 → D2 0.061. Steven does not always copy.
- **Counterexample: rule category error.** "No hay significativo derrame articular" reads as rule Effusion POS, but
  D2's Effusion label is 0.17 and Synovitis is 0.16–0.21.
- **Independent supervision.** "hypertrophy of the synovium, indicative of synovitis" → all readers POS, D2 0.92.
  "Derrame con sinovitis" → D2 0.92.
- **Explicit synovitis but low label.**
  - "Differential diagnoses such as focal synovitis and PVNS are less probable" → HEDGE, D2 0.29. Correctly low.
  - "R/O synovitis" with small effusion → HEDGE, D2 0.50.
- **Explicit negatives.** "Small knee joint effusion. No significant synovial thickening" → D2 0.039.
  "Trace. No synovitis" → D2 0.032.
- **laymond POS without synovitis wording.**
  - "Pinzamiento de la almohadilla grasa de Hoffa" → laymond POS.
  - Turkish "Derin infrapateller bursit vardır…" → teammate and laymond POS, D2 0.46.
- **Rule-silent synov- term handled by readers.** "Umjereni izljev… uz blažu hipertrofiju sinovije" (synovial
  hypertrophy) → readers POS, D2 0.82. "bez signifikantne proliferacije sinovije" (no synovial proliferation) → D2
  0.04.

## Is there a minimal Synovitis-only candidate? Not yet

Two definition-based previews were computed. Both are counts only and write no label file; the other 11 targets are
unchanged by construction.

| Preview | Training cells changed | No-text cells ≥ 0.5 | Synovitis–Effusion correlation (no-text / all training) |
|---|---|---|---|
| D2 as trained | — | 59 | 0.672 / 0.704 |
| P1: no synovitis text → constant 0.2209 (training mean of D2 over those cells) | 2,930 (924 up > 0.1) | 0 | — / 0.380 |
| P2: LLM-reader positive-direction votes on no-text reports treated as silent | 1,481 (549 down > 0.1) | 8 | **0.860** / **0.840** |

In P2, the votes dropped on training reports are: teammate 935, laymond 548, vmohitrao 266, pilkwang 15.

**Neither preview is minimal.**
- **P1** follows the host's definition: silence carries no synovitis information, and effusion is a separate
  finding. But it rewrites 83% of training cells and raises every effusion-negative report. It also discards the
  effusion–synovitis association entirely, and that association may be partly real.
- **P2** removes reader votes that are not grounded in synovitis wording. But those cells then fall back to HPO-v1, so
  effusion copying gets stronger. The two non-evidence mechanisms are entangled: removing one amplifies the other.

**Conclusion: the evidence supports no specific minimal change.**
- The hypothesis that the v3.1/D2 changes introduced an effusion shortcut is not supported at label level:
  - those changes touch only explicit-synovitis reports;
  - Effusion coupling is strongest in HPO-v1.
- The earlier gold diagnostic found the opposite: the HPO-v1-trained model kept more synovitis-specific signal within
  effusion-positive knees (0.661 vs 0.549). The label coupling here does not explain that difference. Possible causes:
  - seed or run variance;
  - noise from the v3 reader votes on no-text cells;
  - chance with 35 studies.

## Open questions and what would resolve them

1. **How should silent Synovitis be supervised?** This is a policy decision, not something these data can settle:
   - from effusion (inherited from HPO-v1, and correlated with synovitis);
   - as uninformative (the host's definition separates the two findings);
   - from fat-pad or bursa surrogates (laymond's votes).
2. **Whether the HPO-v1 vs D2 model difference on gold is real.** A second seed of either run, or the existing
   ensembles, would show whether the within-effusion gap holds before any label is built for it. This needs compute or
   uses gold as development evidence, so it is the user's call.
3. **Rule-extractor gaps.** These affect the category boundaries only, not the conclusion:
   - Effusion negation in Spanish ("No hay significativo derrame");
   - Turkish and Croatian fluid wording;
   - synovial hypertrophy or proliferation in Croatian, which readers already catch.
4. **Validation cannot measure any Synovitis label change.** Fold-0 with the Steven-v4 mask cannot detect effusion
   copying (section 6). Any Synovitis experiment would need public LB and, as development evidence only, the 58 gold
   studies.
