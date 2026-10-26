# Review of external agent's HPO-v3 candidate

Reviewed `../label_rules/build_hpo_v3.py`, `rule_extractor.py`, generated files,
rule evidence, largest changes and `../KNEE_MRI_LABELING_GUIDE.md`. Rebuilt both
JSON files on CPU into `data/label_audits/hpo_v3/rebuilt`: byte-identical to the
submitted candidates. No training or source modification was performed.

Structural checks: 4349 unique report-only IDs, exact HPO-v1 ID order, zero
overlap with 58 gold IDs, finite probabilities in [0,1], matching target order.
Uniform candidate weights are all 1. The weighted variant has identical targets
and exactly 3573 cells at weight 0.25, affecting 1527 studies. This proves file
compatibility and reproducibility, not label accuracy.

## Useful direction

Distinguishing explicit absence, below-threshold findings, uncertainty and silence
is more principled than interpreting every report mention as a qualifying positive.
The builder preserves HPO-v1 with weight two and adds up to five categorical readers.
Evidence clauses and source hashes enable review. No gold targets enter the builder.
Readers are correlated and their probability mappings are heuristic; the arithmetic
blend is not a calibrated posterior or seven independent expert assessments.

## Reproducible defects and limitations

1. Severity scope leaks between findings in a clause. `Small joint effusion and
   a large Baker cyst.` yields BORD for both. `Moderate joint effusion and a small
   Baker cyst.` likewise yields BORD for both. The guide implies different states
   for the two findings. A real saved clause `moderate joint effusion is seen with
   mild synovial proliferation at suprapatellar burse` is Effusion BORD. Severity
   needs attachment to the relevant finding, not a clause-wide small/large search.
2. `Mild synovitis.` yields BORD. The local guide/source extraction prompt defines
   definite synovitis of any severity as positive. There are 114 BORD Synovitis
   rule cells; examples include mild reactive synovitis and diffuse synovitis next
   to mild effusion. These need review for assertion, anatomy and scope, not an
   indiscriminate promotion of every synovial/cyst term.
3. `Grade III chondromalacia of the medial femoral condyle involving a 3 mm focus.`
   yields POS, as does a 2 cm lesion. The stated guide requires sufficient depth
   and extent. The OA rule promotes grade >=3 without testing extent or grading
   system. Explicitly subthreshold extent must not be overridden by grade.
4. Baker size handling uses local 15/30 mm cutoffs and clause-wide dimensions;
   these are not established as host thresholds by code comments. Other lesion
   dimensions can contaminate cyst grading, and rupture is automatically POS.
   Such heuristics need explicit provenance and conservative uncertainty handling.
5. Fracture/Contusion cap is broader than the builder description. It checks only
   the rules reader's fracture and contusion states, not whether other contusion
   readers refer to the fracture or to a distinct site. Twenty-six cells are capped.
   A fracture can coexist with a separate contusion. Remove unsupported caps or
   require site-specific evidence rather than whole-report co-occurrence.
6. Co-occurrence regressions are fitted using all 4349 reports, including the fixed
   825 validation reports, with global standardization and silence-mean calibration.
   This is not gold leakage, but it makes label-refinement parameters transductive.
   For a conventional held-out training experiment fit these parameters on the
   3524 training studies only, then apply without refitting. Preserve means within
   training silence groups, not the combined train/validation cohort.

The host webpage returned no readable text through the web tool in this review.
Severity comparisons above are against the local guide and saved prompts; their
host attribution was not independently reverified live. The local guide combines
host-attributed criteria with interpretation, so these must not be conflated.

## Evidence strength and evaluation

The extractor's 0.92–0.99 AUC is agreement against an LLM-consensus subset used
while fixing rules, not independent clinical accuracy. Its consensus negative
definition also includes silent states. Gold-equivalent quality cannot be inferred.

Effusion rank correlation with HPO-v1 remains 0.984 despite positive-at-0.5 share
falling from 48.8% to 32%. Meniscus changes likewise mostly preserve ordering.
Prevalence changes alone do not prove AUC improvement. Synovitis (rank correlation
0.822) and Fracture (0.670) deserve particular attention to changed ranking, not
just total positives. Silence co-occurrence assigns probable associations, not
observed abnormalities, and can strengthen dataset shortcuts.

Recommendation: prioritize a corrected label experiment over the weak sampling
hypothesis, but do not train this exact candidate unchanged. Fix demonstrable scope
and definition errors, constrain or remove unsupported heuristics, and fit learned
co-occurrence only on training IDs. First use the all-weight-1 candidate so label
replacement is the experimental factor. Test silence downweighting separately.
Keep the 0.924 ImageNet-initialized ResNet50 recipe, original sampling/BCE/scheduler,
3524 training IDs and fixed 825 Steven-v4 validation. Fresh training, not continuation.
Record all per-target metrics and predictions; fixed Steven validation measures
report agreement and may penalize legitimate definition changes. Do not discard
unfavorable targets after seeing results. Final submission scoring is necessary
to establish utility beyond that ruler; no public-score gain is guaranteed.
