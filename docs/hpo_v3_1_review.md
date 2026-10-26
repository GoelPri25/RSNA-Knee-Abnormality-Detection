# Independent HPO-v3.1 review

Ran the 190 regression checks locally: all pass. Verified every entry in
`../label_rules/hpo_v3_1/SHA256SUMS`: no mismatches. Rebuilt the label builder
into `data/label_audits/hpo_v3_1/rebuilt`; both trainable JSONs and frozen
co-occurrence parameters are byte-identical. Inspected the train-only masks,
standardization and calibration code. Confirmed 4349 unique report-only IDs,
zero overlap with gold IDs, all-one primary weights and identical probabilities
between primary and optional silence-downweighted variants.

Independent probes confirm corrected finding-specific severity, separate-site
fracture/contusion handling, 3 mm cartilage extent handling and reduced certainty
for dimension-only Baker classification. The previously reported defects are
substantially addressed. This is verification of implementation, not clinical
validation. Source correlation and heuristic probability mappings remain.

Definition decisions for the experiment:
- D1: keep non-joint/cyst-wall synovitis uncertain when joint involvement is
  unresolved; do not treat it as definite joint synovitis or proof of absence.
- D2: definite minimal/trace synovitis should be positive under the provided
  any-severity definition. Severity alone is not uncertainty about existence.
  Promote only the five reviewed definite assertions; preserve actual hedging
  and non-joint anatomy. Recommend this small correction before final freeze.
- D3: retain uncertainty for borderline extent and unspecified moderate OA;
  severe/advanced OA as positive is a documented interpretation, not a verified
  universal equivalence. Explicit contrary depth/extent should take precedence.
- D4: use explicit qualifying size description; rupture alone is uncertain,
  explicitly small ruptured cyst stays below threshold. Probe confirms code
  does the latter, although the D4 question wording suggests borderline.
  Dimension-only LIKELY=0.75 is an experimental heuristic.
- D5: retain uncertainty when grade supplies depth but no extent.
- D6: do not reinstate whole-report caps across readers lacking lesion sites.
  Keep site-specific handling in the rules reader; association does not establish
  that edema at one site explains a separate contusion elsewhere.

No further broad redesign is justified before testing. Primary trial should
use all weights one, fresh ImageNet ResNet50 and the actual single-T4 0.924
recipe, original 24-window sampling, BCE, scheduler and fixed 825 Steven-v4
validation. The referenced `hpo_v1_blend_resnet50_t4x2/run.py` wraps the trunk
in DataParallel: changing only its labels/hash preserves that two-GPU recipe,
not the original single-GPU computation/BatchNorm behavior. Avoid claiming an
exact reproduction of the original setup if switching hardware distribution.

Large departures from HPO-v1 remain unproven, especially Synovitis and Fracture.
Small corrections versus v3.0 do not imply small differences versus HPO-v1.
Track every target and retain predictions; report-validation disagreements
cannot be selectively excluded to declare success. Submission score determines
public utility, without guaranteeing hidden/private generalization.

No training was launched, original label files were not edited. Current primary
hash (before proposed D2 correction):
`12e97fbf78548bd1b6e93d13cf68bf0031f2957737f73df7d28bed136c31697e`.
