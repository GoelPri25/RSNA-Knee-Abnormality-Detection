# Development-only label audit — first pass

Kaggle kernel: [RSNA Report Label Source Audit](https://www.kaggle.com/code/tumis001/rsna-report-label-source-audit) (CPU, completed, version 4).

The comparison used only the existing 29-study development split. The other 29 gold studies were not scored. We compared report-derived scores with image-derived gold labels; this measures agreement with the competition's labeling target, not clinical truth. The challenge organizers explicitly state that gold labels were independently assigned from MRI by two MSK radiologists and adjudicated, and that report-image disagreements are expected ([label definitions and organizer clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343), [report disagreement discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)).

## Results

| Source | Development macro AUC | Mean target coverage | Finding |
| --- | ---: | ---: | --- |
| Our saved conservative-v2 development extraction, scored through the Confidence-V2 severity map | 0.881 | 64.1% | Partial abstention; this is a prompt-audit artifact, not the training-label CSV itself |
| Flight hybrid v4 | 0.891 | 100% | Clean candidate; on the exact same internal-covered decisions, macro AUC 0.913 vs 0.881 for our reference (+0.032) |
| Steven Lee Hans v4 blend | 0.890 | 100% | On the same decisions, macro AUC 0.915 (+0.033); likely related lineage to Flight |
| Riad/JEV hybrid scores | 0.899 | 100% | On the same decisions, macro AUC 0.918 (+0.037); slightly stronger here, but less independent provenance established |
| Yehezkiel `data.csv` | 1.000 | 100% | Exact copy of all development gold labels; excluded as leakage, not a valid label teacher |

The internal macro AUC is computed only where the extraction emitted a resolved present/absent decision; mean coverage is 64.1%. The matched-coverage comparison removes this denominator difference. All differences still come from only 29 studies and are noisy; the full-coverage AUC values should not be interpreted as proof that one teacher will yield a better trained model.

## Decision

Run a controlled, CPU-only A/B on the existing frozen MedSigLIP slice embeddings with the exact original hierarchical MIL recipe. Arm A uses our current Confidence-V2 training labels from `report_structured_labels.csv`; arm B uses Flight hybrid v4. Same architecture, seed, schedule, and all non-gold studies. Score only the existing development 29. The evaluation 29 remain unused. This matched head-only test is now running as [RSNA Report Label A/B on MedSigLIP Baseline](https://www.kaggle.com/code/tumis001/rsna-report-label-a-b-on-medsiglip-baseline).

The teammate dataset `priyanjaliml/skanee-blended-soft-labels` could not be attached: Kaggle's file-manifest API returned HTTP 403. It was excluded rather than silently substituting another source.
