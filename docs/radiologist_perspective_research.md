# Radiologist perspective on knee MRI findings (research 2026-10-19)

Web research only (three agents: per-finding criteria, report language, published models). No project data left the machine. Items the agents could not source are omitted or marked.

## Per-finding decisive evidence and label ceilings

| Finding | Decisive sequence / plane | Key secondary signs and mimics | Reader ceiling |
|---|---|---|---|
| ACL | Sagittal FS; coronal / axial confirm the femoral footprint | Anterior tibial translation > 7 mm, posterior horn of the lateral meniscus uncovered > 3.5 mm, PCL buckling, pivot-shift contusions, Segond (~95% ACL) | Sens 87%, spec 93% vs arthroscopy (Phelan 2016) |
| MCL | Coronal FS; axial confirms the layers | Fluid around the MCL also comes from OA, parameniscal cysts, bursitis: low grades are over-called; grade 1 vs 2 is fuzzy | Sens 7–100% across studies (no standard grading; Meyer 2021, PMC9596543) |
| Menisci | Sagittal + coronal; thin axial for radial / root | Two-slice touch (PPV medial 91→94%, lateral 83→96%; De Smet AJR 2006); bucket-handle signs; parameniscal cyst; mimics: transverse ligament, meniscofemoral ligaments, popliteus, magic angle (short-TE only) | Medial 89 / 88, lateral 78 / 95; lateral worse with an ACL tear (72.9 vs 88.0% accuracy) |
| OA (3) | Cartilage on FS; osteophytes on non-FS T1 / PD; BML on FS | ESSR advises against writing "OA" on MRI: labels depend on wording | Cartilage sens 0.70–0.80; MOAKS tibial cartilage κ 0.36, osteophytes 0.49 |
| Effusion | Sagittal suprapatellar recess, axial FS | ≥ 10 mm AP pathological; "small / physiologic" = normal amount | MOAKS effusion-synovitis κ 0.72 |
| Synovitis | Sagittal FS (Hoffa signal), effusion-synovitis | Non-contrast proxy only; Hoffa spec 47–64% vs contrast | Hoffa intra-rater κ 0.42: lowest ceiling |
| Baker's | Axial FS: neck between medial gastrocnemius and semimembranosus | Mimics without the neck: parameniscal cyst, ganglion, bursitis | Unsourced |
| Contusion | FS / STIR high signal, T1 reticular without a line | Pattern after ACL tear: lateral tibial plateau 83%, lateral femoral condyle 71%; mimics: red marrow (T1 > muscle), OA BMLs | Time since injury changes presence (resolves 6 weeks to 12 months) |
| Fracture | **T1 / non-FS shows the line**; FS shows oedema | Avulsions have little oedema (2.5 mm vs 31 mm for impaction); MRI missed 4 / 7 Segond | — |

ESSR consensus (PMC11399221): 3 planes, ≤ 3 mm slices (4 mm acceptable), T1 required for marrow and fracture lines.
Thin slices help menisci: 3 mm sagittal sens medial 90.6% / lateral 72.7% vs 1 mm axial 97.3 / 95.7% (Gökalp 2012, PMC3474033).

## How reports are written (silent and mild cells)

- Templates state ACL / MCL / menisci as "intact": silence there is usually a real negative. Effusion, synovitis, Baker's, trace oedema and early OA are omitted when judged physiologic or trivial (convention; partly quantified: a UK audit found 50% of BML + cyst cases not called OA, ESSR 2019 P-0132).
- Chest X-ray analogue (Olatunji 2019, arXiv 1905.02283): report recall vs image labels 0.11 for borderline cardiomegaly; omissions are selective (borderline, age-related, not actionable), not random.
- Hedge scale (Panicek & Hricak AJR 2016): "consistent with" > 90%, "suspicious / probable" ~75%, "possible" ~50%, "unlikely" < 10%; usage is inconsistent between radiologists.
- Meniscal grade 2 intrasubstance signal (not a tear) in 24–41% of asymptomatic knees; tears in 19–56% of people aged 50–90 (Englund NEJM 2008): degenerative tears are base-rate findings whose mention depends on the radiologist.

## Published mechanisms with matched ablations

- **CoPAS** (Nat Commun 2024, doi 10.1038/s41467-024-51888-4): 1,748 patients, 12 knee findings, 3D ResNet18 from scratch. Mean AUC multitask 0.744 → + cross-plane attention 0.774 → + cross-sequence (FS ↔ non-FS) attention 0.799 → + finding-correlation module 0.812; per-finding models −0.004. One seed, ~220 test cases, weak baseline.
- Multi-view mammography: cross-view transformer +0.034 over late fusion (arXiv 2206.10096); same-breast view matching +2.2 recall (arXiv 2105.10160).
- ORACLE-CT (preprint arXiv 2606.05460): anatomy masks restrict per-label pooling, +0.02 internal / +0.03 external.
- CheXpert (arXiv 1901.07031): the best uncertain-label policy is finding-specific (U-Ones atelectasis +0.047, consolidation −0.033). Pham (arXiv 1911.06475): conditional training + label smoothing on uncertain labels +0.034 (small validation set).
- Missing-label losses (Hill / SPLC, arXiv 2112.07368) gain +2–5 mAP on COCO with random missing labels; untested under selective omission.
- No gain: 3D vs 2D for ACL (Namiri 2020: 89 vs 92%); MRNet-validation architectures did not replicate on CoPAS data; logit-level label graphs ≈ +0.00x on strong baselines.
- Foundation models (OrthoFoundation 2026 MIT, KneePreM, MARS): no matched comparison against generic initialisation on knee classification.

## Mapping to our ledger (2026-10-19)

| Idea | Status in our work |
|---|---|
| Finding co-occurrence (CoPAS CRM, label graphs) | JF-0: no gain at logit level; MCL the exception (+0.005 / +0.012), consistent with the MCL pitfall above |
| FS ↔ non-FS at the same location (CoPAS CSA) | MC-PAIR early fusion: partner path never learned (conv1 partner half 3% of own norm). Mid-level fusion untested |
| Cross-plane correspondence (CoPAS CPA, mammography) | MP-0 frozen probe borderline (+0.0028, 89%); MS1-GL failed for other reasons. Trained cross-plane binding untested |
| Anatomy-restricted pooling (ORACLE-CT) | Close to guided windows (B+D), spotlights (v3), per-finding spatial attention: no gain |
| Two-slice touch / thin slices | SK-0 underpowered; menisci trend negative with skipped slices |
| Finding-specific silent / hedge policy | silw (global) at the floor; label-v2 already has HEDGE / BORD / BELOW / LIKELY states |
| Foundation models | Frozen OrthoFoundation weak (Steven fold 0 0.851); fine-tune untested (~15× r50 FLOPs) |
