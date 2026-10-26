# Improvement tracker (Priyanjali, 2026-10-21)

Every fix proposed to raise the single-model score past E4 (LB 0.946) and to speed up the efficiency submission. Source: Ledger = from findings in `experiment_ledger.md`; New = proposed in Priyanjali's 2026-10-21 review. Expected gains are rough macro-AUC estimates vs E4 and overlap. Spreadsheet version with drop-downs: `docs/improvement_tracker.xlsx`.

## Accuracy fixes

| ID | Fix | Source | Type | What changes | Expected gain | Status | Owner | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | Start student from the teacher (rap2 D) | Ledger | Training | CoAtNet-384 starts from Raptor's 0.949 weights, fine-tuned 6 epochs on the full 0.958-blend targets | +0.004 to +0.009 vs E4 | Blocked |  | OAI teacher risk (weights start from the OAI-trained model) |
| 2 | Online distillation | Ledger | Training | Teacher scores the student's exact augmented windows every step; R6/reader part offline, shifted per view | 0 to +0.003 | Done | Priyanjali | Fold 0 fused 0.9499 vs E4 0.9471 (+0.0028, P 1.00); global 0.9501 vs 0.9447 (+0.0054); gold 0.9216 vs 0.9181; LB probe pending |
| 3 | Mixup / strong augmentation with the online teacher | New | Training | Mix two studies' windows; heavier augmentation; teacher labels each mixed view | +0.002 to +0.005 | Not started |  |  |
| 4 | Logit-matching loss / temperature | New | Training | Add MSE on teacher logits (or BCE at T = 2-4) | +0.001 to +0.003 | Not started |  |  |
| 5 | Attention distillation | Ledger | Training | KL between student and teacher window attention | +0.001 to +0.003 | Deferred | Sumit | KD-ATT-T free test (ledger 2026-10-21): inconclusive, leaning against; no paid lane yet |
| 6 | 48 training windows | Ledger | Training | 48 instead of 16 windows per study per step | +0.001 to +0.002 | In progress | Sumit | KD-W48 pre-registered and launched (ledger commit db00851) |
| 7 | Axial quotas + series dropout | Ledger | Training | Every draw includes axial windows; whole series dropped at random | +0.001 to +0.004 | Not started |  |  |
| 8 | Train on all data | Ledger | Training | Fold 0 + 58 gold added: 3,524 -> 4,407 studies | +0.002 to +0.004 | Done | Sumit | E4-ALL trained (Modal H100); kernel tumis001/rsna-pubkd-e4all-submission pushed; LB submission awaits go |
| 9 | Confidence-weighted teacher / report mix | New | Training | Per-cell mix: label-v2 where the report is explicit, teacher where silent | +0.001 to +0.002 | Not started |  |  |
| 10 | Bigger student (ResNet-101 / ConvNeXt-S 22k) | New | Training | Higher-capacity backbone | +0.002 to +0.004 | Not started |  |  |
| 11 | Ranking (AUC) loss | New | Training | Pairwise loss on teacher-ordered pairs, with a memory of recent scores | +0.001 to +0.002 | Not started |  |  |
| 12 | Severity / tag auxiliary heads | New | Training | Extra heads on severity_aux_targets.json and tag_aux_targets.json | +0.001 to +0.002 | Not started |  |  |
| 13 | Native-resolution crops | New | Training + Test | Crops from original DICOM pixels, not the 0.365 mm cache | Uncertain; potentially large for menisci / ACL / cartilage | Deferred |  | After deadline |
| 14 | Mirror averaging at test time | Ledger | Test-time | Average plain + mirrored predictions | 0 to +0.002 | Not started |  |  |
| 15 | Shifted-window averaging | New | Test-time | Average over 2 offsets of the 94 window centres | 0 to +0.001 | Not started |  |  |
| 16 | Per-finding fused vs global | Ledger | Test-time | Pick the better head per finding from fold 0 | 0 to +0.001 | Not started |  |  |
| 17 | Anatomical slice priors for crops | New | Test-time | Crop slices chosen inside per-finding anatomical bands | 0 to +0.002 | Not started |  |  |
| 18 | PF OA / contusion / fracture crop fixes | New | Test-time | Patella-centred PF OA crop; 2 boxes for contusion / fracture | 0 to +0.001 | Not started |  |  |

## Efficiency fixes

| ID | Fix | Source | Type | What changes | Expected gain | Status | Owner | Notes |
|---|---|---|---|---|---|---|---|---|
| E1 | Measure each inference stage | New | Test-time | Time headers, decode, global pass, placement, crops on ~200 training studies (T4 x2) | - (tells what to fix) | In progress | Sumit | FAST-G eval kernel: per-stage timing on fold-0 studies |
| E2 | Cheaper DICOM header reading | New | Test-time | Parse only needed tags, no double opens, more threads | Several minutes of runtime | In progress | Sumit | FAST-G: header reads limited to IOP/IPP/PixelSpacing/InstanceNumber, one job per (study, series) over all CPUs |
| E3 | Both T4s + decoding overlapped with GPU | New | Test-time | Two model copies, one per GPU; CPU decodes ahead | Up to ~half of GPU time | In progress | Sumit | FAST-G: decoding starts before GPU init; single GPU so far |
| E4 | Cheaper crops or none at inference | New | Test-time | 1 crop per finding / only where helpful / none if > ~70 s total | ~1/3 of GPU work | In progress | Sumit | FAST-G = F1 global head only (no crops), LB 0.946 |
| E5 | No package installs at run time | New | Test-time | Use preinstalled timm or torchvision ResNet-50 | ~1 min | Done | Sumit | FAST-G: plain-torch ResNet-50, BN folded into fp16 convs, no timm / pip / hashing |
| E6 | 48 instead of 94 evaluation windows | New | Test-time | Fewer windows in the global pass | ~1/2 of global pass | Not started |  | Only if fold 0 holds |
| E7 | torch.compile | New | Test-time | Compile the model | 10-30% of GPU time | Not started |  |  |

- kd_e4cnxs (2026-10-22): ConvNeXt-S ImageNet-22k backbone (convnext_small.fb_in22k_ft_in1k) instead of ResNet-50; fused label-v2 +0.0058 / Steven +0.0051 vs E4; dataset `priyanjaliml/rsna-pubkd-e4cnxs-fold0`.

- kd_e4rk (2026-10-22): pairwise ranking loss (0.3 x RankNet on target-ordered pairs, batch + 512-study memory bank, margin 0.2) added to BCE; fused label-v2 +0.0015 / Steven +0.0002 vs E4; dataset `priyanjaliml/rsna-pubkd-e4rk-fold0`.
