# RSNA Knee Abnormality Detection

Deep-learning pipeline for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition (2026): detect 12 abnormalities from multi-series knee MRI studies, scored by macro ROC-AUC on a
hidden test set, with a separate efficiency prize that trades AUC against inference runtime on 2x T4 GPUs.

**Findings:** ACL, MCL, medial / lateral meniscus, medial / lateral / patellofemoral osteoarthritis, effusion,
synovitis, Baker's cyst, contusion, fracture. Training labels are derived from free-text radiology reports.

This repository is a cleaned snapshot of a two-person team project: training code, Kaggle submission and evaluation
kernels, cloud launchers, and the research documentation behind every decision. No competition data, labels or model
weights are included.

---

## Approach

```
DICOM study ──► study volume ──► 94 three-slice windows ──► CNN backbone ──► per-finding attention (MIL) ──► 12 logits
 (5 series)     96 slices,        (c-1, c, c+1), 320 px      ResNet-50 / BiT-101 /    a weight per window           │
                140 mm crop,      centre crop                ConvNeXt-S / CoAtNet      and finding                   │
                2/98 window                                                     │                                    │
                                                    zoomed crops of the top-attention windows ──► crop branch ───────┘ (fused head)
```

1. **Study volume.** Five series slots (sagittal fluid / non-fluid, coronal fluid / non-fluid, axial) are sampled
   evenly over the 2–98 % range of each stack, intensity-windowed per series (2nd–98th percentile), cropped to a
   140 mm field of view and resized to 384 px.
2. **Multiple-instance learning.** Each 3-slice window is encoded by a CNN; a learned attention head weights the
   windows separately for every finding, so each finding reads the slices that matter for it.
3. **Global + local heads.** An optional crop branch (its own copy of the backbone's last stage) reads zoomed crops
   at the windows the attention selects; fused output = sigmoid((global + local) / 2).
4. **Knowledge distillation.** Students are trained on soft targets from a strong public ensemble (its rank order,
   with values from a CoAtNet teacher, mixed 70 / 30 with report-derived labels); online distillation lets the
   frozen teacher score the student's own augmented views each step.
5. **Inference.** Percentile-rank blending with per-finding weights, anatomical mirror test-time augmentation,
   start-up fidelity checks against the training pipeline, and fast torch-free TensorRT kernels for the efficiency
   track.

## Results

Public-leaderboard progression of selected single models and ensembles (full record in
[docs/experiment_ledger.md](docs/experiment_ledger.md)):

| Model | Public LB (macro AUC) |
|---|---|
| ResNet-34, report labels (first recipe) | 0.918 |
| ResNet-50, refined report labels | 0.925 |
| ResNet-50 on the d96 study volume (Exp 1e) | 0.936 |
| ResNet-50 global + crop heads (v4 l4copy) | 0.939 |
| Ensemble of 7 earlier models | 0.943 |
| ResNet-50 student distilled from the public ensemble (E4) | 0.946 |
| ConvNeXt-S (ImageNet-22k) student, global head | 0.947 |
| CoAtNet student initialised from a public 384 px model (rap2-D) | 0.948 |

Fold-0 comparison of the final distilled students (825 held-out studies, confident cells, paired bootstrap with
2,000 resamples; two independent label sets):

| Student (same recipe, backbone swapped) | Report-label AUC | Stricter label set AUC |
|---|---|---|
| ResNet-50 (E4) | 0.9471 | 0.9144 |
| BiT ResNet-101 (ImageNet-21k) | 0.9515 | 0.9181 |
| ConvNeXt-S (ImageNet-22k) | **0.9529** | **0.9196** |
| ConvNeXt-S 0.7 + BiT-101 0.3 (rank blend, global heads) | 0.9539 | 0.9210 |

Notable measured negatives (kept in the ledger on purpose): online distillation gained +0.0028 on fold 0 but nothing
on the leaderboard; 48 instead of 16 training windows gained only +0.001 at 3x compute; using one crop per finding
instead of four cost about −0.003.


## Getting started

**Environment.** Python 3.11; the training stack pins torch 2.5.1, timm 1.0.29 and numpy 2.4.6.

```sh
uv sync --extra test       # or: docker compose build && docker compose run --rm app
uv run pytest              # unit tests on synthetic DICOMs
```

**Data.** Accept the competition rules on Kaggle and download the data there; it is not redistributed here. The
d96 study-volume cache is built by the cache kernels in `kaggle/experiment_kernels/` (`coatnet_d96_cache`).

**Training.** The final students are trained by `scripts/r50_gl4_runner.py` (configured entirely by environment
variables; see the env files in `scripts/meluxina/env/` and the lane launchers):
- Modal (H100): `scripts/online_kd/modal_lanes/modal_e4_lane.py` (download / verify step + training step)
- Google Colab Pro via the Colab CLI: `scripts/online_kd/colab_cli/`
- Slurm (MeluXina): `scripts/meluxina/*.sbatch`

**Evaluation.** `scripts/online_kd/score_fold0.py` (fold-0 macro AUC with paired bootstrap).

**Inference.** `scripts/online_kd/inference/` holds the shared inference core and the rank-blend / mirror-TTA
kernels; `kaggle/submission_kernels/` holds every submitted kernel, including the fast efficiency-track kernels.

## Documentation

Start with [docs/README.md](docs/README.md). Key documents: the [experiment ledger](docs/experiment_ledger.md)
(every experiment, its pre-registered decision rule and outcome), the
[project learning synthesis](docs/project_learning_synthesis.md), the
[pipeline evolution study](docs/pipeline_evolution_study.md) and the
[public-solution notes](docs/COMPETITIVE_NOTEBOOK_NOTES.md).

## Notes

- Competition data, report-derived labels, caches and trained weights are excluded (competition data licence).
- Kaggle dataset and kernel references in the code point to the team's own (mostly private) Kaggle resources and
  to public resources by other competitors, credited where used.
- External pretrained weights (ImageNet, public competition checkpoints) are used under their own licences.

## License

[MIT](LICENSE)
