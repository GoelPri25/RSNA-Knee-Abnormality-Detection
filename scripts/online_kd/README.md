# Online distillation for the E4 student (ONLINE-KD) — Priyanjali, 2026-10-21

Ledger row: **ONLINE-KD** in `docs/experiment_ledger.md`. Tracker: `docs/improvement_tracker.{md,xlsx}` (item 2).

## Idea
E4 (`pubkd_kd_e4`) trains the r50-GL4 `l4copy` student on **offline** targets: one number per study and finding (full
0.958-blend rank order with Raptor values, mixed 0.7 / 0.3 with label-v2). Each step the student only sees 16 random,
augmented windows, so the fixed target can ask for a confident score when the evidence is not in those windows.

**Online** distillation recomputes the target for the exact view the student sees. The frozen Raptor teacher
(nartaa CoAtNet-rmlp-2 384 + 224) scores the student's own augmented windows every step. R6 and the 2.5D reader cannot
run on d96 windows, so the blend part stays offline and is shifted by the Raptor's view-vs-clean-view change:

```
v      = 0.65 * z384(view) + 0.35 * z224(view)            384: 320 centre crop of the augmented 384 window
                                                          224: 0.8 centre crop resized to 224
c      = same blend on the stored clean view               raptor{384,224}_{0,1}.json, mirror-averaged, 94 windows
y_view = sigmoid(logit(y_pure) + clamp(v - c, -4, 4))      y_pure = pubkd_targets_fullorder_rapvalues.json
y      = 0.7 * y_view + 0.3 * y_labelv2  where label-v2 weight > 0, else y_view     (E4's mix73 rule)
```
With a zero shift this reproduces E4's offline target file exactly. Nothing else changes: architecture, seed 2026,
folds 1–5, 16 windows, augmentation, 25 epochs, SWA 21–25, BN refresh. **No teacher at inference.**

## Files
| File | Role |
|---|---|
| `online_kd.py` | `OnlineTeacher` (teacher loading with sha check, per-view targets, identity check, clean-view parity gate, per-epoch shift stats) and `LogitTargets` (offline logit matching) |
| `apply_online_patch.py` | 5 asserted edits to `r50_gl4_runner.py`: `V4_ONLINE=1` lane flag (in G4 / CFG, so a resume with a different setup is refused), target replacement in `train_step_loss`, indexed dataset + teacher in `main`, epoch stats. Unset = E4 unchanged |
| `apply_logit_patch.py` | Optional second patch: `V4_LOGIT=<w>` adds `w * MSE(student logit, teacher logit)` (offline: logit of the pure teacher target; online: the shifted view logit). Built and CPU-tested; its run was lost to a reclaimed runtime and not repeated |
| `score_fold0.py` | Fold-0 macro AUC on label-v2 and Steven-v4 confident cells (≤ 0.3 / ≥ 0.7), paired study bootstrap (2,000) between prediction files |
| `make_global_slim.py` | Full checkpoint → slim fp16 global-only weights for FAST-G (same conversion as `e4all_slim/convert.py`), with a folding check |
| `colab_cli/` | Colab Pro runs driven from a laptop with Google's Colab CLI (below) |

## Running on Colab Pro with the Colab CLI
Needs `google-colab-cli` (`colab --auth oauth2 …`, one-time browser sign-in with all scopes ticked), the teammate kit
`tumis001/rsna-pubkd-teammate-kit`, the d96 shard kernels `tumis001/rsna-coatnet-d96-cache-shard-0..3`, the public
`nartaa/rsna-knee-publication-swa-weights-20261007` and a Kaggle API token.

1. `colab new -s <name> --gpu A100 --high-mem`; upload `/content/okd/*` (`online_kd.py`, both patches, `score_fold0.py`,
   `colab_cli/vm_*`, `kit_seq.py`); write `/content/kaggle.env` (`KAGGLE_API_TOKEN=…`) on the VM only.
2. `vm_fast_cache.sh` (4 cache shards in parallel, ~12 MB/s each, ~25 min) → `vm_prepare.sh` (kit file by file with
   `kit_seq.py`, Raptor weights, competition `train.csv`, the teammate's `colab_pubkd_setup.py` with the torch 2.5.1
   pin, `apply_online_patch.py`, lane env `pubkd_kd_e4on`).
3. `vm_check.sh` (GPU gates: identity check, 384 teacher parity ≤ 0.03, teacher time per micro-batch).
4. `colab_pubkd_lane.sh kd_e4on` (smoke, then the lane, detached). `vm_status.sh` per-epoch summary.
5. After DONE: `vm_score.sh` (fold 0 vs E4 / cv1, gold via `gold_gl4_predictions.py` + `score_gold.py`).

Lessons (cost us time):
- **Kaggle:** the whole-dataset download of the kit returns 404, and parallel per-file downloads hit HTTP 429 →
  `kit_seq.py` fetches sequentially. Strip the CSV header line from the file list.
- **Colab reclaims runtimes whose kernel looks idle** (training runs outside the kernel). The L4 lane was lost in epoch 1
  and the A100 after epoch 23 (it came back as a CPU runtime). Run `kernel_busy.py` through `colab exec -f … --timeout`
  for the whole run, and copy `recovery.pt` + `weights_epoch*.pt` off the VM every few minutes (`vmpull.sh`).
  `recovery.pt` restores model, AdamW state and LR schedule (not RNG), so a resume on a new VM continues exactly.
- **One SSH connection per runtime at a time** (HTTP 429 otherwise); `pkill -f` patterns must not match the SSH command itself.
- **Costs:** A100 high-mem ~6.8 compute units/h; L4 ~1.5 units/h (≈3.5× slower); H100 was not available on the account.

## Result (seed 2026, fold 0, 825 studies, paired bootstrap ×2,000)
| | online (`kd_e4on`) | E4 | diff (95% CI, P>0) |
|---|---|---|---|
| fused, label-v2 cells | **0.9499** | 0.9471 | **+0.0028** [+0.0010, +0.0048], 1.00 (vs cv1 +0.0024, 0.98) |
| global, label-v2 cells | **0.9501** | 0.9447 | **+0.0054** [+0.0029, +0.0085], 1.00 (vs cv1 +0.0037, 0.99) |
| fused, Steven-v4 cells | 0.9149 | 0.9144 | +0.0005, 0.74 |
| global, Steven-v4 cells | 0.9148 | 0.9133 | +0.0014, 0.92 |
| gold-58, global head | 0.9216 | 0.9181 | descriptive (cv1 0.9275; gold has not tracked the LB) |

Per finding (fused, label-v2): MCL +0.019, PF OA +0.005, fracture +0.005, synovitis +0.004, contusion +0.003, medial
meniscus +0.003; ACL −0.002, medial OA −0.001, Baker's −0.001. Fused ahead of E4 in 24 of 25 epochs, no dips (E4 dipped
at epochs 7 and 9), smaller train−fold-0 gap. Mean per-view shift |d| = 0.27 logits, 0% clipped. Training ~377 s/epoch
on an A100 (E4 ~171 s); the teacher costs 0.115 s per micro-batch of 2 studies × 16 windows.

Caveats: one seed; the teacher was trained on fold-0 studies, so fold-0 gains of teacher-trained students may be
optimistic. The LB probe decides.

## Inference (no teacher needed)
- **Fused, main track:** `priyanjaliml/rsna-pubkd-online-e4on-fold0` — `weights_swa21_25.pt`, sha256
  `5ad74809a0caccfe40b317148fa1559289c91765a9c42577eeae5b6eb75fd8d0`. Drop-in for E4 (same 579 keys and shapes):
  `kaggle/submission_kernels/pubkd_kd_e4_submission/submission.py` with `WEIGHTS` pointed at this dataset and sha.
- **Global only, efficiency (FAST-G):** `priyanjaliml/rsna-pubkd-online-e4on-global` — `online_e4on_global_fp16.pt`,
  sha256 `b66acd5c947118f7ba9c9411b124240f84707f511675918a75b3ea7de77ba57f` (folding check: rel diff 2.5e-4). Run
  FAST-G's fidelity gate against the included `val_swa_global.json`.

## 2026-10-22 additions (Priyanjali)

### Inference: rank blend and mirror TTA (`inference/`)
- `gl4_core.py`: one inference core for the GL4 students: archs `r50` (E4 family), `cnxs` (KD-E4CNXS), `r101`, `bit101`,
  `rap2` (rap2-D, global head only). The E4 kernel's d96 builder, self-located crops and fp16-with-fp32-retry path,
  unchanged. Global-only checkpoints load strict (crop-branch keys checked and dropped). Checked: r50 = the E4 submission
  kernel exactly (max |d| 0.0); cnxs and bit101 global / crop features = `r50_gl4_runner.RaptorGL4` (0.0 / 7e-8);
  rap2 = `pubkd_rap2d_global_submission` (0.0).
- `rank_blend_submission.py`: every model on every study, per-finding percentile ranks over the test studies, per-finding
  weighted mean (`BLEND = "prob"` for the A/B). Default ConvNeXt-S global (LB 0.947) + rap2-D global (0.948), 0.5 / 0.5.
  Fold 0, saved predictions: ConvNeXt-S + an E4-recipe student adds +0.0005 at most (rank = prob to 4 decimals).
- `mirror_tta_submission.py`: one model, plain + anatomical mirror (the training augmentation), mean of probabilities.
- Both: MODE "fold0" re-predicts the fold-0 studies from DICOM and writes val-format JSONs for `score_fold0.py`; before the
  test set each model must match 8 of its own validation-fold studies (saved `val_swa_global.json`) within 0.01.
  `build_kernels.py` inlines the core and those references into `inference/kernels/<slug>/submission.py` (reads the local
  handoff packages under `reports/`; rebuild after editing).

### E4 lanes on Modal (`modal_lanes/`)
- `modal_e4_lane.py` ports the Colab pipeline: `prepare` (CPU: d96 shards + teammate kit from Kaggle, MeluXina manifest and
  pinned-input checks, extra ImageNet weights) and `train` (H100: env = kit `pubkd_kd_e4.env` + `V4_BB` / `V4_VAL_FOLD` /
  seed, runner patched in the container, smoke first, volume committed every 5 min, resumes from recovery.pt).
  Kaggle token = Modal secret `kaggle`. `spawn.py` starts a deployed function detached.
- Patches for the kit's runner copy: `apply_bit101_patch.py` (BiT ResNet-101 21k, `resnetv2_101x1_bit.goog_in21k_ft_in1k`,
  GroupNorm, mean / std 0.5, crop copy of stages.3 + norm), `apply_valfold_patch.py` (validation fold; the branch runner
  has its own `V4_VAL_FOLD`).
- **KD-E4BIT101-F1** (done): BiT-101 on the E4 recipe, validation fold 1, SWA 21-25 fold-1 label-v2 (runner readout, all
  cells) fused 0.9630 / global 0.9627; 2.6 h on one H100. Kaggle `priyanjaliml/rsna-pubkd-e4bit101-fold1` (fused and
  global-only weights). Not comparable with fold-0 numbers (fold 1 is easier; see the fold-0 twin below).
- **KD-E4BIT101** (done, fold 0, same recipe, Modal H100 ~2.7 h): Kaggle `priyanjaliml/rsna-pubkd-e4bit101-fold0`
  (fused sha 19cb975f…, global-only sha 9a2ed00c…). `score_fold0.py`, 825 studies, confident cells, bootstrap x2,000:

  | head | ruler | BiT-101 | vs E4 | vs ConvNeXt-S |
  |---|---|---|---|---|
  | fused | label-v2 | 0.9515 | +0.0044 (P 1.00) | −0.0014 (P 0.12) |
  | fused | Steven-v4 | 0.9181 | +0.0037 (P 1.00) | −0.0014 (P 0.10) |
  | global | label-v2 | 0.9514 | +0.0067 (P 1.00) | −0.0020 (P 0.06) |
  | global | Steven-v4 | 0.9178 | +0.0045 (P 1.00) | −0.0023 (P 0.04) |

  Second-best student (E4 < BiT-101 < ConvNeXt-S). Blend with ConvNeXt-S (global heads, percentile rank; probability
  averaging within 0.0002): 0.7 / 0.3 = label-v2 0.9539 (+0.0006, P 0.92), Steven 0.9210 (+0.0009 [+0.0001, +0.0017],
  P 0.99): the only blend of our students with a clear Steven gain, still ~+0.001. ConvNeXt-S + an E4-recipe ResNet-50:
  ≤ +0.0006; equal-weight blends of all five fold-0 students lose (−0.0026 label-v2).
