# r50-GL v4 rev 2 — implementation / feasibility review, round 2 (2026-10-18)

Scope: `docs/exp_v4_design_rev2.md` (protected global, detached-shared-weight local with LoRA + dual BN, OOF-residual
fusion, batch A teachers → batch B). Read: `coatnet_d96_runner.py` (teacher phases `:331-347`, `teacher_ids` `:409-420`,
`main` `:1173-1479`), `r50_gl_runner.py` (v2), `run_cnxa_teacher*/`, `run_r50_pertarget/run_summary.json`,
`kaggle/submission_kernels/d96_labelv2_fold0_eval/`, `assets/teacher_splits.csv`, ledger row 34 (teammate label-v2) and
MP-0. Checks were numpy only, with no network.

## Verdict

Implementable in about a day of code and tests. One MeluXina allocation of about 3.5 h runs batch A and then batch B. I
found **four new correctness traps**. Each would silently break the claim that the global path is "protected by
construction": shared BatchNorm affine weights (N1), joint gradient clipping (N2), SWA of LoRA factors (N3) and the
baseline phase not scoring epochs 10–12 (N4). All are cheap to fix. Two design numbers need correcting: the LoRA size is
≈ 1.1 M parameters at rank 32 on the 1×1 convolutions, not 3–4 M, and the teachers split training folds **1–5** into 4
splits, not "folds 1–4".

## 1. Batch A: label-v2 r50 teachers

**Which phase reproduces label-v2 r50?** No current phase is "plain r50 on label-v2".
- `kaggle_fold0_labelv2` is the **teammate's** checkpoint (`priyanjaliml/rsna-knee-labelv2-checkpoint`,
  `r50_labelv2_swa21_25.pt`, sha 1c6006b1). Its fold-0 predictions came from our inference-only T4 kernel
  (`kaggle/submission_kernels/d96_labelv2_fold0_eval/run.py`). **Its training config was never recorded.** Ledger row 34
  says only "Exp 1e retrained on label-v2".
- Exp 1e's recorded CFG (`run_r50_pertarget/run_summary.json`) and the `r50_gl` CFG (`coatnet_d96_runner.py:373-377`) are
  **identical except** `version`, the label file and sha (any_model d40803 → latD2 dd0ab9), and `eval_epochs`
  ([5, 10, 15, 20, 25] in r50_gl).
- So the recipe is reproducible: same arch, LRs, schedule, SWA, augmentation and windows. It is not reproducible bitwise
  (unknown teammate code version and hardware, cuDNN non-determinism, `eval_epochs` changing CPU RNG draws).
- MP-0 showed our inference path reproduces the teammate's SWA weights within max |Δp| 0.006 on 20 fold-0 studies, so
  the 0.9425 bar is comparable.

**Teachers.** Extend the `_teacher` family (`coatnet_d96_runner.py:334-347`) with `r50lv2`: arch `resnet50.tv_in1k`,
labels `soft_labels_final_per_target_latD2.json` with sha dd0ab9ab (checked, the local assets match), version
`d96_r50lv2_teacher_v1`. Keep `teacher_splits.csv` sha 7cb597a7 (checked), `eval_epochs` [5, …, 25] for monitoring only,
no TTA. The rest is the existing teacher path:
- `teacher_ids` asserts the splits cover exactly the training folds, fit and oof are disjoint, and there is no fold 0
  (`:409-420`).
- Outputs: `oof_swa.json` (the held-out split), `val_swa.json`, `gold_swa.json`.
- I checked the existing cnxa teachers: the 4 oof files cover the 3,524 training studies exactly once, from folds 1–5,
  none from fold 0 (split sizes ~882).

**Seed override.** `CFG["seed"]` is fixed at 2026 (`:52`). Add `D96_SEED`: it sets `CFG["seed"]` and appends
`_seed{n}` to `version`, so the recovery assert sees it.

**Baseline phase `r50_lv2`** (label-v2 r50 seed 2027): 1e CFG plus label-v2, **without `eval_epochs`**, so fold 0 is
scored every epoch. The early-stop rule needs epochs 10–12 (see N4).

**Cost on A100:** fit 2,642 studies → about 50–65 min per teacher (1e took 2,590 s on H100 for 3,524 studies; A100 is
about ×2). The design's "2–2.5 h" for batch A is too high: **~1 h**.

A free alternative, a methods call: the existing CK-A (cnxa) teachers already give cross-fitted label-v2 OOF logits.
They are the wrong architecture to offset an r50 global, but usable as a descriptive control.

## 2. v4 runner, built on `r50_gl_runner` (v2)

**Detach mechanism: generalise v2's `crop` flag rather than `torch.func.functional_call`.** `encode(x, crop)` already
toggles `DualBN.crop` on every BatchNorm (`r50_gl_runner.py:258-264`). Do the same for convolutions: wrap every `Conv2d` in
the backbone as `SharedConv`. Its forward is `F.conv2d(x, w.detach() if self.crop else w, …)`, plus, when `crop` is set
and the conv is a layer3 / layer4 1×1, a LoRA residual `B(A(x))·α/r`. Use stride on A for the downsample convs, and
zero-init B. This is equivalent to `functional_call` with a detached parameter dict but:
1. composes with the existing DualBN flags;
2. keeps state-dict names, so SWA, recovery and loading the global into a plain `Raptor` still work;
3. avoids `functional_call`'s handling of buffers. Passing buffers in the dict makes BatchNorm update the dict's tensors,
   not the module's.

If `functional_call` is preferred, the dict must include **every** shared parameter, BatchNorm weight and bias included.

**N1 (blocker): shared BatchNorm affine leaks.** The v2/v3 `DualBN` crop branch calls
`F.batch_norm(x, crop_mean, crop_var, self.weight, self.bias, …)` (`r50_gl_runner.py:194`). Reused unchanged, the local
loss trains the **shared** γ and β, which the global path uses, so protection fails. Use one of these in the crop branch:
- `self.weight.detach(), self.bias.detach()`; or
- **own crop affine** (`crop_weight`, `crop_bias`, initialised from ImageNet), recommended **only in layer3 / layer4**,
  where the LoRA lives.

With shared-detached affine in stem–layer2 and no trainable crop parameters there, run **stem → layer2 of the crop pass
under `torch.no_grad()`**. Only layer3 / layer4 keep activations, and the backward computes activation and LoRA gradients
but no shared-weight gradients. That makes the crop pass cost about half of v2's per image (below).

**LoRA size.** Rank 32 on the layer3 / layer4 1×1 convolutions (conv1, conv3, downsample: 20 convs) is **1.08 M**
parameters. Adding the 3×3 convolutions gives 2.06 M. Rank 96 or 128 on the 1×1 convolutions gives 3.2 M or 4.3 M.
Choose explicitly and fix the design's "≈ 3–4 M".

**Optimiser groups.** In 1e, the head group is "names not starting with `backbone.`" (`r50_gl3_runner.py:209`). LoRA and
crop-affine parameters registered inside the backbone would otherwise land in the backbone LR group (1e-4). Put them in
an explicit local group, with the LR stated in the CFG (1e-3 as the head, or 2–5e-4).

**N2 (blocker): gradient clipping couples the losses.** `clip_grad_norm_(model.parameters(), 1.0)` (`r50_gl_runner.py:394`)
computes **one** norm over all parameters. Local gradients then rescale the shared weights' update, so the local loss
reaches the global path. Clip twice: (shared backbone + global head) and (LoRA + crop affine + local heads), each with max
norm 1.0. AdamW is per-parameter, so it is fine, and the shared parameters' weight decay is unchanged versus label-v2 r50.

**Heads and loss.**
- `norm_l` → two per-finding vectors on the crop feature: `clsV` (standalone l) and `clsR` (residual r, zero-init so that
  fused = g at step 0).
- Per finding, both l and r are averaged with weights `wc` over the crops (1 in training; 4 MS1-weighted in evaluation,
  v2's `scatter_add`).
- Loss = BCE(global) + BCE(σ(g_src + r), y)·mask10 + 0.5·BCE(σ(l), y)·mask10. g_src = g_oof (v4), or
  `gz.detach()` (v4-noOOF). The global BCE is the only path into shared weights.
- No dropout in the local head, as v2.

**Verifying zero gradient into shared weights** (CPU, real timm, random init):
- **T1:** `zero_grad(set_to_none=True)`, backward the local losses alone. Assert `p.grad is None` for every shared
  parameter (backbone convs, all shared BatchNorm γ/β, `norm`, `att`, `clsW`, `clsb`). Assert the local parameters have
  non-zero gradients. Note that A's gradient is 0 at step 0 because B = 0, so check after one step or with B ~ N(0, 1e-3)
  in the test.
- **T2:** same batch and seed, one optimiser step on (global) and one on (global + local). Assert the shared parameters
  **and the global BatchNorm running statistics** are bitwise equal, and that only the crop statistics and local
  parameters differ. This catches N1 and N2.
- **T3:** global BatchNorm buffers are unchanged by a crop-only forward in train mode.
- **T4:** fused = g exactly at init, and on effusion / synovitis always.
- On the GPU, repeat T1 in the Modal smoke and log the result in `training.log`.

Optional and stronger: draw crops with a dedicated `np.random.Generator` per (epoch, study), and initialise the extra
modules under `torch.random.fork_rng()`. Then v4's data order, windows and augmentation are the same as a plain label-v2
run with the same seed, and v4 s2027's global can be compared like-for-like with `r50_lv2` s2027 (bitwise on CPU for a few
steps).

**BatchNorm statistics.**
- In train mode the crop pass updates only `crop_mean / crop_var` (DualBN routes by flag).
- After SWA: reset both sets (`DualBN.reset_running_stats` already resets both), refresh global statistics from global
  windows at 384, then **crop statistics from all 4 crops per finding** (the evaluation distribution, as v3). v2 used the
  top-1 crop.
- The refresh must run **after** the averaged LoRA is loaded, because LoRA changes the crop activations.

**N3: SWA with LoRA.** The SWA average of A and of B is not the average of B·A, since the residual is bilinear. Primary:
for each LoRA conv, average the dense ΔW_e = B_e A_e·α/r over epochs 21–25 and evaluate in a "dense delta" mode
(~12 M floats, 48 MB). Log the factor-averaged version descriptively. This is exact SWA of the local path's weights.

**Memory and time on A100.** v2 on H100: 99 s per epoch with 16 global + 10 crops per study, 6.9 GB peak.
- v4 training: 16 global (full) + 10 crops at about half cost → about 75 s per epoch on H100.
- Evaluation: 94 windows + 40 crops, about 0.034 s per study → 28 s on H100.
- One run: about 45–55 min on H100 → **1.5–2 h on A100**; peak ≈ 6 GB.
- Label-v2 baseline: about 1.4 h on A100. Teachers: about 1 h.
- bf16 autocast is fine on A100.

## 3. g_oof lookup

Confirmed: g_oof is a lookup table, logit = log(p/(1−p)) with p clipped to [1e-6, 1−1e-6], from the 4 teachers'
`oof_swa.json`. The stored probabilities are float32-precision JSON floats, which is enough. Build it once on CPU with
`build_goof.py` → `goof_r50lv2.npz` (ids, logits [3524, 12], sha). Assert:
- each training id appears exactly once;
- no fold-0 or gold id;
- each id comes from the teacher whose split held it, checked against `teacher_splits.csv`.

The dataset returns g_oof[12] as an extra tensor (zeros outside the training ids). Pitfalls:
- **Protocol:** g_oof is an eval-protocol logit (SWA, 94 windows, 320 crop, no TTA). v4's fold-0 global uses the same
  protocol, so it matches. Keep mirror TTA **off** for v4's fused readout. r was trained against non-TTA logits, so if a
  submission uses TTA, apply it to g and r alike or not at all. The 50% training mirror does not touch g_oof (labels are
  unchanged by mirroring), which is fine.
- **Fold-0 evaluation uses v4's own SWA global:** fused = g_own + r. Never use the teachers' `val_swa`, except as a
  descriptive readout (mean teacher g + r).
- **Strength mismatch** (risk 2): teachers fit 3/4 of the training data. Log per-finding mean |g_oof| and BCE on training
  studies against v4's global on fold 0, and r's mean / sd for training versus fold 0.
- **v4-noOOF confound:** `gz.detach()` is a training-mode logit (16 augmented windows, attention dropout, moving). It
  differs from g_oof in two ways at once, in-sample and protocol. A cleaner control is an in-sample **eval-protocol**
  logit: run the teammate's label-v2 checkpoint (`data/teammate_labelv2/r50_labelv2_swa21_25.pt`, already local) on the
  training studies, about 3 min on an A100, which can go in batch A. This is a methods call; flag it to the methods
  reviewer.
- The 825-study training-sample gap logging uses g_oof for fused.

## 4. One MeluXina job (batch A → batch B)

The cache is in `~/rsna/d96`. It must keep the `shard_{0..3}/{manifest.jsonl, cache_summary.json, vol_*.npy}` layout
that `read_manifest` expects (`coatnet_d96_runner.py:446-450`). Re-run the Modal transfer checks in the preflight:
summary shard / study counts, array shapes, 4,407 studies.

```bash
#!/bin/bash -l
#SBATCH -J v4ab -A $SLURM_ACCOUNT -p gpu -q default -N 1 --ntasks-per-node=4 --cpus-per-task=16
#SBATCH --gpus-per-task=1            # or --gres=gpu:4; steps then use the matching flag
#SBATCH -t 06:30:00 -o /home/users/$USER/rsna/logs/%x-%j.out
set -uo pipefail
module load Apptainer/1.2.4-GCCcore-12.3.0
R=$HOME/rsna; SIF=$R/rsna_v4.sif; OUT=$R/out; CACHE=$R/d96
if [[ "${STAGE_LOCAL:-1}" == 1 ]]; then      # per-job scratch only, auto-cleaned; nothing persists outside home
  t0=$SECONDS; rsync -a "$R/d96/" "$LOCALSCRATCH/d96/" && CACHE=$LOCALSCRATCH/d96; echo "stage $((SECONDS-t0))s"
fi
ax() { apptainer exec --nv --cleanenv -B "$CACHE":/data/d96:ro -B "$R":/work --pwd /work/scripts "$@"; }
step() {  # name script
  mkdir -p "$OUT/$1"
  srun --exact -n1 -c16 --gpus-per-task=1 --mem=110G --cpu-bind=cores \
    apptainer exec --nv --cleanenv -B "$CACHE":/data/d96:ro -B "$R":/work --pwd /work/scripts \
      --env-file "$R/env/$1.env" "$SIF" python -u "/work/scripts/$2" > "$OUT/$1/train.log" 2>&1
}
waitall() { local f=0; for p in "$@"; do wait "$p" || f=1; done; return $f; }

# preflight (~5 min, 1 GPU): imports, offline weights + sha, cache layout/shape checks, GPU UUID, v4 GL_SMOKE with dummy g_oof
srun --exact -n1 -c16 --gpus-per-task=1 --mem=110G ax --env-file "$R/env/preflight.env" "$SIF" \
  python -u /work/scripts/preflight_v4.py > "$OUT/preflight.log" 2>&1 || exit 1

# batch A: 4 label-v2 r50 teachers (~1 h)
p=(); for k in 0 1 2 3; do step "r50lv2_teacher$k" coatnet_d96_runner.py & p+=($!); done
waitall "${p[@]}" || { echo "batch A failed"; exit 1; }
ax "$SIF" python -u /work/scripts/build_goof.py /work/out /work/out/goof_r50lv2.npz || exit 1   # CPU, asserts coverage

# batch B (~2 h): v4 s2026, v4 s2027, v4-noOOF, label-v2 r50 s2027
p=(); for n in v4_s2026 v4_s2027 v4_nooof; do step "$n" r50_gl4_runner.py & p+=($!); done
step lv2_s2027 coatnet_d96_runner.py & p+=($!)
waitall "${p[@]}"
```

Env files (`$R/env/<run>.env`, one `KEY=VALUE` per line; `--env-file` avoids `--env`'s comma splitting):
- **Common:** `D96_CACHE=/data/d96`, `D96_ASSETS=/work/assets`, `PYTHONPATH=/work/scripts`, `PYTHONUNBUFFERED=1`,
  `OMP_NUM_THREADS=4`, `HF_HUB_OFFLINE=1`, `D96_PRETRAINED_FILE=/opt/weights/resnet50_tv_in1k.safetensors` (B6).
- **Teachers:** `D96_PHASE=r50lv2_teacher{k}`, `D96_OUT=/work/out/r50lv2_teacher{k}`.
- **Baseline:** `D96_PHASE=r50_lv2`, `D96_SEED=2027`, `D96_OUT=/work/out/lv2_s2027`.
- **v4:** `D96_PHASE=r50_gl`, `D96_SEED=…`, `GL_OUT=/work/out/<run>`, `GL_CROPS=/work/ms1_crops_slim`,
  `V4_GOOF=/work/out/goof_r50lv2.npz`, `V4_RESID=oof|self`.

Notes:
- **Per-step `--mem` is required**, or the first step can claim all job memory and the other steps wait.
- Each run prints `CUDA_VISIBLE_DEVICES` and the GPU UUID. The preflight asserts the GPU is visible.
- Fallback if step binding misbehaves: one task starts 4 processes with `CUDA_VISIBLE_DEVICES=i` and `taskset`.
- Workers: 8 per run.
- The early-stop rule is applied **post hoc** (compute is free and the runs are concurrent; there is no automatic kill).
  It needs per-epoch fold-0 AUC from v4 (every epoch) and from `r50_lv2` (every epoch, N4).

**Home budget:** cache 58.1 GiB + SIF ~4 GiB + outputs (4 teachers × ~0.9 GiB, 4 batch-B runs × ~1 GiB) ≈ 8 GiB + slim
MS1 table, g_oof and assets < 0.1 GiB ≈ **70 GiB**. Optionally delete the teachers' `weights_epoch2*.pt` after the SWA
file exists (−2 GiB). Do not stage the 1.3 GiB `crop_probe`: use the slim table.

**Smoke on Modal** (per-second billing): add `r50_gl4_runner.py` and a dummy `goof` to the Modal image, then run
`GL_SMOKE=1` on H100 for about 10 min (~$1). It covers training, SWA with dense LoRA delta, BN refresh, predict and the T1
gradient check on GPU. The teacher and `r50_lv2` phases need only a CPU config / split test: the training path is the
already-run cnxa teacher path. The MeluXina preflight then covers what Modal cannot: the SIF, offline weights, the
cache on MeluXina and GPU binding.

## 5. Round-1 blockers: status

| Round 1 | Status under rev 2 |
|---|---|
| B1 geometry rows / cols | **Gone** (no cross-plane). Submission note: inference still needs MS1's crop table (MS1 at ~0.43 s/study on H100) unless SL-0 passes. |
| B2 same-step data flow, B3 mirror / affine inverse | **Gone**: crops come from the fixed MS1 table through v2's tested `GLStudies` / `prepare_gl`. |
| B4 MS1 table fold-0 rows | **Changed**: v2 uses MS1 crops for fold-0 evaluation by design. That is fold-safe, since MS1 was trained on folds ≠ 0. Training-study points are in-sample but label-free. Action: ship a **slim** `cp_*.npz` (ids, centres, a_top only, ~2 MB; `crop_table` reads only those keys, `r50_gl_runner.py:80-83`), drop the MS1 logits and features, and assert coverage of all 4,349. |
| B5 seed / baseline recipe | **Applies**: add `D96_SEED`, the `r50lv2` teachers and the `r50_lv2` phase (section 1). |
| B6 offline weights | **Applies** to every phase: `timm.create_model(pretrained=True)` (`coatnet_d96_runner.py:661`). Bake `resnet50.tv_in1k` into the SIF, add an env-guarded `pretrained_cfg_overlay=dict(file=…)` and log the sha256. |
| Container | **Applies**. apptainer 1.4.5 is available locally: `FROM localhost/rsna-local-global-checks` + `timm==1.0.29 huggingface_hub==0.35.3 safetensors==0.6.2` + weights → `podman save` (docker-archive) → `apptainer build` (no `--oci`). Run the CPU tests in **this** image with `--network none --read-only` and **real timm, random init**, not a stub. A stub can get timm's ResNet module names wrong (`layerN[i].conv1/conv3`, `downsample.0`), and those names are what the LoRA wrapping depends on. |

New in round 2: **N1** shared-BatchNorm-affine leak, **N2** joint gradient clipping, **N3** SWA of LoRA factors, **N4**
the baseline phase must score every epoch (the `r50_gl` CFG's `eval_epochs` skips epochs 11–12, which the early-stop rule
needs).

## Minimal files to write

1. `scripts/coatnet_d96_runner.py` (edit): `r50lv2` in `_teacher`, phase `r50_lv2`, `D96_SEED`,
   `D96_PRETRAINED_FILE` hook.
2. `scripts/r50_gl4_runner.py` (new): `SharedConv` + LoRA, `DualBN4` (detached or own crop affine), `RaptorGL4`
   (`no_grad` stem–layer2 for crops), heads l / r, g_oof dataset, split clipping and optimiser groups, dense-delta SWA,
   BN refresh with 4 crops, predict / score (fused, global, local, mean(g, l)), recovery, smoke.
3. `scripts/build_goof.py` (new, numpy): teachers' oof → `goof_r50lv2.npz`, with coverage and leak asserts.
4. `scripts/slim_ms1_crops.py` (new, numpy): `cp_*.npz` → slim copies.
5. `scripts/test_r50_gl4_cpu.py` (new): T1–T4, the teacher / `r50_lv2` CFG and split checks, the goof checks, an
   end-to-end CPU smoke.
6. `scripts/preflight_v4.py` (new): imports, weights sha, cache verification, GPU UUID, `GL_SMOKE` with dummy goof.
7. `meluxina/Containerfile`, `meluxina/v4ab.sbatch`, `meluxina/env/*.env` (10 files).
8. `scripts/modal_coatnet_d96.py` (edit): `r50_gl4` smoke function plus image files.

## Time estimate

| step | estimate |
|---|---|
| runner edits (phases, seed, weights hook) + CPU config test | 1 h |
| `r50_gl4_runner.py` | 4–6 h |
| CPU tests T1–T4 + goof / split tests | 2–3 h |
| image + SIF + upload (parallel; networked build run by the user) | 1–2 h |
| Modal smoke | ~15 min, ~$1 |
| MeluXina job: staging ≤ 10 min, preflight 5 min, batch A ~1 h, goof < 1 min, batch B ~2 h | **~3.5 h wall** (request 6.5 h) |

Code to results is about 1–1.5 days, so results are possible on 2026-10-20.
