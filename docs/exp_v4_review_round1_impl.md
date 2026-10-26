# r50-GL v4 (cross-plane) design — implementation / feasibility review, round 1 (2026-10-18)

Scope: `docs/exp_v4_crossplane_design.md` revision 1 (implementation, compute, MeluXina, code reuse, leakage). Methods /
statistics are reviewed separately. Read: `scripts/r50_gl3_runner.py`, `r50_gl_runner.py`, `coatnet_d96_runner.py`,
`multiplane_points.py`, `selfloc_r50_gl3_gpu.py`, `d96_positions.py`, `modal_coatnet_d96.py`, ledger rows MS1-CP / MP-0 /
r50-GL v3 / r50-GL3-XAI / SL-0, `run_r50_gl3/{training.log,run_summary.json}`, local manifests, `multiplane_points.npz`,
the 10-study smoke cache. Checks run on the host (numpy only) and in podman with `--network none`.

## Verdict

Feasible before the deadline, with changes. Six blocking items (B1–B6). The biggest are the data flow (crops depend on
the step's own forward pass), the mirror / affine inverse that this creates, getting the 58 GiB cache onto MeluXina,
and the Kaggle submission path, which lacks per-series rows / cols for the geometry. The runtime estimate is optimistic.
Expect **2.5–3.5 h per v4 run on an A100-40GB**, not 2–2.5 h. Request 5 h of wall time.

Recommended implementation: **periodic point-table refresh (Option B below) instead of same-step self-location.** It
reuses v3's tested DataLoader and crop path almost unchanged, makes the mirror / affine edge cases disappear, and puts
the training points on the same protocol as inference. If the design must keep same-step self-location (Option A), it is
implementable. It needs the whole volume on the GPU, geometry tensors on the GPU and an exact inverse of the global
augmentation, which adds about half a day of code and one more test class.

## Blocking issues

| # | Issue | Evidence | Fix |
|---|---|---|---|
| B1 | **The geometry needs each series' native rows / cols, and the d96 manifest does not hold them.** `Slice` computes the cache square from `rows, cols` (`multiplane_points.py:46-47`). Those come from `data/external/rsna-knee-crop-fov-check/series_fov.csv` (`:108-109`). The manifest `geometry` entries hold only `ipp / iop / ps` (checked on `manifests/manifest_0.jsonl`). | For the fold-0 experiment: precompute a geometry table offline (below) and ship it. **For a Kaggle submission:** the inference kernel's cache builder must also record rows / cols per series, or read them from the DICOM headers. Otherwise v4 cannot run at test time. This must be solved before PASS → LB fits the deadline. Scope it now. |
| B2 | **Data flow: the crops depend on the model's own global forward.** v3 gathers region slices inside the DataLoader worker (`r50_gl3_runner.py:90-102`, `vol[np.stack([S-1,S,S+1],1)]`) because the region table is fixed. Same-step self-location knows the slices only after the global forward on the GPU. | Option A: the DataLoader returns the whole study volume and the slices are gathered on the GPU after the forward. Option B: refresh a point table between epochs and keep v3's worker-side gather. Details below. |
| B3 | **Mirror / affine: training centroids live in the augmented frame.** `selfloc.locate` converts cells to pixels as `LO + (cy+0.5)*px` (`selfloc_r50_gl3_gpu.py:65`), which is right only for un-augmented eval windows (320 centre crop, LO = 32). Training windows pass through mirror plus a per-study affine (rotation ±10°, scale ±10%, shift ±12% ≈ ±46 px; `r50_gl_runner.py:133-147`). Reused as is in training, every point is displaced. Worse, in the 50% of mirrored studies the coronal / axial x is reflected, so a 3D point maps to the **opposite compartment** (medial ↔ lateral) in other series. This is a silent bug, not noise. | Option A: map each cell centroid back as raw_norm = A·u + t (the `affine_grid` convention, input = θ·[u;1]; no matrix inverse needed). Then negate x if the study was flipped and the window is not sagittal, convert to raw px, and do the 3D mapping in raw coordinates. Then cut the crops from raw slices and apply the same per-study flip and s·A as `prepare_gl` already does (`:135-137, :147`). Option B avoids this entirely: points are located on un-augmented eval-protocol windows and crops go through v3's tested path. Either way, test T4 below is mandatory. |
| B4 | **The MS1 table covers all 4,349 studies, fold 0 included.** `crop_table` reads every `cp_*.npz` (`r50_gl_runner.py:76-84`). The design allows MS1 points for training studies only. | Derive a small `ms1_points_train.npz` (ids, centres, a_top for the 3,524 training studies only, about 1 MB) offline. At runtime, assert `set(ids) ∩ set(val_ids) == ∅`. Do not ship `cp_*.npz` (220 MB, it also holds MS1 logits and features). |
| B5 | **Seeds and the baseline recipe.** `CFG["seed"]` is hard-coded to 2026 (`coatnet_d96_runner.py:52`), and the `r50_gl` phase does not override it (`:373-377`). The label-v2 r50 reference (0.9425) is the teammate's Kaggle run (`kaggle_fold0_labelv2/val_swa.json`, note "Kaggle d96 path"), not a run of this code on this hardware. | Add `V4_SEED` (or a phase) that overrides `CFG["seed"]` and enters `version` and the recovery assert. Add a `D96_PHASE` that reproduces label-v2 r50 with a seed override, and confirm it is the same recipe as the Kaggle run (labels sha dd0ab9ab, SWA 21–25, 1% floor). Use the idle GPU time: chain **label-v2 seed 2026 and 2027 on the same A100s**, so the baseline is on the same platform and precision as v4 (see compute). |
| B6 | **Pretrained weights are downloaded at runtime.** `timm.create_model(CFG["arch"], pretrained=True)` (`coatnet_d96_runner.py:661`) fetches `timm/resnet50.tv_in1k` from the HF hub. MeluXina compute nodes are offline. timm is not in the local podman image. Modal installs `timm==1.0.29` (`modal_coatnet_d96.py:21`). No local HF cache of resnet50.tv_in1k exists (`~/.cache/huggingface/hub` has 3 text models). | Bake the weights into the image (below). Prefer loading from a file: an env-guarded `pretrained_cfg_overlay=dict(file=...)` in `Raptor.__init__`. That is robust on a read-only SIF, where HF lock-file writes can fail. Log the weights' sha256. |

## 1. Geometry and data flow

### What is already right

- The `Slice` mapping (`multiplane_points.py:38-57`) is an affine per cached slice: `to_patient(r, c) = O + U·c + V·r`,
  where U = u·psc·f, V = v·psr·f and O absorbs the x0 / y0 / half-pixel terms. `from_patient` gives `c = (P−O)·U/|U|²`,
  `r = (P−O)·V/|V|²` and `dist = (P−O)·n`. Precompute **per study, per cache row: O, U, V, n (12 floats) + a valid flag**,
  plus per slot: first / last row and spacing. 4,407 studies × 96 × 13 × 4 B ≈ 22 MB. Mapping one point into the 5 slots is
  then one `[96]` dot product, a masked argmin per slot and two dot products. That costs microseconds per point in numpy
  and is trivial on the GPU as `[b, 8, 96]` tensors.
- Series presence (all 4,407 local manifests): slot 0 sagittal fat-sat 100%, slot 1 sagittal non-fat-sat 97.3%, slot 2
  coronal fluid 100%, slot 3 coronal other **74.5%**, slot 4 axial 100%. Every present slot has full geometry. 72% of
  studies have all 5 series and 28% have 4. Masking is needed in about 30% of studies, so it is not an edge case.

### Edge cases (all need code and a test)

1. **Missing series:** mask the token. For fixed shapes, see compute below.
2. **Point outside the target series' slab:** in `multiplane_points.npz`, |dist| > spacing for 0.04% (sagittal), 0.37%
   (coronal) and 0.64% (axial) of points, with a maximum of 208 / 65 / 77 mm. Nearest-slice still returns a slice, but of
   different anatomy. **Mask the token when |dist| > 1.0 × that slot's spacing**, or when the in-plane point falls outside
   [0, 384). MP-0's "99–100% inside" is an in-plane check only (`multiplane_points.py:98`).
3. **Duplicated picks:** 31% of sagittal fat-sat slots, 11% sagittal non-fat-sat, 8% coronal fluid and 2% axial contain
   repeated picks (the series is shorter than the slot; e.g. 16 files → k = 18). The argmin ties harmlessly. However, the
   slot spacing `median(diff(sorted ipp·n))` (`:99-100`) must use **unique** positions. Otherwise duplicates push it
   toward 0 (it is 0 in 0.02% of points already) and the dist mask would drop the whole slot.
4. **Series edges:** clip the target slice to `[first+1, last−1]` **of that slot**, as v3 does for Baker's
   (`r50_gl3_runner.py:75`). Clipping to [1, 94] would let the 3-slice window straddle two series. Apply the ±1-slice
   training jitter before the clip. Skip slots with k < 3. Note that the global windows from `centres()` are clipped only
   to [1, MAXS−2] (`coatnet_d96_runner.py:584`) and can straddle a series boundary. Their centre slice is still the right
   source row for `to_patient`.
5. **Mirror** (Option A only, see B3). `mirror()` reverses the 3 slice channels of sagittal windows and flips the width of
   the others (`coatnet_d96_runner.py:603-605`). This is a local L/R reflection, not a 3D one: a sagittal window stays at
   its slice index. Geometry is therefore valid only in **raw, un-mirrored** coordinates: un-flip the centroid, map, cut
   raw slices, then mirror the crops with the same per-study flag (`prepare_gl` already flips the crop centre for
   non-sagittal crops, `r50_gl_runner.py:137`). Do not feed absolute in-plane coordinates to the transformer. If you must,
   use the flipped `px` that `prepare_gl` returns, as the v2 reader did.
6. **Affine** (Option A only): use the global's own draw, raw_norm = A·u + t, then the crops get θ_c = [s·A | p_raw]. The
   shift t is not applied to crops, which are centred on the point (as r50-GL). Rotating crops from other planes by the
   source plane's A is just augmentation.
7. **The 320 eval centre crop:** eval / refresh centroids use a 10×10 grid offset by LO = 32 (`selfloc:31,56,65`). The BN
   refresh globals run at 384 (`gres=CFG["train_res"]`, `r50_gl3_runner.py:187`): a 12×12 grid with LO = 0. Make
   `locate()` take `(res, lo)` explicitly. Today it reads the module-level `RES / LO`.
8. **Baker's:** drop selfloc's sagittal-only restriction (`selfloc:50-52`). v4 maps the 3D point into every series.

### The self-location pass is analytic: no backward

`locate` takes `autograd.grad` of `a_kn · (LN(mean_hw fm) · W_n)` with respect to `fm` (`selfloc:54-61`). Mean pooling
makes that gradient **spatially uniform**: ∂/∂fm[c,h,w] = a_kn/(HW) · (J_LNᵀ W_n)_c. The positive scale a_kn/(HW) cancels
in the centroid, so the grad×act map is a CAM through the LayerNorm Jacobian:
`map[k,n,h,w] = Σ_c G[k,n,c] · fm[k,c,h,w]`, with `G = (γ⊙W_n − mean(γ⊙W_n) − x̂·mean(γ⊙W_n⊙x̂)) / σ`, where x̂ and σ come
from LN of the pooled vector. Closed form verified against finite differences (|Δ| 4e-9) and against the pooled-feature
derivative on a 12×12 map. Implementation: one `einsum` over `fm.detach().float()` under `no_grad`, batched over windows ×
8 findings. There is no `retain_graph` loop, it does not interact with the training graph, and its cost is negligible.
Get `fm` from `backbone.forward_features` and pool with `backbone.forward_head(fm)`, which equals timm's
`backbone(x)` for `num_classes=0`. `RaptorGL3.encode` (`r50_gl3_runner.py:121-127`) needs a variant that returns `fm`.

### Option A: same-step self-location (as the design is written)

DataLoader → `vol [96,384,384] uint8` (14.2 MB per study), stratified centres `c [16]`, `sag`, y, w, ctx and the study
index. That is 2× the bytes of today's 16 × 3 slices: about 200 MB/s per run on an A100, 0.8 GB/s for 4 runs, from page
cache. Fine. Pinned host→device copy is about 28 MB per step, about 2 ms. On the GPU:
1. `x = vol[c−1:c+2]` gather → `aug = draw_aug(b)` → `apply_global(x, sag, aug)`. Split `prepare_gl` into `draw_aug` +
   `apply_global` + `apply_crops`, keeping its exact random-draw order (flip, angle, scale, tx, ty, gamma, gain) so a
   regression test can assert bit-equality with `prepare_gl`.
2. Global forward (`fm`, a, gz).
3. Under `no_grad`: analytic CAM centroids on the top-k = 4 windows by a_kn, sample one ∝ a_kn (epochs ≥ 4; MS1 table
   before), inverse affine + un-flip → raw (slice, row, col), jitter ±16 px / ±1 slice.
4. Geometry on the GPU (per-study tensors gathered from a resident `[N,96,13]` table) → per slot (row, r, c, dist, ok) →
   mask.
5. `xc = vol[s−1:s+2]` gather → `apply_crops(…, aug)` → `encode(crop=True)` → tokens → transformer → lz.

Weakness: training points come from 16 randomly stratified, augmented windows. Inference picks among 94 un-augmented
windows. Point quality and protocol differ between training and inference. Log the per-finding CAM mass and top-1
attention for both.

### Option B (recommended): periodic point-table refresh

- Epochs 1–3: the table is MS1's training-study points (B4). After epoch 3, then every 2 epochs (3, 5, …, 23), run an
  **eval-protocol locate pass** (eval mode, 94 windows, 320 crop, LO = 32, no augmentation) over the 3,524 training
  studies. Keep the top-4 windows per finding with a_kn and CAM centroids, map each into the 5 slots (numpy, vectorised),
  and store `[N, 8, 4, 5, (row, r, c, dist, ok)]`. Set `train_ds.tab = new_tab` before building the next epoch's
  DataLoader. Workers are re-forked each epoch (`r50_gl3_runner.py:236`), so they see the new table.
- The Dataset samples one of 4 ∝ a_kn per finding, applies ±16 px / ±1-slice jitter within the slot, gathers
  `[8, 5, 3, 384, 384]` raw slices, and builds the mask (missing / out-of-slab / plane dropout). Augmentation then goes
  through `prepare_gl`'s existing crop path. **Mirror and affine are handled exactly as in v3. No inverse is needed.**
- Inference uses the same locate on fold 0 with the SWA weights, so training and inference use the same protocol.
  Training points are at most 2 epochs stale. That is the design change to accept, and the methods reviewer should confirm
  it.
- MS1 warm-up, self from epoch 1 and "MS1 throughout + self at inference" (the design's open question) are all the
  same code path with a different table, so one flag switches between them.
- Cost: about 0.05 s/study on an A100 for a global-only 94-window pass → about 3 min per refresh, 11 refreshes ≈ 32 min.
  Using every other eval centre (47 windows) halves it. The refresh can also produce the 825-study training-sample gap
  logging if the local path runs on that subset in the same pass.
- **SWA tail order:** average weights → refresh global BN statistics (globals only) → locate the 256 refresh studies
  with the SWA model → refresh crop BN statistics on their crops → locate fold 0 → predict.

## 2. Compute and memory (A100-40GB vs v3 on H100)

Measured on H100 (Modal): plain label-v2 r50 59 s/epoch (16 images per study); v2 99 s (26); v3 149 s (48), peak 12.2
GB at 2 studies × 48 = 96 images per micro-step. v3 eval was 0.032 s/study and the whole run 4,472 s = 74.5 min. A linear
fit gives about 14 s + 2.81 s per image per study per epoch.

| v4 variant | images per study (training) | H100 train / epoch | A100 train / epoch (×1.8–2.1) |
|---|---|---|---|
| all 5 × 8 crops encoded (fixed shape) | 56 | ~171 s | 310–360 s |
| masked crops dropped before encoding (missing ~28% of slot 3, plane dropout p_keep 0.7) | ~42 | ~132 s | 240–280 s |
| v4-1P (8 crops) | 24 | ~81 s | 145–170 s |
| label-v2 baseline | 16 | 59 s | 105–125 s |

Per v4 run on an A100: training 100–150 min, plus fold-0 eval about 1 min/epoch (25 min), plus the training-sample gap
(about 1 min each; schedule 8 of them, not 25: epochs 3, 6, 9, 12, 15, 18, 21, 25), plus refreshes (Option B, 16–32 min),
plus the SWA tail (5 min). Total **≈ 2.5–3.5 h** (Option A ≈ 0.3–0.5 h less). v4-1P ≈ 2 h, baseline ≈ 1.2 h. The node is
billed for the longest run: about 3.5–4 node-hours with staging, plus the smoke. **Use the idle GPUs:** chain label-v2
seed 2026 and then seed 2027 on GPU 3 (2 × 1.2 h fits inside v4's wall time), so the baseline has two same-platform seeds
at no extra node cost.

Memory: about 12.2 × 112/96 ≈ 14–15 GB peak, plus full-slice float buffers before the crop `grid_sample` (2 × 40 ×
3 × 384² × 4 B ≈ 140 MB), plus 2 GPU-resident volumes in Option A (28 MB). Eval `fm` for 94 windows is about 38 MB in
bf16. **It fits 40 GB with a wide margin.** bf16 autocast is native on Ampere, which is the same numeric family as the H100
runs (cuDNN TF32 convolutions are on by default on both).

The self-location pass: analytic, about 0 cost (section 1). The design's "extra self-location pass" exists only in
Option B, as the refresh.

Shapes:
- `cudnn.benchmark = True` (`r50_gl3_runner.py:195`) re-tunes for every new crop batch size. With dropped masked crops,
  N varies (≤ 80 distinct values for b = 2). That is a one-time cost of seconds per shape, acceptable.
- Do **not** pad missing crops with zero images into the encoder: in training mode, crop-BN batch statistics would include
  the blanks. Either drop masked crops before encoding and scatter the tokens back into `[b, 8, 5, d]`, or pad with a
  duplicate of a real crop and mask its token.
- Transformer: `enable_nested_tensor=False` (as in `r50_gl_runner.py:218`), `key_padding_mask` per `[b·8]` sequence, and a
  guard so the source-slot token can never be masked. An all-masked row gives NaN.
- The zero-initialised local logit (`fused = global` at step 0) needs an INIT_CHECK in the smoke, like v3's
  (`r50_gl3_runner.py:218-223`).

## 3. MeluXina

**Container.** `apptainer` 1.4.5 **is installed locally** (`/usr/bin/apptainer`; `singularity` → apptainer), so build
the SIF on the workstation:
1. Containerfile `FROM localhost/rsna-local-global-checks` (torch 2.5.1+cu124, torchvision 0.20.1, Python 3.11.10,
   numpy 2.1.2, PyYAML; **no timm, huggingface_hub or safetensors**, checked offline). Add
   `RUN pip install --no-cache-dir timm==1.0.29 huggingface_hub==0.35.3 safetensors==0.6.2` (pins as Modal). Add a
   build-time download of `resnet50.tv_in1k` to `/opt/weights/` with its sha256 and
   `ENV HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4`. The build needs network, so the user runs it.
2. `podman save --format docker-archive -o <scratch>/rsna_v4.tar localhost/rsna-v4:1` →
   `apptainer build rsna_v4.sif docker-archive://<scratch>/rsna_v4.tar` (unprivileged, local). **Do not use `--oci`**:
   OCI-SIF (1.3+) does not run on 1.2.4. A plain squashfs SIF from 1.4.5 should run on 1.2.4, but verify with
   `apptainer exec rsna_v4.sif python -c "import torch, timm"` on a login node or in the smoke. The local disk has 106 GB
   free, enough for the 6.5 GB tar and the ~3.5–4.5 GB SIF. Upload only the SIF and delete the tar.
3. The cu124 wheels need NVIDIA driver ≥ 525.60 (CUDA 12 minor-version compatibility). Print `nvidia-smi` in the smoke.

**Data.** The d96 cache is 4,407 studies × 96 × 384² B = **62.4 GB = 58.1 GiB** (4 shards). It exists only on the Modal
volume and as Kaggle kernel outputs (`tumis001/rsna-coatnet-d96-cache-shard-{0..3}`). It is **not local**: only the
10-study smoke cache `data/coatnet_d96_cache_smoke/d96` is. Uploading 58 GiB from a home connection may take 10+ h, so
this is the **critical path. Start it first.** The preferred route is `kaggle kernels output` on a MeluXina **login
node** (shards in sequence straight into home, 3 retries). This assumes login nodes have outbound internet. It also needs
a Kaggle token on MeluXina, which is the user's call. Then re-run the shard verification from `modal_coatnet_d96.py:128-140`
(summary shard / study counts, `.npy` shape and dtype, total 4,407). Fallback: `modal volume get` to local → `rsync -P`.
zlib-1 compresses the cache only 1.33×, which is not worth it unless quota is tight.

**Home budget (100 GiB / 100k files):**

| item | size |
|---|---|
| d96 cache (4 shards, ~48 files) | 58.1 GiB |
| SIF | ~3.5–4.5 GiB |
| code, assets (13 MB), geometry table (~22 MB), MS1 training table (~1 MB) | < 0.1 GiB |
| outputs per run: 5 epoch weights (~0.1 GiB each) + recovery.pt (~0.32 GiB) + SWA + JSONs | ~1 GiB × 4–5 runs |
| smoke outputs | ~0.5 GiB |
| **total** | **~67–69 GiB**, with ~30 GiB headroom if home is otherwise empty |

Check `myquota` first. Do not keep the docker-archive tar, Kaggle download temps, `cp_*.npz` or the native d96n cache in
home. File count is well under 1k.

**Job.** `#!/bin/bash -l`, `-A $SLURM_ACCOUNT -p gpu -q default -N 1 --ntasks-per-node=4 --cpus-per-task=16`, 4 GPUs (gres per
MeluXina docs), `-t 05:00:00`, `module load Apptainer/1.2.4-GCCcore-12.3.0`. Then:
1. `rsync -a $HOME/rsna_v4/d96/ $LOCALSCRATCH/d96/` (58 GiB, budget ≤ 10 min; log the time). The 4 runs mmap the same
   files, so the page cache holds one shared ~58 GiB copy out of 512 GB.
2. Per run: `srun -n1 -c16 --gpus=1 --mem=120G --exact --cpu-bind=cores apptainer exec --nv --cleanenv -B $LOCALSCRATCH/d96:/data/d96:ro -B $HOME/rsna_v4:/work --env-file /work/env/<run>.env /work/rsna_v4.sif python -u /work/scripts/r50_gl4_runner.py > /work/out/<run>/train.log 2>&1 &`, then `wait`.
   - **Set `--mem` per step.** Without it the first step can claim all job memory and the others block ("step creation
     temporarily disabled").
   - Use `--env-file`: the `--env` comma lists break on values containing commas.
   - Each run prints `CUDA_VISIBLE_DEVICES`, `torch.cuda.get_device_properties(0)`, the GPU UUID and the core affinity.
     The smoke asserts 4 distinct UUIDs.
   - Fallback if step binding misbehaves: one task launches 4 processes with explicit `CUDA_VISIBLE_DEVICES=i` and
     `taskset -c 16i–16i+15` (NUMA-aware ranges).
3. Env per run: `D96_CACHE=/data/d96`, `D96_ASSETS=/work/assets`, `D96_PHASE=r50_gl`, `GL_OUT=/work/out/<run>`,
   `GL_CROPS` (still read at import, `r50_gl_runner.py:59`; point it at the MS1 table dir or refactor),
   `V4_GEOM=/work/geom_v4.npz`, `V4_MS1=/work/ms1_points_train.npz`, `V4_SEED`, `V4_PLANES=all|source`, `GL_SMOKE`,
   `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=4`, `PYTHONUNBUFFERED=1`. Workers: 8–10 per run.
4. Outputs go straight to home: small per-epoch writes, and `recovery.pt` survives a wall-time kill. `$LOCALSCRATCH`
   holds only the cache copy. On a timeout, resubmit: `recovery.pt` resumes, with the config assert kept.

**Smoke (`test` QOS, 30 min, 1 job).** Bind the cache **from home** (no 58 GiB copy). Run 4 concurrent `GL_SMOKE=1` runs
(32 steps + 16-study eval + SWA of 1 + BN refresh on 8 + one refresh pass on ~64 studies). This checks GPU binding, offline
weight loading, `INIT_CHECK`, peak memory, s/study on the A100 (extrapolate before the full launch) and
`nvidia-smi dmon` GPU utilisation (is the loader a bottleneck?).

## 4. Code reuse, new code, CPU tests

**Reuse unchanged:** `coatnet_d96_runner` (`read_manifest`, `centres`, `Studies`, `mirror`, `Raptor` + 1e head, labels /
AUC / BCE / save, CFG `r50_gl`); `r50_gl_runner` (`DualBN`, `dual_bn`, `bce`, `to_dev`, `BOX/BOXN`, crop half of
`prepare_gl`); `r50_gl3_runner` (R8 / RF / RMASK, `encode`, main loop skeleton: optimiser groups, schedule, recovery, SWA
average, BN refresh, `score`, `V3Studies` pattern); `selfloc.locate` logic (top-k windows, centroid of the positive map);
`multiplane_points.Slice / study_slices` (offline table build).

**Minimal new code** (`scripts/r50_gl4_runner.py`, plus 2 offline numpy scripts):
1. `build_geom_v4.py` (offline): manifests + `series_fov.csv` → `geom_v4.npz` (per row O / U / V / n / valid, per slot
   first / last / unique-spacing, ids and a mask hash checked against the cache manifest at runtime).
2. `build_ms1_train_table.py` (offline): `cp_*.npz` → training-only (ids, eval-centre row, r, c, a_top). Assert no fold-0
   ids.
3. `map_slots(geom, src_row, r, c) → [5, (row, r, c, dist, ok)]` (numpy; torch twin only for Option A).
4. `cam_centroids(fm, model, rf, res, lo, topk)` (analytic CAM, batched).
5. `refresh_table(model, ids, …)` (Option B) or `draw_aug / apply_global / apply_crops / unaug_points` (Option A).
6. `V4Studies(r.Studies)`: sampling, jitter, slot clipping, masks, plane dropout, `[8, 5, 3, 384, 384]` raw slices.
7. `RaptorGL4(r.Raptor)`: dual BN, 2048→256 projection + slot embedding + finding query, 2-layer TransformerEncoder
   (d 256, 4 heads, dropout 0.2), zero-initialised per-finding logit, `fused = gz + lz` on RF.
8. `predict_v4` (top-1 and top-4-weighted variants logged), `score_v4` (+ mask rates per slot, per-slot attention of the
   query, point dist), `bn_refresh_v4`, early-stop check at epoch 12.
9. A baseline phase (label-v2 r50 with a seed override) in `coatnet_d96_runner.py` (B5); the `pretrained_cfg_overlay`
   env hook (B6).

**CPU tests** (in the **new** image, `podman run --network none --read-only`, which reproduces the SIF's offline,
read-only conditions; 10-study smoke cache as shard_0 with empty shards 1–3, randomly initialised r50):
- T1 geometry: `map_slots` reproduces `multiplane_points.npz["points"]` on all 4,349 × 12 MS1 top-1 points (same row, r / c
  and dist within 1e-3) for the 3 fluid slots. Round trip source → target → source within 1 px when dist < 0.5 mm.
- T2 visual: 3 smoke studies × 8 findings × 5 slots crop panels (PNG), checked by eye (same anatomy and side, as in MP-0).
- T3 CAM: analytic centroids equal `selfloc.locate` (autograd) within 0.5 px, at 320 / LO 32 and 384 / LO 0.
- T4 augmentation (Option A; for Option B the `prepare_gl` regression is enough): a synthetic blob at raw (s, r, c) →
  `apply_global` with 200 random draws (sagittal and non-sagittal, flipped and not) → argmax in output → `unaug_points`
  within 2 px. Also `draw_aug + apply_*` equals `prepare_gl` bit-for-bit under the same RNG state.
- T5 Dataset: shapes; a missing slot 3 is masked; window rows stay inside slot bounds; dist > spacing is masked; jitter
  ranges; MS1 table ∩ fold 0 = ∅; plane dropout never empties a row.
- T6 model: `INIT_CHECK` fused = global exactly; findings without regions give fused = global; perturbing finding f's
  crops moves only lz[f]; v4-1P masks every slot except the source; no NaN under maximal masking; global and crop BN
  statistics update separately.
- T7 end-to-end `GL_SMOKE` on CPU (2 steps, predict, refresh, SWA, BN refresh); recovery resume; offline pretrained load
  from the baked file with sha256 logged.

## 5. Leakage

- **MS1 points:** from MS1-CP. Frozen MS1 SWA (trained on folds ≠ 0) on all 4,349 studies, 94 eval windows,
  grad×act of MS1's own logit contribution. No labels are read, so the points are label-free per study. Training-study
  points are **in-sample** (MS1 saw those labels). MP-0 follow-up (b) found no placement advantage on training studies
  (crop − global −0.119 training vs −0.109 fold 0) and LOC-0-MS1 found the maps mark anatomy (positive − negative
  2–4 mm). **Fold-safe for fold 0 provided B4 is enforced** (the file contains fold-0 rows).
- **Self-located training points are in-sample too** (the in-training model fits those studies), in both Options A and
  B. This cannot leak fold 0, but it is a train / inference shift in placement quality, the same family as v3's local
  memorisation. Log the per-finding point stability between refreshes and the CAM mass, training sample vs fold 0.
- The early stop at epoch 12 reads fold 0, but only to stop, not to select. Acceptable as pre-registered.
- The open "per-finding scalar learned on a held-out slice of the training folds" needs a slice **not trained on**. Either
  shrink the training set (which changes the recipe against the baseline) or cross-fit. Fitting it on training studies
  repeats v3's in-sample gate failure. Prefer a descriptive cross-fit inside fold 0, as the XAI ceilings did.
- The geometry and MS1 tables are built from manifests and images only. The fixed 825-study training sample is
  label-free. BN refresh uses training studies. All fine.
- Design-level: several choices (k = 4, boxes, plane facts) came from fold-0 diagnostics (MP-0, XAI). That is the methods
  reviewer's call, but it should be noted in the pre-registration.

## 6. Plan and time

| step | effort | notes |
|---|---|---|
| 0. Start the cache transfer to MeluXina home | 0.5 h hands-on, hours of wall time | critical path; start before coding |
| 1. Image + SIF (timm, weights baked), upload SIF | 1–2 h | user runs the networked build |
| 2. Offline tables (geometry, MS1 training-only) + T1 / T2 | 2 h | numpy, host |
| 3. `r50_gl4_runner.py` Option B (+ baseline phase, seed, weights hook) | 5–7 h | Option A +3–4 h |
| 4. CPU tests T3–T7 in podman, offline and read-only | 2–3 h | |
| 5. `test`-QOS smoke (4 concurrent runs) | 0.5 h + queue | go / no-go on s/study, memory, binding |
| 6. Full node job: v4 s2026, v4 s2027, v4-1P, label-v2 s2026→s2027 chained | ~4 h wall, ~4 node-h | confirm the spec and cost with the user before launch |

Code to results ≈ 1.5–2 days, so results on 2026-10-20/21. That leaves about 1 day for the Kaggle inference port. Its
geometry needs rows / cols at test time (B1) plus the self-location code, and that is the riskiest step for the 10-22
deadline. Scope it in parallel with step 3.
