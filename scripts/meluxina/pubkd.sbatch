#!/bin/bash -l
# PUB-KD students (ledger row PUB-KD): four v4 l4copy lanes trained on the public 0.958 blend teacher targets.
# Usage: sbatch -J rsna-kd pubkd.sbatch kd_a kd_b kd_c kd_d; lane x reads env/pubkd_x.env, its smoke env/pubkd_smoke_x.env.
# Each lane first runs its 32-step smoke (GL_SMOKE=1); a lane whose smoke fails is dropped, the others go on.
# Pretrained weights: $R/hf (timm / HF cache built offline-ready, includes resnet50.tv_in1k) bound over the image's /opt/hf.
# After a lane exits 0 with its SWA weights written, its weights_epoch*.pt and recovery.pt are deleted (home quota).
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 09:00:00
#SBATCH -o /home/users/$USER/rsna/logs/%x-%j.out
set -uo pipefail
module load env/release/2023.1 Apptainer/1.2.4-GCCcore-12.3.0
R=$HOME/rsna; SIF=$R/rsna-mlx-train.sif; OUT=$R/out; mkdir -p "$OUT"
LANES=("$@"); [ ${#LANES[@]} -ge 1 ] && [ ${#LANES[@]} -le 4 ] || { echo "need 1-4 lanes"; exit 1; }
echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) $(date -u +%FT%TZ): ${LANES[*]}"; nvidia-smi -L
echo "ce4473c71f3ea100db021805750c6bb9038bc7b0e56a143c81516edf2d29e77b  $R/assets/rule_states.csv
dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d  $R/assets/soft_labels_final_per_target_latD2.json
2dd1044adcc54ca4d9bcb9c9fd3b910a6784bd2b99e792bbeaf84a42d417f06a  $R/assets/folds.csv
7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef  $R/pubckpt/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt" | sha256sum -c || exit 1
sha256sum -c "$R/pubkd/targets.sha256" || exit 1                 # teacher target files, pinned when built
sha256sum "$R"/scripts/r50_gl4_runner.py "$R"/scripts/r50_gl_runner.py "$R"/scripts/coatnet_d96_runner.py
for m in resnet50.tv_in1k convnext_tiny.in12k_ft_in1k_384 coatnet_0_rw_224.sw_in1k tf_efficientnetv2_s.in21k_ft_in1k \
         vit_small_patch14_reg4_dinov2.lvd142m vit_small_patch16_dinov3.lvd1689m; do
  [ -d "$R/hf/hub/models--timm--$m" ] || { echo "missing weights $m"; exit 1; }; done
for l in "${LANES[@]}"; do [ -f "$R/env/pubkd_$l.env" ] && [ -f "$R/env/pubkd_smoke_$l.env" ] || { echo "missing env for $l"; exit 1; }
  [ -e "$OUT/pubkd_$l" ] && { echo "$OUT/pubkd_$l exists (would resume or mix logs); move it first"; exit 1; }
  rm -rf "$OUT/pubkd_smoke_$l"; done                     # fresh smoke logs: no stale SMOKE_DONE
t0=$SECONDS; mkdir -p "$LOCALSCRATCH/d96"
rsync -a "$R/d96/" "$LOCALSCRATCH/d96/" || { echo "staging failed"; exit 1; }
CACHE=$LOCALSCRATCH/d96; echo "staged $(du -sh $CACHE | cut -f1) in $((SECONDS - t0))s"

run() {  # run <env name> <gpu index>; CUDA_VISIBLE_DEVICES pinned per lane (MeluXina steps see all 4 GPUs)
  mkdir -p "$OUT/$1"
  srun --exact -n1 -c16 --gpus-per-task=1 --mem=110G \
    apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$2 -B "$CACHE":/data/d96:ro -B "$R/hf":/opt/hf:ro -B "$R":/work \
      --pwd /work/scripts --env-file "$R/env/$1.env" "$SIF" python -u /work/scripts/r50_gl4_runner.py >> "$OUT/$1/train.log" 2>&1
  local rc=$?; echo "$(date -u +%FT%TZ) $1 (GPU $2) exit $rc"
  if [ $rc -eq 0 ] && ls "$OUT/$1"/weights_swa*.pt >/dev/null 2>&1; then
    rm -f "$OUT/$1"/weights_epoch*.pt "$OUT/$1/recovery.pt"; echo "$1: per-epoch weights and recovery.pt removed"
  fi
}

for i in "${!LANES[@]}"; do ( run pubkd_smoke_${LANES[$i]} $i ) & done; wait
OK=()
for l in "${LANES[@]}"; do
  if grep -q SMOKE_DONE "$OUT/pubkd_smoke_$l/train.log"; then
    OK+=("$l"); grep -E "FUSE|s/study" "$OUT/pubkd_smoke_$l/train.log" | tail -3
  else
    echo "smoke failed, lane dropped: $l"; tail -30 "$OUT/pubkd_smoke_$l/train.log"
  fi
done
[ ${#OK[@]} -ge 1 ] || { echo "all smokes failed"; exit 1; }
echo "$(date -u +%FT%TZ) smokes OK: ${OK[*]}"

for i in "${!OK[@]}"; do ( run pubkd_${OK[$i]} $i ) & done
( sleep 300
  nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv > "$OUT/gpu_map_$SLURM_JOB_NAME.txt" 2>&1
  np_=$(tail -n +2 "$OUT/gpu_map_$SLURM_JOB_NAME.txt" | wc -l); n=$(tail -n +2 "$OUT/gpu_map_$SLURM_JOB_NAME.txt" | cut -d, -f2 | sort -u | wc -l)
  echo "$(date -u +%FT%TZ) GPU guard: $np_ processes on $n distinct GPUs"   # cancel only if lanes share a GPU
  [ "$np_" -le "$n" ] || { echo "GPU guard failed: lanes share GPUs"; cat "$OUT/gpu_map_$SLURM_JOB_NAME.txt"; scancel "$SLURM_JOB_ID"; } ) &
nvidia-smi dmon -s um -d 60 > "$OUT/gpu_util_$SLURM_JOB_NAME.log" 2>&1 &
DMON=$!
for j in $(jobs -p); do [ "$j" = "$DMON" ] || wait "$j"; done
kill "$DMON" 2>/dev/null
echo "done $(date -u +%FT%TZ) after $((SECONDS / 60)) min"
