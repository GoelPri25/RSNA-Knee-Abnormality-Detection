#!/bin/bash -l
# SURPASS-F Job 2 (ledger row SURPASS-F): students on the Raptor teacher's targets over competition + fastMRI studies.
#   GPU 0  F1  v4 l4copy r50 (ra recipe, 25 epochs) + fastMRI, one source per batch, fastMRI = global loss only
#   GPU 1  F2  F1 + student noise (drop path 0.2, feature dropout 0.3, stronger augmentation)
#   GPU 2  F3  rap2 (Raptor 384 init) + fastMRI + noise; epochs 4 (6 / 4 / 3 chosen from its smoke speed, F3_BUDGET_H)
#   GPU 3  ra's weights on the fastMRI holdout (eval only), then F4 pre (fastMRI only, 20 epochs, SWA 16-20),
#          then F4 fine (ra recipe on competition only, from F4 pre's SWA; crop layer 4 = copy of layer 4)
# Lane x reads env/sf_x.env; its smoke env is derived here (GL_SMOKE=1, own GL_OUT; F4 fine smoke from F4 pre's smoke).
# Each lane first runs its 32-step smoke; a lane whose smoke fails is dropped, the others go on.
# Caches staged to $LOCALSCRATCH when it has room (competition 59 GB + fastMRI ~140 GB), else fastMRI read from Lustre.
# Outputs in $PROJECT_DIR/out/sf_*; per-epoch weights / recovery.pt removed after a lane's SWA weights exist.
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 10:00:00
#SBATCH -o /home/users/$USER/rsna/logs/%x-%j.out
set -uo pipefail
module load env/release/2023.1 Apptainer/1.2.4-GCCcore-12.3.0
R=$HOME/rsna; SIF=$R/rsna-mlx-train.sif; F=$PROJECT_DIR; OUT=$F/out; mkdir -p "$OUT" "$R/out"
F3_BUDGET_H=${F3_BUDGET_H:-8.5}
echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) $(date -u +%FT%TZ)"; nvidia-smi -L
echo "dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d  $R/assets/soft_labels_final_per_target_latD2.json
2dd1044adcc54ca4d9bcb9c9fd3b910a6784bd2b99e792bbeaf84a42d417f06a  $R/assets/folds.csv
7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef  $R/pubckpt/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt" | sha256sum -c || exit 1
sha256sum -c "$F/fastmri_targets/targets.sha256" || exit 1          # targets + holdout, pinned when built
sha256sum "$R"/scripts/r50_gl4_runner.py "$R"/scripts/r50_gl_runner.py "$R"/scripts/coatnet_d96_runner.py "$R/out/pubkd_kd_ra/weights_swa21_25.pt"
[ -d "$R/hf/hub/models--timm--resnet50.tv_in1k" ] || { echo "missing resnet50 weights"; exit 1; }
LANES=(f1 f2 f3 f4pre f4fine raeval)
for l in "${LANES[@]}"; do [ -f "$R/env/sf_$l.env" ] || { echo "missing env sf_$l"; exit 1; }
  grep -q PIN_AFTER_JOB1 "$R/env/sf_$l.env" && { echo "sf_$l.env: targets sha not pinned"; exit 1; }
  [ -e "$OUT/sf_$l" ] && { echo "$OUT/sf_$l exists (would resume or mix logs); move it first"; exit 1; }
  rm -rf "$OUT/sf_smoke_$l"
  sed -e "s#^GL_OUT=.*#GL_OUT=/fm/out/sf_smoke_$l#" -e "s#^V4_INIT=.*#V4_INIT=/fm/out/sf_smoke_f4pre/weights_swa1_1.pt#" \
      "$R/env/sf_$l.env" > "$R/env/sf_smoke_$l.env"; echo "GL_SMOKE=1" >> "$R/env/sf_smoke_$l.env"
done

t0=$SECONDS; mkdir -p "$LOCALSCRATCH/d96"
rsync -a "$R/d96/" "$LOCALSCRATCH/d96/" || { echo "staging failed"; exit 1; }
CACHE=$LOCALSCRATCH/d96; echo "staged competition $(du -sh $CACHE | cut -f1) in $((SECONDS - t0))s"
avail=$(df --output=avail -BG "$LOCALSCRATCH" | tail -1 | tr -dc 0-9); need=$(( $(du -s -BG "$F/fastmri_d96" | cut -f1 | tr -dc 0-9) + 20 ))
echo "local scratch free ${avail}G, fastMRI cache needs ${need}G"
if [ "$avail" -ge "$need" ]; then
  t1=$SECONDS; mkdir -p "$LOCALSCRATCH/fm"
  rsync -a --include='shard_*/' --include='*.npy' --include='manifest.jsonl' --exclude='*/*' --exclude='*' \
    "$F/fastmri_d96/" "$LOCALSCRATCH/fm/" || { echo "fastMRI staging failed"; exit 1; }
  FMC=$LOCALSCRATCH/fm; echo "staged fastMRI $(du -sh $FMC | cut -f1) in $((SECONDS - t1))s"
else
  FMC=$F/fastmri_d96; echo "fastMRI read from Lustre ($FMC)"
fi

run() {  # run <env name> <gpu index>; CUDA_VISIBLE_DEVICES pinned per lane (MeluXina steps see all 4 GPUs)
  mkdir -p "$OUT/$1"
  srun --exact -n1 -c16 --gpus-per-task=1 --mem=110G \
    apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$2 -B "$CACHE":/data/d96:ro -B "$FMC":/data/fm:ro \
      -B "$R/hf":/opt/hf:ro -B "$R":/work -B "$F":/fm --pwd /work/scripts --env-file "$R/env/$1.env" "$SIF" \
      python -u /work/scripts/r50_gl4_runner.py >> "$OUT/$1/train.log" 2>&1
  local rc=$?; echo "$(date -u +%FT%TZ) $1 (GPU $2) exit $rc"
  if [ $rc -eq 0 ] && ls "$OUT/$1"/weights_swa*.pt >/dev/null 2>&1; then
    rm -f "$OUT/$1"/weights_epoch*.pt "$OUT/$1/recovery.pt"; echo "$1: per-epoch weights and recovery.pt removed"
  fi
  return $rc
}
smoke_ok() { grep -q SMOKE_DONE "$OUT/sf_smoke_$1/train.log" 2>/dev/null; }

# ---- smokes (GPU 3: F4 pre, F4 fine from it, then ra on the holdout = a full eval-only run of ~2 min)
( run sf_smoke_f1 0 ) & ( run sf_smoke_f2 1 ) & ( run sf_smoke_f3 2 ) &
( run sf_smoke_f4pre 3 && run sf_smoke_f4fine 3; run sf_raeval 3 ) &
wait
OK=()
for l in f1 f2 f3 f4pre f4fine; do
  if smoke_ok $l; then OK+=("$l"); grep -E "EXT |s/study" "$OUT/sf_smoke_$l/train.log" | tail -2
  else echo "smoke failed: $l"; tail -30 "$OUT/sf_smoke_$l/train.log"; fi
done
grep -q "EVAL_ONLY" "$OUT/sf_raeval/train.log" && ls -la "$OUT/sf_raeval/fm_holdout_global.json" || { echo "ra holdout eval failed"; tail -20 "$OUT/sf_raeval/train.log"; }
echo "$(date -u +%FT%TZ) smokes OK: ${OK[*]}"
has() { local x; for x in "${OK[@]}"; do [ "$x" = "$1" ] && return 0; done; return 1; }

# ---- F3 epochs from its smoke speed: largest of 6 / 4 / 3 whose estimate (+ 15% evaluation) fits F3_BUDGET_H
if has f3; then
  sps=$(grep -oE "[0-9.]+s/study" "$OUT/sf_smoke_f3/train.log" | tail -1 | sed 's#s/study##')
  ntr=$(grep -oE "EXT [0-9]+ studies in [0-9]+ shards: [0-9]+ with targets" "$OUT/sf_smoke_f3/train.log" | awk '{print $7}')
  E3=$(awk -v s="$sps" -v n=$((3524 + ntr)) -v b="$F3_BUDGET_H" \
    'BEGIN { e = 0; split("6 4 3", c, " "); for (i = 1; i <= 3; i++) if (!e && s * n * c[i] * 1.15 / 3600 <= b) e = c[i]; print e }')
  echo "F3: smoke ${sps}s/study x $((3524 + ntr)) studies -> epochs $E3 (budget ${F3_BUDGET_H} h)"
  if [ "$E3" -gt 0 ]; then sed -i "s/^V4_EPOCHS=.*/V4_EPOCHS=$E3/" "$R/env/sf_f3.env"; else echo "F3 does not fit: dropped"; fi
fi

# ---- full runs
has f1 && ( run sf_f1 0 ) &
has f2 && ( run sf_f2 1 ) &
has f3 && [ "${E3:-0}" -gt 0 ] && ( run sf_f3 2 ) &
has f4pre && ( run sf_f4pre 3 && { has f4fine && run sf_f4fine 3; } ) &
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
