#!/bin/bash -l
# V4-IMP (docs/v4_improve_spec.md v2, ledger V4-IMP): four single-change v4 l4copy runs on one node.
#   GPU 0: imp_silw_s2026     silent assumed-negative cells x 0.25
#   GPU 1: imp_hill_s2026     silent assumed-negative cells, Hill negative term
#   GPU 2: imp_mcpair_s2026   FS + NF co-registered 6-channel input
#   GPU 3: imp_mcpair_s2027   replicate seed
# Each lane first runs its 32-step smoke (GL_SMOKE=1, about 1 min); the job stops unless all four print SMOKE_DONE.
# Cache staged to node-local SSD; outputs to ~/rsna/out; compute nodes treated as offline.
#SBATCH -J rsna-imp
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 03:30:00
#SBATCH -o /home/users/$USER/rsna/logs/%x-%j.out
set -uo pipefail
module load env/release/2023.1 Apptainer/1.2.4-GCCcore-12.3.0
R=$HOME/rsna; SIF=$R/rsna-mlx-train.sif; OUT=$R/out; mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname) $(date -u +%FT%TZ)"; nvidia-smi -L
echo "ce4473c71f3ea100db021805750c6bb9038bc7b0e56a143c81516edf2d29e77b  $R/assets/rule_states.csv
dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d  $R/assets/soft_labels_final_per_target_latD2.json
2dd1044adcc54ca4d9bcb9c9fd3b910a6784bd2b99e792bbeaf84a42d417f06a  $R/assets/folds.csv" | sha256sum -c || exit 1
t0=$SECONDS; mkdir -p "$LOCALSCRATCH/d96"
rsync -a "$R/d96/" "$LOCALSCRATCH/d96/" || { echo "staging failed"; exit 1; }
CACHE=$LOCALSCRATCH/d96; echo "staged $(du -sh $CACHE | cut -f1) in $((SECONDS - t0))s"

run() {  # run <name> <gpu index>; CUDA_VISIBLE_DEVICES pinned per lane (MeluXina steps see all 4 GPUs)
  mkdir -p "$OUT/$1"
  srun --exact -n1 -c16 --gpus-per-task=1 --mem=110G \
    apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$2 -B "$CACHE":/data/d96:ro -B "$R":/work \
      --pwd /work/scripts --env-file "$R/env/$1.env" "$SIF" python -u /work/scripts/r50_gl4_runner.py >> "$OUT/$1/train.log" 2>&1
  local rc=$?; echo "$(date -u +%FT%TZ) $1 (GPU $2) exit $rc"
}
LANES=(silw_s2026 hill_s2026 mcpair_s2026 mcpair_s2027)

for i in 0 1 2 3; do ( run imp_smoke_${LANES[$i]} $i ) & done; wait
for l in "${LANES[@]}"; do
  grep -q SMOKE_DONE "$OUT/imp_smoke_$l/train.log" || { echo "smoke failed: $l"; tail -30 "$OUT/imp_smoke_$l/train.log"; exit 1; }
  grep -E "MCPAIR|SILENT|s/study" "$OUT/imp_smoke_$l/train.log" | tail -3
done
echo "$(date -u +%FT%TZ) smokes OK"

for i in 0 1 2 3; do ( run imp_${LANES[$i]} $i ) & done
( sleep 300
  nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv > "$OUT/gpu_map_imp.txt" 2>&1
  n=$(tail -n +2 "$OUT/gpu_map_imp.txt" | cut -d, -f2 | sort -u | wc -l); echo "$(date -u +%FT%TZ) GPU guard: $n distinct GPUs busy"
  [ "$n" -ge 4 ] || { echo "GPU guard failed"; cat "$OUT/gpu_map_imp.txt"; scancel "$SLURM_JOB_ID"; } ) &
nvidia-smi dmon -s um -d 60 > "$OUT/gpu_util_imp.log" 2>&1 &
DMON=$!
for j in $(jobs -p); do [ "$j" = "$DMON" ] || wait "$j"; done
kill "$DMON" 2>/dev/null
echo "done $(date -u +%FT%TZ) after $((SECONDS / 60)) min"
