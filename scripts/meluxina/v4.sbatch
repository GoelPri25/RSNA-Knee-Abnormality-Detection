#!/bin/bash -l
# r50-GL v4 (ledger r50-GL4): one GPU node (4 x A100-40GB), 4 lanes of 2 runs each, one allocation.
#   lane 1: v4 shared s2026  -> label-v2 r50 s2027
#   lane 2: v4 shared s2027  -> label-v2 r50 s2028
#   lane 3: v4 adapter s2026 -> v4 shared s2028     (adapter is the fastest v4 run: lanes balanced at ~3.4-3.6 h)
#   lane 4: v4 l4copy s2026  -> label-v2 r50 s2029
# Cache staged to node-local SSD; outputs to ~/rsna/out (home only). Compute nodes are treated as offline.
#SBATCH -J rsna-v4
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 06:30:00
#SBATCH -o /home/users/$USER/rsna/logs/%x-%j.out
set -uo pipefail
module load env/release/2023.1 Apptainer/1.2.4-GCCcore-12.3.0   # Apptainer needs the 2023.1 release env
R=$HOME/rsna; SIF=$R/rsna-mlx-train.sif; OUT=$R/out; mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname) $(date -u +%FT%TZ)"; nvidia-smi -L
t0=$SECONDS; mkdir -p "$LOCALSCRATCH/d96"
rsync -a "$R/d96/" "$LOCALSCRATCH/d96/" || { echo "staging failed"; exit 1; }
CACHE=$LOCALSCRATCH/d96; echo "staged $(du -sh $CACHE | cut -f1) in $((SECONDS - t0))s"

run() {  # run <name> <script> <gpu index>
  # MeluXina does not confine a step to its GPUs (every step sees all 4) and --cleanenv drops Slurm's
  # CUDA_VISIBLE_DEVICES, so each lane pins its own GPU explicitly (job 5355465 ran all 4 runs on GPU 0).
  mkdir -p "$OUT/$1"
  srun --exact -n1 -c16 --gpus-per-task=1 --mem=110G \
    apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$3 -B "$CACHE":/data/d96:ro -B "$R":/work \
      --pwd /work/scripts --env-file "$R/env/$1.env" "$SIF" python -u "/work/scripts/$2" >> "$OUT/$1/train.log" 2>&1
  local rc=$?; echo "$(date -u +%FT%TZ) $1 (GPU $3) exit $rc"
}

run preflight preflight_v4.py 0
grep -q PREFLIGHT_OK "$OUT/preflight/train.log" || { echo "preflight failed"; tail -30 "$OUT/preflight/train.log"; exit 1; }

( run v4_shared_s2026 r50_gl4_runner.py 0;  run lv2_s2027 coatnet_d96_runner.py 0 ) &
( run v4_shared_s2027 r50_gl4_runner.py 1;  run lv2_s2028 coatnet_d96_runner.py 1 ) &
( run v4_adapter_s2026 r50_gl4_runner.py 2; run v4_shared_s2028 r50_gl4_runner.py 2 ) &
( run v4_l4copy_s2026 r50_gl4_runner.py 3;  run lv2_s2029 coatnet_d96_runner.py 3 ) &
( sleep 300                                  # guard: 4 training processes on 4 distinct GPUs, else stop the job
  nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv > "$OUT/gpu_map.txt" 2>&1
  n=$(tail -n +2 "$OUT/gpu_map.txt" | cut -d, -f2 | sort -u | wc -l); echo "$(date -u +%FT%TZ) GPU guard: $n distinct GPUs busy"
  [ "$n" -ge 4 ] || { echo "GPU guard failed"; cat "$OUT/gpu_map.txt"; scancel "$SLURM_JOB_ID"; } ) &
nvidia-smi dmon -s um -d 60 > "$OUT/gpu_util.log" 2>&1 &  # per-GPU SM / memory utilisation every minute
DMON=$!
for j in $(jobs -p); do [ "$j" = "$DMON" ] || wait "$j"; done
kill "$DMON" 2>/dev/null
echo "done $(date -u +%FT%TZ) after $((SECONDS / 60)) min"
