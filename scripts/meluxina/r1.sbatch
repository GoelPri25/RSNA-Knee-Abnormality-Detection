#!/bin/bash -l
# L4C-R1 (ledger L4C-R1, round 1 on v4 l4copy): one GPU node (4 x A100-40GB), 4 lanes, one allocation.
#   lane 0: r1_self_s2026     MS1-free l4copy: own crop table rebuilt every epoch (random crops epochs 1-2)  ~3.5 h
#   lane 1: r1_selfjit_s2026  the same + slice jitter +-2 inside the series                                   ~3.5 h
#   lane 2: v4_l3copy_s2026   -> v4_l3copy_s2027   crops get own layer3 + layer4 (MS1 table, v4 recipe)        ~1.8 + 1.8 h
#   lane 3: v4_l4copy_s2027   -> (none)            the l4copy seed reference (MS1 table, v4 recipe)             ~1.7 h
# Cache staged to node-local SSD; outputs to ~/rsna/out (home only). Compute nodes are treated as offline.
#SBATCH -J rsna-r1
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 05:30:00
#SBATCH -o /home/users/$USER/rsna/logs/%x-%j.out
set -uo pipefail
module load env/release/2023.1 Apptainer/1.2.4-GCCcore-12.3.0   # Apptainer needs the 2023.1 release env
R=$HOME/rsna; SIF=$R/rsna-mlx-train.sif; OUT=$R/out; mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname) $(date -u +%FT%TZ)"; nvidia-smi -L
t0=$SECONDS; mkdir -p "$LOCALSCRATCH/d96"
rsync -a "$R/d96/" "$LOCALSCRATCH/d96/" || { echo "staging failed"; exit 1; }
CACHE=$LOCALSCRATCH/d96; echo "staged $(du -sh $CACHE | cut -f1) in $((SECONDS - t0))s"

run() {  # run <name> <script> <gpu index>   (each lane pins its own GPU: MeluXina steps see all 4, job 5355465)
  mkdir -p "$OUT/$1"
  srun --exact -n1 -c16 --gpus-per-task=1 --mem=110G \
    apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$3 -B "$CACHE":/data/d96:ro -B "$R":/work \
      --pwd /work/scripts --env-file "$R/env/$1.env" "$SIF" python -u "/work/scripts/$2" >> "$OUT/$1/train.log" 2>&1
  local rc=$?; echo "$(date -u +%FT%TZ) $1 (GPU $3) exit $rc"
}

run preflight_r1 preflight_r1.py 0
grep -q PREFLIGHT_OK "$OUT/preflight_r1/train.log" || { echo "preflight failed"; tail -30 "$OUT/preflight_r1/train.log"; exit 1; }

( run r1_self_s2026 l4copy_r1_runner.py 0 ) &
( run r1_selfjit_s2026 l4copy_r1_runner.py 1 ) &
( run v4_l3copy_s2026 r50_gl4_runner.py 2; run v4_l3copy_s2027 r50_gl4_runner.py 2 ) &
( run v4_l4copy_s2027 r50_gl4_runner.py 3 ) &
( sleep 600                                  # guard: 4 training processes on 4 distinct GPUs, else stop the job
  nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv > "$OUT/gpu_map_r1.txt" 2>&1
  n=$(tail -n +2 "$OUT/gpu_map_r1.txt" | cut -d, -f2 | sort -u | wc -l); echo "$(date -u +%FT%TZ) GPU guard: $n distinct GPUs busy"
  [ "$n" -ge 4 ] || { echo "GPU guard failed"; cat "$OUT/gpu_map_r1.txt"; scancel "$SLURM_JOB_ID"; } ) &
nvidia-smi dmon -s um -d 60 > "$OUT/gpu_util_r1.log" 2>&1 &
DMON=$!
for j in $(jobs -p); do [ "$j" = "$DMON" ] || wait "$j"; done
kill "$DMON" 2>/dev/null
echo "done $(date -u +%FT%TZ) after $((SECONDS / 60)) min"
