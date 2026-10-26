#!/bin/bash -l
# L4C-LOC (docs/l4c_loc_spec.md v2, ledger L4C-LOC): learned crop locator on the frozen v4 l4copy, 1 node, 1 node-hour.
#   lane 0: loc_s2026   locator for the LB-0.939 weights (sha 2a570689...)      ~25 min
#   lane 1: loc_s2027   the replication seed (sha cd36f3f3...)                   ~25 min
# Uploaded beforehand to ~/rsna: scripts/{l4c_loc_runner.py,l4copy_xai.py,l4copy_diag_core.py}, v4art/{v4_submission.py,
# cp_*.npz}, v4w/{s2026,s2027}.pt, loc/quota_prior.json. Cache staged to node-local SSD; compute nodes offline.
#SBATCH -J rsna-loc
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH -o /home/users/$USER/rsna/logs/%x-%j.out
set -uo pipefail
module load env/release/2023.1 Apptainer/1.2.4-GCCcore-12.3.0
R=$HOME/rsna; SIF=$R/rsna-mlx-train.sif; OUT=$R/out; mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname) $(date -u +%FT%TZ)"; nvidia-smi -L
echo "2a570689d07a3a01caa96ca058076a7d695181848033a9e9dcd0f4b3a68d063b  $R/v4w/s2026.pt
cd36f3f3f9311de214a66b04b8f2d21be97542a38038ae51909b47bc4b3f0ecc  $R/v4w/s2027.pt" | sha256sum -c || exit 1
t0=$SECONDS; mkdir -p "$LOCALSCRATCH/d96"
rsync -a "$R/d96/" "$LOCALSCRATCH/d96/" || { echo "staging failed"; exit 1; }
CACHE=$LOCALSCRATCH/d96; echo "staged $(du -sh $CACHE | cut -f1) in $((SECONDS - t0))s"

lane() {  # lane <tag> <gpu>
  local n=loc_$1; mkdir -p "$OUT/$n"
  srun --exact -n1 -c16 --gpus-per-task=1 --mem=110G \
    apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$2 -B "$CACHE":/data/d96:ro -B "$R":/work --pwd /work/scripts \
      --env D96_CACHE=/data/d96,V4_ART=/work/v4art,LOC_WEIGHTS=/work/v4w/$1.pt,LOC_TAG=$1,LOC_OUT=/work/out/$n,LOC_FOLDS=/work/assets/folds.csv,LOC_QPRIOR=/work/loc/quota_prior.json,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=8 \
      "$SIF" python -u /work/scripts/l4c_loc_runner.py >> "$OUT/$n/train.log" 2>&1
  local rc=$?; echo "$(date -u +%FT%TZ) $n (GPU $2) exit $rc"
}
( lane s2026 0 ) &
( lane s2027 1 ) &
wait
echo "done $(date -u +%FT%TZ)"
