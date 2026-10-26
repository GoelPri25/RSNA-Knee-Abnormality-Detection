#!/bin/bash -l
# PUB-KD teacher, Raptor slot (ledger row PUB-KD): nartaa's public CoAtNet 384 / 224 checkpoints on every study of the
# d96 cache (scripts/pubkd_raptor_teacher.py; their inference: 94 windows, 320 crop / 0.8 crop + 224, mirror TTA, fp16).
# 4 processes, one GPU each: 384 shards 0/1, 224 shards 0/1. Outputs $R/out/pubkd_teacher/raptor{384,224}_{0,1}.json.
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 01:30:00
#SBATCH -o /home/users/$USER/rsna/logs/%x-%j.out
set -uo pipefail
module load env/release/2023.1 Apptainer/1.2.4-GCCcore-12.3.0
R=$HOME/rsna; SIF=$R/rsna-mlx-train.sif; O=$R/out/pubkd_teacher
echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) $(date -u +%FT%TZ)"; nvidia-smi -L
echo "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef  $R/pubckpt/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6  $R/pubckpt/raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt" | sha256sum -c || exit 1
sha256sum "$R"/scripts/pubkd_raptor_teacher.py "$R"/scripts/coatnet_d96_runner.py
[ -e "$O" ] && { echo "$O exists; move it first"; exit 1; }; mkdir -p "$O"
t0=$SECONDS; mkdir -p "$LOCALSCRATCH/d96"
rsync -a "$R/d96/" "$LOCALSCRATCH/d96/" || { echo "staging failed"; exit 1; }
echo "staged $(du -sh $LOCALSCRATCH/d96 | cut -f1) in $((SECONDS - t0))s"
run() {  # run <model> <shard> <gpu>
  local ck=raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt; [ "$1" = 224 ] && ck=raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt
  srun --exact -n1 -c16 --gpus-per-task=1 --mem=110G \
    apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$3 -B "$LOCALSCRATCH/d96":/data/d96:ro -B "$R":/work --pwd /work/scripts \
      --env D96_CACHE=/data/d96,D96_ASSETS=/work/assets,PYTHONPATH=/work/scripts,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=4 \
      --env PUBKD_MODEL=$1,PUBKD_SHARD=$2,PUBKD_NSHARD=2,PUBKD_CKPT=/work/pubckpt/$ck,PUBKD_OUT=/work/out/pubkd_teacher \
      "$SIF" python -u /work/scripts/pubkd_raptor_teacher.py >> "$O/raptor$1_$2.log" 2>&1
  echo "$(date -u +%FT%TZ) raptor$1 shard $2 (GPU $3) exit $?"
}
run 384 0 0 & run 384 1 1 & run 224 0 2 & run 224 1 3 & wait
ls -la "$O"; echo "done $(date -u +%FT%TZ) after $((SECONDS / 60)) min"
