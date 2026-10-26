#!/bin/bash -l
# SURPASS-F Job 1 (ledger row SURPASS-F): fastMRI knee DICOM batches -> d96 cache -> dedupe vs our cache -> Raptor teacher.
# 1. both archives streamed (xz -> tar) through scripts/fastmri_build_cache.py in parallel: batch 2 -> shard_0..1,
#    batch 1 -> shard_2..7 under $F/fastmri_d96 (competition builder e76793ca, series table from DICOM headers).
# 2. scripts/fastmri_gate.py (revised before the full build, 10-20: our own cache fills all 5 slots in only 71.8%):
#    (a) decode errors < 1% of picked slices; (b) >= 95% sag-fluid slot filled and >= 90% with >= 3 non-empty slots;
#    else stop here (no teacher). Per-slot fill vs our cache and a 4-study montage are logged (descriptive).
# 3. scripts/fastmri_dedupe.py vs ~/rsna/d96 (4 shards) -> $F/fastmri_d96/fm_dupes.json.
# 4. scripts/pubkd_raptor_teacher.py (nartaa 384 / 224, mirror TTA) on the fastMRI cache, 4 GPUs x (384 then 224),
#    -> $F/fastmri_teacher/raptor{384,224}_{0..3}.json.
# No CPU allocation on <project> (cpu partition: AssocGrpGRESMinutes), so the CPU build runs on this GPU node too.
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -t 03:00:00
#SBATCH -o /home/users/$USER/rsna/logs/%x-%j.out
set -uo pipefail
module load env/release/2023.1 Apptainer/1.2.4-GCCcore-12.3.0
R=$HOME/rsna; SIF=$R/rsna-mlx-train.sif; F=$PROJECT_DIR; C=$F/fastmri_d96; T=$F/fastmri_teacher
echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) $(date -u +%FT%TZ)"; nvidia-smi -L; nproc; free -g | head -2
grep -q ALL_DONE $F/fastmri_raw/get.log || { echo "download not finished"; exit 1; }
ls -la $F/fastmri_raw/*.tar.xz
echo "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef  $R/pubckpt/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6  $R/pubckpt/raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt" | sha256sum -c || exit 1
sha256sum $R/scripts/fastmri_build_cache.py $R/scripts/fastmri_gate.py $R/scripts/fastmri_dedupe.py $R/scripts/pubkd_raptor_teacher.py $R/scripts/coatnet_d96_runner.py
[ -e "$C/shard_0" ] && { echo "$C exists; move it first"; exit 1; }; mkdir -p "$C"
[ -e "$T" ] && { echo "$T exists; move it first"; exit 1; }
AX="apptainer exec --cleanenv -B $F:/fm -B $R:/work --pwd /work/scripts --env FM_PYLIBS=/fm/pylibs,FM_STAGE=/dev/shm/fm_stage_$SLURM_JOB_ID,OMP_NUM_THREADS=1 $SIF"

t0=$SECONDS
build() {  # build <archive> <tag> <offset> <nshard> <workers>
  xz -T4 -dc "$F/fastmri_raw/$1" | $AX python -I /work/scripts/fastmri_build_cache.py - /fm/fastmri_d96 $2 $3 $4 $5 > "$C/build_$2.log" 2>&1
  echo "$(date -u +%FT%TZ) build $2 exit ${PIPESTATUS[*]} after $(((SECONDS - t0) / 60)) min"
}
build knee_DICOMs_batch2.tar.xz b2 0 2 24 &
build knee_DICOMs_batch1.tar.xz b1 2 6 88 &
wait
rm -rf /dev/shm/fm_stage_$SLURM_JOB_ID
tail -5 "$C/build_b2.log" "$C/build_b1.log"

$AX python -I /work/scripts/fastmri_gate.py /work/d96 4 /fm/fastmri_d96 8 b2 b1 || { echo "GATE FAILED: no teacher"; exit 2; }

$AX python -I /work/scripts/fastmri_dedupe.py /work/d96 4 /fm/fastmri_d96 8 /fm/fastmri_d96/fm_dupes.json
echo "$(date -u +%FT%TZ) dedupe done after $(((SECONDS - t0) / 60)) min"

mkdir -p "$T"
teach() {  # teach <shard = gpu>
  for m in 384 224; do
    local ck=raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt; [ $m = 224 ] && ck=raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt
    srun --exact -n1 -c32 --gpus-per-task=1 --mem=110G \
      apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$1 -B $F:/fm -B $R:/work --pwd /work/scripts \
        --env D96_CACHE=/fm/fastmri_d96,D96_ASSETS=/work/assets,PYTHONPATH=/work/scripts,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=4 \
        --env PUBKD_CACHE_SHARDS=8,PUBKD_MODEL=$m,PUBKD_SHARD=$1,PUBKD_NSHARD=4,PUBKD_CKPT=/work/pubckpt/$ck,PUBKD_OUT=/fm/fastmri_teacher \
        "$SIF" python -u /work/scripts/pubkd_raptor_teacher.py >> "$T/raptor${m}_$1.log" 2>&1
    echo "$(date -u +%FT%TZ) raptor$m shard $1 exit $?"
  done
}
teach 0 & teach 1 & teach 2 & teach 3 & wait
ls -la "$T"; du -sh "$C"; echo "done $(date -u +%FT%TZ) after $(((SECONDS - t0) / 60)) min"
