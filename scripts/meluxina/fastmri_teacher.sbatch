#!/bin/bash -l
# SURPASS-F Job 1b (ledger row SURPASS-F): rerun of Job 1's teacher stage, then the Job 2 targets.
# Job 1 (5363573) stopped at the teacher: 4 exams sit in both fastMRI archives, so their study ids were listed twice and
# read_manifest refused them. Those rows were dropped from the manifests on 2026-10-20 (originals kept as
# manifest.jsonl.job1; ids in fastmri_d96/fm_dup_archive_ids.json), leaving 10,008 studies.
# 1. scripts/pubkd_raptor_teacher.py (nartaa 384 then 224, mirror TTA), 4 GPUs -> $F/fastmri_teacher/raptor{384,224}_{0..3}.json;
#    success = the json exists (Job 1's log reported exit 0 for the failed runs).
# 2. scripts/meluxina/sf_prepare.sh (targets + holdout + sha pins in env/sf_*.env): runs here because the login node
#    cannot mount the SIF (no squashfuse).
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -t 01:30:00
#SBATCH -o /home/users/$USER/rsna/logs/%x-%j.out
set -uo pipefail
module load env/release/2023.1 Apptainer/1.2.4-GCCcore-12.3.0
R=$HOME/rsna; SIF=$R/rsna-mlx-train.sif; F=$PROJECT_DIR; C=$F/fastmri_d96; T=$F/fastmri_teacher
echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) $(date -u +%FT%TZ)"; nvidia-smi -L
echo "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef  $R/pubckpt/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6  $R/pubckpt/raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt" | sha256sum -c || exit 1
sha256sum $R/scripts/pubkd_raptor_teacher.py $R/scripts/coatnet_d96_runner.py $R/scripts/build_fm_targets.py $R/sf_prepare.sh
python3 -I -c "
import json, collections
c = collections.Counter(json.loads(l)['study_id'] for i in range(8) for l in open('$C/shard_%d/manifest.jsonl' % i))
d = [s for s, n in c.items() if n > 1]; print('manifest rows', sum(c.values()), 'duplicate ids', len(d)); assert not d" || exit 1
ls "$T"/raptor*.json >/dev/null 2>&1 && { echo "$T already has teacher json; move it first"; exit 1; }
mkdir -p "$T"

t0=$SECONDS
teach() {  # teach <shard = gpu>
  for m in 384 224; do
    local ck=raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt; [ $m = 224 ] && ck=raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt
    srun --exact -n1 -c32 --gpus-per-task=1 --mem=110G \
      apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$1 -B $F:/fm -B $R:/work --pwd /work/scripts \
        --env D96_CACHE=/fm/fastmri_d96,D96_ASSETS=/work/assets,PYTHONPATH=/work/scripts,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=4 \
        --env PUBKD_CACHE_SHARDS=8,PUBKD_MODEL=$m,PUBKD_SHARD=$1,PUBKD_NSHARD=4,PUBKD_CKPT=/work/pubckpt/$ck,PUBKD_OUT=/fm/fastmri_teacher \
        "$SIF" python -u /work/scripts/pubkd_raptor_teacher.py >> "$T/raptor${m}_$1.log" 2>&1
    local rc=$?; echo "$(date -u +%FT%TZ) raptor$m shard $1 exit $rc after $(((SECONDS - t0) / 60)) min"
    [ -s "$T/raptor${m}_$1.json" ] || { echo "raptor$m shard $1: no json"; tail -15 "$T/raptor${m}_$1.log"; return 1; }
  done
}
teach 0 & teach 1 & teach 2 & teach 3 & wait
n=$(ls "$T"/raptor{384,224}_{0,1,2,3}.json 2>/dev/null | wc -l)
echo "teacher json files: $n / 8"; [ "$n" -eq 8 ] || { echo "TEACHER INCOMPLETE: no targets"; exit 2; }
ls -la "$T"

bash -l "$R/sf_prepare.sh" || { echo "sf_prepare failed"; exit 3; }
cat "$F/fastmri_targets/build_fm_targets.txt"
echo "done $(date -u +%FT%TZ) after $(((SECONDS - t0) / 60)) min"
