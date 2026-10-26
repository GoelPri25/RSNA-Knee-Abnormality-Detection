#!/bin/bash -l
# L4C-R2 crop arm on the idle GPU 3 of job 5356217 (srun --overlap step, same allocation): parent s2026 then s2027.
module load env/release/2023.1 Apptainer/1.2.4-GCCcore-12.3.0
R=$HOME/rsna; SIF=$R/rsna-mlx-train.sif; CACHE=$LOCALSCRATCH/d96
echo "$(date -u +%FT%TZ) r2 lane on $(hostname), cache $CACHE ($(ls $CACHE | wc -l) entries)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
for n in r2_crop_p2026 r2_crop_p2027; do
  mkdir -p "$R/out/$n"
  apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=3 -B "$CACHE":/data/d96:ro -B "$R":/work \
    --pwd /work/scripts --env-file "$R/env/$n.env" "$SIF" python -u /work/scripts/l4copy_r2_runner.py >> "$R/out/$n/train.log" 2>&1
  echo "$(date -u +%FT%TZ) $n (GPU 3) exit $?"
done
