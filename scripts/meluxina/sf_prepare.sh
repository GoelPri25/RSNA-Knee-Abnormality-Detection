#!/bin/bash -l
# SURPASS-F Job 2 preparation (login node, after Job 1): joint Raptor targets over competition + fastMRI studies with the
# patient-level holdout (scripts/build_fm_targets.py, numpy only, ~1 min), leaving out the dupes and fm_exclude.json
# (fm_misslot.json reason "negation": NON FAT SAT series slotted as fat-sat; its "fs_fallback" studies are kept), their sha pinned in targets.sha256 and in
# env/sf_*.env (V4_TARGETS_SHA). Refuses to overwrite an existing target dir.
set -euo pipefail
module load env/release/2023.1 Apptainer/1.2.4-GCCcore-12.3.0
R=$HOME/rsna; SIF=$R/rsna-mlx-train.sif; F=$PROJECT_DIR; O=$F/fastmri_targets
[ -e "$O" ] && { echo "$O exists; move it first"; exit 1; }
ls "$F"/fastmri_teacher/raptor384_{0,1,2,3}.json "$F"/fastmri_teacher/raptor224_{0,1,2,3}.json "$F/fastmri_d96/fm_dupes.json" "$F/fastmri_d96/fm_exclude.json" >/dev/null
echo "6ed9225434bf8cf77cc2a3ded2c47a8443fc0967a1a215c081da482e1f3bfa49  $R/pubkd/pubkd_targets_raptor_pure.json" | sha256sum -c
apptainer exec --cleanenv --env FM_EXCLUDE=/fm/fastmri_d96/fm_exclude.json -B "$F":/fm -B "$R":/work "$SIF" python -I /work/scripts/build_fm_targets.py \
  /work/out/pubkd_teacher /work/pubkd/pubkd_targets_raptor_pure.json /fm/fastmri_teacher /fm/fastmri_d96 8 \
  /fm/fastmri_d96/fm_dupes.json /fm/fastmri_targets 500
(cd "$O" && sha256sum fm_targets_raptor_pure.json fm_holdout.json fm_holdout_teacher.json > targets.sha256 && cat targets.sha256)
sed -i "s#^V4_TARGETS=.*#V4_TARGETS=/fm/fastmri_targets/fm_targets_raptor_pure.json#" "$R"/env/sf_*.env
sed -i "s#^V4_TARGETS_SHA=.*#V4_TARGETS_SHA=$(sha256sum "$O/fm_targets_raptor_pure.json" | cut -d' ' -f1)#" "$R"/env/sf_*.env
sed -i "s#  #  $O/#" "$O/targets.sha256"                     # absolute paths for sha256sum -c in the job
grep -H V4_TARGETS_SHA "$R"/env/sf_*.env
