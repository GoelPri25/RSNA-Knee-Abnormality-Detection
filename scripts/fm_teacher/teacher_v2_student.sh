#!/bin/bash
# KF-CNXS -> teacher v2 -> final student (ledger rows KF-CNXS, 2026-10-22). Steps (each re-runnable):
#   fetch   the 5 fold members' val_swa_global.json from the Modal volume (fold 0 = the teammate's KD_E4CNXS)
#   build   teacher v2 targets (build_teacher_v2.py, w = $W, default 0.15 by the pre-registered rule) -> data/teacher_v2/
#   launch  upload targets + runner to Colab VM r7, write the student's env files (R7-D recipe, teacher v2 targets,
#           fastMRI joint, fold 0 held out), start lane fbf_r7cnxv2 (smoke first) - refuses while another runner trains
# Usage: teacher_v2_student.sh fetch|build|launch [w]
set -euo pipefail
R=.; cd $R
STEP=$1; W=${2:-0.15}; D=data/teacher_v2; V=scripts/fm_teacher/vmssh.sh; L=fbf_r7cnxv2
T=$D/teacher_v2_w${W}_mix73.json
case $STEP in
fetch)
  source ~/.config/rsna/modal-personal.env
  mkdir -p $D/oof
  cp data/_session_tmp/teammate_e4cnxs/dl/fold0_predictions/val_swa_global.json $D/oof/vf0_val_swa_global.json
  for k in 1 2 3 4 5; do
    modal volume get --force rsna-coatnet-d96-cache experiments/pubkd_modal/pubkd_kfcnxs_vf$k/val_swa_global.json $D/oof/vf${k}_val_swa_global.json >/dev/null
    modal volume get --force rsna-coatnet-d96-cache experiments/pubkd_modal/pubkd_kfcnxs_vf$k/run_summary.json $D/oof/vf${k}_run_summary.json >/dev/null
  done
  ls -la $D/oof ;;
build)
  python3 -I scripts/fm_teacher/build_teacher_v2.py data/fbf_targets/pool data/coatnet_d96_fold0/assets/soft_labels_final_per_target_latD2.json \
      $T $W --oof $D/oof/vf{0,1,2,3,4,5}_val_swa_global.json --fm $D/fm/fm_all_*.json | tee $D/build_w${W}.txt ;;
launch)
  [ -f $T ] || { echo "no $T"; exit 1; }
  n=$($V r7 'pgrep -f "python -u /work/scripts/r50_gl4_runner.py" | wc -l' 2>/dev/null | tail -1)
  [ "$n" = 0 ] || { echo "r7 busy: $n runner processes"; exit 1; }
  SHA=$(sha256sum $T | cut -c1-64)
  put() { $V r7 "mkdir -p $(dirname $2); cat > $2" < "$1" 2>/dev/null
          [ "$($V r7 "sha256sum $2" 2>/dev/null | cut -c1-64)" = "$(sha256sum "$1" | cut -c1-64)" ] || { echo "MISMATCH $2"; exit 1; }; echo "ok $2"; }
  put $T /work/fbf/teacher_v2_mix73.json
  put scripts/r50_gl4_runner.py /work/scripts/r50_gl4_runner.py
  put scripts/colab_pubkd_lane.sh /work/scripts/colab_pubkd_lane.sh
  $V r7 "cd /work/env; for sm in '' smoke_; do grep -vE '^(GL_OUT|V4_TARGETS|V4_TARGETS_SHA)=' pubkd_\${sm}fbf_r7cnx.env > pubkd_\${sm}$L.env
    printf 'V4_TARGETS=/work/fbf/teacher_v2_mix73.json\nV4_TARGETS_SHA=$SHA\nGL_OUT=/work/out/pubkd_\${sm}$L\n' >> pubkd_\${sm}$L.env; done
    grep -c GL_SMOKE pubkd_smoke_$L.env; bash /work/scripts/colab_pubkd_lane.sh $L" 2>&1 | grep -v Pseudo
  echo "launched $L targets sha $SHA at $(TZ=Europe/Luxembourg date +%H:%M)" ;;
*) echo "fetch|build|launch"; exit 1 ;;
esac
