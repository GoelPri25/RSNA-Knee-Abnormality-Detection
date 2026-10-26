#!/bin/bash
# KF-1..KF-5 (ledger KF-CNXS) on every fastMRI exam (teacher v2's fastMRI team = mean of KF-0..KF-5; no leakage: fastMRI
# carries no competition labels). Per fold: SWA weights from the Modal volume -> local -> Colab VM r7, then
# fmeval.sh (smoke 16 exams, then all 9,693) with V4_BB=cnxs, V4_EVAL_ONLY, V4_EVAL_SET=all; at most $PAR at once.
# Results: r7 /work/out/fmeval_kf<k>/fm_all_global.json -> data/teacher_v2/fm/fm_all_kf<k>.json (step pull).
# Usage: kf_fastmri.sh start "1 2 3 4 5" [PAR] | pull
set -euo pipefail
R=.; cd $R
V=scripts/fm_teacher/vmssh.sh; W=data/teacher_v2/weights; mkdir -p $W data/teacher_v2/fm
put() { $V r7 "mkdir -p $(dirname $2); cat > $2" < "$1" 2>/dev/null
        [ "$($V r7 "sha256sum $2" 2>/dev/null | cut -c1-64)" = "$(sha256sum "$1" | cut -c1-64)" ] || { echo "MISMATCH $2"; exit 1; }; echo "ok $2"; }
case $1 in
start)
  FOLDS=$2; PAR=${3:-3}
  source ~/.config/rsna/modal-personal.env
  for k in $FOLDS; do
    [ -f $W/kf$k.pt ] || modal volume get --force rsna-coatnet-d96-cache experiments/pubkd_modal/pubkd_kfcnxs_vf$k/weights_swa21_25.pt $W/kf$k.pt >/dev/null
    put $W/kf$k.pt /work/init/kf$k.pt
    $V r7 "cd /work/env; for sm in '' smoke_; do grep -vE '^(GL_OUT|V4_EVAL_ONLY)=' fmeval_\${sm}cnxs_f0.env > fmeval_\${sm}kf$k.env
      printf 'V4_EVAL_ONLY=/work/init/kf$k.pt\nGL_OUT=/work/out/fmeval_\${sm}kf$k\n' >> fmeval_\${sm}kf$k.env; done" 2>/dev/null
  done
  # queue on the VM: PAR parallel slots, each slot runs its folds in sequence
  $V r7 "cd /work/out; i=0; for k in $FOLDS; do echo \$k >> /tmp/kfq_\$((i % $PAR)); i=\$((i+1)); done
    for q in /tmp/kfq_*; do nohup bash -c \"for k in \\\$(cat \$q); do bash /work/scripts/fmeval.sh kf\\\$k; done; rm -f \$q\" > /work/out/kfq_\$(basename \$q).nohup 2>&1 & done
    sleep 2; ls /tmp/kfq_* 2>/dev/null | wc -l" 2>&1 | grep -v Pseudo
  echo "started folds [$FOLDS] PAR $PAR at $(TZ=Europe/Luxembourg date +%H:%M)" ;;
pull)
  for k in 1 2 3 4 5; do
    s=$($V r7 "cat /work/out/fmeval_kf$k.status 2>/dev/null" 2>/dev/null | tail -1); echo "kf$k $s"
    [ "$s" = DONE ] && $V r7 "cat /work/out/fmeval_kf$k/fm_all_global.json" 2>/dev/null > data/teacher_v2/fm/fm_all_kf$k.json
  done
  cp -n data/teacher_v2/fm/fm_all_cnxs_f0.json data/teacher_v2/fm/fm_all_kf0.json 2>/dev/null || true
  ls -la data/teacher_v2/fm ;;
esac
