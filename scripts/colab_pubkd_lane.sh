#!/bin/bash
# PUB-KD lane on a Colab VM (after scripts/colab_pubkd_setup.py printed SETUP_OK): pubkd.sbatch's run() for one lane,
# started detached (nohup) so it does not depend on the CLI connection. Smoke first (GL_SMOKE=1), the lane only if
# SMOKE_DONE; env = the lane's env file only (as apptainer --cleanenv). After exit 0 with SWA weights, per-epoch weights
# are removed (recovery.pt kept since 2026-10-22, for continuations). Status: /work/out/<lane>.status (RUNNING / DONE / FAILED <stage>).
# Usage (on the VM): bash /work/scripts/colab_pubkd_lane.sh <lane>      e.g. kd_e4s27
set -uo pipefail
L=$1; PY=/content/venv/bin/python; O=/work/out
[ -f /work/env/pubkd_$L.env ] && [ -f /work/env/pubkd_smoke_$L.env ] || { echo "missing env for $L"; exit 1; }
[ -e $O/pubkd_$L/run_summary.json ] && { echo "pubkd_$L already finished"; exit 1; }
run() {  # run <env name>
  mkdir -p $O/$1
  env -i PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/bin:/bin HOME=/root LD_LIBRARY_PATH=/usr/lib64-nvidia \
      $(grep -vE '^\s*(#|$)' /work/env/$1.env | xargs) $PY -u /work/scripts/r50_gl4_runner.py >> $O/$1/train.log 2>&1
  local rc=$?; echo "$(date -u +%FT%TZ) $1 exit $rc" >> $O/$L.driver.log
  if [ $rc -eq 0 ] && ls $O/$1/weights_swa*.pt >/dev/null 2>&1; then rm -f $O/$1/weights_epoch*.pt; fi   # recovery.pt kept (2026-10-22: Adam state for continuations)
  return $rc
}
nohup bash -c "
  $(declare -f run); O=$O; PY=$PY; L=$L
  echo RUNNING > $O/$L.status
  if [ ! -e $O/pubkd_$L/recovery.pt ]; then
    rm -rf $O/pubkd_smoke_$L
    run pubkd_smoke_$L && grep -q SMOKE_DONE $O/pubkd_smoke_$L/train.log || { echo 'FAILED smoke' > $O/$L.status; exit 1; }
  fi
  run pubkd_$L && echo DONE > $O/$L.status || echo 'FAILED lane' > $O/$L.status
" > $O/$L.nohup.log 2>&1 &
echo "started $L pid $!"
