#!/bin/bash
# FB-F: get a Colab GPU VM and start a teacher lane on it, unattended.
#  1. allocation: H100 (2 tries), else A100 High-RAM (3 tries), else A100 standard (user 2026-10-21: A100 / H100 only)
#  2. upload: Kaggle token (deleted by the setup), fastMRI URL list (mode 600), setup / driver / teacher; sha256-checked
#  3. setup (scripts/fm_teacher/fm_vm_setup.py), then the driver on <buckets>, then the host guard (keep-alive + pulls)
# Usage: scripts/fm_teacher/fm_vm_up.sh <session> <buckets e.g. 12-23> [chunk=400]
set -uo pipefail
S=$1; B=$2; CH=${3:-400}; K=$HOME/.ssh/id_ed25519_colab
R=$(cd "$(dirname "$0")/../.." && pwd); OUT=$R/data/_session_tmp/colab/fmteach/$S; mkdir -p "$OUT"; LOG=$OUT/up.log
log() { echo "$(TZ=Europe/Luxembourg date +%F' '%H:%M:%S) $*" | tee -a "$LOG"; }
vm() { "$R/scripts/fm_teacher/vmssh.sh" "$S" "$@" 2>/dev/null; }   # retries Colab's one-ssh-per-runtime 429
put() {  # put <local> <remote>
  vm "mkdir -p /content/fm; umask 077; cat > $2" < "$1"
  [ "$(vm "sha256sum $2 | cut -c1-64")" = "$(sha256sum "$1" | cut -c1-64)" ] || { log "upload $2 sha mismatch"; return 1; }
}
have() { colab sessions 2>&1 | grep -q "^\[$S\]"; }
try() {  # try <label> <colab new args...>
  log "request $1"
  COLAB_CLI_REQUEST_TIMEOUT=600 timeout 700 colab new -s "$S" "${@:2}" 2>&1 | grep -v -iE "eyJ|token|nbh=" | tail -1 >> "$LOG"
  have
}
have || try H100 --gpu H100 || { sleep 60; try H100 --gpu H100; } || try A100-HM --gpu A100 --high-mem \
  || { sleep 60; try A100-HM --gpu A100 --high-mem; } || { sleep 60; try A100-HM --gpu A100 --high-mem; } \
  || try A100 --gpu A100 || { log "no GPU allocated"; exit 2; }
log "allocated: $(colab sessions 2>&1 | grep "^\[$S\]" | cut -d'|' -f2-)"
ssh-keygen -f "$HOME/.config/colab-cli/known_hosts" -R "colab-$S" > /dev/null 2>&1
put "$HOME/.config/rsna/kaggle_colab.env" /content/fm/kaggle.env || exit 1
put "$R/data/meluxina_mirror/project_sumit/fastmri_raw/urls.txt" /content/fm/urls.txt || exit 1
for f in fm_vm_setup.py fm_vm_driver.py teacher_fm.py; do put "$R/scripts/fm_teacher/$f" /content/fm/$f || exit 1; done
(colab exec -s "$S" --timeout 86400 -f "$R/scripts/fm_teacher/ka_generic.py" >> "$OUT/keepalive.log" 2>&1 &)
log "setup start"
vm "cd /content/fm && python3 fm_vm_setup.py > setup.log 2>&1; tail -n 1 setup.log" | tee -a "$LOG" | grep -q SETUP_OK \
  || { log "setup failed: $(vm 'tail -n 3 /content/fm/setup.log')"; exit 1; }
vm "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | awk '/Mem/{print \$2\" GB RAM\"}'" | tr '\n' ' ' | sed 's/^/hw: /' >> "$LOG"; echo >> "$LOG"
vm "cd /content/fm && (nohup /content/fmv/bin/python -u fm_vm_driver.py --buckets $B --chunk $CH --loader-workers 6 > driver.log 2>&1 &)"
log "driver started on buckets $B; guard follows"
exec "$R/scripts/fm_teacher/fm_guard.sh" "$S" "$OUT"
