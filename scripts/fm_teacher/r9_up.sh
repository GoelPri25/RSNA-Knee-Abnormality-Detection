#!/bin/bash
# RUN-C-H on a third Colab VM (2026-10-22): allocate (H100, else A100 high-mem, else A100), bring it to r8's state (setup,
# fastMRI d96 cache built on the VM), then lane runch = RUN-C without the feature KD, from R7-D-C (continue mode).
# Usage: scripts/fm_teacher/r9_up.sh r9      log: data/_session_tmp/colab/r9vm/up.log
set -uo pipefail
S=$1; R=.; cd $R
D=data/_session_tmp/colab/${S}vm; LOG=$D/up.log; L=runch; mkdir -p $D
log() { echo "$(TZ=Europe/Luxembourg date +%H:%M:%S) $*" | tee -a $LOG; }
vm() { scripts/fm_teacher/vmssh.sh $S "$@" 2>/dev/null; }
put() { vm "mkdir -p $(dirname $2); umask 077; cat > $2" < "$1"
        [ "$(vm "sha256sum $2" | cut -c1-64)" = "$(sha256sum "$1" | cut -c1-64)" ] || { log "MISMATCH $2"; exit 1; }; }
wait_for() {  # wait_for <label> <remote test command> <max minutes>
  for i in $(seq $(($3 * 2))); do vm "$2" && { log "$1 ok"; return 0; }; sleep 30; done; log "$1 TIMEOUT"; exit 1; }
have() { colab sessions 2>&1 | grep -q "^\[$S\]"; }
try() { log "request $1"; COLAB_CLI_REQUEST_TIMEOUT=600 timeout 700 colab new -s "$S" "${@:2}" 2>&1 | grep -v -iE "eyJ|token|nbh=|https" | tail -1 >> $LOG; have; }
have || try H100 --gpu H100 || try A100-HM --gpu A100 --high-mem || { sleep 60; try A100-HM --gpu A100 --high-mem; } \
  || { sleep 60; try A100-HM --gpu A100 --high-mem; } || try A100 --gpu A100 || { log "no GPU allocated"; exit 2; }
ssh-keygen -f "$HOME/.config/colab-cli/known_hosts" -R "colab-$S" > /dev/null 2>&1
log "hw: $(vm "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | awk '/Mem/{print \$2\" GB RAM\"}'" | tr '\n' ' ')"
(colab exec -s $S --timeout 86400 -f scripts/fm_teacher/ka_generic.py >> $D/keepalive.log 2>&1 &)
put ~/.config/rsna/kaggle_colab.env /content/kaggle.env
put data/_session_tmp/colab/bundle.tgz /content/bundle.tgz
put scripts/colab_pubkd_setup.py /content/colab_pubkd_setup.py
put data/_session_tmp/colab/r7vm/fmb.tar /content/fmb.tar
log "uploads ok; setup start"
vm "cd /content && (nohup python3 colab_pubkd_setup.py > setup.log 2>&1 &)"
wait_for venv "/content/venv/bin/python -c 'import torch, timm' 2>/dev/null" 30
vm "cd /content && tar xf fmb.tar && rm -f fmb.tar; ram=\$(free -g | awk '/Mem/{print \$2}'); [ \$ram -ge 150 ] || sed -i 's#^for k in 0 1 2; do mkdir -p /dev/shm.*##' fmb/fm_cache_run.sh
  (nohup bash fmb/fm_cache_run.sh > fmb/run.log 2>&1 &); echo ram \$ram"
log "fastMRI cache build started"
wait_for setup "grep -q SETUP_OK /content/setup.log" 60
put data/label_v3a/v3a_targets_mix73.json /work/fbf/v3a_targets_mix73.json
put data/label_v3a/hydra_members.json /work/fbf/hydra_members.json
put data/_session_tmp/colab/r8vm/fm_holdout.json /work/fm/fm_holdout.json
put data/_session_tmp/colab/r8vm/r7soup4_timm.pt /work/pubckpt/r7soup4_timm.pt
put data/_session_tmp/colab/r8vm/r7dc_swa6_8.pt /work/init/r7dc_swa6_8.pt
put scripts/r50_gl4_runner.py /work/scripts/r50_gl4_runner.py
put scripts/colab_pubkd_lane.sh /work/scripts/colab_pubkd_lane.sh
for sm in "" smoke_; do put $D/pubkd_${sm}$L.env /work/env/pubkd_${sm}$L.env; done
log "work inputs ok; waiting for the fastMRI cache"
wait_for fmcache "grep -q 'ALL END' /content/fmb/STATUS.txt" 150
vm "grep -c OK /content/fmb/STATUS.txt" | grep -q 2 || { log "fastMRI cache FAILED: $(vm 'cat /content/fmb/STATUS.txt')"; exit 1; }
vm "ln -sfn /content/fmcache /data/fm"
log "fm manifests: $(vm 'for k in 0 1 2 3 4 5 6 7; do sha256sum /data/fm/shard_$k/manifest.jsonl | cut -c1-16; done' | tr '\n' ' ')"
log "lane start: $(vm "bash /work/scripts/colab_pubkd_lane.sh $L")"
