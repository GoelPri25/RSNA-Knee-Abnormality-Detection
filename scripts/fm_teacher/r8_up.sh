#!/bin/bash
# SOUP-R7 B (2026-10-22): bring a fresh Colab VM to r7's state and start lane fbf_r7soupv3a (soup init + label v3a).
#  1. upload Kaggle token, PUB-KD bundle + setup (venv, d96 cache from our Kaggle shard kernels, /work), fastMRI builder
#     (fmb.tar: builder, pylibs, signed URL list); keep-alive
#  2. setup and the fastMRI d96 cache build (fm_cache_run.sh: batch 1 + 2 streamed, ~80 min on r7) in parallel
#  3. r7's FB-F inputs (teacher v1 + v3a targets, fastMRI holdout, R7 soup encoder, runner, lane script, env files)
#  4. lane: smoke, then the run (colab_pubkd_lane.sh)
# Usage: scripts/fm_teacher/r8_up.sh <session>       log: data/_session_tmp/colab/<session>vm/up.log
set -uo pipefail
S=$1; R=.; cd $R
ST=data/_session_tmp/colab/r8vm; D=data/_session_tmp/colab/${S}vm; LOG=$D/up.log; L=fbf_r7soupv3a; mkdir -p $D
log() { echo "$(TZ=Europe/Luxembourg date +%H:%M:%S) $*" | tee -a $LOG; }
vm() { scripts/fm_teacher/vmssh.sh $S "$@" 2>/dev/null; }
put() { vm "mkdir -p $(dirname $2); umask 077; cat > $2" < "$1"
        [ "$(vm "sha256sum $2" | cut -c1-64)" = "$(sha256sum "$1" | cut -c1-64)" ] || { log "MISMATCH $2"; exit 1; }; }
wait_for() {  # wait_for <label> <remote test command> <max minutes>
  for i in $(seq $(($3 * 2))); do vm "$2" && { log "$1 ok"; return 0; }; sleep 30; done; log "$1 TIMEOUT"; exit 1; }
ssh-keygen -f "$HOME/.config/colab-cli/known_hosts" -R "colab-$S" > /dev/null 2>&1
log "hw: $(vm "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | awk '/Mem/{print \$2\" GB RAM\"}'; df -h / | tail -1" | tr '\n' ' ')"
(colab exec -s $S --timeout 86400 -f scripts/fm_teacher/ka_generic.py >> $D/keepalive.log 2>&1 &)
put ~/.config/rsna/kaggle_colab.env /content/kaggle.env
put data/_session_tmp/colab/bundle.tgz /content/bundle.tgz
put scripts/colab_pubkd_setup.py /content/colab_pubkd_setup.py
put data/_session_tmp/colab/r7vm/fmb.tar /content/fmb.tar
log "uploads ok; setup start"
vm "cd /content && (nohup python3 colab_pubkd_setup.py > setup.log 2>&1 &)"
wait_for venv "/content/venv/bin/python -c 'import torch, timm' 2>/dev/null" 30
# fastMRI cache: shards 0-2 in /dev/shm as on r7 when RAM allows, else all on disk
vm "cd /content && tar xf fmb.tar && rm -f fmb.tar; ram=\$(free -g | awk '/Mem/{print \$2}'); [ \$ram -ge 150 ] || sed -i 's#^for k in 0 1 2; do mkdir -p /dev/shm.*##' fmb/fm_cache_run.sh
  (nohup bash fmb/fm_cache_run.sh > fmb/run.log 2>&1 &); echo ram \$ram"
log "fastMRI cache build started"
wait_for setup "grep -q SETUP_OK /content/setup.log" 60
put data/fbf_targets/pool/pubkd_targets_fullorder_rapvalues_mix73.json /work/fbf/fbf_targets_mix73.json
put data/label_v3a/v3a_targets_mix73.json /work/fbf/v3a_targets_mix73.json
put $ST/fm_holdout.json /work/fm/fm_holdout.json
put $ST/r7soup4_timm.pt /work/pubckpt/r7soup4_timm.pt
put scripts/r50_gl4_runner.py /work/scripts/r50_gl4_runner.py
put scripts/colab_pubkd_lane.sh /work/scripts/colab_pubkd_lane.sh
SHA=$(sha256sum data/label_v3a/v3a_targets_mix73.json | cut -c1-64)
for sm in "" smoke_; do
  grep -vE '^(GL_OUT|V4_TARGETS|V4_TARGETS_SHA)=' $ST/pubkd_${sm}fbf_r7soup.env > $D/pubkd_${sm}$L.env
  printf 'V4_TARGETS=/work/fbf/v3a_targets_mix73.json\nV4_TARGETS_SHA=%s\nGL_OUT=/work/out/pubkd_%s%s\n' $SHA "$sm" $L >> $D/pubkd_${sm}$L.env
  put $D/pubkd_${sm}$L.env /work/env/pubkd_${sm}$L.env
done
log "work inputs ok; waiting for the fastMRI cache"
wait_for fmcache "grep -q 'ALL END' /content/fmb/STATUS.txt" 150
vm "grep -c OK /content/fmb/STATUS.txt" | grep -q 2 || { log "fastMRI cache FAILED: $(vm 'cat /content/fmb/STATUS.txt')"; exit 1; }
vm "ln -sfn /content/fmcache /data/fm; ls /data/fm | wc -l"
log "lane start: $(vm "bash /work/scripts/colab_pubkd_lane.sh $L")"
