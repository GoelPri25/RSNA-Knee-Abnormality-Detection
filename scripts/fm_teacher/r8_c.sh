#!/bin/bash
# RUN-C on r8 (2026-10-22): after r8_up.sh's setup (stopped before its lane start), put run C's runner and env files,
# wait for r8's fastMRI d96 cache, then start lane runc (smoke, then the run).   log: data/_session_tmp/colab/r8vm/up.log
set -uo pipefail
S=r8; R=.; cd $R
D=data/_session_tmp/colab/r8vm; LOG=$D/up.log; L=runc
log() { echo "$(TZ=Europe/Luxembourg date +%H:%M:%S) $*" | tee -a $LOG; }
vm() { scripts/fm_teacher/vmssh.sh $S "$@" 2>/dev/null; }
put() { vm "mkdir -p $(dirname $2); umask 077; cat > $2" < "$1"
        [ "$(vm "sha256sum $2" | cut -c1-64)" = "$(sha256sum "$1" | cut -c1-64)" ] || { log "MISMATCH $2"; exit 1; }; }
put scripts/r50_gl4_runner.py /work/scripts/r50_gl4_runner.py
put data/label_v3a/hydra_members.json /work/fbf/hydra_members.json
for sm in "" smoke_; do put $D/pubkd_${sm}$L.env /work/env/pubkd_${sm}$L.env; done
log "run C inputs ok; waiting for the fastMRI cache"
for i in $(seq 300); do vm "grep -q 'ALL END' /content/fmb/STATUS.txt" && break; sleep 30; done
vm "grep -c OK /content/fmb/STATUS.txt" | grep -q 2 || { log "fastMRI cache FAILED or timeout: $(vm 'cat /content/fmb/STATUS.txt')"; exit 1; }
vm "ln -sfn /content/fmcache /data/fm"
log "fm manifests: $(vm 'for k in 0 1 2 3 4 5 6 7; do sha256sum /data/fm/shard_$k/manifest.jsonl | cut -c1-16; done' | tr '\n' ' ')"
log "lane start: $(vm "bash /work/scripts/colab_pubkd_lane.sh $L")"
