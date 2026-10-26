#!/bin/bash
# RUN-C-H on r7 (2026-10-22): Hydra-only arm of RUN-C (no feature KD), started when lane fbf_r7soup (SOUP-R7 A) is DONE.
# log: data/_session_tmp/colab/r7vm/ch.log
set -uo pipefail
S=r7; R=.; cd $R
D=data/_session_tmp/colab/r7vm; LOG=$D/ch.log; L=runch
log() { echo "$(TZ=Europe/Luxembourg date +%H:%M:%S) $*" | tee -a $LOG; }
vm() { scripts/fm_teacher/vmssh.sh $S "$@" 2>/dev/null; }
put() { vm "mkdir -p $(dirname $2); umask 077; cat > $2" < "$1"
        [ "$(vm "sha256sum $2" | cut -c1-64)" = "$(sha256sum "$1" | cut -c1-64)" ] || { log "MISMATCH $2"; exit 1; }; }
put scripts/r50_gl4_runner.py /work/scripts/r50_gl4_runner.py
put data/label_v3a/v3a_targets_mix73.json /work/fbf/v3a_targets_mix73.json
put data/label_v3a/hydra_members.json /work/fbf/hydra_members.json
for sm in "" smoke_; do put $D/pubkd_${sm}$L.env /work/env/pubkd_${sm}$L.env; done
log "C-H inputs ok; waiting for lane fbf_r7soup"
for i in $(seq 400); do st=$(vm "cat /work/out/fbf_r7soup.status"); [ "$st" != RUNNING ] && break; sleep 30; done
log "fbf_r7soup status: $st"
[ "$st" = DONE ] || exit 1
log "lane start: $(vm "bash /work/scripts/colab_pubkd_lane.sh $L")"
