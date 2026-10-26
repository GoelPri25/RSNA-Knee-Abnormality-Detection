#!/bin/bash
# C-A on r7 (2026-10-22): RUN-C + attention distillation (V4_AKD_W 0.2), from R7-D-C, started when lane fbf_r7soup (SOUP-R7 A) is DONE.
# log: data/_session_tmp/colab/r7vm/ca.log
set -uo pipefail
S=r7; R=.; cd $R
D=data/_session_tmp/colab/r7vm; LOG=$D/ca.log; L=runca
log() { echo "$(TZ=Europe/Luxembourg date +%H:%M:%S) $*" | tee -a $LOG; }
vm() { scripts/fm_teacher/vmssh.sh $S "$@" 2>/dev/null; }
put() { vm "mkdir -p $(dirname $2); umask 077; cat > $2" < "$1"
        [ "$(vm "sha256sum $2" | cut -c1-64)" = "$(sha256sum "$1" | cut -c1-64)" ] || { log "MISMATCH $2"; exit 1; }; }
put scripts/r50_gl4_runner.py /work/scripts/r50_gl4_runner.py
put data/label_v3a/v3a_targets_mix73.json /work/fbf/v3a_targets_mix73.json
put data/label_v3a/hydra_members.json /work/fbf/hydra_members.json
vm "mkdir -p /work/init && cp -n /work/out/pubkd_fbf_r7cnxc/weights_swa6_8.pt /work/init/r7dc_swa6_8.pt; sha256sum /work/init/r7dc_swa6_8.pt" | grep -q ^c110911cbf27ee7f || { log "R7-D-C init MISMATCH"; exit 1; }
for sm in "" smoke_; do put $D/pubkd_${sm}$L.env /work/env/pubkd_${sm}$L.env; done
log "C-A inputs ok; waiting for lane fbf_r7soup"
for i in $(seq 400); do st=$(vm "cat /work/out/fbf_r7soup.status"); [ "$st" != RUNNING ] && break; sleep 30; done
log "fbf_r7soup status: $st"
[ "$st" = DONE ] || exit 1
log "lane start: $(vm "bash /work/scripts/colab_pubkd_lane.sh $L")"
