#!/bin/bash
# Runs ON the VM after vm_prepare.sh: logit-matching patch + lane env kd_e4lg (= E4 + V4_LOGIT=0.5, offline teacher logits).
set -euo pipefail
cp /content/okd/online_kd.py /content/okd/apply_logit_patch.py /work/scripts/
grep -q V4_LOGIT /work/scripts/r50_gl4_runner.py || python3 /work/scripts/apply_logit_patch.py /work/scripts/r50_gl4_runner.py
L=kd_e4lg
LG="V4_LOGIT=0.5
V4_ONLINE_PURE=/work/pubkd/pubkd_targets_fullorder_rapvalues.json"
grep -v '^GL_OUT=' /work/env/pubkd_kd_e4.env > /work/env/pubkd_$L.env
printf '%s\nGL_OUT=/work/out/pubkd_%s\n' "$LG" "$L" >> /work/env/pubkd_$L.env
grep -v '^GL_OUT=' /work/env/pubkd_smoke_kd_e4.env > /work/env/pubkd_smoke_$L.env
printf '%s\nGL_OUT=/work/out/pubkd_smoke_%s\n' "$LG" "$L" >> /work/env/pubkd_smoke_$L.env
cat /work/env/pubkd_$L.env
