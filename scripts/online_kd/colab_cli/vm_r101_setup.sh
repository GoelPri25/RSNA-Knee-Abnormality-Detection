#!/bin/bash
# Runs ON the VM after vm_prepare.sh: ResNet-101 backbone for the E4 recipe (lane kd_e4r101) + its ImageNet weights.
set -euo pipefail
cp /content/okd/apply_r101_patch.py /work/scripts/
grep -q '"r101"' /work/scripts/r50_gl4_runner.py || python3 /work/scripts/apply_r101_patch.py /work/scripts/r50_gl4_runner.py
HF_HOME=/opt/hf HF_HUB_OFFLINE=0 /content/venv/bin/python -c "import timm; m = timm.create_model('resnet101.tv_in1k', pretrained=True); print('resnet101.tv_in1k', sum(p.numel() for p in m.parameters()) / 1e6, 'M params')"
for kind in "" "smoke_"; do
  grep -v '^GL_OUT=' /work/env/pubkd_${kind}kd_e4.env > /work/env/pubkd_${kind}kd_e4r101.env
  printf 'V4_BB=r101\nGL_OUT=/work/out/pubkd_%skd_e4r101\n' "$kind" >> /work/env/pubkd_${kind}kd_e4r101.env
done
cat /work/env/pubkd_kd_e4r101.env
