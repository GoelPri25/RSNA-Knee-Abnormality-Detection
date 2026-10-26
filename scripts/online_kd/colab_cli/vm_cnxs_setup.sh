#!/bin/bash
# Runs ON the VM after vm_prepare.sh: ConvNeXt-S 22k backbone for the E4 recipe (lane kd_e4cnxs) + its ImageNet weights.
set -euo pipefail
cp /content/okd/apply_cnxs_patch.py /work/scripts/
grep -q '"cnxs"' /work/scripts/r50_gl4_runner.py || python3 /work/scripts/apply_cnxs_patch.py /work/scripts/r50_gl4_runner.py
HF_HOME=/opt/hf HF_HUB_OFFLINE=0 /content/venv/bin/python -c "import timm; m = timm.create_model('convnext_small.fb_in22k_ft_in1k', pretrained=True, num_classes=0); print('convnext_small.fb_in22k_ft_in1k', round(sum(p.numel() for p in m.parameters()) / 1e6, 1), 'M params, features', m.num_features)"
for kind in "" "smoke_"; do
  grep -v '^GL_OUT=' /work/env/pubkd_${kind}kd_e4.env > /work/env/pubkd_${kind}kd_e4cnxs.env
  printf 'V4_BB=cnxs\nGL_OUT=/work/out/pubkd_%skd_e4cnxs\n' "$kind" >> /work/env/pubkd_${kind}kd_e4cnxs.env
done
cat /work/env/pubkd_kd_e4cnxs.env
