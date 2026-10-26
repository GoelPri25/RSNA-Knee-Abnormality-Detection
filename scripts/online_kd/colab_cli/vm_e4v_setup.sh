#!/bin/bash
# Runs ON the VM after vm_prepare.sh: E4 version 1 (pairwise ranking loss) and version 2 (rank-mapped targets).
set -euo pipefail
cp /content/okd/online_kd.py /content/okd/apply_rank_patch.py /work/scripts/
grep -q V4_RANK /work/scripts/r50_gl4_runner.py || python3 /work/scripts/apply_rank_patch.py /work/scripts/r50_gl4_runner.py
cp /content/okd/pubkd_targets_fullorder_lv2values_mix73.json /work/pubkd/
SHA=$(sha256sum /work/pubkd/pubkd_targets_fullorder_lv2values_mix73.json | cut -d' ' -f1)
[ "$SHA" = d4610941a091ab4646f20b4e99bbce956a762311685f83e2653d076dc46ffe62 ] || { echo "rank-mapped target sha differs: $SHA"; exit 1; }
mk() {  # mk <lane> <extra lines> <target file> <target sha>
  for kind in "" "smoke_"; do
    src=/work/env/pubkd_${kind}kd_e4.env; dst=/work/env/pubkd_${kind}$1.env
    grep -vE '^(GL_OUT|V4_TARGETS|V4_TARGETS_SHA)=' $src > $dst
    printf 'V4_TARGETS=%s\nV4_TARGETS_SHA=%s\n%sGL_OUT=/work/out/pubkd_%s%s\n' "$3" "$4" "$2" "$kind" "$1" >> $dst
  done
}
E4T=/work/pubkd/pubkd_targets_fullorder_rapvalues_mix73.json; E4S=f07f202b846b99272b6fe34bcb1d30d4d71e18a1870bcc7d3ce61559737ccd34
mk kd_e4rk $'V4_RANK=0.3\n' $E4T $E4S
mk kd_e4qm '' /work/pubkd/pubkd_targets_fullorder_lv2values_mix73.json $SHA
echo "--- kd_e4rk"; cat /work/env/pubkd_kd_e4rk.env; echo "--- kd_e4qm"; cat /work/env/pubkd_kd_e4qm.env
