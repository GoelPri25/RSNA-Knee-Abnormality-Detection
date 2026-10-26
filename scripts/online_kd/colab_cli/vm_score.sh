#!/bin/bash
# Runs ON the VM after the lane is DONE: fold-0 vs E4 / cv1, and gold-58 (global head).
L=kd_e4on; O=/work/out/pubkd_$L; K=/content/kit/fold0_predictions; G=/work/out/gold_$L
/content/venv/bin/python /work/scripts/score_fold0.py /work/assets $O/val_swa_fused.json $K/e4/val_swa_fused.json $K/cv1/val_swa_fused.json
echo '--- global head only'
/content/venv/bin/python /work/scripts/score_fold0.py /work/assets $O/val_swa_global.json $K/e4/val_swa_global.json
cd /work/scripts
env $(grep -vE '^\s*(#|$)' /work/env/pubkd_$L.env | xargs) /content/venv/bin/python -u gold_gl4_predictions.py $G \
    online=$O/weights_swa21_25.pt:$O/val_swa_global.json | tail -2
mkdir -p /content/goldroot/reports_raw && cp /content/comp/train.csv /content/goldroot/reports_raw/train.csv
/content/venv/bin/python /content/kit/scoring/rsna_root/repository/data/coatnet_d96_fold0/pubkd/gold/score_gold.py /content/goldroot \
    online=$G/gold_online.json e4=$K/gold/gold_e4.json cv1=$K/gold/gold_cv1.json
sha256sum $O/weights_swa21_25.pt
