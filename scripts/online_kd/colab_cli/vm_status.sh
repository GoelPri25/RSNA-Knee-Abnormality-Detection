#!/bin/bash
# Runs ON the VM: prepare / lane status and the per-epoch summary.
L=kd_e4on; O=/work/out
echo "prepare: $(cat /content/okd/prepare.status 2>/dev/null)"; tail -2 /content/okd/prepare.log 2>/dev/null
echo "lane: $(cat $O/$L.status 2>/dev/null)"; tail -2 $O/$L.driver.log 2>/dev/null
grep -E "ONLINE_KD|SMOKE_DONE|Error|Traceback" $O/pubkd_smoke_$L/train.log 2>/dev/null | tail -4
grep -E "ONLINE_KD|RESUMED|step .*/|Error|Traceback" $O/pubkd_$L/train.log 2>/dev/null | tail -3
grep "^EPOCH" $O/pubkd_$L/train.log 2>/dev/null | python3 -c "
import sys, json
for l in sys.stdin:
    _, n, d = l.split(' ', 2); d = json.loads(d)
    print('epoch', n, 'train_s', d.get('train_s'), 'lv2 fused/global', round(d.get('lv2_fused', 0), 4), round(d.get('lv2_global', 0), 4),
          'mean|d|', d.get('online_mean_abs_d'), 'mem', d.get('mem_gb'))"
df -h /content | tail -1; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
