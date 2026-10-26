#!/bin/bash
# fast_global/submission.py -> eval variant (EVAL_N studies from EVAL_IDS: stage timing, fidelity vs REF, raw reader vs
# pydicom); BENCH=1 appends the startup profiler (bench_block.py)
cd "$(dirname "$0")"
sed -e 's/^MODE = "submit"/MODE = "eval"/' -e "s/^EVAL_N = 825/EVAL_N = ${EVAL_N:-120}/" \
    -e "s|^REF = None|REF = \"${REF:-tumis001/rsna-fast-eval/f1_val_swa_global.json}\"|" \
    ../fast_global/submission.py > submission.py
[ "${BENCH:-0}" = 1 ] && cat bench_block.py >> submission.py
true
