#!/bin/sh
# Run inside a Kaggle notebook terminal after cloning/installing this repository.
set -eu
: "${1:?Usage: sh scripts/kaggle_audit.sh /kaggle/input/competition-directory}"
python -m rsna_knee.data.audit "$1" --config configs/audit.json --output /kaggle/working/knee-audit
