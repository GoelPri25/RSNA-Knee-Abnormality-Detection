#!/bin/bash
# Online distillation (E4 + live Raptor teacher) on a Colab Pro A100, driven from this machine with Google's Colab CLI.
#   one-time:  ~/.venvs/colab-cli/bin/colab --auth oauth2 sessions        (browser sign-in, paste the code)
#   steps:     bash run_online_kd.sh new | upload | token | prepare | check | launch | status | score | fetch | stop
# token needs KAGGLE_API_TOKEN in this shell's environment; it is written only on the VM (/content/kaggle.env), and the
# teammate's setup deletes it after the cache download.
set -euo pipefail
C="$HOME/.venvs/colab-cli/bin/colab --auth oauth2"; S=${COLAB_SESSION:-rsna_okd}; L=kd_e4on
HERE="$(cd "$(dirname "$0")" && pwd)"; OKD="$HERE/.."
vm() { echo "import subprocess; r = subprocess.run($1, shell=True, capture_output=True, text=True); print(r.stdout[-6000:]); print(r.stderr[-3000:])" \
         | $C exec -s $S --timeout "${2:-120}"; }
case "${1:-}" in
  new)     $C new -s $S --gpu A100 --high-mem && $C status -s $S ;;
  upload)  vm "'mkdir -p /content/okd'"
           for f in "$OKD/online_kd.py" "$OKD/apply_online_patch.py" "$OKD/score_fold0.py" \
                    "$HERE/vm_prepare.sh" "$HERE/vm_check.py" "$HERE/vm_check.sh" "$HERE/vm_status.sh" "$HERE/vm_score.sh"; do
             $C upload -s $S "$f" "/content/okd/$(basename "$f")"; done
           vm "'ls -la /content/okd'" ;;
  token)   : "${KAGGLE_API_TOKEN:?export KAGGLE_API_TOKEN first}"
           echo "import os; open('/content/kaggle.env', 'w').write('KAGGLE_API_TOKEN=' + os.environ['KAGGLE_API_TOKEN'] + '\n'); print('token written on the VM')" \
             | $C exec -s $S --env "KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN" ;;
  prepare) vm "'nohup bash /content/okd/vm_prepare.sh > /content/okd/prepare.log 2>&1 &'"; echo "prepare started; poll with: $0 status" ;;
  check)   vm "'bash /content/okd/vm_check.sh'" 900 ;;
  launch)  vm "'bash /work/scripts/colab_pubkd_lane.sh $L'" ;;
  status)  vm "'bash /content/okd/vm_status.sh'" ;;
  score)   vm "'bash /content/okd/vm_score.sh'" 3600 ;;
  fetch)   mkdir -p "$HERE/out_$L"
           for f in weights_swa21_25.pt val_swa_fused.json val_swa_global.json history.json train.log; do
             $C download -s $S "/work/out/pubkd_$L/$f" "$HERE/out_$L/$f"; done; ls -la "$HERE/out_$L" ;;
  stop)    $C stop -s $S ;;
  *)       echo "usage: $0 new|upload|token|prepare|check|launch|status|score|fetch|stop"; exit 1 ;;
esac
