#!/bin/bash
# Host-side guard for a detached PUB-KD lane on a Colab CLI session (after scripts/colab_pubkd_lane.sh started it).
#  1. Keep-alive: the Colab CLI keeps a session only while its Jupyter kernel is active. A lane started with nohup leaves
#     the kernel idle, and the VM was reclaimed within ~30 min of the last kernel execution (w48 lost at epoch 7, 10-21,
#     after E4-S27's keep-alive exec ended). So a kernel cell polls the lane status until DONE / FAILED, and this guard
#     restarts that exec whenever it ends early.
#  2. Per-epoch sync: after every new epoch in history.json, recovery.pt (and weights_epochNN.pt for the SWA epochs) is
#     snapshotted on the VM, downloaded and sha256-checked here, so a lost VM can resume on a new one (upload the files to
#     /work/out/pubkd_<lane>/ after setup and rerun colab_pubkd_lane.sh: it skips the smoke when recovery.pt exists).
#  3. On DONE: downloads the lane outputs with sha256 checks, then stops the session (an idle VM burns units).
#     COLAB_KEEP=1: the session is kept for the next job instead, with a generic keep-alive exec (ka_generic.py) on it.
#     KA_PID=<pid>: adopt a keep-alive exec already running (a restarted guard), instead of starting a second one.
# Usage: scripts/colab_lane_guard.sh <lane> <local dir> [swa_first=21]     e.g. w48 data/_session_tmp/colab/w48run/sync
set -uo pipefail
L=$1; OUT=$2; SWA0=${3:-21}; S=${COLAB_SESSION:-rsna}; K=$HOME/.ssh/id_ed25519_colab
R=/work/out/pubkd_$L; ST=/work/out/$L.status
mkdir -p "$OUT"; LOG=$OUT/guard.log; KA=$OUT/keepalive.py
log() { echo "$(TZ=Europe/Luxembourg date +%F' '%H:%M:%S) $*" >> "$LOG"; }
vm() {  # vm <command>: run on the VM over SSH, no tty (binary-safe output)
  timeout 120 ssh -i "$K" -o IdentitiesOnly=yes -o BatchMode=yes -o ProxyCommand="colab ssh --proxy-mode -s $S -i $K" \
    -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.config/colab-cli/known_hosts" \
    -o ServerAliveInterval=30 "root@colab-$S" "$@" 2>/dev/null
}
pull() {  # pull <remote file> <local name>: snapshot on the VM, download, compare sha256 (3 tries)
  local snap=/work/sync/$2 want got
  want=$(vm "mkdir -p /work/sync && cp $1 $snap && sha256sum $snap | cut -c1-64") || return 1
  for t in 1 2 3; do
    timeout 1800 colab download -s "$S" "$snap" "$OUT/$2.part" > /dev/null 2>&1
    got=$(sha256sum "$OUT/$2.part" 2>/dev/null | cut -c1-64)
    if [ -n "$want" ] && [ "$got" = "$want" ]; then mv "$OUT/$2.part" "$OUT/$2"; echo "$want  $2" >> "$OUT/sha256.txt"; return 0; fi
    log "pull $2 try $t: sha mismatch or download failed"
  done
  return 1
}
cat > "$KA" <<EOF
import json, pathlib, time
st, h = pathlib.Path("$ST"), pathlib.Path("$R/history.json")
while True:
    s = st.read_text().strip() if st.exists() else "?"
    if s.startswith(("DONE", "FAILED")):
        print("STATUS", s, flush=True); break
    n = len(json.loads(h.read_text())) if h.exists() else 0
    print(time.strftime("%H:%M"), "status", s, "epochs", n, flush=True)
    time.sleep(60)
EOF
ka_pid=${KA_PID:-}; last=0; fails=0
log "guard start lane $L session $S"
while true; do
  if [ -z "$ka_pid" ] || ! kill -0 "$ka_pid" 2>/dev/null; then
    [ -n "$ka_pid" ] && log "keep-alive exec ended; restarting"
    colab exec -s "$S" --timeout 86400 -f "$KA" >> "$OUT/keepalive.log" 2>&1 & ka_pid=$!
  fi
  st=$(vm "cat $ST 2>/dev/null; python3 -c 'import json;print(len(json.load(open(\"$R/history.json\"))))' 2>/dev/null || echo 0")
  if [ -z "$st" ]; then
    fails=$((fails + 1)); log "VM unreachable ($fails)"
    if [ $fails -ge 10 ] && ! colab sessions 2>&1 | grep -q "$S"; then log "SESSION GONE; last synced epoch $last"; exit 2; fi
    sleep 60; continue
  fi
  fails=0; status=$(echo "$st" | head -1); n=$(echo "$st" | tail -1)
  if [ "$n" -gt "$last" ] 2>/dev/null; then
    ok=1
    pull "$R/recovery.pt" recovery.pt || ok=0
    for e in $(seq $SWA0 "$n"); do
      f=$(printf 'weights_epoch%02d.pt' "$e"); [ -f "$OUT/$f" ] || pull "$R/$f" "$f" || ok=0
    done
    pull "$R/history.json" history.json; pull "$R/train.log" train.log
    [ $ok = 1 ] && { last=$n; log "epoch $n synced (status $status)"; } || log "epoch $n sync incomplete; retry next round"
  fi
  case "$status" in
    DONE)
      ok=1
      for f in $(vm "cd $R && ls weights_swa*.pt val_swa*.json run_summary.json history.json train.log 2>/dev/null"); do
        pull "$R/$f" "$f" || ok=0
      done
      if [ $ok = 1 ] && [ "${COLAB_KEEP:-0}" = 1 ]; then
        (nohup colab exec -s "$S" --timeout 86400 -f "$(dirname "$0")/fm_teacher/ka_generic.py" >> "$OUT/keepalive_after.log" 2>&1 &)
        log "DONE: outputs pulled, sha256 match; session kept (COLAB_KEEP=1, generic keep-alive started)"; exit 0
      fi
      if [ $ok = 1 ]; then log "DONE: outputs pulled, sha256 match; stopping session"; colab stop -s "$S" >> "$LOG" 2>&1; exit 0; fi
      log "DONE but a pull failed; session kept for a manual fetch"; exit 1 ;;
    FAILED*) log "lane $status; session kept for inspection"; exit 1 ;;
  esac
  sleep 120
done
