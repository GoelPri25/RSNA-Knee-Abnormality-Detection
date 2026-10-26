#!/bin/bash
# Host-side guard for an FB-F teacher VM (scripts/fm_teacher/fm_vm_driver.py running detached on Colab session $1).
#  1. Keep-alive: a kernel cell (ka_generic.py) polls /content/fm/STATUS until ALLDONE / FAILED; restarted if it ends early
#     (a Colab CLI VM whose kernel is idle is reclaimed within ~30 min, see scripts/colab_lane_guard.sh).
#  2. Every finished chunk (out/cNNN/DONE) is tarred on the VM, downloaded and sha256-checked, and unpacked into $2/cNNN.
#  3. ALLDONE: last chunks + done.txt / nodata.txt pulled, then the session is stopped. FAILED: pulled, session kept.
# Usage: scripts/fm_teacher/fm_guard.sh <session> <local dir>      e.g. tC data/_session_tmp/colab/fmteach/tC
set -uo pipefail
S=$1; OUT=$2; K=$HOME/.ssh/id_ed25519_colab; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; LOG=$OUT/guard.log
log() { echo "$(TZ=Europe/Luxembourg date +%F' '%H:%M:%S) $*" >> "$LOG"; }
vm() {
  timeout 120 ssh -i "$K" -o IdentitiesOnly=yes -o BatchMode=yes -o ProxyCommand="colab ssh --proxy-mode -s $S -i $K" \
    -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.config/colab-cli/known_hosts" \
    -o ServerAliveInterval=30 "root@colab-$S" "$@" 2>/dev/null
}
pull() {  # pull <remote file> <local file>: download, compare sha256 (3 tries)
  local want got
  want=$(vm "sha256sum $1 | cut -c1-64") || return 1
  for t in 1 2 3; do
    timeout 1800 colab download -s "$S" "$1" "$2.part" > /dev/null 2>&1
    got=$(sha256sum "$2.part" 2>/dev/null | cut -c1-64)
    if [ -n "$want" ] && [ "$got" = "$want" ]; then mv "$2.part" "$2"; return 0; fi
    log "pull $1 try $t: sha mismatch or download failed"
  done
  return 1
}
sync_chunks() {
  local c ok=0
  for c in $(vm "cd /content/fm/out && for d in c[0-9][0-9][0-9]; do [ -f \$d/DONE ] && echo \$d; done"); do
    [ -f "$OUT/$c/DONE" ] && continue
    vm "mkdir -p /content/fm/sync && tar czf /content/fm/sync/$c.tgz -C /content/fm/out $c" || continue
    if pull "/content/fm/sync/$c.tgz" "$OUT/$c.tgz" && tar xzf "$OUT/$c.tgz" -C "$OUT"; then
      rm -f "$OUT/$c.tgz"; log "$c pulled: $(cat "$OUT/$c/chunk.json" | tr -d '\n ')"
    else ok=1; fi
  done
  for f in done.txt nodata.txt; do vm "test -f /content/fm/out/$f" && pull "/content/fm/out/$f" "$OUT/$f"; done
  return $ok
}
log "guard start session $S"; fails=0
while true; do
  if ! pgrep -f "colab exec -s $S " > /dev/null; then
    log "keep-alive exec not running; starting"
    (colab exec -s "$S" --timeout 86400 -f "$HERE/ka_generic.py" >> "$OUT/keepalive.log" 2>&1 &)
  fi
  st=$(vm "cat /content/fm/STATUS 2>/dev/null || echo setup")
  if [ -z "$st" ]; then
    fails=$((fails + 1)); log "VM unreachable ($fails)"
    if [ $fails -ge 10 ] && ! colab sessions 2>&1 | grep -qw "$S"; then log "SESSION GONE"; exit 2; fi
    sleep 60; continue
  fi
  fails=0; echo "$st" > "$OUT/STATUS"
  sync_chunks
  case "$st" in
    ALLDONE*) if sync_chunks; then log "ALLDONE: all chunks pulled; stopping session"; colab stop -s "$S" >> "$LOG" 2>&1; exit 0; fi
              log "ALLDONE but a pull failed; retrying" ;;
    FAILED*) log "driver $st; session kept for inspection"; exit 1 ;;
  esac
  sleep 300
done
