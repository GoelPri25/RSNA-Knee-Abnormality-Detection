#!/bin/bash
# vmssh.sh <session> <command...>: ssh to a Colab CLI VM, no tty (binary-safe). Colab allows one `colab ssh` per runtime
# at a time (HTTP 429 when the guard of that VM is connected), so a refused connection is retried (10 x 15 s).
S=$1; shift; K=$HOME/.ssh/id_ed25519_colab
for t in $(seq 10); do
  err=$(mktemp)
  timeout ${VMSSH_TIMEOUT:-900} ssh -o IdentityAgent=none -i "$K" -o IdentitiesOnly=yes -o BatchMode=yes -o ProxyCommand="colab ssh --proxy-mode -s $S -i $K" \
    -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.config/colab-cli/known_hosts" "root@colab-$S" "$@" 2> "$err"
  rc=$?
  if [ $rc -eq 255 ] && grep -q "429\|Already-active" "$err"; then rm -f "$err"; sleep 15; continue; fi
  grep -v -iE "eyJ|token" "$err" >&2; rm -f "$err"; exit $rc
done
echo "vmssh: $S busy (429) after 10 tries" >&2; exit 255
