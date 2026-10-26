#!/bin/bash
# Run a command on the Colab VM over SSH (independent of the Jupyter kernel). Usage: vmssh.sh '<command>'
S=${COLAB_SESSION:-rsna_okd}; K=$HOME/.ssh/id_ed25519_colab
exec ssh -i "$K" -o IdentitiesOnly=yes -o ProxyCommand="$HOME/.venvs/colab-cli/bin/colab --auth oauth2 ssh --proxy-mode -s $S -i $K" \
  -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.config/colab-cli/known_hosts" -o ConnectTimeout=60 \
  -o ServerAliveInterval=30 -o BatchMode=yes "root@colab-$S" "$@"
