#!/bin/bash
# Copy files from the Colab VM to this machine over the CLI's SSH proxy. Usage: vmpull.sh <remote path or glob> <local dir>
S=${COLAB_SESSION:-rsna_okd}; K=$HOME/.ssh/id_ed25519_colab
exec rsync -a --partial -e "ssh -i $K -o IdentitiesOnly=yes -o ProxyCommand='$HOME/.venvs/colab-cli/bin/colab --auth oauth2 ssh --proxy-mode -s $S -i $K' -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.config/colab-cli/known_hosts -o BatchMode=yes" "root@colab-$S:$1" "$2"
