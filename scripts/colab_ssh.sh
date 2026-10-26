#!/bin/bash
# SSH into a Colab CLI session (separate from its kernel, so it works while an exec runs), via the CLI's proxy mode.
# Key: ~/.ssh/id_ed25519_colab (only for Colab VMs). Usage:
#   scripts/colab_ssh.sh                                    interactive shell on session "rsna"
#   scripts/colab_ssh.sh tail -f /work/out/pubkd_kd_e4s27/train.log      stream a lane's training log
#   COLAB_SESSION=other scripts/colab_ssh.sh nvidia-smi
S=${COLAB_SESSION:-rsna}; K=$HOME/.ssh/id_ed25519_colab
exec ssh -i "$K" -o IdentitiesOnly=yes -o ProxyCommand="colab ssh --proxy-mode -s $S -i $K" \
  -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.config/colab-cli/known_hosts" \
  -o ServerAliveInterval=30 -t "root@colab-$S" "$@"
