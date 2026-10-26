#!/bin/bash
# Runs ON the VM: switch the d96 cache download from one shard at a time to all remaining shards in parallel
# (each Kaggle download stream runs at ~12 MB/s, so parallel streams cut the wait). Keeps the shard already in flight,
# then re-runs vm_prepare.sh, which skips every finished part and continues with the setup.
set -uo pipefail
export KAGGLE_API_TOKEN=$(cut -d= -f2- /content/kaggle.env)
cd /content
pkill -f "bash /content/okd/vm_prepare.sh" || true          # stop the sequential loop; its running kaggle child continues
echo RUNNING > /content/okd/prepare.status
fetch() {  # fetch <shard>
  local i=$1
  [ -f d96/shard_$i/cache_summary.json ] && return 0
  if pgrep -f "rsna-coatnet-d96-cache-shard-$i -p _dl_$i" > /dev/null; then    # already downloading: wait for it
    while pgrep -f "rsna-coatnet-d96-cache-shard-$i -p _dl_$i" > /dev/null; do sleep 20; done
  else
    for t in 1 2 3; do
      rm -rf _dl_$i && mkdir _dl_$i
      kaggle kernels output tumis001/rsna-coatnet-d96-cache-shard-$i -p _dl_$i > /dev/null && break
    done
  fi
  [ -f _dl_$i/d96/cache_summary.json ] || { echo "shard $i incomplete"; return 1; }
  rm -rf d96/shard_$i && mv _dl_$i/d96 d96/shard_$i && rm -rf _dl_$i && echo "cache shard $i ok"
}
mkdir -p d96
for i in 0 1 2 3; do fetch $i & done
wait
for i in 0 1 2 3; do [ -f d96/shard_$i/cache_summary.json ] || { echo "FAILED cache shard $i" > /content/okd/prepare.status; exit 1; }; done
exec bash /content/okd/vm_prepare.sh
