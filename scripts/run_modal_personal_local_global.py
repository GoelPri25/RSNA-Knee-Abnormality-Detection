"""Deploy and launch the local+global job using personal credentials."""
import argparse
import json
import subprocess
from pathlib import Path
from run_modal_personal_efficientnet_v31 import personal_env

ROOT = Path(__file__).resolve().parents[1]
APP = 'rsna-local-global-native-triplets'
SCRIPT = ROOT/'scripts/modal_local_global.py'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['transfer', 'transfer-all', 'train'], default='transfer')
    parser.add_argument('--shard', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    env = personal_env()
    listing = subprocess.run(['modal','app','list','--json'], env=env, cwd=ROOT,
                             check=True, capture_output=True, text=True)
    rows = json.loads(listing.stdout)
    matching = [row for row in rows if row.get('description') == APP]
    if any(row.get('state','').lower() != 'stopped' for row in matching):
        raise RuntimeError('Local+global app already active; refusing duplicate launch.')
    subprocess.run(['modal','deploy',str(SCRIPT)], env=env, cwd=ROOT, check=True)
    subprocess.run(['modal','run','--detach',str(SCRIPT), '--stage', args.stage,
                    '--shard', str(args.shard)], env=env, cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
