"""Deploy and spawn the Phase 1 full-band run on the personal Modal profile, independent of the local CLI."""
import json
import subprocess
from pathlib import Path
from run_modal_personal_efficientnet_v31 import personal_env

ROOT = Path(__file__).resolve().parents[1]
APP = 'rsna-r50-fullband224-seed2026'
SCRIPT = ROOT/'scripts/modal_fullband224.py'

def main():
    env = personal_env()
    listing = subprocess.run(['modal','app','list','--json'], env=env, cwd=ROOT,
                             check=True, capture_output=True, text=True)
    rows = json.loads(listing.stdout)
    matching = [row for row in rows if row.get('description') == APP]
    if any(row.get('state','').lower() != 'stopped' for row in matching):
        raise RuntimeError('App is already active; refusing duplicate launch.')
    subprocess.run(['modal','deploy',str(SCRIPT)], env=env, cwd=ROOT, check=True)
    # This entrypoint only spawns the deployed coordinator and then exits.
    subprocess.run(['modal','run','--detach',str(SCRIPT)], env=env, cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
