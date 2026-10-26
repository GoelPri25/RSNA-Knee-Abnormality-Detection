"""Launch a single detached HPO-validation job using personal Modal credentials."""
import json
import subprocess
from pathlib import Path
from run_modal_personal_efficientnet_v31 import personal_env

ROOT = Path(__file__).resolve().parents[1]
APP = 'rsna-local-global-hpo-validation'
SCRIPT = ROOT / 'scripts/modal_local_global_hpo_validation.py'

def main():
    env = personal_env()
    listing = subprocess.run(['modal','app','list','--json'], env=env, cwd=ROOT,
                             check=True, capture_output=True, text=True)
    rows = json.loads(listing.stdout)
    if any(row.get('description') == APP and row.get('state','').lower() != 'stopped' for row in rows):
        raise RuntimeError('HPO-validation app already exists; refusing duplicate launch.')
    active = [row for row in rows if row.get('state','').lower() != 'stopped' and int(row.get('tasks',0)) > 0]
    if active: raise RuntimeError('Another personal Modal app has active tasks; inspect before starting.')
    subprocess.run(['modal','deploy',str(SCRIPT)], env=env, cwd=ROOT, check=True)
    subprocess.run(['modal','run','--detach',str(SCRIPT)], env=env, cwd=ROOT, check=True)

if __name__ == '__main__': main()
