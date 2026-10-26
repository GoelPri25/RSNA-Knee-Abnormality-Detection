"""Sequential CPU cache preparation, private Kaggle publication and Modal upload.

Never starts GPU training. Resume-safe at the completed-shard boundary.
"""
from pathlib import Path
import json
import subprocess
import time
from run_modal_personal_efficientnet_v31 import personal_env

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/highres336'
VOLUME = 'rsna-cnn-140mm-336px-cache-v1'
KERNELS = ['tumis001/rsna-cnn-140mm-336px-cache'] + [
    f'tumis001/rsna-cnn-140mm-336px-cache-shard-{i}' for i in (1, 2)]


def command(args, modal=False, capture=False):
    return subprocess.run(args, cwd=ROOT, env=personal_env() if modal else None,
                          check=True, text=True, capture_output=capture)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    command(['modal', 'volume', 'put', '-f', VOLUME,
             str(DATA/'experiment_assets'), '/experiment_assets'], modal=True)
    for shard, kernel in enumerate(KERNELS):
        marker = DATA / f'shard_{shard}_published.json'
        if marker.is_file():
            continue
        if shard and not (DATA / f'shard_{shard}_launched.json').is_file():
            folder = ROOT / f'kaggle/experiment_kernels/highres336_cache_shard{shard}'
            command(['kaggle', 'kernels', 'push', '-p', str(folder)])
        while True:
            status = command(['kaggle', 'kernels', 'status', kernel], capture=True).stdout
            print(status.strip(), flush=True)
            if 'COMPLETE' in status:
                break
            if 'ERROR' in status or 'CANCEL' in status:
                raise RuntimeError(f'Cache builder failed: {status}')
            time.sleep(60)
        output = DATA / f'shard_{shard}_output'
        command(['kaggle', 'kernels', 'output', kernel, '-p', str(output)])
        complete = json.loads((output/'highres336/completion.json').read_text())
        assert complete['complete'] and complete['shards'][0]['shard'] == shard
        cache = output / f'highres336/shard_{shard}/slice_cache'
        dataset = f'tumis001/rsna-cnn-140mm-336px-cache-shard-{shard}'
        (cache/'dataset-metadata.json').write_text(json.dumps({
            'id': dataset, 'title': f'RSNA CNN 140mm 336px Cache Data Part {shard}',
            'licenses': [{'name': 'other'}]}))
        probe = subprocess.run(['kaggle','datasets','files',dataset],cwd=ROOT,
                               capture_output=True)
        args = (['kaggle','datasets','create','-p',str(cache)] if probe.returncode
                else ['kaggle','datasets','version','-p',str(cache),'-m','336px cache'])
        command(args)
        command(['modal','volume','put','-f',VOLUME,str(cache),f'/shard_{shard}/slice_cache'],modal=True)
        marker.write_text(json.dumps({'dataset':dataset,'kernel':kernel,
                                     'volume':VOLUME,'completion':complete},indent=2)+'\n')
        print(f'Shard {shard} published and uploaded to Modal',flush=True)
    print('All 3 cache shards and experiment assets ready; GPU training was not launched.',flush=True)


if __name__ == '__main__':
    main()
