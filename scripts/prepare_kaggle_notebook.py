"""Bundle only project source into a private, self-contained Kaggle audit notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT / 'src').rglob('*.py'))}
files['configs/audit.json'] = (ROOT / 'configs/audit.json').read_text()
cells = []


def cell(kind, source):
    item = {'cell_type': kind, 'id': f'cell-{len(cells)}', 'metadata': {}, 'source': source.splitlines(True)}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell('markdown', '''# RSNA knee: frozen Pillar-0 experiment, data gate
Private first run: inventory competition data, sample DICOM headers, reconstruct up to
three volumes. All competition files and derived artifacts remain on Kaggle. No full
preprocessing, encoder substitution, pseudo-label generation or training is performed.
The next experiment step depends on the actual audit and visual geometry review.
''')
cell('code', '''import sys, subprocess, json, os
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pydicom==3.0.1'], check=True)
PROJECT = Path('/kaggle/working/project')
''' + 'SOURCE_FILES = ' + repr(files) + '''
for name, source in SOURCE_FILES.items():
    dest = PROJECT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(source)
sys.path.insert(0, str(PROJECT / 'src'))
print('Project source installed; no competition files copied.')
''')
cell('code', '''from rsna_knee.data.audit import audit
mount = Path('/kaggle/input')
roots = [p for p in [mount / 'rsna-knee-abnormality-detection', mount / 'competitions/rsna-knee-abnormality-detection'] if p.is_dir()]
if len(roots) != 1:
    raise RuntimeError(f'Expected one competition mount, found {len(roots)}; inspect mounted inputs')
DATA = roots[0]
OUTPUT = Path('/kaggle/working/knee-audit')
OUTPUT.mkdir(exist_ok=True)
config = json.loads((PROJECT / 'configs/audit.json').read_text())
config['sample_series'] = 32
result = audit(DATA, config)
(OUTPUT / 'audit.json').write_text(json.dumps(result, indent=2))
# Operational log contains aggregates/schema only, no individual identifiers or reports.
summary = {k: result[k] for k in ['status', 'file_extensions', 'dicom_series_directory_candidates', 'sample_duplicate_sop_uids']}
summary['tables'] = result['tables']
summary['geometry_status'] = dict(__import__('collections').Counter(x['status'] for x in result['sampled_series']))
summary['geometry_errors'] = dict(__import__('collections').Counter(x.get('error', '') for x in result['sampled_series'] if x['status'] != 'valid_geometry'))
summary['planes'] = dict(__import__('collections').Counter(x['geometry']['plane'] for x in result['sampled_series'] if x['status'] == 'valid_geometry'))
print('AUDIT_SUMMARY_BEGIN')
print(json.dumps(summary, indent=2))
print('AUDIT_SUMMARY_END')
if result['status'] != 'sampled_audit':
    raise RuntimeError('No DICOM recognized: inspect extension inventory before proceeding')
''')
cell('code', '''import numpy as np
from rsna_knee.preprocessing.dicom import reconstruct, visualize
from IPython.display import Image, display
valid = [x for x in result['sampled_series'] if x['status'] == 'valid_geometry']
completed = 0
for index, item in enumerate(valid[:3]):
    paths = sorted(p for p in (DATA / item['directory']).iterdir() if p.suffix.lower() in config['dicom_suffixes'])
    try:
        volume, metadata = reconstruct(paths)
        dest = OUTPUT / f'volume-{index}'
        dest.mkdir(exist_ok=True)
        np.save(dest / 'volume.npy', volume)
        (dest / 'geometry.json').write_text(json.dumps(metadata, indent=2))
        visualize(volume, metadata, dest / 'views.png')
        display(Image(filename=str(dest / 'views.png')))
        print('VOLUME_OK', index, list(volume.shape), metadata['plane'], metadata['spacing_mm'])
        completed += 1
    except Exception as exc:
        (OUTPUT / f'volume-{index}-error.json').write_text(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))
        print('VOLUME_FAILED', index, type(exc).__name__)
print('RECONSTRUCTED_VOLUMES', completed)
if completed == 0:
    raise RuntimeError('No reconstructed volumes; resolve audit/decoder failures first')
''')
cell('markdown', '''## Next gate
Review private audit artifacts and real-volume images here on Kaggle. Verify official
label schema, patient grouping and metric. Then pin and load the gated
`YalaLab/Pillar0-BreastMRI` checkpoint with frozen weights and reviewed preprocessing.
No end-to-end classification result is claimed by this audit notebook.
''')
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = ROOT / 'notebooks/kaggle'
out.mkdir(exist_ok=True)
(out / 'audit.ipynb').write_text(json.dumps(nb, indent=2) + '\n')
print(out / 'audit.ipynb')
