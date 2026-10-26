"""Inventory all filenames/CSV rows, sample DICOM series headers; never decode pixels."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import platform

import numpy as np
import pydicom

from rsna_knee.preprocessing.dicom import geometry

FIELDS = ('PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
          'Modality', 'SeriesDescription', 'ProtocolName', 'ScanningSequence',
          'SequenceVariant', 'ScanOptions', 'ImageType', 'EchoTime', 'RepetitionTime',
          'InversionTime', 'MagneticFieldStrength', 'Manufacturer', 'ManufacturerModelName',
          'InstitutionName', 'Rows', 'Columns', 'PixelSpacing', 'SliceThickness',
          'SpacingBetweenSlices', 'ImageOrientationPatient', 'ImagePositionPatient',
          'InstanceNumber', 'AcquisitionNumber', 'TemporalPositionIdentifier', 'NumberOfFrames')


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_audit(path, study_column=None, targets=()):
    missing, values, ids, hashes = Counter(), defaultdict(Counter), Counter(), Counter()
    n = 0
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if study_column and study_column not in columns:
            raise ValueError(f'{path}: missing study column {study_column}')
        if set(targets) - set(columns):
            raise ValueError(f'{path}: missing target columns')
        for row in reader:
            n += 1
            hashes[hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()] += 1
            for key in columns:
                value = row[key]
                if value is None or value.strip().lower() in ('', 'nan', 'null', 'none', 'na'):
                    missing[key] += 1
                elif key in targets:
                    values[key][value] += 1
            if study_column and row[study_column]:
                ids[row[study_column]] += 1
    return {'rows': n, 'columns': columns, 'missing': dict(missing),
            'duplicate_rows': sum(v - 1 for v in hashes.values()),
            'unique_studies': len(ids) if study_column else None,
            'duplicate_study_rows': sum(v - 1 for v in ids.values()) if study_column else None,
            'target_values': dict(values), 'sha256': digest_file(path)}


def audit(root, config):
    if not root.is_dir():
        raise ValueError(f'Dataset directory does not exist: {root}')
    groups, extensions, tables, archives = defaultdict(list), Counter(), {}, []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        extensions[path.suffix.lower()] += 1
        if path.suffix.lower() in config['dicom_suffixes']:
            groups[str(rel.parent)].append(path)
        elif path.suffix.lower() == '.csv':
            is_labels = str(rel) == config.get('labels_csv')
            tables[str(rel)] = csv_audit(path, config.get('study_column') if is_labels else None,
                                        config.get('target_columns', []) if is_labels else [])
        elif path.suffix.lower() in ('.zip', '.gz', '.tar', '.7z'):
            archives.append(str(rel))
    rng = random.Random(config['seed'])
    chosen = sorted(rng.sample(sorted(groups), min(config['sample_series'], len(groups))))
    sampled, seen_sop = [], Counter()
    for directory in chosen:
        entry = {'directory': directory, 'files': len(groups[directory])}
        try:
            headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in groups[directory]]
            entry['fields_available'] = sorted({d.keyword for h in headers for d in h if d.keyword})
            entry['missing_fields'] = {key: sum(not hasattr(h, key) for h in headers) for key in FIELDS}
            entry['values'] = {key: dict(Counter(str(getattr(h, key, '')) for h in headers)) for key in FIELDS}
            seen_sop.update(str(h.SOPInstanceUID) for h in headers if hasattr(h, 'SOPInstanceUID'))
            _, meta = geometry(headers)
            entry['geometry'] = meta
            entry['status'] = 'valid_geometry'
        except Exception as exc:
            entry['status'] = 'flagged'
            entry['error'] = f'{type(exc).__name__}: {exc}'
        sampled.append(entry)
    # Directory names are candidates only until checked against DICOM UIDs.
    studies = defaultdict(list)
    for directory, paths in groups.items():
        studies[str(Path(directory).parent)].append(len(paths))
    inventory = [{'study_directory_candidate': key, 'series_directories': len(v),
                  'slice_file_counts': v} for key, v in sorted(studies.items())]
    return {
        'status': 'sampled_audit' if groups else 'no_dicom_found',
        'root': str(root), 'config': config,
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pydicom': pydicom.__version__},
        'file_extensions': dict(extensions), 'archives_not_opened': archives, 'tables': tables,
        'directory_inventory': inventory, 'dicom_series_directory_candidates': len(groups),
        'sampled_series': sampled, 'sample_duplicate_sop_uids': sum(v - 1 for v in seen_sop.values()),
        'unverified': ['Directory hierarchy is not authoritative study/series identity.',
                       'Train/test counts and labelled/unlabelled joins require reviewed table schema.',
                       'Patient IDs may be missing or anonymized; no safe split approved yet.',
                       'Site/protocol distribution is sampled; no leakage conclusion yet.',
                       'Fluid sensitivity/fat suppression require sequence review; no text heuristics applied.',
                       'Reports remain text; target names and metric require competition verification.',
                       'Only configured suffixes inventoried as DICOM; inspect extension counts for others.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/audit.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['sample_series'] < 1:
        parser.error('sample_series must be positive')
    result = audit(args.root, config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2))
    lines = ['# Dataset audit', '', f'Status: {result["status"]}',
             f'DICOM series directory candidates: {result["dicom_series_directory_candidates"]}',
             f'Series sampled: {len(result["sampled_series"])}', '', '## Tables', '']
    lines += [f'- {key}: {value["rows"]} rows; columns: {value["columns"]}' for key, value in result['tables'].items()]
    lines += ['', '## Not yet verified', ''] + [f'- {x}' for x in result['unverified']]
    (args.output / 'AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(f'Audit written to {args.output}; status={result["status"]}')
    if not result['dicom_series_directory_candidates']:
        sys.exit(2)


if __name__ == '__main__':
    main()
