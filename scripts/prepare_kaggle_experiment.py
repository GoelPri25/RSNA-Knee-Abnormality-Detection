"""Build the readable, cell-by-cell Kaggle milestone notebook."""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_experiment"
OUT.mkdir(parents=True, exist_ok=True)
cells = []


def markdown(source: str) -> None:
    cells.append({"cell_type": "markdown", "id": f"cell-{len(cells):02d}", "metadata": {}, "source": dedent(source).splitlines(True)})


def code(source: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "id": f"cell-{len(cells):02d}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": dedent(source).splitlines(True),
        }
    )


markdown("""
# Frozen Pillar-0 knee MRI: first small-sample experiment

This notebook executes the first milestone in explicit stages. It keeps all competition
data and derived image artifacts on Kaggle. It does not scale to the full dataset.

The checkpoint is `YalaLab/Pillar0-BreastMRI`. It was trained on three-channel breast
DCE MRI, while each knee series here is a single MRI acquisition. For this transfer
probe, the same windowed knee series is repeated across the three input channels. This
is an explicit domain/input mismatch, not a claim that the channels represent DCE phases.
""")

markdown("## 1. Environment and reproducibility")
code("""
%pip install -q pydicom==3.0.1 transformers==4.56.2 accelerate==1.10.1
""")
code("""
import gc
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATA = Path('/kaggle/input/rsna-knee-abnormality-detection')
if not DATA.is_dir():
    alternate = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
    DATA = alternate if alternate.is_dir() else DATA
assert DATA.is_dir(), f'Competition mount not found: {DATA}'

WORK = Path('/kaggle/working/pillar0-small-sample')
WORK.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]

RUN_CONFIG = {
    'seed': SEED,
    'checkpoint': 'YalaLab/Pillar0-BreastMRI',
    'checkpoint_revision': 'main',
    'target_spacing_mm': [1.0, 1.0, 1.0],
    'target_shape_dhw': [192, 384, 384],
    'mr_window': 'high_contrast_percentiles_20_99',
    'channel_adaptation': 'repeat one knee series into 3 channels',
    'max_studies': 4,
    'max_series_per_study': 3,
}
(WORK / 'run_config.json').write_text(json.dumps(RUN_CONFIG, indent=2))
print('device:', DEVICE)
print(json.dumps(RUN_CONFIG, indent=2))
""")

markdown("## 2. Tables, targets and supervision")
code("""
train = pd.read_csv(DATA / 'train.csv')
series = pd.read_csv(DATA / 'train_series.csv')
test = pd.read_csv(DATA / 'test.csv')
test_series = pd.read_csv(DATA / 'test_series.csv')

label_mask = train[TARGETS].notna()
labeled_any = label_mask.any(axis=1)
labeled_all = label_mask.all(axis=1)

print('train studies:', train.StudyInstanceUID.nunique())
print('test rows visible to this notebook:', test.StudyInstanceUID.nunique())
print('train series:', series.SeriesInstanceUID.nunique())
print('studies with any gold label:', int(labeled_any.sum()))
print('studies with all 12 gold labels:', int(labeled_all.sum()))
print('reports missing:', int(train.Report.isna().sum()))
print('duplicate train study IDs:', int(train.StudyInstanceUID.duplicated().sum()))
print('duplicate train series IDs:', int(series.SeriesInstanceUID.duplicated().sum()))

prevalence = pd.DataFrame({
    'known': label_mask.sum(),
    'positive': train[TARGETS].eq(1).sum(),
    'negative': train[TARGETS].eq(0).sum(),
})
prevalence['positive_rate'] = prevalence.positive / prevalence.known
display(prevalence)

series_per_study = series.groupby('StudyInstanceUID').size()
display(series_per_study.describe(percentiles=[.01, .25, .5, .75, .99]).to_frame('series_per_study'))
display(series[['Anatomical_Plane', 'Fluid_Sensitive', 'Fat_Suppression']].value_counts().to_frame('series'))

join_check = series.StudyInstanceUID.isin(train.StudyInstanceUID)
print('series rows whose study appears in train.csv:', int(join_check.sum()), '/', len(series))
assert set(series.StudyInstanceUID).issubset(set(train.StudyInstanceUID))
assert set(train.loc[labeled_any, TARGETS].stack().unique()).issubset({0, 1})
""")

markdown("## 3. DICOM geometry and physical reconstruction")
code("""
class GeometryError(ValueError):
    pass


def discover_image_layout():
    # Infer UID directory levels from one DICOM without scanning pixel data.
    first_dicom = next(DATA.rglob('*.dcm'))
    all_series = set(series.SeriesInstanceUID.astype(str)) | set(test_series.SeriesInstanceUID.astype(str))
    all_studies = set(train.StudyInstanceUID.astype(str)) | set(test.StudyInstanceUID.astype(str))
    current = first_dicom.parent
    while current != DATA:
        if current.name in all_series:
            if current.parent.name in all_studies:
                return {
                    'root': current.parent.parent,
                    'has_study_level': True,
                    'example': first_dicom.relative_to(DATA),
                }
            return {
                'root': current.parent,
                'has_study_level': False,
                'example': first_dicom.relative_to(DATA),
            }
        current = current.parent
    raise RuntimeError(f'Could not identify a SeriesInstanceUID in {first_dicom.relative_to(DATA)}')


IMAGE_LAYOUT = discover_image_layout()
print('discovered DICOM example:', IMAGE_LAYOUT['example'])
print('image root:', IMAGE_LAYOUT['root'])
print('study directory level:', IMAGE_LAYOUT['has_study_level'])


def series_dir(study_uid, series_uid):
    root = IMAGE_LAYOUT['root']
    path = root / str(study_uid) / str(series_uid) if IMAGE_LAYOUT['has_study_level'] else root / str(series_uid)
    if not path.is_dir():
        raise FileNotFoundError(f'Series directory not found under inferred layout: {path}')
    return path


def reconstruct_series(folder):
    paths = sorted(path for path in folder.iterdir() if path.suffix.lower() == '.dcm')
    if len(paths) < 2:
        raise GeometryError(f'Need at least two DICOM slices: {folder}')

    headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in paths]
    iop = np.asarray(headers[0].ImageOrientationPatient, dtype=np.float64)
    row_direction, column_direction = iop[:3], iop[3:]
    normal = np.cross(row_direction, column_direction)
    positions = np.asarray([h.ImagePositionPatient for h in headers], dtype=np.float64)
    pixel_spacing = np.asarray(headers[0].PixelSpacing, dtype=np.float64)

    if not np.allclose([np.linalg.norm(row_direction), np.linalg.norm(column_direction),
                        np.dot(row_direction, column_direction)], [1, 1, 0], atol=1e-4):
        raise GeometryError('ImageOrientationPatient is not orthonormal')
    if any(not np.allclose(h.ImageOrientationPatient, iop, atol=1e-4) for h in headers):
        raise GeometryError('Orientation varies within series')
    if any(not np.allclose(h.PixelSpacing, pixel_spacing, atol=1e-4) for h in headers):
        raise GeometryError('PixelSpacing varies within series')
    if len({str(h.SOPInstanceUID) for h in headers}) != len(headers):
        raise GeometryError('Duplicate SOPInstanceUID')

    projection = positions @ normal
    order = np.argsort(projection, kind='stable')
    gaps = np.diff(projection[order])
    slice_spacing = float(np.median(gaps))
    if np.any(gaps < 1e-3):
        raise GeometryError('Repeated physical slice positions')
    if not np.allclose(gaps, slice_spacing, atol=.05, rtol=.01):
        raise GeometryError('Irregular physical slice spacing')

    slices = []
    for index in order:
        dcm = pydicom.dcmread(paths[index])
        pixels = dcm.pixel_array.astype(np.float32)
        pixels = pixels * float(getattr(dcm, 'RescaleSlope', 1))
        pixels += float(getattr(dcm, 'RescaleIntercept', 0))
        slices.append(pixels)

    volume = np.stack(slices)
    metadata = {
        'spacing_dhw_mm': [slice_spacing, float(pixel_spacing[0]), float(pixel_spacing[1])],
        'shape_dhw': list(volume.shape),
        'normal_lps': normal.tolist(),
        'first_position_lps': positions[order[0]].tolist(),
        'ordered_files': [str(paths[i]) for i in order],
    }
    return volume, metadata
""")

markdown("## 4. Reconstruct and visually check representative series")
code("""
labeled_studies = train.loc[labeled_all, 'StudyInstanceUID'].astype(str).sort_values().tolist()
representatives = (
    series[series.StudyInstanceUID.astype(str).isin(labeled_studies)]
    .sort_values(['Anatomical_Plane', 'StudyInstanceUID', 'SeriesInstanceUID'])
    .groupby('Anatomical_Plane', as_index=False)
    .head(1)
)

fig, axes = plt.subplots(len(representatives), 3, figsize=(12, 4 * len(representatives)))
axes = np.atleast_2d(axes)
for row_index, row in enumerate(representatives.itertuples(index=False)):
    volume, metadata = reconstruct_series(series_dir(row.StudyInstanceUID, row.SeriesInstanceUID))
    finite = volume[np.isfinite(volume)]
    lo, hi = np.percentile(finite, [1, 99])
    for axis, slice_index in zip(axes[row_index], np.linspace(0, len(volume) - 1, 3).astype(int)):
        axis.imshow(volume[slice_index], cmap='gray', vmin=lo, vmax=hi)
        axis.set_title(f'{row.Anatomical_Plane}; slice {slice_index}; {metadata["spacing_dhw_mm"]}')
        axis.axis('off')
plt.tight_layout()
plt.savefig(WORK / 'representative_series.png', dpi=140)
plt.show()
""")

markdown("## 5. Physically meaningful preprocessing for Pillar-0")
code("""
TARGET_DHW = tuple(RUN_CONFIG['target_shape_dhw'])
TARGET_SPACING = tuple(RUN_CONFIG['target_spacing_mm'])


def center_pad_or_crop(volume, target_shape):
    result = volume
    for axis, target in enumerate(target_shape):
        current = result.shape[axis]
        if current > target:
            start = (current - target) // 2
            indices = [slice(None)] * 3
            indices[axis] = slice(start, start + target)
            result = result[tuple(indices)]
    padding = []
    for current, target in reversed(list(zip(result.shape, target_shape))):
        amount = max(target - current, 0)
        padding.extend([amount // 2, amount - amount // 2])
    return F.pad(result[None, None], padding, value=0).squeeze(0).squeeze(0)


def preprocess_for_pillar(volume, spacing_dhw):
    # Return [1,3,H,W,D]; preserve spacing before center pad/crop.
    tensor = torch.from_numpy(volume).float()[None, None]
    resampled_shape = [
        max(1, int(round(size * spacing / target_spacing)))
        for size, spacing, target_spacing in zip(volume.shape, spacing_dhw, TARGET_SPACING)
    ]
    tensor = F.interpolate(tensor, size=resampled_shape, mode='trilinear', align_corners=False)
    tensor = tensor.squeeze(0).squeeze(0)

    nonzero = tensor[tensor != 0]
    values = nonzero if nonzero.numel() else tensor.flatten()
    lo, hi = torch.quantile(values, torch.tensor([.20, .99]))
    tensor = ((tensor - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    tensor = center_pad_or_crop(tensor, TARGET_DHW)

    # Pillar0-BreastMRI uses three DCE channels. A knee series has one channel.
    # Repetition is the simplest frozen-transfer adapter and is recorded in RUN_CONFIG.
    tensor = tensor.permute(1, 2, 0).contiguous()  # D,H,W -> H,W,D
    return tensor[None, None].repeat(1, 3, 1, 1, 1)


example_row = representatives.iloc[0]
example_volume, example_metadata = reconstruct_series(
    series_dir(example_row.StudyInstanceUID, example_row.SeriesInstanceUID)
)
example_input = preprocess_for_pillar(example_volume, example_metadata['spacing_dhw_mm'])
print('native:', example_volume.shape, example_metadata['spacing_dhw_mm'])
print('model input:', tuple(example_input.shape), example_input.dtype,
      float(example_input.min()), float(example_input.max()))
assert example_input.shape == (1, 3, 384, 384, 192)
del example_volume, example_input
gc.collect()
""")

markdown("""
## 6. Load the frozen Pillar-0 checkpoint

This requires a private Kaggle secret named `HF_TOKEN` from a Hugging Face account that
has accepted the gated checkpoint conditions. The cell fails clearly if access is absent.
""")
code("""
from kaggle_secrets import UserSecretsClient
from transformers import AutoModel

try:
    HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN')
except Exception as exc:
    raise RuntimeError(
        'Add an HF_TOKEN Kaggle secret and accept access at '
        'https://huggingface.co/YalaLab/Pillar0-BreastMRI'
    ) from exc

pillar = AutoModel.from_pretrained(
    RUN_CONFIG['checkpoint'],
    revision=RUN_CONFIG['checkpoint_revision'],
    token=HF_TOKEN,
    trust_remote_code=True,
    low_cpu_mem_usage=False,
).eval().to(DEVICE)
for parameter in pillar.parameters():
    parameter.requires_grad_(False)

print('parameters:', sum(p.numel() for p in pillar.parameters()))
print('trainable parameters:', sum(p.numel() for p in pillar.parameters() if p.requires_grad))
""")

markdown("## 7. Extract and cache a few series embeddings")
code("""
@torch.inference_mode()
def extract_embedding(row):
    volume, geometry = reconstruct_series(series_dir(row.StudyInstanceUID, row.SeriesInstanceUID))
    model_input = preprocess_for_pillar(volume, geometry['spacing_dhw_mm']).to(DEVICE)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
        embedding = pillar.extract_vision_feats({'breast_mr': model_input})
    embedding = embedding.squeeze(0).float().cpu()
    if embedding.ndim != 1 or not torch.isfinite(embedding).all():
        raise RuntimeError(f'Invalid embedding shape/values: {tuple(embedding.shape)}')
    del model_input, volume
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return embedding, geometry


selected_studies = labeled_studies[:RUN_CONFIG['max_studies']]
selected_series = (
    series[series.StudyInstanceUID.astype(str).isin(selected_studies)]
    .sort_values(['StudyInstanceUID', 'Anatomical_Plane', 'SeriesInstanceUID'])
    .groupby('StudyInstanceUID', as_index=False)
    .head(RUN_CONFIG['max_series_per_study'])
)

cache_rows = []
for index, row in enumerate(selected_series.itertuples(index=False), start=1):
    embedding, geometry = extract_embedding(row)
    path = WORK / f'{row.StudyInstanceUID}__{row.SeriesInstanceUID}.pt'
    record = {
        'study_uid': str(row.StudyInstanceUID),
        'series_uid': str(row.SeriesInstanceUID),
        'embedding': embedding,
        'anatomical_plane': str(row.Anatomical_Plane),
        'fluid_sensitive': str(row.Fluid_Sensitive),
        'fat_suppression': str(row.Fat_Suppression),
        'geometry': geometry,
        'preprocessing': RUN_CONFIG,
    }
    torch.save(record, path)
    cache_rows.append({'study_uid': record['study_uid'], 'series_uid': record['series_uid'],
                       'embedding_dim': embedding.numel(), 'cache_path': str(path)})
    print(f'{index}/{len(selected_series)}', tuple(embedding.shape), path.name)

cache_index = pd.DataFrame(cache_rows)
cache_index.to_csv(WORK / 'embedding_cache.csv', index=False)
display(cache_index)
assert cache_index.embedding_dim.nunique() == 1
""")

markdown("## 8. Small variable-series study aggregator")
code("""
PLANE_TO_INDEX = {'axial': 0, 'coronal': 1, 'sagittal': 2}


def binary_metadata(value):
    text = str(value).strip().lower()
    if text in {'1', '1.0', 'true', 'yes', 'y'}:
        return 1.0
    if text in {'0', '0.0', 'false', 'no', 'n'}:
        return 0.0
    return 0.5


def metadata_vector(record):
    plane = torch.zeros(4)
    plane[PLANE_TO_INDEX.get(record['anatomical_plane'].lower(), 3)] = 1
    spacing = torch.tensor(record['geometry']['spacing_dhw_mm']).log1p() / 3
    return torch.cat([
        plane,
        torch.tensor([binary_metadata(record['fluid_sensitive']),
                      binary_metadata(record['fat_suppression'])]),
        torch.tensor([np.log1p(record['geometry']['shape_dhw'][0]) / 5]),
        spacing,
    ]).float()


records_by_study = {}
for path in cache_index.cache_path:
    record = torch.load(path, weights_only=False)
    records_by_study.setdefault(record['study_uid'], []).append(record)

label_lookup = train.set_index(train.StudyInstanceUID.astype(str))[TARGETS]
embedding_dim = int(cache_index.embedding_dim.iloc[0])
first_record = next(iter(records_by_study.values()))[0]
metadata_dim = len(metadata_vector(first_record))


class StudyAggregator(nn.Module):
    def __init__(self, embedding_dim, metadata_dim, width=128, heads=4, labels=12):
        super().__init__()
        self.series_projection = nn.Linear(embedding_dim, width)
        self.metadata_projection = nn.Linear(metadata_dim, width)
        self.study_token = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width * 2, dropout=.1,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, labels)

    def forward(self, series_embeddings, series_metadata, padding_mask):
        tokens = self.series_projection(series_embeddings) + self.metadata_projection(series_metadata)
        query = self.study_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([query, tokens], dim=1)
        query_mask = torch.zeros((padding_mask.size(0), 1), dtype=torch.bool, device=padding_mask.device)
        encoded = self.encoder(tokens, src_key_padding_mask=torch.cat([query_mask, padding_mask], dim=1))
        z_study = self.norm(encoded[:, 0])
        return self.head(z_study), z_study


def make_batch(records_by_study):
    studies = sorted(records_by_study)
    max_series = max(len(records_by_study[study]) for study in studies)
    embeddings = torch.zeros(len(studies), max_series, embedding_dim)
    metadata = torch.zeros(len(studies), max_series, metadata_dim)
    padding_mask = torch.ones(len(studies), max_series, dtype=torch.bool)
    labels = torch.zeros(len(studies), len(TARGETS))
    label_mask = torch.zeros_like(labels, dtype=torch.bool)
    for batch_index, study in enumerate(studies):
        records = records_by_study[study]
        for series_index, record in enumerate(records):
            embeddings[batch_index, series_index] = record['embedding']
            metadata[batch_index, series_index] = metadata_vector(record)
            padding_mask[batch_index, series_index] = False
        values = label_lookup.loc[study].to_numpy(dtype=np.float32)
        known = np.isfinite(values)
        labels[batch_index, known] = torch.from_numpy(values[known])
        label_mask[batch_index] = torch.from_numpy(known)
    return studies, embeddings, metadata, padding_mask, labels, label_mask


batch = make_batch(records_by_study)
study_ids, series_embeddings, series_metadata, padding_mask, labels, known = batch
aggregator = StudyAggregator(embedding_dim, metadata_dim).to(DEVICE)
print('aggregator trainable parameters:', sum(p.numel() for p in aggregator.parameters()))
print('batch shapes:', series_embeddings.shape, series_metadata.shape, labels.shape)
""")

markdown("## 9. Tiny overfit sanity check and study-embedding export")
code("""
series_embeddings = series_embeddings.to(DEVICE)
series_metadata = series_metadata.to(DEVICE)
padding_mask = padding_mask.to(DEVICE)
labels = labels.to(DEVICE)
known = known.to(DEVICE)

optimizer = torch.optim.AdamW(aggregator.parameters(), lr=3e-3, weight_decay=1e-4)
losses = []
aggregator.train()
for step in range(101):
    logits, z_study = aggregator(series_embeddings, series_metadata, padding_mask)
    loss = F.binary_cross_entropy_with_logits(logits[known], labels[known])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
    if step % 20 == 0:
        print(step, losses[-1])

assert losses[-1] < losses[0], (losses[0], losses[-1])
aggregator.eval()
with torch.inference_mode():
    logits, z_study = aggregator(series_embeddings, series_metadata, padding_mask)
    probabilities = logits.sigmoid()

torch.save({'model': aggregator.state_dict(), 'config': RUN_CONFIG}, WORK / 'aggregator_sanity.pt')
np.savez_compressed(
    WORK / 'study_embeddings.npz',
    study_uid=np.asarray(study_ids),
    z_study=z_study.cpu().numpy(),
    probabilities=probabilities.cpu().numpy(),
    labels=labels.cpu().numpy(),
    label_mask=known.cpu().numpy(),
    target_names=np.asarray(TARGETS),
)
pd.DataFrame({'step': range(len(losses)), 'loss': losses}).to_csv(WORK / 'loss.csv', index=False)

print('initial/final loss:', losses[0], losses[-1])
print('logits:', tuple(logits.shape))
print('probabilities:', tuple(probabilities.shape))
print('z_study:', tuple(z_study.shape))
print('finite:', bool(torch.isfinite(logits).all() and torch.isfinite(z_study).all()))
""")

markdown("""
## Interpretation boundary

This tiny overfit is only a wiring test. It is not a validation score and cannot support
a claim about transfer quality. The next run should scale the embedding cache only after
reviewing the reconstructed anatomy, checkpoint input behavior, label coverage, patient
identity, and the exact competition metric.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(OUT / "pillar0_small_sample.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")

metadata = {
    "id": "tumis001/rsna-knee-pillar0-small-sample",
    "title": "RSNA Knee Pillar0 Small Sample",
    "code_file": "pillar0_small_sample.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "kernel_sources": [],
    "model_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(OUT)
