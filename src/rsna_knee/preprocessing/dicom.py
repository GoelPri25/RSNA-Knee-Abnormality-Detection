"""Classic single-frame MR reconstruction. Array axes are (slice, row, column)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pydicom


class GeometryError(ValueError):
    """Unsupported or inconsistent geometry; never fall back to filenames."""


def geometry(datasets):
    if len(datasets) < 2:
        raise GeometryError('At least two slices required to establish physical spacing')
    required = ('ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing',
                'Rows', 'Columns', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID')
    for d in datasets:
        if any(not hasattr(d, key) for key in required):
            raise GeometryError('Missing required geometry or UID fields')
        if getattr(d, 'Modality', '') != 'MR' or int(getattr(d, 'NumberOfFrames', 1)) != 1:
            raise GeometryError('Only single-frame MR is supported; separate enhanced MR explicitly')
    for key in ('StudyInstanceUID', 'SeriesInstanceUID', 'Rows', 'Columns'):
        if len({str(getattr(d, key)) for d in datasets}) != 1:
            raise GeometryError(f'Inconsistent {key}')
    if len({str(d.SOPInstanceUID) for d in datasets}) != len(datasets):
        raise GeometryError('Duplicate SOPInstanceUID')
    iop = np.asarray(datasets[0].ImageOrientationPatient, dtype=float)
    spacing = np.asarray(datasets[0].PixelSpacing, dtype=float)
    positions = np.asarray([d.ImagePositionPatient for d in datasets], dtype=float)
    if iop.shape != (6,) or spacing.shape != (2,) or positions.shape != (len(datasets), 3):
        raise GeometryError('Invalid geometry dimensions')
    if not all(np.isfinite(x).all() for x in (iop, spacing, positions)) or (spacing <= 0).any():
        raise GeometryError('Nonfinite geometry or nonpositive pixel spacing')
    col_direction, row_direction = iop[:3], iop[3:]
    if not np.allclose([np.linalg.norm(col_direction), np.linalg.norm(row_direction),
                        np.dot(col_direction, row_direction)], [1, 1, 0], atol=1e-4):
        raise GeometryError('Orientation cosines are not orthonormal')
    for d in datasets:
        if not np.allclose(d.ImageOrientationPatient, iop, atol=1e-4, rtol=0):
            raise GeometryError('Varying orientation (localizer or mixed acquisition)')
        if not np.allclose(d.PixelSpacing, spacing, atol=1e-4, rtol=0):
            raise GeometryError('Varying pixel spacing')
    for key in ('EchoTime', 'TemporalPositionIdentifier', 'AcquisitionNumber'):
        if len({str(getattr(d, key, '')) for d in datasets}) > 1:
            raise GeometryError(f'Mixed acquisition: {key}')
    normal = np.cross(col_direction, row_direction)
    order = np.argsort(positions @ normal, kind='stable')
    positions = positions[order]
    gaps = np.diff(positions @ normal)
    dz = float(np.median(gaps))
    if (gaps < 1e-3).any():
        raise GeometryError('Repeated slice positions; split echoes/timepoints explicitly')
    if not np.allclose(gaps, dz, atol=0.05, rtol=0.01):
        raise GeometryError('Irregular slice spacing or missing slices')
    if not np.allclose(np.diff(positions, axis=0), normal * dz, atol=0.05, rtol=0.01):
        raise GeometryError('In-plane slice displacement; unsupported sheared volume')
    affine = np.eye(4)
    affine[:3, 0] = normal * dz
    affine[:3, 1] = row_direction * spacing[0]
    affine[:3, 2] = col_direction * spacing[1]
    affine[:3, 3] = positions[0]
    plane = ['sagittal', 'coronal', 'axial'][int(np.argmax(np.abs(normal)))]
    if np.max(np.abs(normal)) < 0.95:
        plane = 'oblique'
    metadata = {
        'array_axes': ['slice', 'row', 'column'], 'world_coordinates': 'DICOM LPS (mm)',
        'affine_array_to_lps': affine.tolist(), 'spacing_mm': [dz, *spacing.tolist()],
        'normal_lps': normal.tolist(), 'plane': plane, 'slice_count': len(datasets),
        'positions_lps': positions.tolist(), 'orientation_patient': iop.tolist(),
        'study_uid': str(datasets[0].StudyInstanceUID),
        'series_uid': str(datasets[0].SeriesInstanceUID),
        'series_description': str(getattr(datasets[0], 'SeriesDescription', '')),
        'slice_thickness': str(getattr(datasets[0], 'SliceThickness', '')),
        'spacing_between_slices': str(getattr(datasets[0], 'SpacingBetweenSlices', '')),
        'transformations': ['sort by IPP dot cross(IOP[:3], IOP[3:])',
                            'apply per-slice RescaleSlope and RescaleIntercept'],
    }
    return order, metadata


def reconstruct(paths):
    paths = list(map(Path, paths))
    headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths]
    order, metadata = geometry(headers)
    slices = []
    for index in order:
        d = pydicom.dcmread(paths[index])
        if int(getattr(d, 'SamplesPerPixel', 1)) != 1 or getattr(d, 'PhotometricInterpretation', '') != 'MONOCHROME2':
            raise GeometryError('Only MONOCHROME2 grayscale MR is supported')
        if hasattr(d, 'ModalityLUTSequence') or hasattr(d, 'RealWorldValueMappingSequence'):
            raise GeometryError('Explicit LUT/real-world mapping requires a reviewed conversion')
        pixels = d.pixel_array.astype(np.float32)
        if pixels.shape != (int(d.Rows), int(d.Columns)):
            raise GeometryError('Pixel shape differs from header')
        if hasattr(d, 'PixelPaddingValue'):
            low = min(float(d.PixelPaddingValue), float(getattr(d, 'PixelPaddingRangeLimit', d.PixelPaddingValue)))
            high = max(float(d.PixelPaddingValue), float(getattr(d, 'PixelPaddingRangeLimit', d.PixelPaddingValue)))
            pixels[(pixels >= low) & (pixels <= high)] = np.nan
        pixels = pixels * float(getattr(d, 'RescaleSlope', 1)) + float(getattr(d, 'RescaleIntercept', 0))
        if not np.isfinite(pixels).any():
            raise GeometryError('No finite non-padding pixels in slice')
        slices.append(pixels)
    metadata['ordered_files'] = [str(paths[i]) for i in order]
    metadata['padding_policy'] = 'DICOM padding becomes NaN; excluded from display percentiles'
    return np.stack(slices), metadata


def visualize(volume, metadata, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    finite = volume[np.isfinite(volume)]
    lo, hi = np.percentile(finite, [1, 99])
    dz, dy, dx = metadata['spacing_mm']
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, index in zip(axes[0], np.linspace(0, len(volume) - 1, 3).astype(int)):
        ax.imshow(volume[index], cmap='gray', vmin=lo, vmax=hi, aspect=dy / dx)
        ax.set_title(f'Slice {index}; LPS {np.round(metadata["positions_lps"][index], 1)}')
        ax.set_xlabel('column'); ax.set_ylabel('row')
    views = [(volume[:, volume.shape[1] // 2, :], dz / dx, 'column'),
             (volume[:, :, volume.shape[2] // 2], dz / dy, 'row')]
    for ax, (pixels, aspect, xlabel) in zip(axes[1], views):
        ax.imshow(pixels, cmap='gray', vmin=lo, vmax=hi, aspect=aspect)
        ax.set_xlabel(xlabel); ax.set_ylabel('physically ordered slice')
    axes[1, 2].axis('off')
    axes[1, 2].text(0, 0.5, f'{metadata["plane"]}\nSpacing z,y,x: {dz:g}, {dy:g}, {dx:g} mm\n'
                           'Native acquisition axes; not canonical anatomy.\nDisplay only: 1–99% intensity window.\nNo resampling, cropping or normalization.', wrap=True)
    fig.tight_layout(); fig.savefig(output, dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('series', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(p for p in args.series.iterdir() if p.is_file())
    volume, metadata = reconstruct(paths)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'volume.npy', volume)
    (args.output / 'geometry.json').write_text(json.dumps(metadata, indent=2))
    visualize(volume, metadata, args.output / 'views.png')


if __name__ == '__main__':
    main()
