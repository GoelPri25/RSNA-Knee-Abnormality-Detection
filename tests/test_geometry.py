import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from rsna_knee.preprocessing.dicom import GeometryError, geometry, reconstruct, visualize
from rsna_knee.data.audit import audit


def headers():
    result = []
    for z in (4, 0, 2):
        d = Dataset()
        d.Modality = 'MR'
        d.StudyInstanceUID = '1.2.3'
        d.SeriesInstanceUID = '1.2.3.4'
        d.SOPInstanceUID = generate_uid()
        d.Rows, d.Columns = 5, 7
        d.ImagePositionPatient = [0, 0, z]
        d.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        d.PixelSpacing = [0.7, 0.4]
        d.InstanceNumber = 100 - z
        result.append(d)
    return result


def test_physical_order_and_affine():
    order, meta = geometry(headers())
    assert order.tolist() == [1, 2, 0]
    np.testing.assert_allclose(np.array(meta['affine_array_to_lps']) @ [1, 2, 3, 1], [1.2, 1.4, 2, 1])


@pytest.mark.parametrize('fault', ['duplicate', 'orientation', 'gap', 'missing', 'shear', 'echo', 'spacing', 'frames'])
def test_rejects_unsafe_geometry(fault):
    ds = headers()
    if fault == 'duplicate': ds[0].ImagePositionPatient = ds[1].ImagePositionPatient
    if fault == 'orientation': ds[0].ImageOrientationPatient = [0, 1, 0, 1, 0, 0]
    if fault == 'gap': ds[0].ImagePositionPatient = [0, 0, 8]
    if fault == 'missing': del ds[0].ImagePositionPatient
    if fault == 'shear': ds[0].ImagePositionPatient = [1, 0, 4]
    if fault == 'echo': ds[0].EchoTime = 40
    if fault == 'spacing': ds[0].PixelSpacing = [-1, 0.4]
    if fault == 'frames': ds[0].NumberOfFrames = 2
    with pytest.raises(GeometryError): geometry(ds)


def test_oblique_affine():
    ds = headers()
    c = np.sqrt(0.5)
    for d, z in zip(ds, (4, 0, 2)):
        d.ImageOrientationPatient = [c, c, 0, 0, 0, 1]
        d.ImagePositionPatient = [c*z, -c*z, 0]
    _, meta = geometry(ds)
    assert meta['plane'] == 'oblique'
    np.testing.assert_allclose(np.array(meta['affine_array_to_lps']) @ [1, 0, 0, 1], [2*c, -2*c, 0, 1])


def test_dicom_roundtrip_and_audit(tmp_path):
    series = tmp_path / 'train' / 'study' / 'series'
    series.mkdir(parents=True)
    for i, h in enumerate(headers()):
        fm = FileMetaDataset()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.MediaStorageSOPClassUID = MRImageStorage
        fm.MediaStorageSOPInstanceUID = h.SOPInstanceUID
        d = FileDataset(None, {}, file_meta=fm, preamble=b'\0'*128)
        d.update(h)
        d.SOPClassUID = MRImageStorage
        d.SamplesPerPixel = 1
        d.PhotometricInterpretation = 'MONOCHROME2'
        d.BitsAllocated = d.BitsStored = 16
        d.HighBit = 15
        d.PixelRepresentation = 0
        d.RescaleSlope, d.RescaleIntercept = 2, -3
        d.PixelData = np.full((5, 7), int(h.ImagePositionPatient[2]), dtype=np.uint16).tobytes()
        d.save_as(series / f'{i}.dcm', enforce_file_format=True)
    vol, meta = reconstruct(sorted(series.glob('*.dcm')))
    np.testing.assert_array_equal(vol[:, 0, 0], [-3, 1, 5])
    assert vol.shape == (3, 5, 7)
    visualize(vol, meta, tmp_path / 'synthetic.png')
    assert (tmp_path / 'synthetic.png').stat().st_size > 1000
    (tmp_path / 'labels.csv').write_text('study,label\ns1,1\ns2,\ns1,1\n')
    config = {'seed': 42, 'sample_series': 2, 'dicom_suffixes': ['.dcm'],
              'labels_csv': 'labels.csv', 'study_column': 'study', 'target_columns': ['label']}
    result = audit(tmp_path, config)
    assert result['sampled_series'][0]['status'] == 'valid_geometry'
    assert result['tables']['labels.csv']['missing']['label'] == 1
    assert result['tables']['labels.csv']['duplicate_rows'] == 1
    assert audit(tmp_path, config) == result
