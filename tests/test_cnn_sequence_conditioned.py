from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


SCRIPT = (Path(__file__).resolve().parents[1]
          / "kaggle/experiment_kernels/hpo_v1_sequence_conditioned/source/skanee/scripts/train_cnn_sequence_conditioned.py")
SPEC = importlib.util.spec_from_file_location("train_cnn_sequence_conditioned", SCRIPT)
TRAIN = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.path.insert(0, str(SCRIPT.parents[1]))
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)
from src.images.dataset import Study


def fake_torchvision() -> None:
    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.fc = torch.nn.Linear(512, 1000)
            self.projection = torch.nn.Linear(6, 512)

        def forward(self, image):
            return self.projection(image.mean(dim=(-1, -2)))

    torchvision = types.ModuleType("torchvision")
    torchvision.models = types.SimpleNamespace(resnet34=lambda weights=None: FakeBackbone())
    sys.modules["torchvision"] = torchvision


class MemoryCache:
    def __init__(self, study, manifest):
        self._study = study
        self.manifest = {study.study_id: manifest}

    def study(self, study_id, labels):
        return self._study


def test_acquisition_categories_accept_normalized_metadata_and_unknowns():
    assert TRAIN.sequence_category("Sagittal", "True") == 1
    assert TRAIN.sequence_category("coronal", "false") == 2
    assert TRAIN.sequence_category("AXIAL", "1") == 5
    assert TRAIN.sequence_category("transverse", "yes") == 5
    assert TRAIN.sequence_category("oblique", "true") == 6
    assert TRAIN.sequence_category("sagittal", "unspecified") == 6


def test_zero_initialized_correction_preserves_warm_start_logits():
    fake_torchvision()
    torch.manual_seed(7)
    model = TRAIN.SliceModel("resnet34", False, 0.0, targets=12).eval()
    slices = torch.randn(5, 6, 16, 16)
    metadata = torch.eye(7)[:5]
    owner = torch.zeros(5, dtype=torch.long)
    with torch.no_grad():
        encoded = model.backbone(slices)
        baseline_window_logits = model.head(encoded)
        conditioned_logits = model(slices, metadata, owner, studies=1)
        baseline_logits = TRAIN.target_topk_mean(
            baseline_window_logits, owner, 1, model.top_k_fraction)
    assert torch.count_nonzero(model.sequence_correction.weight) == 0
    assert torch.allclose(conditioned_logits, baseline_logits, atol=1e-7)


def test_series_metadata_is_repeated_for_its_physical_slice_windows():
    slot_ids = np.repeat(np.arange(3, dtype=np.int16), 12)
    pixels = np.full((36, 16, 16), 127, dtype=np.uint8)
    study = Study("study", pixels, np.full_like(pixels, 255), slot_ids,
                  np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32))
    manifest = [
        {"slot": 0, "count": 12, "plane": "Sagittal", "fluid_sensitive": "True"},
        {"slot": 1, "count": 12, "plane": "Coronal", "fluid_sensitive": "False"},
        {"slot": 2, "count": 12, "plane": "Axial", "fluid_sensitive": "True"},
    ]
    labels = {"study": (study.target, study.weight)}
    batch, sequence, owner, *_ = TRAIN.prepare_batch(
        MemoryCache(study, manifest), labels, ["study"], None, None,
        torch.device("cpu"))

    assert tuple(batch.shape) == (36, 6, 16, 16)
    assert tuple(sequence.shape) == (36, 7)
    assert torch.equal(owner, torch.zeros(36, dtype=torch.long))
    assert torch.equal(sequence[:12].argmax(dim=1), torch.ones(12, dtype=torch.long))
    assert torch.equal(sequence[12:24].argmax(dim=1), torch.full((12,), 2))
    assert torch.equal(sequence[24:].argmax(dim=1), torch.full((12,), 5))
    assert torch.equal(sequence.sum(dim=1), torch.ones(36))
