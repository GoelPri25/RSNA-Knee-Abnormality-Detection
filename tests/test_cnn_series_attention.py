from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


SCRIPT = (Path(__file__).resolve().parents[1]
          / "kaggle/experiment_kernels/hpo_v1_series_attention/source/skanee/scripts/train_cnn_series_attention.py")
SPEC = importlib.util.spec_from_file_location("train_cnn_series_attention", SCRIPT)
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
            self.projection = torch.nn.Linear(8, 512)

        def forward(self, image):
            return self.projection(image.mean(dim=(-1, -2)))

    torchvision = types.ModuleType("torchvision")
    torchvision.models = types.SimpleNamespace(resnet34=lambda weights=None: FakeBackbone())
    sys.modules["torchvision"] = torchvision


class MemoryCache:
    def __init__(self, study):
        self._study = study

    def study(self, study_id, labels):
        return self._study


def test_balanced_sampler_covers_every_series_and_position_bin():
    rng = np.random.default_rng(42)
    slot_ids = np.repeat(np.arange(4, dtype=np.int16), 12)
    image = np.arange(48 * 16 * 16, dtype=np.uint8).reshape(48, 16, 16)
    support = np.full_like(image, 255)
    study = Study("study", image, support, slot_ids,
                  np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32))
    labels = {"study": (study.target, study.weight)}
    augmentation = {
        "crop_area_min": 1.0, "translate_fraction": 0.0, "rotation_degrees": 0.0,
        "intensity_jitter": 0.0, "brightness_shift": 0.0,
        "erase_probability": 0.0, "erase_fraction": 0.25,
    }
    batch, study_ids, *_ = TRAIN.prepare_batch(
        MemoryCache(study), labels, ["study"], 24, rng, torch.device("cpu"), augmentation)

    assert tuple(batch.shape) == (24, 8, 16, 16)
    assert torch.equal(batch[:, 6, 0, 0].long().bincount(minlength=4), torch.tensor([6, 6, 6, 6]))
    for slot in range(4):
        slot_rows = batch[:, 6, 0, 0].long() == slot
        source_positions = (batch[slot_rows, 7, 0, 0] * 11).round().long().tolist()
        assert len(source_positions) == 6
        assert sorted(position // 2 for position in source_positions) == list(range(6))
    assert study_ids.tolist() == [0] * 24


def test_reader_outputs_study_logits_and_starts_with_closed_residual_gate():
    fake_torchvision()
    model = TRAIN.SliceModel("resnet34", False, 0.0, targets=12)
    slices = torch.randn(24, 8, 16, 16)
    slices[:, 6] = torch.arange(4).repeat_interleave(6)[:, None, None]
    slices[:, 7] = torch.linspace(0, 1, 6).repeat(4)[:, None, None]
    owner = torch.zeros(24, dtype=torch.long)
    logits = model(slices, owner, studies=1)
    baseline_logits = model.head(model.backbone(slices))
    expected = TRAIN.target_topk_mean(baseline_logits, owner, 1, model.top_k_fraction)
    assert tuple(logits.shape) == (1, 12)
    assert torch.isfinite(logits).all()
    assert model.reader_gate.item() == 0.0
    assert torch.allclose(logits, expected, atol=1e-6)
