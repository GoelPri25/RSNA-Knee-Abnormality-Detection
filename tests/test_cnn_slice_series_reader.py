import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn


SOURCE = Path(__file__).parents[1] / "kaggle/experiment_kernels/cnn_slice_series_reader/source/skanee"
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location(
    "cnn_slice_series_reader", SOURCE / "scripts/train_cnn_slice_series.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class _TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16, 1000)

    def forward(self, image):
        features = torch.relu(self.conv1(image))
        features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.fc(features)


def test_stratified_slice_budget_keeps_series_and_physical_order():
    study = types.SimpleNamespace(
        slices=np.zeros((48, 8, 8), dtype=np.uint8),
        slot_of_slice=np.repeat(np.arange(4), 12),
    )
    selected = reader.stratified_indices(study, 24, np.random.default_rng(4))
    assert len(selected) == 24
    assert np.all(selected[:-1] < selected[1:])
    np.testing.assert_array_equal(np.bincount(study.slot_of_slice[selected]), [6, 6, 6, 6])


def test_reader_returns_study_logits_and_backpropagates(monkeypatch):
    fake_vision = types.ModuleType("torchvision")
    fake_vision.models = types.SimpleNamespace(resnet34=lambda weights=None: _TinyResNet())
    monkeypatch.setitem(sys.modules, "torchvision", fake_vision)
    config = {
        "model": {"backbone": "resnet34", "pretrained": False,
                  "dropout": 0.1, "top_k_fraction": 0.1},
        "data": {"series_slots": 4, "slices_per_slot": 12},
    }
    model = reader.OrderedSeriesReader(config, targets=12)
    # Two studies with two independent five-slice series apiece.
    image = torch.randn(20, 2, 32, 32)
    study = torch.tensor([0] * 10 + [1] * 10)
    slot = torch.tensor(([0] * 5 + [1] * 5) * 2)
    rank = torch.tensor(([0, 1, 2, 3, 4] * 2) * 2)
    position = torch.tensor(([0, .25, .5, .75, 1] * 2) * 2, dtype=torch.float32)
    plane = torch.tensor([[1, 2, 0, 0], [1, 2, 0, 0]])
    fluid = torch.tensor([[1, 2, 0, 0], [1, 2, 0, 0]])
    logits = model(image, study, slot, rank, position, plane, fluid, studies=2)
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert model.series_query.grad is not None
    assert torch.isfinite(model.series_query.grad).all()
