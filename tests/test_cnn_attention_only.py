from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


SCRIPT = (Path(__file__).resolve().parents[1]
          / "kaggle/experiment_kernels/hpo_v1_attention_only/source/skanee/scripts/train_cnn_attention_only.py")
SPEC = importlib.util.spec_from_file_location("train_cnn_attention_only", SCRIPT)
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


def test_sampler_matches_uniform_baseline_window_selection():
    seed = 42
    expected = np.sort(np.random.default_rng(seed).choice(48, 24, replace=False))
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
    batch, *_ = TRAIN.prepare_batch(
        MemoryCache(study), labels, ["study"], 24,
        np.random.default_rng(seed), torch.device("cpu"), augmentation)

    expected_slot_counts = np.bincount(slot_ids[expected], minlength=4)
    observed_slot_counts = batch[:, 6, 0, 0].long().bincount(minlength=4).numpy()
    assert tuple(batch.shape) == (24, 8, 16, 16)
    assert np.array_equal(observed_slot_counts, expected_slot_counts)
    expected_positions = (expected % 12) / 11.0
    assert np.allclose(batch[:, 7, 0, 0].numpy(), expected_positions, atol=1e-6)


def test_active_attention_receives_gradients_and_changes_slice_logits():
    fake_torchvision()
    torch.manual_seed(4)
    model = TRAIN.SliceModel("resnet34", False, 0.0, targets=12,
                             attention_residual_scale=0.1)
    slices = torch.randn(24, 8, 16, 16)
    slices[:, 6] = torch.arange(4).repeat_interleave(6)[:, None, None]
    slices[:, 7] = torch.linspace(0, 1, 6).repeat(4)[:, None, None]
    owner = torch.zeros(24, dtype=torch.long)

    logits, ratio = model(slices, owner, studies=1, return_attention_stats=True)
    baseline_logits = model.head(model.backbone(slices))
    baseline_pooled = TRAIN.target_topk_mean(baseline_logits, owner, 1,
                                             model.top_k_fraction)
    assert tuple(logits.shape) == (1, 12)
    assert torch.isfinite(logits).all()
    assert ratio.item() > 0
    assert not torch.allclose(logits, baseline_pooled)

    logits.square().mean().backward()
    grad_norm = model.reader.in_proj_weight.grad.norm()
    assert torch.isfinite(grad_norm)
    assert grad_norm.item() > 0
