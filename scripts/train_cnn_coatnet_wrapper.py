#!/usr/bin/env python3
"""Load the audited CNN trainer with an ImageNet-pretrained CoAtNet-0 reader."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
import timm


SOURCE = Path("/root/rsna/source")
TRAINER_PATH = SOURCE / "scripts/train_cnn_baseline.py"
sys.path.insert(0, str(SOURCE))

spec = importlib.util.spec_from_file_location("train_cnn_baseline", TRAINER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load the audited trainer at {TRAINER_PATH}")
trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trainer
spec.loader.exec_module(trainer)


class CoAtNetSliceModel(torch.nn.Module):
    """CoAtNet-0 window encoder with the original support-aware target reader."""

    def __init__(self, backbone: str, pretrained: bool, dropout: float,
                 targets: int, top_k_fraction: float = 0.1) -> None:
        super().__init__()
        expected = "coatnet_0_rw_224.sw_in1k"
        if backbone != expected:
            raise ValueError(f"Expected {expected}, received {backbone}")
        self.backbone = timm.create_model(
            expected, pretrained=pretrained, num_classes=0
        )

        # The 0.918 recipe has three neighbouring image channels and three
        # binary support channels. Preserve the pretrained RGB response and
        # initialize the newly added mask channels to zero, exactly as in the
        # ResNet reader.
        old = self.backbone.stem.conv1
        new = torch.nn.Conv2d(
            6, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, dilation=old.dilation,
            groups=old.groups, bias=old.bias is not None,
            padding_mode=old.padding_mode, device=old.weight.device,
            dtype=old.weight.dtype,
        )
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3].copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.backbone.stem.conv1 = new

        self.head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.backbone.num_features, targets),
        )
        if not 0 < top_k_fraction <= 1:
            raise ValueError("top-k fraction must be in (0, 1]")
        self.top_k_fraction = float(top_k_fraction)

    def forward(self, slices: torch.Tensor, study_of_slice: torch.Tensor,
                studies: int, fixed_windows_per_study: int | None = None,
                fixed_window_valid: torch.Tensor | None = None) -> torch.Tensor:
        # Keep the ResNet baseline's complete cache and augmentation path fixed,
        # then convert only the three intensity channels to CoAtNet's pretrained
        # ImageNet input convention (mean/std 0.5/0.5). The support channels stay
        # in [0, 1]. Inputs are batch-local and do not require gradients.
        image = slices[:, :3]
        image.mul_(trainer.IMAGENET_STD).add_(trainer.IMAGENET_MEAN)
        image.sub_(0.5).div_(0.5)
        window_logits = self.head(self.backbone(slices))
        if fixed_windows_per_study is not None:
            if (window_logits.shape[0] != studies * fixed_windows_per_study
                    or fixed_window_valid is None
                    or fixed_window_valid.numel() != window_logits.shape[0]):
                raise ValueError("Fixed-window inputs have inconsistent shapes")
            grouped = window_logits.reshape(studies, fixed_windows_per_study, -1)
            valid = fixed_window_valid.reshape(studies, fixed_windows_per_study).bool()
            valid_count = valid.sum(dim=1).clamp_min(1)
            k = torch.ceil(valid_count.float() * self.top_k_fraction).long().clamp_min(1)
            max_k = max(1, math.ceil(fixed_windows_per_study * self.top_k_fraction))
            top = grouped.masked_fill(~valid.unsqueeze(-1), -torch.inf).topk(max_k, dim=1).values
            ranks = torch.arange(max_k, device=top.device).view(1, max_k, 1)
            use = ranks < k.view(-1, 1, 1)
            return top.masked_fill(~use, 0).sum(dim=1) / k.float().unsqueeze(-1)
        return trainer.target_topk_mean(
            window_logits, study_of_slice, studies, self.top_k_fraction
        )


trainer.SliceModel = CoAtNetSliceModel

if __name__ == "__main__":
    print(f"CoAtNet library version: {timm.__version__}", flush=True)
    print("Reader: coatnet_0_rw_224.sw_in1k, pretrained ImageNet-1K; six-channel stem initialized with zeroed support-mask weights.", flush=True)
    trainer.main()
