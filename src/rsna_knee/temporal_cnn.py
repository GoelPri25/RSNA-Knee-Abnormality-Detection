"""A slice-preserving temporal CNN reader for frozen MedSigLIP features.

The model applies temporal convolutions independently to each MRI series. It
keeps one contextual vector and one set of target logits per real slice, then
mean-pools logits over the complete study. It never convolves across series
boundaries and masks padded positions from study pooling.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _edge_pad_temporal_input(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Replicate each sequence's endpoints before a valid width-3 convolution.

    ``x`` is ``[batch_of_series, channels, padded_slices]``. The collator uses
    prefix masks, so the last real token is at ``length - 1``. Replicating that
    token avoids teaching the convolution that batch padding is anatomy.
    """
    batch, channels, steps = x.shape
    positions = torch.arange(steps, device=x.device)[None, :]
    valid = positions < lengths[:, None]
    last_index = (lengths - 1).clamp_min(0)[:, None, None].expand(-1, channels, 1)
    last = x.gather(2, last_index)
    last = torch.where((lengths > 0)[:, None, None], last, torch.zeros_like(last))
    first = torch.where((lengths > 0)[:, None, None], x[:, :, :1], torch.zeros_like(x[:, :, :1]))
    x = torch.where(valid[:, None, :], x, last)
    return torch.cat((first, x, last), dim=2)


def _temporal_conv(conv: nn.Conv1d, x: torch.Tensor,
                   lengths: torch.Tensor) -> torch.Tensor:
    """Apply an unpadded width-3 convolution with per-series edge replication."""
    return conv(_edge_pad_temporal_input(x, lengths))


class ResidualTemporalBlock(nn.Module):
    """Pre-normalized depthwise-separable convolution with an identity path."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width, width, kernel_size=3, groups=width, padding=0, bias=False
        )
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Sequence-first LayerNorm, then channels-first temporal convolution.
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = _temporal_conv(self.depthwise, y, lengths)
        y = self.pointwise(F.gelu(y)).transpose(1, 2)
        return F.gelu(residual + self.dropout(y))


class TemporalCNNReader(nn.Module):
    """CNN-inspired reader that predicts from every cached slice embedding.

    Args:
        input_dim: MedSigLIP feature width (1152 for the current cache).
        hidden_dim: Per-slice contextual width after the temporal stem.
        targets: Number of study-level findings.
        plane_count: Number of known imaging planes; one extra index is unknown.
        fluid_states: Number of fluid-sensitive metadata states, including unknown.
        dropout: Drop probability in the residual blocks and slice head.

    Inputs have shapes ``embeddings [B,S,N,D]``, ``positions [B,S,N]``,
    ``plane/fluid [B,S]``, ``slice_mask [B,S,N]`` and ``series_mask [B,S]``.
    Masks must mark a contiguous prefix of real slices in each series.
    """

    def __init__(
        self,
        input_dim: int = 1152,
        hidden_dim: int = 384,
        targets: int = 12,
        plane_count: int = 3,
        fluid_states: int = 3,
        dropout: float = 0.2,
        residual_blocks: int = 2,
        feature_dropout: float = 0.0,
        factorized_stem: bool = False,
        linear_position: bool = False,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, targets, plane_count, fluid_states) <= 0:
            raise ValueError("Model dimensions and metadata cardinalities must be positive")
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be nonnegative")
        if not 0.0 <= feature_dropout < 1.0:
            raise ValueError("feature_dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.targets = targets
        self.feature_norm = nn.LayerNorm(input_dim)
        self.feature_dropout = nn.Dropout(feature_dropout)
        # The compact profile projects each slice independently, then applies
        # temporal convolution at the smaller width. The original profile keeps
        # its wider convolutional stem for exact reproduction.
        self.factorized_stem = factorized_stem
        self.temporal_stem = (
            nn.Linear(input_dim, hidden_dim) if factorized_stem else
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=0)
        )

        # Metadata is a parallel input path, broadcast to each slice and added
        # after the feature stem. Position is normalized within each series.
        self.position = (
            nn.Linear(1, hidden_dim) if linear_position else
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )
        self.plane = nn.Embedding(plane_count + 1, hidden_dim)
        self.fluid = nn.Embedding(fluid_states, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(hidden_dim, dropout) for _ in range(residual_blocks)
        )
        self.slice_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, targets),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        plane: torch.Tensor,
        fluid: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if embeddings.ndim != 4:
            raise ValueError("embeddings must have shape [B,S,N,D]")
        batch, series, steps, feature_dim = embeddings.shape
        if feature_dim != self.input_dim:
            raise ValueError(f"Expected feature width {self.input_dim}; got {feature_dim}")
        expected = (batch, series, steps)
        if positions.shape != expected or slice_mask.shape != expected:
            raise ValueError("positions and slice_mask must have shape [B,S,N]")
        if plane.shape != (batch, series) or fluid.shape != (batch, series):
            raise ValueError("plane and fluid must have shape [B,S]")
        if series_mask.shape != (batch, series):
            raise ValueError("series_mask must have shape [B,S]")

        mask = slice_mask.bool() & series_mask.bool().unsqueeze(-1)
        lengths = mask.sum(dim=-1).reshape(batch * series)
        if bool((mask.sum(dim=(1, 2)) == 0).any()):
            raise ValueError("Every study must contain at least one real slice")

        # Fold only the batch and series axes. Each series remains a distinct
        # sequence throughout every temporal convolution.
        flat_embeddings = embeddings.reshape(batch * series, steps, feature_dim)
        flat_positions = positions.reshape(batch * series, steps, 1)
        flat_mask = mask.reshape(batch * series, steps)
        flat_valid = torch.arange(steps, device=embeddings.device)[None, :] < lengths[:, None]

        normalized = self.feature_dropout(self.feature_norm(flat_embeddings))
        if self.factorized_stem:
            stem = self.temporal_stem(normalized)
        else:
            stem = _temporal_conv(
                self.temporal_stem, normalized.transpose(1, 2), lengths
            ).transpose(1, 2)

        metadata = self.position(flat_positions)
        metadata = metadata + self.plane(plane.reshape(-1))[:, None, :]
        metadata = metadata + self.fluid(fluid.reshape(-1))[:, None, :]
        x = F.gelu(stem + metadata)
        x = torch.where(flat_valid.unsqueeze(-1), x, torch.zeros_like(x))
        for block in self.blocks:
            x = block(x, lengths)
            x = torch.where(flat_valid.unsqueeze(-1), x, torch.zeros_like(x))

        slice_logits = self.slice_head(x).reshape(batch, series, steps, self.targets)
        slice_logits = slice_logits * mask.unsqueeze(-1).to(slice_logits.dtype)
        denominator = mask.sum(dim=(1, 2)).clamp_min(1).unsqueeze(-1)
        study_logits = slice_logits.sum(dim=(1, 2)) / denominator
        return {"logits": study_logits, "slice_logits": slice_logits, "slice_mask": mask}


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated by the optimizer."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
