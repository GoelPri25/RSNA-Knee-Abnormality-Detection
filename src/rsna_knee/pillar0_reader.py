"""Small target-query reader for cached multiscale Pillar-0 tokens."""

from __future__ import annotations

import math

import torch
from torch import nn


def grid_coordinates(side: int) -> torch.Tensor:
    """Return XYZ cell centres in [-1, 1], flattened in ZYX tensor order."""
    axis = (torch.arange(side, dtype=torch.float32) + 0.5) * (2.0 / side) - 1.0
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((x, y, z), dim=-1).reshape(-1, 3)


def fourier_coordinates(xyz: torch.Tensor) -> torch.Tensor:
    """Fixed low-frequency XYZ features; no learned coordinate scale."""
    frequencies = xyz.new_tensor((1.0, 2.0, 4.0)) * math.pi
    phase = xyz.unsqueeze(-1) * frequencies
    return torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(start_dim=-2)


class Pillar0TargetQueryReader(nn.Module):
    """Read all series tokens in a study with one learned query per target.

    Pillar-0 has already contextualized tokens within each series. This reader
    projects the three scales, adds scale/position/protocol information, and
    uses target queries to select evidence across the study without quadratic
    self-attention over thousands of tokens.
    """

    def __init__(
        self,
        targets: int = 12,
        width: int = 128,
        heads: int = 4,
        dropout: float = 0.15,
        feature_dim: int = 384,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.targets = targets
        self.width = width
        self.heads = heads
        self.head_dim = width // heads

        self.scale_norms = nn.ModuleList(nn.LayerNorm(feature_dim) for _ in range(3))
        self.scale_projections = nn.ModuleList(
            nn.Linear(feature_dim, width) for _ in range(3)
        )
        self.scale_embeddings = nn.Embedding(3, width)
        self.position_projection = nn.Sequential(
            nn.Linear(18, width), nn.GELU(), nn.Linear(width, width)
        )
        self.plane_embedding = nn.Embedding(5, 12, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, 6, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, 6, padding_idx=0)
        self.protocol_projection = nn.Sequential(
            nn.Linear(24, width), nn.GELU(), nn.Linear(width, width)
        )

        self.queries = nn.Parameter(torch.empty(targets, width))
        nn.init.normal_(self.queries, std=0.02)
        self.query_projection = nn.Linear(width, width, bias=False)
        self.key_projection = nn.Linear(width, width, bias=False)
        self.value_projection = nn.Linear(width, width, bias=False)
        self.output_projection = nn.Linear(width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(width)
        self.classifier = nn.Sequential(
            nn.Linear(width, width // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width // 2, 1),
        )

    def _project_scales(self, tokens: torch.Tensor, scale_ids: torch.Tensor) -> torch.Tensor:
        batch, count, _ = tokens.shape
        result = tokens.new_zeros((batch, count, self.width), dtype=torch.float32)
        flat_tokens = tokens.reshape(-1, tokens.shape[-1]).float()
        flat_scales = scale_ids.reshape(-1)
        flat_result = result.reshape(-1, self.width)
        for scale in range(3):
            selected = flat_scales == scale
            if selected.any():
                values = self.scale_norms[scale](flat_tokens[selected])
                # Linear layers return FP16/BF16 under autocast, while the
                # mixed-scale staging buffer is deliberately FP32.
                flat_result[selected] = self.scale_projections[scale](values).float()
        result = result + self.scale_embeddings(scale_ids)
        return result

    def forward(
        self,
        tokens: torch.Tensor,
        scale_ids: torch.Tensor,
        xyz: torch.Tensor,
        protocol_ids: torch.Tensor,
        support: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return [batch, targets] logits.

        ``support`` is a fractional coverage map in [0, 1]. Its log is an
        additive attention prior, clamped at 0.05 so contextual tokens at
        unsupported locations remain available at reduced weight.
        ``padding_mask`` is True only for batch padding, not image padding.
        """
        x = self._project_scales(tokens, scale_ids)
        pos = fourier_coordinates(xyz.float())
        p_plane = self.plane_embedding(protocol_ids[..., 0].long())
        p_fluid = self.fluid_embedding(protocol_ids[..., 1].long())
        p_fat = self.fat_embedding(protocol_ids[..., 2].long())
        protocol = self.protocol_projection(torch.cat((p_plane, p_fluid, p_fat), dim=-1))
        x = x + self.position_projection(pos) + protocol
        x = self.dropout(x)

        batch, count, _ = x.shape
        q = self.query_projection(self.queries).view(
            self.targets, self.heads, self.head_dim
        )
        q = q.unsqueeze(0).expand(batch, -1, -1, -1)
        k = self.key_projection(x).view(batch, count, self.heads, self.head_dim)
        v = self.value_projection(x).view(batch, count, self.heads, self.head_dim)

        logits = torch.einsum("bqhd,bnhd->bqhn", q, k).float()
        logits = logits * (self.head_dim ** -0.5)
        coverage_bias = support.float().clamp(min=0.05, max=1.0).log()
        logits = logits + coverage_bias[:, None, None, :]
        logits = logits.masked_fill(padding_mask[:, None, None, :], -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        context = torch.einsum("bqhn,bnhd->bqhd", weights, v.float())
        context = context.reshape(batch, self.targets, self.width)
        target_features = self.attention_norm(
            self.queries.unsqueeze(0) + self.output_projection(context)
        )
        target_features = self.output_norm(self.dropout(target_features))
        return self.classifier(target_features).squeeze(-1)


class Pillar0RegionalTargetQueryReader(nn.Module):
    """Two-stage target-query reader over scale-aligned spatial regions.

    Input is grouped as ``[batch, regions, 129 tokens, feature_dim]``. Each
    region contains 64 fine 8^3 tokens, 64 native 8^3 tokens, and its aligned
    coarse 2^3 token. Stage one reads local multiscale evidence per region;
    stage two reads those region summaries across every series in the study.
    """

    def __init__(
        self,
        targets: int = 12,
        width: int = 128,
        heads: int = 4,
        dropout: float = 0.15,
        feature_dim: int = 384,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.targets = targets
        self.width = width
        self.heads = heads
        self.head_dim = width // heads

        self.scale_norms = nn.ModuleList(nn.LayerNorm(feature_dim) for _ in range(3))
        self.scale_projections = nn.ModuleList(nn.Linear(feature_dim, width) for _ in range(3))
        self.scale_embeddings = nn.Embedding(3, width)
        self.position_projection = nn.Sequential(
            nn.Linear(18, width), nn.GELU(), nn.Linear(width, width)
        )
        self.plane_embedding = nn.Embedding(5, 12, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, 6, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, 6, padding_idx=0)
        self.protocol_projection = nn.Sequential(
            nn.Linear(24, width), nn.GELU(), nn.Linear(width, width)
        )

        self.local_queries = nn.Parameter(torch.empty(targets, width))
        self.study_queries = nn.Parameter(torch.empty(targets, width))
        nn.init.normal_(self.local_queries, std=0.02)
        nn.init.normal_(self.study_queries, std=0.02)

        self.local_query_projection = nn.Linear(width, width, bias=False)
        self.local_key_projection = nn.Linear(width, width, bias=False)
        self.local_value_projection = nn.Linear(width, width, bias=False)
        self.local_output_projection = nn.Linear(width, width)
        self.local_attention_norm = nn.LayerNorm(width)
        self.local_output_norm = nn.LayerNorm(width)

        self.study_query_projection = nn.Linear(width, width, bias=False)
        self.study_key_projection = nn.Linear(width, width, bias=False)
        self.study_value_projection = nn.Linear(width, width, bias=False)
        self.study_output_projection = nn.Linear(width, width)
        self.study_attention_norm = nn.LayerNorm(width)
        self.study_output_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(width, width // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width // 2, 1),
        )

    def _embed_tokens(
        self,
        tokens: torch.Tensor,
        scale_ids: torch.Tensor,
        xyz: torch.Tensor,
        protocol_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, regions, count, feature_dim = tokens.shape
        result = tokens.new_zeros((batch, regions, count, self.width), dtype=torch.float32)
        flat_tokens = tokens.reshape(-1, feature_dim).float()
        flat_scales = scale_ids.reshape(-1)
        flat_result = result.reshape(-1, self.width)
        for scale in range(3):
            selected = flat_scales == scale
            if selected.any():
                normalized = self.scale_norms[scale](flat_tokens[selected])
                flat_result[selected] = self.scale_projections[scale](normalized).float()
        result = result + self.scale_embeddings(scale_ids)

        position = fourier_coordinates(xyz.float())
        plane = self.plane_embedding(protocol_ids[..., 0].long())
        fluid = self.fluid_embedding(protocol_ids[..., 1].long())
        fat = self.fat_embedding(protocol_ids[..., 2].long())
        protocol = self.protocol_projection(torch.cat((plane, fluid, fat), dim=-1))
        return self.dropout(result + self.position_projection(position) + protocol)

    def forward(
        self,
        tokens: torch.Tensor,
        scale_ids: torch.Tensor,
        xyz: torch.Tensor,
        protocol_ids: torch.Tensor,
        support: torch.Tensor,
        region_padding_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return study logits and optionally local/global attention weights.

        ``support`` remains a fractional-coverage prior with the same 0.05
        floor as the first reader. It is not a hard validity mask because the
        inputs are already contextualized by the frozen encoder.
        """
        batch, regions, count, _ = tokens.shape
        x = self._embed_tokens(tokens, scale_ids, xyz, protocol_ids)

        local_query = self.local_query_projection(self.local_queries).view(
            self.targets, self.heads, self.head_dim
        )
        local_key = self.local_key_projection(x).view(
            batch, regions, count, self.heads, self.head_dim
        )
        local_value = self.local_value_projection(x).view(
            batch, regions, count, self.heads, self.head_dim
        )
        local_logits = torch.einsum(
            "qhd,bgthd->bgqht", local_query, local_key
        ).float() * (self.head_dim ** -0.5)
        local_logits = local_logits + support.float().clamp(min=0.05, max=1.0).log()[:, :, None, None, :]
        local_logits = local_logits.masked_fill(
            region_padding_mask[:, :, None, None, None], -1.0e4
        )
        local_weights = torch.softmax(local_logits, dim=-1)
        local_context = torch.einsum(
            "bgqht,bgthd->bgqhd", local_weights, local_value.float()
        ).reshape(batch, regions, self.targets, self.width)
        regional = self.local_attention_norm(
            self.local_queries[None, None] + self.local_output_projection(local_context)
        )
        regional = self.local_output_norm(self.dropout(regional))

        study_query = self.study_query_projection(self.study_queries).view(
            self.targets, self.heads, self.head_dim
        )
        study_key = self.study_key_projection(regional).view(
            batch, regions, self.targets, self.heads, self.head_dim
        )
        study_value = self.study_value_projection(regional).view(
            batch, regions, self.targets, self.heads, self.head_dim
        )
        study_logits = torch.einsum(
            "qhd,bgqhd->bqhg", study_query, study_key
        ).float() * (self.head_dim ** -0.5)
        study_logits = study_logits.masked_fill(
            region_padding_mask[:, None, None, :], -1.0e4
        )
        study_weights = torch.softmax(study_logits, dim=-1)
        study_context = torch.einsum(
            "bqhg,bgqhd->bqhd", study_weights, study_value.float()
        ).reshape(batch, self.targets, self.width)
        target_features = self.study_attention_norm(
            self.study_queries.unsqueeze(0) + self.study_output_projection(study_context)
        )
        target_features = self.study_output_norm(self.dropout(target_features))
        logits = self.classifier(target_features).squeeze(-1)
        if return_attention:
            return logits, local_weights, study_weights
        return logits


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
