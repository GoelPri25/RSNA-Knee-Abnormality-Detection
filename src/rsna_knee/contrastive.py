"""Target-aware supervised contrastive learning utilities for report labels.

The legacy thresholded objective is retained for exact reproduction. The v2
objective uses soft label probabilities, attenuates uncertain pairs, and
balances positive and negative key mass independently of their counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ContrastiveDiagnostics:
    loss: torch.Tensor
    eligible_anchors: int
    positive_pairs: int
    active_targets: int


class TargetMemoryBank:
    """Detached FIFO keys from recent training batches, with report supervision."""

    def __init__(self, capacity: int = 512) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.embeddings: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None
        self.study_ids: list[str] = []

    def __len__(self) -> int:
        return len(self.study_ids)

    def candidates(self, device: torch.device) -> tuple[torch.Tensor | None,
                                                         torch.Tensor | None,
                                                         torch.Tensor | None,
                                                         list[str]]:
        if self.embeddings is None:
            return None, None, None, []
        return (self.embeddings.to(device), self.targets.to(device),
                self.weights.to(device), list(self.study_ids))

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, targets: torch.Tensor,
                weights: torch.Tensor, study_ids: list[str]) -> None:
        if embeddings.ndim != 3 or targets.shape != weights.shape:
            raise ValueError("Expected embeddings [B,T,D] and targets/weights [B,T]")
        if embeddings.shape[:2] != targets.shape or len(study_ids) != len(targets):
            raise ValueError("Memory-bank batch dimensions do not match")
        if not torch.isfinite(embeddings).all() or not torch.isfinite(targets).all():
            raise ValueError("Memory-bank inputs must be finite")
        emb = embeddings.detach().float().cpu()
        tgt = targets.detach().float().cpu()
        wgt = weights.detach().float().cpu()
        if self.embeddings is not None:
            emb = torch.cat([self.embeddings, emb], dim=0)
            tgt = torch.cat([self.targets, tgt], dim=0)
            wgt = torch.cat([self.weights, wgt], dim=0)
        ids = self.study_ids + [str(uid) for uid in study_ids]
        emb, tgt, wgt, ids = (emb[-self.capacity:], tgt[-self.capacity:],
                              wgt[-self.capacity:], ids[-self.capacity:])
        self.embeddings, self.targets, self.weights, self.study_ids = emb, tgt, wgt, ids


def target_aware_contrastive_loss(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    study_ids: list[str],
    bank: TargetMemoryBank,
    *,
    temperature: float = 0.10,
    positive_threshold: float = 0.70,
    negative_threshold: float = 0.20,
) -> ContrastiveDiagnostics:
    """Compute soft-label-weighted supervised contrastive loss per finding.

    For a given finding, confident present studies attract one another and
    confident absent studies act as negatives.  Intermediate soft labels are
    excluded.  Anchors require at least one positive and one negative key,
    preventing a loss from being reported when the batch/queue cannot form a
    meaningful comparison.  Duplicate study IDs in the queue are excluded.
    """
    if embeddings.ndim != 3 or targets.ndim != 2 or weights.shape != targets.shape:
        raise ValueError("Expected embeddings [B,T,D] and targets/weights [B,T]")
    batch, targets_count, _ = embeddings.shape
    if targets.shape != (batch, targets_count) or len(study_ids) != batch:
        raise ValueError("Input dimensions do not agree")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= negative_threshold < positive_threshold <= 1:
        raise ValueError("Require 0 <= negative threshold < positive threshold <= 1")
    if not torch.isfinite(embeddings).all() or not torch.isfinite(targets).all():
        raise ValueError("Contrastive inputs must be finite")

    normalized = F.normalize(embeddings.float(), dim=-1)
    bank_emb, bank_target, bank_weight, bank_ids = bank.candidates(embeddings.device)
    if bank_emb is None:
        all_embeddings = normalized
        all_targets, all_weights = targets.float(), weights.float()
        all_ids = [str(uid) for uid in study_ids]
    else:
        all_embeddings = torch.cat([normalized, F.normalize(bank_emb.float(), dim=-1)], dim=0)
        all_targets = torch.cat([targets.float(), bank_target.float()], dim=0)
        all_weights = torch.cat([weights.float(), bank_weight.float()], dim=0)
        all_ids = [str(uid) for uid in study_ids] + bank_ids

    anchor_values = targets.float()
    anchor_weights = weights.float().clamp_min(0)
    candidate_values = all_targets.float()
    candidate_weights = all_weights.float().clamp_min(0)
    anchor_positive = anchor_values >= positive_threshold       # [B,T]
    anchor_negative = anchor_values <= negative_threshold       # [B,T]
    candidate_positive = candidate_values >= positive_threshold # [K,T]
    candidate_negative = candidate_values <= negative_threshold # [K,T]
    candidate_valid = ((candidate_positive | candidate_negative)
                       & (candidate_weights > 0))                # [K,T]

    # Study IDs are metadata, so build the self-exclusion mask once on CPU and
    # move it to the active device. Everything else stays vectorized on GPU.
    nonself = torch.as_tensor(
        [[anchor_id != candidate_id for candidate_id in all_ids]
         for anchor_id in map(str, study_ids)],
        dtype=torch.bool, device=embeddings.device,
    )  # [B,K]
    same_class = (
        (anchor_positive[:, :, None] & candidate_positive.T[None, :, :])
        | (anchor_negative[:, :, None] & candidate_negative.T[None, :, :])
    )  # [B,T,K]
    different_class = (
        (anchor_positive[:, :, None] & candidate_negative.T[None, :, :])
        | (anchor_negative[:, :, None] & candidate_positive.T[None, :, :])
    )
    candidate_valid_t = candidate_valid.T[None, :, :]
    positive_mask = (same_class & candidate_valid_t
                     & nonself[:, None, :])
    negative_mask = (different_class & candidate_valid_t
                     & nonself[:, None, :])
    valid_mask = positive_mask | negative_mask
    anchor_valid = ((anchor_positive | anchor_negative)
                    & (anchor_weights > 0))
    eligible = (anchor_valid & positive_mask.any(dim=-1)
                & negative_mask.any(dim=-1))                    # [B,T]

    similarities = torch.einsum("btd,ktd->btk", normalized, all_embeddings) / temperature
    log_key_weight = candidate_weights.T.clamp_min(1e-12).log()[None, :, :]
    # For ineligible anchors, add one dummy key to both sums before logsumexp;
    # this avoids undefined all-negative-infinity rows without affecting loss.
    safe_positive = positive_mask.clone()
    safe_valid = valid_mask.clone()
    safe_positive[:, :, 0] |= ~eligible
    safe_valid[:, :, 0] |= ~eligible
    denominator = torch.logsumexp(
        (similarities + log_key_weight).masked_fill(~safe_valid, -torch.inf), dim=-1)
    numerator = torch.logsumexp(
        (similarities + log_key_weight).masked_fill(~safe_positive, -torch.inf), dim=-1)
    per_anchor = torch.where(eligible, denominator - numerator,
                             torch.zeros_like(denominator))
    certainty = (2 * anchor_values - 1).abs()
    anchor_importance = anchor_weights * certainty * eligible
    per_target_denominator = anchor_importance.sum(dim=0)
    active_target = per_target_denominator > 0
    per_target_loss = (per_anchor * anchor_importance).sum(dim=0) / per_target_denominator.clamp_min(1e-12)
    if bool(active_target.any()):
        loss = per_target_loss[active_target].mean()
    else:
        # Preserve a valid autograd path for batches with no usable pairs.
        loss = embeddings.sum() * 0.0
    return ContrastiveDiagnostics(
        loss=loss,
        eligible_anchors=int(eligible.sum().item()),
        positive_pairs=int((positive_mask & eligible[:, :, None]).sum().item()),
        active_targets=int(active_target.sum().item()),
    )


def soft_mass_contrastive_loss(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    study_ids: list[str],
    bank: TargetMemoryBank,
    *,
    temperature: float = 0.10,
) -> ContrastiveDiagnostics:
    """Confidence-aware soft extension of the weighted multi-positive loss.

    ``targets`` are treated as probabilities that a finding is present. For a
    pair of studies, the probability of sharing a state is
    ``p_i*p_j + (1-p_i)*(1-p_j)``; the probability of opposing states is its
    complement. Pair weights are attenuated by both labels' certainty
    (``abs(2*p-1)``) and their supplied validity/class weights. Thus uncertain
    labels contribute little without a brittle present/absent cutoff.

    The loss is ``logsumexp(similarity + log(r_plus + r_minus)) -
    logsumexp(similarity + log(r_plus))``. It preserves the absolute mass of
    positive and negative evidence: a small disagreement probability remains
    a small contribution rather than being normalized into an artificial
    large share of the denominator. For hard 0/1 labels, it reduces to the
    legacy weighted multi-positive objective, without its arbitrary cutoffs.
    Anchors without positive and negative support are skipped. Same-study
    keys are excluded, including stale/repeated memory-bank entries. The loss
    is auxiliary; BCE remains the primary classification objective.

    This formulation assumes soft labels are meaningful probabilities. If
    their calibration is poor, pair weights may be misleading; validate pair
    coverage and performance before using it in a full run.
    """
    if embeddings.ndim != 3 or targets.ndim != 2 or weights.shape != targets.shape:
        raise ValueError("Expected embeddings [B,T,D] and targets/weights [B,T]")
    batch, target_count, _ = embeddings.shape
    if targets.shape != (batch, target_count) or len(study_ids) != batch:
        raise ValueError("Input dimensions do not agree")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if (not torch.isfinite(embeddings).all()
            or not torch.isfinite(targets).all()
            or not torch.isfinite(weights).all()):
        raise ValueError("Contrastive inputs must be finite")
    if bool(((targets < 0) | (targets > 1)).any()):
        raise ValueError("targets must be probabilities in [0, 1]")
    if bool((weights < 0).any()):
        raise ValueError("weights must be nonnegative")

    normalized = F.normalize(embeddings.float(), dim=-1)
    bank_emb, bank_targets, bank_weights, bank_ids = bank.candidates(embeddings.device)
    if bank_emb is not None:
        # The same study may reappear after a prior epoch. Keep only its latest
        # queue entry, and let the current-batch representation supersede any
        # stale queued copy of a study in this batch.
        current_ids = {str(uid) for uid in study_ids}
        latest_index = {uid: index for index, uid in enumerate(bank_ids)}
        keep = [index for index, uid in enumerate(bank_ids)
                if latest_index[uid] == index and uid not in current_ids]
        keep_tensor = torch.as_tensor(keep, dtype=torch.long, device=embeddings.device)
        bank_emb = bank_emb.index_select(0, keep_tensor)
        bank_targets = bank_targets.index_select(0, keep_tensor)
        bank_weights = bank_weights.index_select(0, keep_tensor)
        bank_ids = [bank_ids[index] for index in keep]
    if bank_emb is None:
        all_embeddings = normalized
        all_targets = targets.float()
        all_weights = weights.float()
        all_ids = [str(uid) for uid in study_ids]
    else:
        all_embeddings = torch.cat(
            [normalized, F.normalize(bank_emb.float(), dim=-1)], dim=0)
        all_targets = torch.cat([targets.float(), bank_targets.float()], dim=0)
        all_weights = torch.cat([weights.float(), bank_weights.float()], dim=0)
        all_ids = [str(uid) for uid in study_ids] + bank_ids

    anchor_p = targets.float()                         # [B,T]
    candidate_p = all_targets.float().T                # [T,K]
    anchor_certainty = (2 * anchor_p - 1).abs()
    candidate_certainty = (2 * candidate_p - 1).abs()
    same_probability = (
        anchor_p[:, :, None] * candidate_p[None, :, :]
        + (1 - anchor_p[:, :, None]) * (1 - candidate_p[None, :, :])
    )                                                    # [B,T,K]
    different_probability = 1 - same_probability

    candidate_valid = all_weights.T > 0                 # [T,K]
    pair_valid = candidate_valid[None, :, :]
    nonself = torch.as_tensor(
        [[anchor_id != candidate_id for candidate_id in all_ids]
         for anchor_id in map(str, study_ids)],
        dtype=torch.bool, device=embeddings.device,
    )[:, None, :]                                        # [B,1,K]
    anchor_factor = (anchor_certainty * weights.float().clamp_min(0))[:, :, None]
    candidate_factor = (
        candidate_certainty * all_weights.float().T.clamp_min(0)
    )[None, :, :]
    pair_factor = anchor_factor * candidate_factor * pair_valid * nonself
    positive_weight = same_probability * pair_factor
    negative_weight = different_probability * pair_factor

    positive_total = positive_weight.sum(dim=-1, keepdim=True)
    negative_total = negative_weight.sum(dim=-1, keepdim=True)
    eligible = ((anchor_certainty * weights.float() > 0)
                & (positive_total.squeeze(-1) > 1e-12)
                & (negative_total.squeeze(-1) > 1e-12))       # [B,T]
    active_target = eligible.any(dim=0)
    if not bool(active_target.any()):
        return ContrastiveDiagnostics(
            loss=embeddings.sum() * 0.0,
            eligible_anchors=0,
            positive_pairs=0,
            active_targets=0,
        )

    # Select eligible rows before logsumexp; this avoids all-masked rows
    # producing -inf/NaN and keeps their gradients out of the objective.
    similarities = torch.einsum("btd,ktd->btk", normalized, all_embeddings) / temperature
    selected_similarity = similarities[eligible]
    selected_batch, selected_target = eligible.nonzero(as_tuple=True)
    selected_positive = positive_weight[eligible]
    selected_total = (positive_weight + negative_weight)[eligible]
    log_positive = torch.where(
        selected_positive > 0,
        selected_positive.log(),
        torch.full_like(selected_positive, -torch.inf),
    )
    log_total = torch.where(
        selected_total > 0,
        selected_total.log(),
        torch.full_like(selected_total, -torch.inf),
    )
    row_loss = (
        torch.logsumexp(selected_similarity + log_total, dim=-1)
        - torch.logsumexp(selected_similarity + log_positive, dim=-1)
    )

    anchor_importance = (anchor_certainty * weights.float()).clamp_min(0) * eligible
    target_losses = []
    for target_index in active_target.nonzero(as_tuple=False).flatten().tolist():
        target_rows = selected_target == target_index
        row_importance = anchor_importance[
            selected_batch[target_rows], selected_target[target_rows]]
        target_losses.append(
            (row_loss[target_rows] * row_importance).sum()
            / row_importance.sum().clamp_min(1e-12)
        )
    loss = torch.stack(target_losses).mean()
    return ContrastiveDiagnostics(
        loss=loss,
        eligible_anchors=int(eligible.sum().item()),
        positive_pairs=int(((positive_weight > 0) & eligible[:, :, None]).sum().item()),
        active_targets=int(active_target.sum().item()),
    )
