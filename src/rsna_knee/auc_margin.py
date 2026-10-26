"""Masked class-conditional AUC margin objective for weak multi-label supervision.

This is an adaptation of Yuan et al. (ICCV 2021), not the unmodified LibAUC
binary-label loss. Cohort rates include unlabelled cells, so uniform minibatches
estimate conditional moments without requiring both classes in every batch.
"""

import torch


class MaskedAUCMargin(torch.nn.Module):
    def __init__(self, positive_rate, negative_rate, margin=0.3):
        super().__init__()
        pos = torch.as_tensor(positive_rate, dtype=torch.float32)
        neg = torch.as_tensor(negative_rate, dtype=torch.float32)
        if pos.shape != neg.shape or pos.ndim != 1 or (pos <= 0).any() or (neg <= 0).any():
            raise ValueError("Each target needs positive and negative cohort support")
        if margin <= 0:
            raise ValueError("Margin must be positive")
        self.register_buffer("positive_rate", pos)
        self.register_buffer("negative_rate", neg)
        self.register_buffer("alpha", torch.full_like(pos, float(margin)))
        self.a = torch.nn.Parameter(torch.full_like(pos, 0.5))
        self.b = torch.nn.Parameter(torch.full_like(pos, 0.5))
        self.margin = float(margin)

    def forward(self, logits, evidence):
        if logits.ndim != 2 or logits.shape != evidence.shape:
            raise ValueError("Expected aligned [studies, targets] logits and evidence")
        scores = logits.float().sigmoid()
        pos = (evidence == 1).float() / self.positive_rate
        neg = (evidence == -1).float() / self.negative_rate
        positive_mean = (scores * pos).mean(0)
        negative_mean = (scores * neg).mean(0)
        variance = (pos * (scores - self.a).square()
                    + neg * (scores - self.b).square()).mean(0)
        gap = self.margin + negative_mean - positive_mean
        objective = variance + 2 * self.alpha * gap - self.alpha.square()
        return objective.mean(), gap.detach()

    @torch.no_grad()
    def ascend(self, gap, learning_rate):
        # Simultaneous projected dual ascent; gap is from the pre-update model.
        self.alpha.add_(2 * learning_rate * (gap - self.alpha)).clamp_(min=0)


def build_evidence(steven, gemini, steven_mask, gemini_mask, low=0.3, high=0.7):
    """+1 / -1 only for two agreeing, addressed sources; 0 stays BCE-only.

    Masks are availability indicators, not proof of direct textual evidence.
    Gemini must address Synovitis too, preventing Steven-only effusion proxies
    from entering the ranking objective. Sentinel conflicts are excluded.
    """
    import numpy as np
    arrays = [np.asarray(x) for x in (steven, gemini, steven_mask, gemini_mask)]
    st, gm, ms, mg = arrays
    if any(x.shape != st.shape for x in arrays):
        raise ValueError("Source arrays must align")
    usable = (ms == 1) & (mg == 1) & ~np.isclose(st, 0.25)
    usable &= np.isfinite(st) & np.isfinite(gm) & (np.abs(st - gm) <= 0.3)
    result = np.zeros(st.shape, dtype=np.int8)
    result[usable & (st >= high) & (gm >= high)] = 1
    result[usable & (st <= low) & (gm <= low)] = -1
    return result
