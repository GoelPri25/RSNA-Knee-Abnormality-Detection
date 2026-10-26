import torch

from rsna_knee.contrastive import (
    TargetMemoryBank,
    soft_mass_contrastive_loss,
    target_aware_contrastive_loss,
)
from scripts.train_steven_contrastive import ContrastiveProjection


def test_projection_produces_unit_vectors_per_target():
    projection = ContrastiveProjection(input_dim=16, targets=12, projection_dim=8)
    study = torch.randn(5, 16, requires_grad=True)

    projected = projection(study)

    assert projected.shape == (5, 12, 8)
    assert torch.allclose(projected.norm(dim=-1), torch.ones(5, 12), atol=1e-6)
    projected[:, 3].sum().backward()
    assert study.grad is not None


def test_target_aware_loss_is_finite_and_backpropagates():
    embeddings = torch.tensor(
        [[[1.0, 0.0]], [[0.9, 0.1]], [[-1.0, 0.0]], [[-0.9, -0.1]]],
        requires_grad=True,
    )
    labels = torch.tensor([[0.9], [0.8], [0.1], [0.05]])
    weights = torch.ones_like(labels)

    result = target_aware_contrastive_loss(
        embeddings, labels, weights, ["a", "b", "c", "d"], TargetMemoryBank()
    )

    assert result.active_targets == 1
    assert result.eligible_anchors == 4
    assert result.positive_pairs == 4
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_ambiguous_soft_labels_are_ignored_without_contrastive_pairs():
    embeddings = torch.randn(4, 2, 8, requires_grad=True)
    labels = torch.full((4, 2), 0.5)
    weights = torch.ones_like(labels)

    result = target_aware_contrastive_loss(
        embeddings, labels, weights, list("abcd"), TargetMemoryBank()
    )

    assert result.active_targets == 0
    assert result.eligible_anchors == 0
    assert result.loss.item() == 0.0
    result.loss.backward()
    assert embeddings.grad is not None


def test_memory_bank_fifo_and_same_study_exclusion():
    bank = TargetMemoryBank(capacity=3)
    first = torch.randn(2, 1, 4)
    first_labels = torch.tensor([[0.9], [0.1]])
    bank.enqueue(first, first_labels, torch.ones_like(first_labels), ["a", "b"])
    bank.enqueue(torch.randn(2, 1, 4), first_labels, torch.ones_like(first_labels), ["c", "d"])
    assert len(bank) == 3
    assert bank.study_ids == ["b", "c", "d"]

    # The current anchor is also present in memory. It cannot count itself as
    # a positive key; the other same-class study still makes a valid pair.
    anchors = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[0.9], [0.1]])
    weights = torch.ones_like(labels)
    bank.enqueue(anchors.detach(), labels, weights, ["a", "e"])
    result = target_aware_contrastive_loss(
        anchors, labels, weights, ["a", "e"], bank
    )
    assert torch.isfinite(result.loss)
    assert result.eligible_anchors > 0


def test_missing_label_weight_masks_anchor_and_key():
    embeddings = torch.randn(4, 1, 8, requires_grad=True)
    labels = torch.tensor([[0.9], [0.8], [0.1], [0.05]])
    weights = torch.tensor([[0.0], [1.0], [1.0], [1.0]])

    result = target_aware_contrastive_loss(
        embeddings, labels, weights, list("abcd"), TargetMemoryBank()
    )

    # The masked positive is neither an anchor nor a key, so only the two
    # confidently absent anchors have another absent positive and a present
    # negative available.
    assert result.eligible_anchors == 2
    assert torch.isfinite(result.loss)


def test_soft_mass_loss_handles_soft_and_ambiguous_labels_with_gradients():
    embeddings = torch.tensor(
        [[[1.0, 0.0]], [[0.9, 0.1]], [[-1.0, 0.0]], [[-0.9, -0.1]],
         [[0.0, 1.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[0.95], [0.75], [0.08], [0.22], [0.50]])
    weights = torch.ones_like(labels)
    result = soft_mass_contrastive_loss(
        embeddings, labels, weights, list("abcde"), TargetMemoryBank()
    )

    assert result.active_targets == 1
    assert result.eligible_anchors == 4
    assert result.positive_pairs > 0
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    # The p=.5 study has no reliable pair or anchor contribution.
    assert torch.allclose(embeddings.grad[4], torch.zeros_like(embeddings.grad[4]))


def test_soft_mass_loss_matches_legacy_on_hard_labels():
    embeddings = torch.tensor(
        [[[1.0, 0.0]], [[0.9, 0.1]], [[-1.0, 0.0]], [[-0.9, -0.1]]]
    )
    labels = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    weights = torch.ones_like(labels)
    ids = list("abcd")
    old = target_aware_contrastive_loss(
        embeddings, labels, weights, ids, TargetMemoryBank()
    )
    revised = soft_mass_contrastive_loss(
        embeddings, labels, weights, ids, TargetMemoryBank()
    )

    assert torch.allclose(old.loss, revised.loss, atol=1e-6)


def test_soft_mass_loss_does_not_promote_tiny_disagreement_to_half_mass():
    embeddings = torch.ones(4, 1, 8)
    labels = torch.full((4, 1), 0.99)
    weights = torch.ones_like(labels)
    result = soft_mass_contrastive_loss(
        embeddings, labels, weights, list("abcd"), TargetMemoryBank()
    )

    expected = -torch.log(torch.tensor(0.99**2 + 0.01**2))
    assert result.eligible_anchors == 4
    assert torch.allclose(result.loss, expected, atol=1e-6)
    assert result.loss.item() < 0.05


def test_soft_mass_loss_excludes_current_studies_from_queue():
    embeddings = torch.tensor(
        [[[1.0, 0.0]], [[0.9, 0.1]], [[-1.0, 0.0]], [[-0.9, -0.1]]]
    )
    labels = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    weights = torch.ones_like(labels)
    ids = list("abcd")
    baseline = soft_mass_contrastive_loss(
        embeddings, labels, weights, ids, TargetMemoryBank()
    )
    bank = TargetMemoryBank(capacity=8)
    bank.enqueue(embeddings, labels, weights, ids)
    queued = soft_mass_contrastive_loss(embeddings, labels, weights, ids, bank)

    assert torch.allclose(baseline.loss, queued.loss, atol=1e-6)
