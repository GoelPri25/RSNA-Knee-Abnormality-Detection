import unittest

import torch
import torch.nn.functional as F

from rsna_knee.temporal_cnn import TemporalCNNReader, count_trainable_parameters


def _batch():
    torch.manual_seed(17)
    embeddings = torch.randn(2, 3, 7, 1152)
    lengths = torch.tensor([[7, 5, 0], [4, 6, 3]])
    slice_mask = torch.arange(7)[None, None, :] < lengths[:, :, None]
    series_mask = lengths > 0
    positions = torch.linspace(0, 1, 7).view(1, 1, 7).expand(2, 3, -1).clone()
    plane = torch.tensor([[0, 1, 3], [2, 0, 1]])
    fluid = torch.tensor([[0, 1, 2], [1, 0, 1]])
    return embeddings, positions, plane, fluid, slice_mask, series_mask


class TemporalCNNReaderTest(unittest.TestCase):
    def test_forward_preserves_slice_logits_and_pools_only_real_slices(self):
        model = TemporalCNNReader(dropout=0.0).eval()
        inputs = _batch()
        output = model(*inputs)

        self.assertEqual(output["slice_logits"].shape, (2, 3, 7, 12))
        self.assertEqual(output["logits"].shape, (2, 12))
        self.assertTrue(torch.equal(
            output["slice_logits"][~inputs[4]],
            torch.zeros_like(output["slice_logits"][~inputs[4]])))
        mask = inputs[4].unsqueeze(-1)
        expected = (output["slice_logits"] * mask).sum((1, 2)) / mask.sum((1, 2))
        torch.testing.assert_close(output["logits"], expected)

    def test_padded_embedding_values_do_not_change_study_logits(self):
        model = TemporalCNNReader(dropout=0.0).eval()
        inputs = list(_batch())
        original = model(*inputs)["logits"]
        invalid = ~inputs[4]
        inputs[0][invalid] = torch.randn_like(inputs[0][invalid]) * 100
        changed = model(*inputs)["logits"]
        torch.testing.assert_close(original, changed, rtol=1e-5, atol=1e-6)

    def test_bce_backward_produces_finite_gradients(self):
        model = TemporalCNNReader(dropout=0.0)
        output = model(*_batch())
        targets = torch.randint(0, 2, output["logits"].shape).float()
        loss = F.binary_cross_entropy_with_logits(output["logits"], targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(count_trainable_parameters(model), 0)

    def test_single_slice_series_is_supported(self):
        model = TemporalCNNReader(dropout=0.0).eval()
        embeddings = torch.randn(1, 1, 1, 1152)
        positions = torch.zeros(1, 1, 1)
        plane = torch.tensor([[0]])
        fluid = torch.tensor([[2]])
        slice_mask = torch.ones(1, 1, 1, dtype=torch.bool)
        series_mask = torch.ones(1, 1, dtype=torch.bool)
        output = model(embeddings, positions, plane, fluid, slice_mask, series_mask)
        self.assertEqual(output["logits"].shape, (1, 12))
        self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_compact_profile_preserves_slice_outputs_with_fewer_parameters(self):
        original = TemporalCNNReader(dropout=0.0)
        compact = TemporalCNNReader(
            hidden_dim=128, residual_blocks=1, dropout=0.35,
            feature_dropout=0.10, factorized_stem=True, linear_position=True,
        ).eval()
        inputs = _batch()
        output = compact(*inputs)

        self.assertEqual(output["slice_logits"].shape, (2, 3, 7, 12))
        self.assertTrue(torch.isfinite(output["logits"]).all())
        self.assertLess(count_trainable_parameters(compact),
                        count_trainable_parameters(original))
        output["logits"].sum().backward()
        gradients = [p.grad for p in compact.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
