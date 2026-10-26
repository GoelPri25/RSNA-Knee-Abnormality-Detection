import unittest

import torch
import torch.nn.functional as F

from rsna_knee.pillar0_reader import Pillar0TargetQueryReader, grid_coordinates


def _inputs():
    torch.manual_seed(29)
    tokens = torch.randn(2, 9, 384, dtype=torch.float16)
    scale_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 2, 2]] * 2)
    xyz = grid_coordinates(2).repeat(3, 1)[:18].reshape(2, 9, 3).to(torch.float16)
    protocol = torch.tensor(
        [[[1, 2, 1]] * 7 + [[0, 0, 0]] * 2] * 2, dtype=torch.long
    )
    support = torch.tensor([[1, 0.5, 1, 0, 1, 1, 1, 1, 1]] * 2, dtype=torch.float16)
    padding = torch.tensor([[False] * 7 + [True] * 2] * 2)
    return tokens, scale_ids, xyz, protocol, support, padding


class Pillar0ReaderTest(unittest.TestCase):
    def test_output_shape_and_padding_invariance(self):
        model = Pillar0TargetQueryReader(dropout=0.0).eval()
        inputs = list(_inputs())
        original = model(*inputs)
        self.assertEqual(original.shape, (2, 12))
        self.assertTrue(torch.isfinite(original).all())
        inputs[0][:, 7:] = torch.randn_like(inputs[0][:, 7:]) * 100
        changed = model(*inputs)
        torch.testing.assert_close(original, changed, rtol=1e-5, atol=1e-6)

    def test_masked_bce_backward_has_finite_nonzero_gradients(self):
        model = Pillar0TargetQueryReader(dropout=0.0)
        logits = model(*_inputs())
        targets = torch.randint(0, 2, logits.shape).float()
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.grad is not None]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(float(model.queries.grad.abs().sum()), 0.0)

    def test_autocast_forward_and_backward(self):
        model = Pillar0TargetQueryReader(dropout=0.0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(*_inputs())
            loss = F.binary_cross_entropy_with_logits(logits.float(), torch.zeros_like(logits).float())
        loss.backward()
        self.assertEqual(logits.shape, (2, 12))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.queries.grad).all())


if __name__ == "__main__":
    unittest.main()
