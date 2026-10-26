import unittest
import numpy as np
import torch
from rsna_knee.auc_margin import MaskedAUCMargin, build_evidence


class AUCMarginTests(unittest.TestCase):
    def test_conditional_objective_and_masked_gradients(self):
        scores = torch.tensor([[.8, .2], [.6, .7], [.3, .4], [.9, .8]])
        evidence = torch.tensor([[1, -1], [1, 1], [-1, 0], [0, 1]])
        loss = MaskedAUCMargin((evidence == 1).float().mean(0), (evidence == -1).float().mean(0))
        with torch.no_grad():
            for j in range(2):
                loss.a[j] = scores[evidence[:, j] == 1, j].mean()
                loss.b[j] = scores[evidence[:, j] == -1, j].mean()
            gap = .3 + loss.b - loss.a
            loss.alpha.copy_(gap.clamp(min=0))
        logits = torch.logit(scores).requires_grad_()
        actual, _ = loss(logits, evidence)
        expected = torch.stack([scores[evidence[:, j] == 1, j].var(unbiased=False)
            + scores[evidence[:, j] == -1, j].var(unbiased=False)
            + gap[j].clamp(min=0).square() for j in range(2)]).mean()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        actual.backward()
        self.assertTrue((logits.grad[evidence == 0] == 0).all())
        loss.ascend(torch.full((2,), -100.), 1.)
        self.assertTrue((loss.alpha == 0).all())

    def test_evidence_excludes_silence_disagreement_and_sentinel(self):
        result = build_evidence(np.array([.9, .1, .25, .9, .8]),
                                np.array([.8, .2, .1, .1, .9]),
                                np.ones(5), np.array([1, 1, 1, 1, 0]))
        np.testing.assert_array_equal(result, [1, -1, 0, 0, 0])
