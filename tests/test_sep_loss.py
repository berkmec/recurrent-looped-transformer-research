"""U4-sep contract (written before the implementation): an auxiliary loss that widens the class-anchor separation
ratio U2d measured, so the correlational finding can be tested causally."""
import unittest

import torch


class SepLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import sep_loss as sl
        except ImportError as e:
            self.fail('horizon.sep_loss is not implemented: ' + str(e))
        return sl

    def test_ratio_matches_the_diagnostic_definition(self):
        sl = self.api()
        # two tight, far-apart classes: ratio = min centre distance / mean within-class spread
        z = torch.tensor([[0., 0.], [0., 0.], [3., 4.], [3., 4.]])
        y = torch.tensor([0, 0, 1, 1])
        r = sl.separation_ratio(z, y, classes=2)
        self.assertGreater(float(r), 1e5)                      # zero spread -> huge ratio
        z2 = torch.tensor([[0., 0.], [2., 0.], [3., 4.], [5., 4.]])   # spread 1.0 each, centres 5 apart
        r2 = sl.separation_ratio(z2, y, classes=2)
        self.assertAlmostEqual(float(r2), 5.0, places=4)

    def test_loss_decreases_as_separation_grows_and_is_differentiable(self):
        sl = self.api()
        y = torch.tensor([0, 0, 1, 1])
        near = torch.tensor([[0., 0.], [1., 0.], [1.5, 0.], [2.5, 0.]], requires_grad=True)
        far = torch.tensor([[0., 0.], [1., 0.], [20., 0.], [21., 0.]], requires_grad=True)
        l_near = sl.separation_loss(near, y, classes=2)
        l_far = sl.separation_loss(far, y, classes=2)
        self.assertLess(float(l_far), float(l_near))
        l_near.backward()
        self.assertIsNotNone(near.grad); self.assertGreater(float(near.grad.abs().sum()), 0.)

    def test_missing_classes_and_degenerate_batches_are_safe(self):
        sl = self.api()
        z = torch.randn(6, 4)
        y = torch.zeros(6, dtype=torch.long)                    # only one class present
        l = sl.separation_loss(z, y, classes=5)
        self.assertTrue(torch.isfinite(l)); self.assertEqual(float(l), 0.)
        single = torch.randn(2, 4); ys = torch.tensor([0, 1])   # one sample per class: spread 0
        l2 = sl.separation_loss(single, ys, classes=2)
        self.assertTrue(torch.isfinite(l2))

    def test_scale_invariance(self):
        """Inflating every state by a constant must not change the ratio — the loss targets geometry, not norm."""
        sl = self.api()
        z = torch.tensor([[0., 0.], [2., 0.], [3., 4.], [5., 4.]])
        y = torch.tensor([0, 0, 1, 1])
        a = sl.separation_ratio(z, y, classes=2)
        b = sl.separation_ratio(z * 7.0, y, classes=2)
        self.assertAlmostEqual(float(a), float(b), places=4)


if __name__ == '__main__':
    unittest.main()
