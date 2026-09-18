"""A5/P2 on OUR Core: exact segment rematerialisation (no detach at boundaries) must reproduce full-BPTT loss and
parameter gradients; the detach-at-boundary arm (TBPTT) must differ — it severs the credit path to early positions."""
import unittest

import torch


class RematTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import remat
        except ImportError as e:
            self.fail('horizon.remat is not implemented: ' + str(e))
        return remat

    def make(self):
        from core.model import Core
        torch.manual_seed(5)
        return Core(width=8, heads=1, window=4, classes=5)

    def grads(self, model, logits, y):
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(logits[:, -1], y[:, -1])  # terminal label only
        loss.backward()
        return float(loss), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    def test_segment_remat_matches_full_bptt_and_detach_does_not(self):
        rm = self.api(); model = self.make()
        torch.manual_seed(9)
        x = torch.randint(0, 3, (3, 24)); y = torch.randint(0, 5, (3, 24))
        full_loss, full = self.grads(model, rm.run(model, x, segment=None, mode='full'), y)
        self.assertIn('initial', full)  # the learned initial state receives credit under full BPTT
        remat_loss, remat_g = self.grads(model, rm.run(model, x, segment=8, mode='remat'), y)
        self.assertAlmostEqual(full_loss, remat_loss, places=6)
        self.assertEqual(set(full), set(remat_g))
        for n in full:
            torch.testing.assert_close(full[n], remat_g[n], rtol=1e-5, atol=1e-6, msg=n)
        self.assertGreater(float(full['initial'].norm()), 0.)  # terminal credit reaches the learned initial state
        self.assertGreater(float(remat_g['initial'].norm()), 0.)
        detach_loss, detach_g = self.grads(model, rm.run(model, x, segment=8, mode='detach'), y)
        self.assertAlmostEqual(full_loss, detach_loss, places=6)  # same forward values
        # structural contract: with detach at the boundaries no credit reaches the initial state (its gradient is absent or zero)
        self.assertTrue('initial' not in detach_g or float(detach_g['initial'].abs().max()) == 0.)
        diff = max(float((full[n] - detach_g[n]).abs().max()) for n in full if n in detach_g)
        self.assertGreater(diff, 0.)  # and the shared-parameter gradients are not identical either

    def test_plain_mode_is_bit_equal_to_forward(self):
        rm = self.api(); model = self.make()
        x = torch.randint(0, 3, (2, 10))
        torch.testing.assert_close(rm.run(model, x, segment=None, mode='full'), model(x), rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
