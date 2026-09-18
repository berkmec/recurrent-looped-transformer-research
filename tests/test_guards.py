"""A8: non-finite loss/gradients never reach the optimizer; skips are counted and reported."""
import unittest

import torch


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import guards
        except ImportError as e:
            self.fail('horizon.guards is not implemented: ' + str(e))
        return guards

    def make(self):
        torch.manual_seed(3)
        lin = torch.nn.Linear(4, 1)
        opt = torch.optim.SGD(lin.parameters(), lr=0.1)
        return lin, opt

    def test_finite_step_updates_and_reports_norms(self):
        g = self.api(); lin, opt = self.make()
        before = lin.weight.detach().clone()
        loss = lin(torch.ones(2, 4)).square().mean()
        report = g.guarded_step(opt, lin.parameters(), loss, clip_norm=1.0)
        self.assertTrue(report['applied']); self.assertTrue(report['finite'])
        self.assertGreater(report['grad_norm_before_clip'], 0.)
        self.assertFalse(torch.equal(before, lin.weight.detach()))

    def test_nonfinite_gradient_is_refused_without_touching_weights(self):
        g = self.api(); lin, opt = self.make()
        before = lin.weight.detach().clone()
        x = torch.ones(2, 4); x[0, 0] = float('inf')
        loss = lin(x).square().mean()
        report = g.guarded_step(opt, lin.parameters(), loss, clip_norm=1.0)
        self.assertFalse(report['applied']); self.assertFalse(report['finite'])
        self.assertTrue(torch.equal(before, lin.weight.detach()))
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in lin.parameters()))

    def test_counter_accumulates_skips(self):
        g = self.api(); lin, opt = self.make()
        c = g.GuardCounter()
        for bad in (False, True, False):
            x = torch.ones(2, 4)
            if bad:
                x[0, 0] = float('nan')
            c.record(g.guarded_step(opt, lin.parameters(), lin(x).square().mean(), clip_norm=1.0))
        self.assertEqual((c.steps, c.skipped), (3, 1))
        self.assertEqual(c.summary()['skipped'], 1)


if __name__ == '__main__':
    unittest.main()
