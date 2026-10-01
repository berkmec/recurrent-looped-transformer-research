"""U2d diagnostic contract (written before the implementation).

Open question from U2c: two models with nearly identical gate hold rates (soft at lr 0.003 vs lr 0.0003) differ
completely in neutral-run robustness (512-middle 1.000 vs 0.182), so the hold rate does not explain robustness.
This module measures what the states themselves do, reusing the U2 measurement ideas on a GatedCore:
committed-state trajectories, per-class anchors, displacement during a neutral run, and state norm growth
(the quantity the Fully Looped Transformer paper calls residual explosion).
"""
import unittest

import torch


class GatedTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import gated_trajectory as gt
        except ImportError as e:
            self.fail('horizon.gated_trajectory is not implemented: ' + str(e))
        return gt

    def model(self, mode='soft'):
        from core.model import Core
        from horizon.gated import GatedCore
        torch.manual_seed(5)
        return GatedCore(Core(width=8, heads=1, window=4, classes=5), mode=mode)

    def test_committed_states_match_the_model_and_carry_gate_decisions(self):
        gt = self.api(); m = self.model()
        x = torch.tensor([[0, 1, 2, 2, 2, 1, 0, 2], [1, 0, 2, 1, 2, 2, 2, 0]])
        out = gt.run_states(m, x, batch=1)
        self.assertEqual(tuple(out['states'].shape), (2, 8, 8))
        self.assertEqual(tuple(out['gates'].shape), (2, 8))
        logits, gates, _ = m(x)
        torch.testing.assert_close(out['logits'], logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(out['gates'], gates, rtol=1e-5, atol=1e-6)

    def test_hard_mode_holds_leave_the_committed_state_bitwise_unchanged(self):
        gt = self.api(); m = self.model('hard_st'); m.close_all()
        x = torch.tensor([[0, 1, 2, 2, 1, 0]])
        out = gt.run_states(m, x, batch=1)
        s = out['states']
        for t in range(1, x.shape[1]):
            self.assertTrue(torch.equal(s[:, t], s[:, 0]), f'state changed at t={t} despite a closed gate')

    def test_norm_and_displacement_curves(self):
        gt = self.api(); m = self.model()
        x = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])
        out = gt.run_states(m, x, batch=1)
        curve = gt.norm_curve(out['states'])
        self.assertEqual(len(curve), x.shape[1])
        self.assertTrue(all(v >= 0 for v in curve))
        # displacement of a neutral run measured against the state just before the run
        xx = torch.tensor([[0, 1, 2, 2, 2, 2]])
        o2 = gt.run_states(m, xx, batch=1)
        d = gt.displacement_curve(o2['states'], start=2)
        self.assertEqual(len(d), 4)
        self.assertAlmostEqual(d[0], float(torch.linalg.vector_norm(o2['states'][0, 2] - o2['states'][0, 1])), places=5)

    def test_anchor_separation_summary(self):
        gt = self.api()
        import numpy as np
        z = np.concatenate([np.zeros((5, 4)), np.ones((5, 4))])
        labels = np.array([0] * 5 + [1] * 5)
        a = gt.anchor_summary(z, labels, classes=2)
        self.assertAlmostEqual(a['min_center_distance'], 2.0, places=5)   # ||0-1|| over 4 dims
        self.assertAlmostEqual(max(a['spread']), 0.0, places=5)
        self.assertGreater(a['separation_ratio'], 1e3)                     # separated centres, zero spread


if __name__ == '__main__':
    unittest.main()
