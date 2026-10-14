"""E-G (hold register): chrono init, exact HOLD, telemetry, reading rule, neutral_p identity.

Written before the E-G runs (pre-registration); the registered rule is tested here, not fitted to results.
"""
import math
import unittest

import numpy as np
import torch


class HoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import hold
        except ImportError as e:
            self.fail('horizon.hold is not implemented: ' + str(e))
        return hold

    def make_core(self, width=8):
        from core.model import Core
        torch.manual_seed(21)
        return Core(width=width, heads=1, window=4, classes=5)

    def test_chrono_bias_range_and_timescales(self):
        hold = self.api()
        rng = np.random.default_rng(0)
        b = hold.chrono_biases(64, 16, rng)
        self.assertEqual(b.shape, (64,))
        self.assertTrue((b <= 0).all() and (b >= -math.log(15)).all())
        tau = 1.0 / (1.0 / (1.0 + np.exp(-b)))  # 1 / sigmoid(b): implied hold timescale
        self.assertLess(float(tau.min()), 2.5)
        self.assertGreater(float(tau.max()), 8.0)
        with self.assertRaises(ValueError):
            hold.chrono_biases(4, 2, rng)

    def test_hold_init_chrono_vs_write_prior(self):
        hold = self.api()
        torch.manual_seed(3)
        chrono = hold.HoldCore(self.make_core(), t_max=16, chrono=True, seed=1)
        torch.manual_seed(3)
        plain = hold.HoldCore(self.make_core(), t_max=16, chrono=False, seed=1)
        self.assertTrue((chrono.head.bias <= 0).all())
        self.assertTrue(torch.allclose(plain.head.bias, torch.full_like(plain.head.bias, 2.)))
        self.assertFalse(torch.allclose(chrono.head.bias, plain.head.bias))

    def test_forward_logits_shape_and_closed_gate_is_exact(self):
        hold = self.api()
        torch.manual_seed(5)
        model = hold.HoldCore(self.make_core(), chrono=False, seed=2)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(-20.)  # gates ~ 0: HOLD everywhere
        x = torch.randint(0, 3, (2, 12))
        model.eval()
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 12, 5))
        first = logits[:, :1, :].expand_as(logits)
        self.assertTrue(torch.equal(logits, first))  # bitwise exact hold, not a mix
        self.assertIsNotNone(model.last_gates)
        self.assertEqual(tuple(model.last_gates.shape), (2, 12, 8))

    def test_hold_stats_and_sign(self):
        hold = self.api()
        gates = torch.full((4, 20, 3), 0.1)   # WRITE 0.1 -> hold 0.9
        mask = torch.zeros(4, 20, dtype=torch.bool)
        mask[:, 10:] = True                    # neutral second half
        stats = hold.hold_stats(gates, neutral_mask=mask)
        self.assertAlmostEqual(stats['hold_mean'], 0.9, places=5)
        self.assertAlmostEqual(stats['hold_neutral'], 0.9, places=5)
        self.assertFalse(stats['sign_ok'])     # no contrast: not ok
        gates2 = torch.full((4, 20, 3), 0.9)
        gates2[:, 10:] = 0.05                  # writes informative, holds neutral
        stats2 = hold.hold_stats(gates2, neutral_mask=mask)
        self.assertTrue(stats2['sign_ok'])
        self.assertGreater(stats2['hold_neutral'], stats2['hold_informative'])

    def test_reading_rule_cases(self):
        hold = self.api()
        self.assertEqual(hold.reading(dict(hold_neutral=0.7, neutral_loss_points=2.0))['verdict'], 'DESTEKLENİR')
        self.assertEqual(hold.reading(dict(hold_neutral=0.05, neutral_loss_points=2.0))['verdict'], 'ÇÜRÜTÜLDÜ')
        self.assertEqual(hold.reading(dict(hold_neutral=0.7, neutral_loss_points=25.0))['verdict'], 'ÇÜRÜTÜLDÜ')
        self.assertEqual(hold.reading(dict(hold_neutral=0.3, neutral_loss_points=2.0))['verdict'], 'MANIPULATION_NOT_APPLIED')
        self.assertEqual(hold.reading(dict(hold_neutral=0.7, neutral_loss_points=10.0))['verdict'], 'KISMEN')
        self.assertEqual(hold.reading(dict(hold_neutral=None, neutral_loss_points=2.0))['verdict'], 'INCOMPLETE')

    def test_neutral_p_zero_is_bit_identical(self):
        from horizon import depth_tasks as dt
        from horizon.depth_run import NAMESPACE
        x0, y0 = dt.generate('perm4', 8, 16, 123)
        x1, y1 = dt.generate('perm4', 8, 16, 123, neutral_p=0.0)
        self.assertTrue(torch.equal(x0, x1) and torch.equal(y0, y1))
        b0 = dt.train_batch(NAMESPACE, 'perm4', 1, 7, 8, length=16)
        b1 = dt.train_batch(NAMESPACE, 'perm4', 1, 7, 8, length=16, neutral_p=0.0)
        self.assertTrue(torch.equal(b0['x'], b1['x']) and b0['source_seed'] == b1['source_seed'])
        x2, y2 = dt.generate('perm4', 64, 32, 123, neutral_p=0.5)
        self.assertFalse(torch.equal(x0[:0], x2[:0]) and torch.equal(x2, dt.generate('perm4', 64, 32, 123)[0]))
        self.assertGreater(float((x2 == 0).float().mean()), 0.3)
        with self.assertRaises(ValueError):
            dt.generate('count8_32', 8, 16, 123, neutral_p=0.5)

    def test_make_arm_hold_wiring(self):
        from horizon import depth_models as dm
        torch.manual_seed(9)
        model = dm.make_arm('hold', 'perm4', 32, 1)
        from horizon.hold import HoldCore
        self.assertIsInstance(model, HoldCore)
        self.assertTrue(model.chrono)
        names = [n for n, _ in model.named_parameters()]
        self.assertIn('head.weight', names)
        torch.manual_seed(9)
        nc = dm.make_arm('hold_nc', 'perm4', 32, 1)
        self.assertIsInstance(nc, HoldCore)
        self.assertFalse(nc.chrono)


if __name__ == '__main__':
    unittest.main()
