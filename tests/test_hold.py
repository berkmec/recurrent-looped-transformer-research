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
        """c8c64fd Z4 branch order and Z5 sign conjunct, thresholds of 319f13c (0.5 / 5 / 20) at their edges."""
        hold = self.api()
        v = lambda h, l, s=True, **kw: hold.reading(dict(hold_neutral=h, neutral_loss_points=l, sign_ok=s, **kw))['verdict']
        self.assertEqual(v(0.7, 2.0), 'DESTEKLENİR')                      # (i)
        self.assertEqual(v(0.5, 5.0), 'DESTEKLENİR')                      # (i) edges inclusive
        self.assertEqual(v(0.7, -50.5), 'DESTEKLENİR')                    # negative loss passes, no floor (Q3)
        self.assertEqual(v(0.7, 25.0), 'ÇÜRÜTÜLDÜ')                       # (ii)
        self.assertEqual(v(0.05, 25.0), 'ÇÜRÜTÜLDÜ')                      # (ii) before (iii)
        self.assertEqual(v(0.7, 20.0), 'KISMEN')                          # 20 is not > 20
        self.assertEqual(v(0.05, 2.0), 'MANIPULATION_NOT_APPLIED')        # Z4: hold < 0.1 is not a refutation any more
        self.assertEqual(v(0.3, 2.0), 'MANIPULATION_NOT_APPLIED')         # (iii)
        self.assertEqual(v(0.4999, 10.0), 'MANIPULATION_NOT_APPLIED')
        self.assertEqual(v(0.7, 10.0), 'KISMEN')                          # (iv)
        self.assertEqual(v(None, 2.0), 'INCOMPLETE')
        self.assertEqual(v(0.7, None), 'INCOMPLETE')
        self.assertEqual(v(0.7, 2.0, None), 'INCOMPLETE')                 # sign_ok is a measurement of (i)

    def test_inverted_gate_is_not_support(self):
        """Z5 negative test (E-A10): a gate that holds informative steps more than neutral ones never reads DESTEKLENİR."""
        hold = self.api()
        mask = torch.zeros(4, 20, dtype=torch.bool)
        mask[:, 10:] = True
        gates = torch.full((4, 20, 3), 0.4)    # neutral: hold 0.6
        gates[:, :10] = 0.1                    # informative: hold 0.9 -> inverted
        stats = hold.hold_stats(gates, neutral_mask=mask)
        self.assertFalse(stats['sign_ok'])
        r = hold.reading(dict(stats, neutral_loss_points=1.0))
        self.assertGreaterEqual(r['hold_neutral'], 0.5)
        self.assertEqual(r['verdict'], 'KISMEN')
        gates[:, :10] = 0.9                    # same neutral hold, correct sign
        self.assertEqual(hold.reading(dict(hold.hold_stats(gates, neutral_mask=mask), neutral_loss_points=1.0))['verdict'], 'DESTEKLENİR')

    def test_hold_nc_reading(self):
        """Z7: the negative control is never support or refutation."""
        hold = self.api()
        r = lambda h, l, s=True: hold.reading(dict(arm='hold_nc', hold_neutral=h, neutral_loss_points=l, sign_ok=s))
        self.assertEqual(r(0.3, 2.0)['verdict'], 'MANIPULATION_NOT_APPLIED')
        self.assertEqual(r(0.05, 30.0)['verdict'], 'MANIPULATION_NOT_APPLIED')
        self.assertEqual(r(0.7, 30.0)['verdict'], 'MANIPULATION_NOT_APPLIED')   # loss > 20: descriptive, control arm
        self.assertEqual(r(0.7, 2.0)['verdict'], 'KISMEN')
        self.assertIn('CHRONO_UNNECESSARY', r(0.7, 2.0)['flags'])
        self.assertEqual(r(0.7, 10.0)['verdict'], 'KISMEN')
        self.assertNotIn('CHRONO_UNNECESSARY', r(0.7, 10.0)['flags'])

    def test_arm_and_eg_combination(self):
        """Z6: seed -> arm (perm4 3 seeds, perm5 2 seeds) -> E-G."""
        hold = self.api()
        S, R, M = (dict(hold_neutral=0.7, neutral_loss_points=2.0, sign_ok=True),
                   dict(hold_neutral=0.7, neutral_loss_points=30.0, sign_ok=True),
                   dict(hold_neutral=0.2, neutral_loss_points=2.0, sign_ok=True))
        seeds = lambda *rs: [dict(r, seed=i + 1, arm='hold') for i, r in enumerate(rs)]
        arm = lambda task, *rs: hold.arm_reading(seeds(*rs), task)
        self.assertEqual(arm('perm4', S, S, M)['verdict'], 'DESTEKLENİR')
        self.assertEqual(arm('perm4', S, M, M)['verdict'], 'MANIPULATION_NOT_APPLIED')
        self.assertEqual(arm('perm4', S, R, M)['verdict'], 'KISMEN')           # 1-1-1
        a = arm('perm4', S, S, R)
        self.assertEqual(a['verdict'], 'KISMEN')                               # a refuted seed blocks arm support
        self.assertEqual(a['refuted_seeds'], [3])
        self.assertEqual(arm('perm4', R, R, S)['verdict'], 'ÇÜRÜTÜLDÜ')
        self.assertEqual(arm('perm5', S, S)['verdict'], 'DESTEKLENİR')
        self.assertEqual(arm('perm5', S, M)['verdict'], 'KISMEN')              # 1-1
        self.assertEqual(arm('perm5', S, S, S)['verdict'], 'INCOMPLETE')       # not the registered seed count
        self.assertEqual(arm('perm4', S, S)['verdict'], 'INCOMPLETE')
        with self.assertRaises(ValueError):
            hold.arm_reading([dict(S, arm='hold_nc', seed=1)] * 3, 'perm4')
        eg = lambda a4, a5: hold.eg_reading(dict(perm4=a4, perm5=a5))
        self.assertEqual(eg('DESTEKLENİR', 'DESTEKLENİR')['verdict'], 'DESTEKLENİR')
        self.assertEqual(eg('ÇÜRÜTÜLDÜ', 'ÇÜRÜTÜLDÜ')['verdict'], 'ÇÜRÜTÜLDÜ')
        self.assertEqual(eg('DESTEKLENİR', 'ÇÜRÜTÜLDÜ')['verdict'], 'KISMEN')
        self.assertTrue(eg('DESTEKLENİR', 'ÇÜRÜTÜLDÜ')['scale_split'])
        self.assertEqual(eg('DESTEKLENİR', 'MANIPULATION_NOT_APPLIED')['verdict'], 'KISMEN')
        self.assertEqual(eg('DESTEKLENİR', 'INCOMPLETE')['verdict'], 'INCOMPLETE')

    def test_ci_flag_is_descriptive(self):
        """Z10: CIs are written, SINIRDA flags a CI across 5 or 20, the label is unchanged."""
        hold = self.api()
        base = dict(hold_neutral=0.7, sign_ok=True, n_loss=200)
        near = hold.reading(dict(base, acc0=0.90, acc9=0.86, neutral_loss_points=4.0))
        self.assertEqual(near['verdict'], 'DESTEKLENİR')
        self.assertIn('SINIRDA', near['flags'])
        lo, hi = near['loss_ci']
        self.assertLess(lo, 4.0)
        self.assertGreater(hi, 5.0)
        far = hold.reading(dict(base, acc0=0.99, acc9=1.0, neutral_loss_points=-1.0))
        self.assertNotIn('SINIRDA', far['flags'])
        self.assertTrue(0.0 <= far['acc9_ci'][0] <= far['acc9_ci'][1] <= 1.0)
        self.assertNotIn('loss_ci', hold.reading(dict(base, neutral_loss_points=4.0)))   # old result files: no CI, same label

    def test_measure_persists_acc_and_scope(self):
        """Q1/Q4: measure stores acc0/acc9 with n, the gate-telemetry n (first batch), length and eval seed."""
        import tempfile
        from pathlib import Path
        from horizon import depth_models as dm
        hold = self.api()
        torch.manual_seed(9)
        model = dm.make_arm('hold', 'perm4', 32, 1)
        meta = dict(arm='hold', task='perm4', width=32, seed=1, hold_tmax=16,
                    parameters=sum(p.numel() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'tiny.pt'
            torch.save(dict(state_dict=model.state_dict(), meta=meta), path)
            r = hold.measure(path, length=16, n=6, seed=3, batch=4, log=lambda m: None)
        self.assertEqual((r['n_loss'], r['n_gate'], r['length'], r['eval_seed'], r['seed']), (6, 4, 16, 3, 1))
        self.assertAlmostEqual(r['neutral_loss_points'], 100.0 * (r['acc0'] - r['acc9']), places=9)
        self.assertEqual(r['reading'], hold.reading(r))                                   # --reading reproduces it

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
