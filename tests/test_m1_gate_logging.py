"""C3/C4 from the independent review: a negative M1 must be interpretable, and "could not measure" is not "no effect".

C3: no gate statistic was recorded anywhere in the M1 runner. Every earlier cycle logged hold_rate per step. Without
it, a null result cannot be told apart from a gate that never closed - and with bias=2.0 at initialisation and no
gate supervision, never closing is the likely outcome.

C4: mline.analyze had no floor branch. If every arm scores zero at the long fillers - the outcome the emergence
probe makes likely at this data budget - the rule returned GATE_DOES_NOT_TRANSFER, converting "we could not measure
it" into "the gate is a toy artefact". Both earlier cycles carried an INTERVENTION_INEFFECTIVE branch for exactly
this; it was left out here.
"""
import unittest

import torch


class GateLoggingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_forward_returns_gates_for_gated_arms_and_none_for_plain(self):
        from mline import lm as ml
        x = torch.randint(0, 256, (2, 16))
        for arm in ('hard_st', 'soft'):
            logits, gates = ml.forward_with_gates(ml.build(arm, width=64, heads=2), x)
            self.assertEqual(logits.shape, (2, 16, 256), arm)
            self.assertEqual(gates.shape, (2, 16), arm)
        logits, gates = ml.forward_with_gates(ml.build('plain', width=64, heads=2), x)
        self.assertEqual(logits.shape, (2, 16, 256))
        self.assertIsNone(gates)

    def test_hold_rate_is_the_fraction_of_closed_gates(self):
        from mline import lm as ml
        self.assertIsNone(ml.hold_rate(None))
        self.assertAlmostEqual(ml.hold_rate(torch.tensor([[1., 1., 0., 0.]])), .5, places=6)
        self.assertAlmostEqual(ml.hold_rate(torch.tensor([[1., 1., 1., 1.]])), 0., places=6)
        self.assertAlmostEqual(ml.hold_rate(torch.tensor([[.2, .3]])), 1., places=6)


class FloorBranchTests(unittest.TestCase):
    def api(self):
        from mline import analyze
        return analyze

    def row(self, bpb, needle):
        return dict(bits_per_byte=bpb, needle={f'{k}@{n}': v for k in ('repeat', 'text')
                                               for n, v in zip((0, 512, 2048), needle)})

    def test_everything_at_floor_is_not_a_verdict_about_the_gate(self):
        ua = self.api()
        arms = dict(plain=self.row(1.9, (0., 0., 0.)), hard_st=self.row(1.9, (0., 0., 0.)), soft=self.row(1.9, (0., 0., 0.)))
        v = ua.verdict(arms, seeds=1)
        self.assertEqual(v['status'], 'NOT_MEASURABLE_AT_THIS_BUDGET')
        self.assertIn('could not', v['meaning'].lower())

    def test_a_control_that_can_do_the_task_makes_a_null_result_meaningful(self):
        ua = self.api()
        arms = dict(plain=self.row(1.9, (.95, .30, .10)), hard_st=self.row(1.9, (.95, .31, .11)), soft=self.row(1.9, (.94, .30, .10)))
        self.assertEqual(ua.verdict(arms, seeds=1)['status'], 'GATE_DOES_NOT_TRANSFER')

    def test_the_floor_threshold_is_declared(self):
        t = self.api().THRESHOLDS
        self.assertIn('measurable_floor', t)
        self.assertGreater(t['measurable_floor'], 0.)


if __name__ == '__main__':
    unittest.main()
