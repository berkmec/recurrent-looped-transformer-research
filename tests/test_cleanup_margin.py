"""E-A11p: the margin gate fires exactly where the margin is large, is the identity at gate 0 and with an infinite threshold, equals the ungated cleanup with a -infinity threshold, and the registered reading is mechanical."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from horizon import cleanup_margin as cm
from horizon import cleanup as cl
from horizon import cleanup_probe as cp
from horizon import depth_models as dm
from horizon import depth_run as dr


def gated(threshold, gate=0.999):
    """A MarginGatedNorm around an RMS norm with two orthogonal prototypes in 4 dimensions."""
    base = dm.make_arm('core', 'perm3', 32, 1)
    protos = torch.zeros(2, 32)
    protos[0, 0], protos[1, 1] = 1.0, 1.0
    out = cp.retrofit(base, protos, gate=gate)
    out.pn.__class__ = cm.MarginGatedNorm
    out.pn.threshold, out.pn.distinct, out.pn.counting, out.pn.fired, out.pn.seen = threshold, torch.nn.functional.normalize(protos, dim=-1), False, 0, 0
    return out.pn


class Gate(unittest.TestCase):
    def states(self):
        s = torch.zeros(3, 32)
        s[0, 0] = 1.0                       # on prototype 0: margin 1 - 0
        s[1, 0], s[1, 1] = 1.0, 0.9         # nearly between the two: a small margin
        s[2, 1] = 1.0                       # on prototype 1
        return s

    def test_it_fires_only_where_the_margin_is_large(self):
        pn, s = gated(0.5), self.states()
        out, plain = pn(s), pn.norm(s)
        self.assertTrue(torch.equal(out[1], plain[1]))                                                       # the ambiguous state passes through untouched
        self.assertFalse(torch.allclose(out[0], plain[0]))                                                    # the confident states are cleaned
        self.assertFalse(torch.allclose(out[2], plain[2]))

    def test_identity_at_gate_zero_and_with_an_infinite_threshold_and_equality_to_the_plain_module_without_a_threshold(self):
        s = self.states()
        for pn in (gated(0.5, gate=0.0), gated(math.inf)):
            self.assertTrue(torch.equal(pn(s), pn.norm(s)))
        pn = gated(-math.inf)
        ungated = cl.CleanupNorm.forward(pn, s)
        self.assertTrue(torch.equal(pn(s), ungated))

    def test_the_firing_fraction_is_counted(self):
        pn, s = gated(0.5), self.states()
        pn.counting = True
        pn(s)
        self.assertEqual((pn.fired, pn.seen), (2, 3))

    def test_the_threshold_is_the_calibration_quantile_and_needs_no_label(self):
        model = dm.make_arm('core', 'perm3', 32, 1).eval()
        g = torch.Generator().manual_seed(1)
        calibration, protos = torch.randn(400, 32, generator=g), torch.randn(6, 32, generator=g)
        patched = cm.retrofit_gated(model, protos, 0.5, calibration)
        self.assertAlmostEqual(patched.pn.threshold, float(torch.quantile(cm.margins(calibration, protos), 0.5)), places=6)
        share = float((cm.margins(calibration, protos) >= patched.pn.threshold).float().mean())
        self.assertTrue(0.49 <= share <= 0.52)                                                                # about half of the calibration states clear the median


def result(task='perm5', width=128, base=(100, 10000), primary=(0, 10000), accuracy=1.0, identity=True, stored=True):
    c = {'base': dict(events=base[0], person_steps=base[1], in_range=1.0), cm.PRIMARY: dict(events=primary[0], person_steps=primary[1], in_range=accuracy, fired=0.7)}
    return dict(weights=f'x/{task}-w{width}.pt', task=task, width=width, seed=1, n=200, length=1024, conditions=c, gate0_identity=identity, kmeans_clusters=120, stored_control=stored)


class Reading(unittest.TestCase):
    def test_the_registered_verdicts(self):
        s4 = [result('perm4', 32, base=(0, 10000), primary=(0, 10000))]
        self.assertEqual(cm.reading([result(), result(width=129)] + s4)['verdict'], 'LABEL-FREE GATED CLEANUP WORKS')
        self.assertEqual(cm.reading([result(primary=(100, 10000)), result(width=129, primary=(100, 10000))] + s4)['verdict'], 'LABEL-FREE GATED CLEANUP DOES NOT WORK')
        self.assertEqual(cm.reading([result(), result(width=129, primary=(100, 10000))] + s4)['verdict'], 'MIXED')
        self.assertEqual(cm.reading([result(primary=(30, 10000)), result(width=129, primary=(30, 10000))] + s4)['verdict'], 'MIXED')               # PARTIAL everywhere
        self.assertTrue(cm.reading([result(accuracy=0.95), result(width=129)] + s4)['verdict'].startswith('INCONCLUSIVE'))
        self.assertTrue(cm.reading([result(), result(width=129), result('perm4', 32, base=(0, 10000), primary=(30, 10000))])['verdict'].startswith('INCONCLUSIVE'))   # an S4 control harmed
        self.assertTrue(cm.reading([result(stored=False), result(width=129)] + s4)['verdict'].startswith('CONTROL FAILED'))
        self.assertTrue(cm.reading([result(identity=False), result(width=129)] + s4)['verdict'].startswith('CONTROL FAILED'))
        self.assertEqual(cm.reading(s4)['verdict'], 'NO S5 MODEL')

    def test_a_failed_control_takes_precedence_over_a_low_accuracy(self):
        self.assertTrue(cm.reading([result(stored=False, accuracy=0.5), result(width=129)])['verdict'].startswith('CONTROL FAILED'))


class EndToEnd(unittest.TestCase):
    def test_a_measure_of_a_tiny_trained_core_has_every_part(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'core', 32, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            r = cm.measure(out.with_suffix('.pt'), n=6, length=100, batch=3, log=lambda m: None)
        self.assertEqual((r['task'], r['width']), ('perm3', 32))
        self.assertEqual(set(r['conditions']), {'base', 'kmeans-q30', 'kmeans-q50', 'kmeans-q70', 'labelled-q50'})
        self.assertTrue(r['gate0_identity'])
        self.assertTrue(all(0.0 <= v['fired'] <= 1.0 for k, v in r['conditions'].items() if k != 'base'))
        self.assertIsNone(r['stored_control'])
        self.assertIn('gate fired on', cm.describe(r))


if __name__ == '__main__':
    unittest.main()
