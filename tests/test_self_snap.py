"""E-A9: the snap replaces the fed-back state exactly as the module says (predicted class or nearest centroid, every k steps, strength lambda), changes nothing when it is off, and the
registered reading is mechanical."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from core.model import Core
from horizon import depth_models as dm
from horizon import depth_run as dr
from horizon import self_snap as sn


def tiny(seed=3):
    return dm.make_arm('core', 'perm3', 32, seed).eval()


def tokens(batch=3, length=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 6, (batch, length), generator=g)


def protos(classes=6, width=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(classes, width, generator=g)


class Snap(unittest.TestCase):
    def spy(self, model, x, **kwargs):
        """Run the configured model and record, per decoder step, the state the sealed decode received (after the snap) and the state it returned."""
        received, returned = {}, {}
        original = Core.decode

        def record(self_, e, s, mk, mv, t):
            if s is not None:
                received[t] = s['s'].clone()
            out = original(self_, e, s, mk, mv, t)
            returned[t] = out[1]['s'].clone()
            return out

        sn.configure(model, **kwargs)
        Core.decode = record
        try:
            with torch.no_grad():
                model(x)
        finally:
            Core.decode = original
        sn.configure(model)
        return received, returned

    def test_the_predicted_class_snap_replaces_the_state_by_the_centroid_of_the_models_own_prediction(self):
        model, x, mu = tiny(), tokens(), protos()
        received, returned = self.spy(model, x, prototypes=mu, selector='class', every=1)
        for t in range(1, x.shape[1]):
            with torch.no_grad():
                chosen = model.read(model.outn(returned[t - 1])).argmax(-1)
            self.assertTrue(torch.equal(received[t], mu[chosen]), t)

    def test_the_nearest_selector_uses_the_state_space_distance_only(self):
        model, x, mu = tiny(), tokens(), protos()
        received, returned = self.spy(model, x, prototypes=mu, selector='nearest', every=1)
        for t in range(1, x.shape[1]):
            chosen = torch.cdist(returned[t - 1], mu).argmin(-1)
            self.assertTrue(torch.equal(received[t], mu[chosen]), t)

    def test_every_k_steps_and_the_soft_strength(self):
        model, x, mu = tiny(), tokens(), protos()
        received, returned = self.spy(model, x, prototypes=mu, selector='class', every=4, strength=1.0)
        for t in range(1, x.shape[1]):
            if t % 4 == 0:
                chosen = model.read(model.outn(returned[t - 1])).argmax(-1)
                self.assertTrue(torch.equal(received[t], mu[chosen]), t)
            else:
                self.assertTrue(torch.equal(received[t], returned[t - 1]), t)                      # untouched between the snaps
        received, returned = self.spy(model, x, prototypes=mu, selector='class', every=1, strength=0.5)
        for t in range(1, x.shape[1]):
            chosen = model.read(model.outn(returned[t - 1])).argmax(-1)
            self.assertTrue(torch.allclose(received[t], 0.5 * returned[t - 1] + 0.5 * mu[chosen], atol=1e-6), t)

    def test_off_or_strength_zero_is_the_sealed_computation(self):
        model, x = tiny(), tokens()
        with torch.no_grad():
            sealed = Core.forward(model, x)
            sn.configure(model)
            self.assertTrue(torch.equal(model(x), sealed))
            sn.configure(model, prototypes=protos(), strength=0.0)
            self.assertTrue(torch.equal(model(x), sealed))
            sn.configure(model, prototypes=protos(), strength=1.0)
            self.assertFalse(torch.allclose(model(x)[:, 5:], sealed[:, 5:], atol=1e-4))             # a real snap does change the run
        sn.configure(model)

    def test_configure_validates(self):
        model = tiny()
        for kwargs in (dict(selector='soft'), dict(every=0), dict(strength=1.5), dict(strength=-0.1)):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                sn.configure(model, prototypes=protos(), **kwargs)
        with self.assertRaises(ValueError):
            sn.configure(dm.make_arm('gru', 'perm3', 16, 1))


def fake(task='perm5', width=128, base=(100, 10000), primary=None, accuracy=1.0, stored=True):
    primary = primary if primary is not None else base
    conditions = {'base': dict(events=base[0], person_steps=base[1]), sn.PRIMARY: dict(events=primary[0], person_steps=primary[1]),
                  'nearest-1': dict(events=base[0], person_steps=base[1])}
    return dict(weights=f'x/{task}-w{width}.pt', task=task, width=width, seed=1, n=200, length=1024, parameters=1, conditions=conditions, stored_control=stored,
                in_range={'base': 1.0, sn.PRIMARY: accuracy, 'nearest-1': 1.0})


class Reading(unittest.TestCase):
    def s5(self, **kw):
        return [fake(width=128 + i, **kw) for i in range(3)]

    def test_verdicts_follow_the_registered_rule(self):
        s4 = [fake('perm4', 32, base=(0, 10000), primary=(0, 10000))]
        self.assertEqual(sn.reading(self.s5(primary=(10, 10000)) + s4)['verdict'], 'SELF-SNAP RESTORES EXACTNESS')
        self.assertEqual(sn.reading(self.s5() + s4)['verdict'], 'SELF-SNAP DOES NOT HELP')
        mixed = [fake(width=128, primary=(10, 10000)), fake(width=129), fake(width=130)]
        self.assertEqual(sn.reading(mixed + s4)['verdict'], 'MIXED')
        self.assertEqual(sn.reading(self.s5(primary=(40, 10000)) + s4)['verdict'], 'MIXED')                  # PARTIAL everywhere
        self.assertTrue(sn.reading(self.s5(primary=(300, 10000)) + s4)['verdict'].startswith('INCONCLUSIVE'))   # the snap harms the S5 models
        harmed_control = [fake('perm4', 32, base=(0, 10000), primary=(30, 10000))]
        self.assertTrue(sn.reading(self.s5() + harmed_control)['verdict'].startswith('INCONCLUSIVE'))
        self.assertEqual(sn.reading(s4)['verdict'], 'NO S5 MODEL')

    def test_failed_controls_void_the_verdict(self):
        s4 = [fake('perm4', 32, base=(0, 10000), primary=(0, 10000))]
        out = sn.reading(self.s5(stored=False) + s4)
        self.assertTrue(out['verdict'].startswith('CONTROL FAILED') and 'stored first-error windows' in out['verdict'])
        out = sn.reading(self.s5(accuracy=0.95) + s4)
        self.assertTrue(out['verdict'].startswith('CONTROL FAILED') and 'in-range accuracy' in out['verdict'])


class EndToEnd(unittest.TestCase):
    def test_a_measure_of_a_tiny_trained_core_has_every_part(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'core', 32, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            with mock.patch.object(sn, 'LONG_LENGTH', 64), mock.patch.object(sn, 'LONG_N', 4), mock.patch.object(sn, 'LONG_EDGES', (16, 32, 64)):
                r = sn.measure(out.with_suffix('.pt'), n=6, length=100, batch=3, log=lambda m: None)
        self.assertEqual((r['task'], r['width'], r['n']), ('perm3', 32, 6))
        self.assertEqual(set(r['conditions']), {'base', 'class-1', 'class-4', 'class-16', 'nearest-1', 'nearest-16', 'class-1-soft0.5'})
        self.assertEqual(set(r['in_range']), set(r['conditions']))
        self.assertIsNone(r['stored_control'])                                                           # no stored 1,024-position curve for a tiny run
        self.assertEqual(set(r['long']), {'base', 'class-1'})
        self.assertEqual(set(r['long']['base']['windows']), {'16-32', '32-64'})
        self.assertIn('class-1', sn.describe(r))
        self.assertEqual(set(sn.labels(r)), {'class-1', 'class-4', 'class-16', 'nearest-1', 'nearest-16', 'class-1-soft0.5'})


if __name__ == '__main__':
    unittest.main()
