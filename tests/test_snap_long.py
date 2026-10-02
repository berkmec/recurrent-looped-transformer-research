"""E-A9L: the combined snap + window class does what both parts do, the labels and the registered verdict are mechanical, a tiny end-to-end run has every part."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from core.model import Core
from horizon import depth_models as dm
from horizon import depth_run as dr
from horizon import snap_long as sl


def tiny(seed=3):
    return dm.make_arm('core', 'perm3', 32, seed).eval()


def tokens(batch=3, length=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 6, (batch, length), generator=g)


class Combined(unittest.TestCase):
    def test_the_snap_and_the_cross_window_act_together_and_the_plain_core_returns(self):
        model, x = tiny(), tokens(3, 26)
        g = torch.Generator().manual_seed(1)
        mu = torch.randn(6, 32, generator=g)
        received, returned, memory = {}, {}, {}
        original = Core.decode

        def record(self_, e, s, mk, mv, t):
            if s is not None:
                received[t] = s['s'].clone()
            memory[t] = mk.shape[2]
            out = original(self_, e, s, mk, mv, t)
            returned[t] = out[1]['s'].clone()
            return out

        with torch.no_grad():
            plain = Core.forward(model, x)
            sl.configure(model, mu, 'class', 4, window=5)
            Core.decode = record
            try:
                model(x)
            finally:
                Core.decode = original
            for t in range(1, x.shape[1]):
                self.assertEqual(memory[t], min(t + 1, 5), t)                               # the cross-attention window
                if t % 4 == 0:
                    chosen = model.read(model.outn(returned[t - 1])).argmax(-1)
                    self.assertTrue(torch.equal(received[t], mu[chosen]), t)               # the predicted-class snap
                else:
                    self.assertTrue(torch.equal(received[t], returned[t - 1]), t)
            sl.configure(model)
            self.assertTrue(torch.equal(model(x), plain))                                  # nothing configured: the sealed computation

    def test_configure_validates(self):
        with self.assertRaises(ValueError):
            sl.configure(tiny(), window=0)
        with self.assertRaises(ValueError):
            sl.configure(dm.make_arm('gru', 'perm3', 16, 1))


class Reading(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(sl.reading_label(0, 1_300_000), 'CLEAN')
        self.assertEqual(sl.reading_label(0, 600_000), 'SHORT')
        self.assertEqual(sl.reading_label(3, 1_300_000), 'SPARSE')
        self.assertEqual(sl.reading_label(5, 1_300_000), 'FLOOR')

    def fake(self, width, labels):
        return dict(weights=f'x/p5-{width}.pt', task='perm5', width=width, seed=1, lengths={str(length): dict(conditions={sl.PRIMARY: dict(label=label)}) for length, label in labels.items()})

    def test_the_verdict_names_the_longest_length_both_models_manage(self):
        both = {32768: 'CLEAN', 131072: 'SPARSE'}
        self.assertEqual(sl.reading([self.fake(128, both), self.fake(129, both)])['verdict'], 'EXACT TO 131,072')
        mid = {32768: 'CLEAN', 131072: 'FLOOR'}
        self.assertEqual(sl.reading([self.fake(128, both), self.fake(129, mid)])['verdict'], 'EXACT TO 32,768')                # one model has a floor at 131,072
        self.assertEqual(sl.reading([self.fake(128, {32768: 'FLOOR', 131072: 'FLOOR'}), self.fake(129, both)])['verdict'], 'NOT EXACT')
        self.assertEqual(sl.reading([dict(task='perm4', lengths={})])['verdict'], 'NO S5 MODEL')
        # blind review F3: read monotonically, not 'the longest length that happens to pass'
        floor_then_ok = {32768: 'FLOOR', 131072: 'SPARSE'}
        self.assertEqual(sl.reading([self.fake(128, floor_then_ok), self.fake(129, {32768: 'CLEAN', 131072: 'CLEAN'})])['verdict'], 'NOT EXACT')
        self.assertEqual(sl.reading([self.fake(128, {32768: 'SHORT', 131072: 'CLEAN'}), self.fake(129, both)])['verdict'], 'NOT EXACT')


class EndToEnd(unittest.TestCase):
    def test_a_tiny_run_has_the_equivalence_check_and_every_length(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'core', 32, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            with mock.patch.object(sl, 'ALL_CONDITIONS_UP_TO', 40), mock.patch.object(sl, 'EDGES', (16, 32, 64)):
                r = sl.measure(out.with_suffix('.pt'), lengths=(40, 60), ns=(4, 2), window=8, batch=2, equivalence_length=48, log=lambda m: None)
        self.assertEqual(set(r['lengths']), {'40', '60'})
        self.assertEqual(set(r['lengths']['40']['conditions']), {'class-16', 'class-1', 'nearest-16'})
        self.assertEqual(set(r['lengths']['60']['conditions']), {'class-16'})                  # only the primary condition beyond ALL_CONDITIONS_UP_TO
        self.assertEqual(r['equivalence']['length'], 48)
        self.assertIn('class-16', sl.describe(r))
        self.assertIn('verdict', sl.reading([dict(r, task='perm5')]))


if __name__ == '__main__':
    unittest.main()
