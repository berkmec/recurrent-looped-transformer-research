"""E-L1X: the long-document probe changes nothing unless configured, resets the carried state exactly where it says, restricts the attention paths to the training window, and the registered
reading is mechanical."""
import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np
import torch

from core.model import Core
from mline import lm
from mline import long_doc as ld


def tiny(seed=5):
    return lm.build('plain', width=32, heads=2, window=4, alpha=0.1, seed=seed).eval()


def small_world():
    """The registered constants scaled down: a 16-byte 'training window' so a 64-byte text is four of them."""
    stack = ExitStack()
    stack.enter_context(mock.patch.object(ld, 'TRAIN_LENGTH', 16))
    stack.enter_context(mock.patch.object(ld, 'MIN_OFFSET', 8))
    stack.enter_context(mock.patch.object(ld, 'BINS', ((0, 16), (16, 32), (32, 64))))
    return stack


def tokens(batch=3, length=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (batch, length), generator=g)


class Conditions(unittest.TestCase):
    def test_full_and_the_default_are_the_sealed_computation_and_win_agrees_inside_the_window(self):
        with small_world():
            model, x = tiny(), tokens()
            with torch.no_grad():
                sealed = Core.forward(model, x)
                ld.configure(model, 'full')
                self.assertTrue(torch.equal(model(x), sealed))
                ld.configure(model, 'chunk512')
                self.assertTrue(torch.equal(model(x), sealed))
                ld.configure(model, 'win512')
                windowed = model(x)
            self.assertTrue(torch.allclose(windowed[:, :16], sealed[:, :16], atol=1e-5))                 # inside the window nothing is cut
            self.assertFalse(torch.allclose(windowed[:, 24:], sealed[:, 24:], atol=1e-4))                # past it the paths really are restricted

    def test_reset_replaces_the_carried_state_by_the_initial_state_every_period_and_nowhere_else(self):
        with small_world():
            model, x = tiny(), tokens(length=40)
            with torch.no_grad():
                model.initial.copy_(torch.randn_like(model.initial))                                       # a distinguishable initial state
            received, returned = {}, {}
            original = Core.decode

            def record(self_, e, s, mk, mv, t):
                if s is not None:
                    received[t] = s['s'].clone()
                out = original(self_, e, s, mk, mv, t)
                returned[t] = out[1]['s'].clone()
                return out

            ld.configure(model, 'reset512')
            Core.decode = record
            try:
                with torch.no_grad():
                    model(x)
            finally:
                Core.decode = original
            for t in range(1, 40):
                if t % 16 == 0:
                    self.assertTrue(torch.equal(received[t], model.initial.expand(3, -1)), t)
                else:
                    self.assertTrue(torch.equal(received[t], returned[t - 1]), t)

    def test_configure_validates_and_restores(self):
        model = tiny()
        with self.assertRaises(ValueError):
            ld.configure(model, 'nope')
        ld.configure(model, 'win512')
        self.assertEqual((model.cross, model.enc, model.reset_every), (ld.TRAIN_LENGTH, ld.TRAIN_LENGTH, None))
        ld.configure(model, 'reset512')
        self.assertEqual((model.cross, model.enc, model.reset_every), (None, None, ld.TRAIN_LENGTH))


def make_result(full, win, reset, control=0.0, n=12, noise=0.0):
    """Per-article bin values built so that D (against a zero reference) is exactly the given mean per bin (plus alternating noise of the given size)."""
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    def values(means):
        return {b: (m + noise * sign).tolist() for b, m in means.items()}
    bins = ('0-512', '512-1024', '1024-2048', '2048-4096')
    zero = {b: 0.0 for b in bins}
    return dict(snapshot='x.pt', n=n, length=4096, control=control,
                per_doc=dict(full=values(dict(zip(bins, full))), win512=values(dict(zip(bins, win))), reset512=values(dict(zip(bins, reset))),
                             chunk512={b: [0.0] * n for b in zero}))


class Reading(unittest.TestCase):
    def verdict(self, **kw):
        return ld.reading(make_result(**kw))

    def test_the_registered_verdicts(self):
        flat = (0, 0, 0, 0)
        self.assertEqual(self.verdict(full=(0, 0.01, 0.02, 0.04), win=flat, reset=flat)['verdict'], 'CONTINUATION IS FINE')
        self.assertEqual(self.verdict(full=(0, 0.01, 0.10, 0.04), win=flat, reset=flat)['verdict'], 'MILD')                 # between the two thresholds
        self.assertEqual(self.verdict(full=(0, 0.20, 0.02, 0.04), win=flat, reset=flat)['verdict'], 'MILD')                 # a bin below 1,024 cannot trigger HURTS ...
        self.assertEqual(self.verdict(full=(0, 0.01, 0.30, 0.40), win=flat, reset=flat)['verdict'], 'CONTINUATION HURTS')
        self.assertEqual(self.verdict(full=(0, 0.01, 0.16, 0.0), win=flat, reset=flat, noise=0.2)['verdict'], 'MILD')       # ... and 3 SE is required (noise 0.2 on 12 articles)
        self.assertEqual(self.verdict(full=(0, 0, 0.2, 0.3), win=flat, reset=flat, control=0.01)['verdict'], 'CONTROL FAILED')

    def test_the_attribution_buckets(self):
        full = (0, 0.1, 0.5, 0.9)
        self.assertEqual(self.verdict(full=full, win=(0, 0, 0.1, 0.1), reset=full)['attribution'], 'MEMORY PATH')           # window removes 0.8 of 0.9, reset nothing
        self.assertEqual(self.verdict(full=full, win=full, reset=(0, 0, 0.1, 0.1))['attribution'], 'STATE PATH')
        self.assertEqual(self.verdict(full=full, win=(0, 0, 0.3, 0.5), reset=(0, 0, 0.3, 0.5))['attribution'], 'BOTH')       # each removes 0.4 (>= a third, < two thirds)
        self.assertEqual(self.verdict(full=full, win=(0, 0, 0.5, 0.8), reset=(0, 0, 0.5, 0.8))['attribution'], 'NEITHER')    # each removes 0.1 (< a third)
        self.assertEqual(self.verdict(full=full, win=(0, 0, 0.1, 0.1), reset=(0, 0, 0.3, 0.5))['attribution'], 'BOTH')       # window 0.8, reset 0.4: both at least a third
        self.assertEqual(self.verdict(full=full, win=(0, 0, 0.3, 0.5), reset=(0, 0, 0.5, 0.8))['attribution'], 'MIXED')      # window removes 0.4 (a third to two thirds), reset only 0.1: none of the four buckets
        self.assertEqual(self.verdict(full=full, win=(0, 0, 0.1, 0.1), reset=(0, 0, 0.5, 0.7))['attribution'], 'MEMORY PATH')   # window 0.8, reset 0.2: memory

    def test_incomplete_without_a_bin_at_or_above_1024(self):
        result = make_result((0, 0.5, 0.5, 0.5), (0,) * 4, (0,) * 4)
        for c in ld.CONDITIONS:
            result['per_doc'][c] = {b: v for b, v in result['per_doc'][c].items() if b in ('0-512', '512-1024')}
        self.assertEqual(ld.reading(result)['verdict'], 'INCOMPLETE')


class EndToEnd(unittest.TestCase):
    def test_a_run_on_a_tiny_model_has_every_part_and_a_clean_control(self):
        with small_world():
            docs = np.random.default_rng(0).integers(0, 256, (4, 65)).astype(np.int64)
            result = ld.run('tiny.pt', docs=docs, model=tiny(), batch=2, log=lambda m: None)
            self.assertLess(result['control'], 1e-3)                                                          # all conditions coincide with the reference on the first window
            self.assertEqual(set(result['per_doc']), set(ld.CONDITIONS))
            self.assertEqual(set(result['per_doc']['full']), {'0-16', '16-32', '32-64'})
            self.assertEqual(len(result['per_doc']['full']['16-32']), 4)
            self.assertEqual(result['reading']['verdict'], 'INCOMPLETE')                                      # no bin at or above 1,024 positions at this scale
            self.assertIn('reading', ld.describe(result))

    def test_bin_means_use_only_positions_with_enough_context(self):
        with small_world():
            loss = np.zeros((2, 64))
            loss[:, [p for p in range(64) if p % 16 < 8]] = 100.0                                              # the early in-chunk offsets are poisoned
            means = ld.bin_means(loss, 64)
            self.assertTrue(all(float(v.max()) == 0.0 for v in means.values()))


if __name__ == '__main__':
    unittest.main()
