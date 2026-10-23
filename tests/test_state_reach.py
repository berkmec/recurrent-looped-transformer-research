"""E-L1X-F: the windowed reset arm differs from E-L1X's windowed arm only in the carried state, resets exactly where it says, loads both checkpoint formats, and the
registered reading is mechanical."""
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np
import torch

from core.model import Core
from mline import lm
from mline import long_doc as ld
from mline import state_reach as sr


def tiny(seed=5):
    return lm.build('plain', width=32, heads=2, window=4, alpha=0.1, seed=seed).eval()


def small_world():
    """E-L1X's constants scaled down: a 16-byte 'training window', so a 64-byte text is four of them and [32,64) is a verdict bin."""
    stack = ExitStack()
    stack.enter_context(mock.patch.object(ld, 'TRAIN_LENGTH', 16))
    stack.enter_context(mock.patch.object(ld, 'MIN_OFFSET', 8))
    stack.enter_context(mock.patch.object(ld, 'BINS', ((0, 16), (16, 32), (32, 64))))
    return stack


def tokens(batch=3, length=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (batch, length), generator=g)


class Conditions(unittest.TestCase):
    def test_configure_sets_the_windows_and_the_reset(self):
        model = tiny()
        with self.assertRaises(ValueError):
            sr.configure(model, 'full')
        sr.configure(model, 'winreset512')
        self.assertEqual((model.cross, model.enc, model.reset_every), (ld.TRAIN_LENGTH, ld.TRAIN_LENGTH, ld.TRAIN_LENGTH))
        sr.configure(model, 'win512')
        self.assertEqual((model.cross, model.enc, model.reset_every), (ld.TRAIN_LENGTH, ld.TRAIN_LENGTH, None))
        sr.configure(model, 'chunk512')
        self.assertEqual((model.cross, model.enc, model.reset_every), (None, None, None))

    def test_the_two_windowed_arms_agree_before_the_first_reset_and_differ_after(self):
        with small_world():
            model, x = tiny(), tokens()
            with torch.no_grad():
                model.initial.copy_(torch.randn_like(model.initial))
                sr.configure(model, 'win512')
                carried = model(x)
                sr.configure(model, 'winreset512')
                reset = model(x)
            self.assertTrue(torch.equal(carried[:, :16], reset[:, :16]))
            self.assertFalse(torch.allclose(carried[:, 16:], reset[:, 16:], atol=1e-6))

    def test_the_reset_arm_replaces_the_state_every_period_under_the_window(self):
        with small_world():
            model, x = tiny(), tokens(length=40)
            with torch.no_grad():
                model.initial.copy_(torch.randn_like(model.initial))
            received, returned, keys = {}, {}, {}
            original = Core.decode

            def record(self_, e, s, mk, mv, t):
                if s is not None:
                    received[t] = s['s'].clone()
                keys[t] = mk.shape[2]
                out = original(self_, e, s, mk, mv, t)
                returned[t] = out[1]['s'].clone()
                return out

            sr.configure(model, 'winreset512')
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
                self.assertLessEqual(keys[t], 16, t)                                                     # the cross-attention window holds


class Loading(unittest.TestCase):
    def test_a_snapshot_and_a_run_checkpoint_load_the_same_weights(self):
        model = tiny(seed=9)
        with tempfile.TemporaryDirectory() as d:
            snap, ckpt = os.path.join(d, 'step004000.pt'), os.path.join(d, 'latest.pt')
            torch.save({k: v.clone() for k, v in model.state_dict().items()}, snap)
            torch.save({'format': 1, 'step': 4321, 'meta': {'note': 'x'}, 'model': model.state_dict(), 'optimizer': {}}, ckpt)
            a, step_a = sr.load(snap, width=32, heads=2)
            b, step_b = sr.load(ckpt, width=32, heads=2)
        self.assertEqual((step_a, step_b), (4000, 4321))
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(a.state_dict()[k], v), k)
            self.assertTrue(torch.equal(b.state_dict()[k], v), k)


class Measures(unittest.TestCase):
    def test_run_on_a_small_world_is_controlled_and_the_profile_starts_away_from_zero(self):
        with small_world():
            model = tiny()
            with torch.no_grad():
                model.initial.copy_(torch.randn_like(model.initial))
            docs = tokens(batch=4, length=65, seed=3).numpy().astype(np.int64)
            with mock.patch.object(sr, 'PROFILE_STEPS', (1, 2, 4, 8, 15)):
                r = sr.run('x', docs=docs, model=model, step=7, log=lambda m: None, profile_docs=2)
        self.assertLess(r['control'], 1e-5)
        self.assertEqual(set(r['per_doc']), set(sr.CONDITIONS))
        self.assertEqual(sorted(r['s_bands']), ['10-12', '12-14', '14-16', '8-10'])
        self.assertEqual(list(r['state_profile']), ['1', '2', '4', '8', '15'])
        self.assertGreater(r['state_profile']['1'], 0.0)
        self.assertIn(r['reading']['verdict'], ('STATE EXTENDS CONTEXT', 'STATE HURTS', 'NO STATE CONTEXT', 'SMALL'))

    def test_band_means_pool_only_the_high_positions_of_each_band(self):
        with small_world():
            diff = np.zeros((2, 64))
            positions = np.arange(64)
            diff[:, (positions >= 32) & (positions % 16 >= 14)] = 1.0                                   # only the last band, only at or above 32
            diff[:, positions < 32] = 5.0                                                               # below 2 x the training length: ignored
            bands = sr.band_means(diff, 64)
        self.assertEqual(bands['14-16'].tolist(), [1.0, 1.0])
        self.assertEqual(bands['8-10'].tolist(), [0.0, 0.0])


def result(s_means, noise=0.0, n=12, control=0.0):
    """per_doc built so that S (winreset512 - win512) per bin is exactly the given mean (plus alternating noise of the given size)."""
    bins = ['0-512', '512-1024', '1024-2048', '2048-4096']
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    win = {b: [1.0] * n for b in bins}
    reset = {b: (1.0 + m + noise * sign).tolist() for b, m in zip(bins, [0.0] + list(s_means))}
    return dict(control=control, per_doc=dict(win512=win, winreset512=reset, chunk512=win))


class Reading(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(sr.reading(result([0.0, 0.0, 0.012]))['verdict'], 'STATE EXTENDS CONTEXT')
        self.assertEqual(sr.reading(result([0.0, 0.010, 0.0]))['verdict'], 'STATE EXTENDS CONTEXT')      # inclusive threshold
        self.assertEqual(sr.reading(result([0.0, 0.0, 0.012], noise=0.05))['verdict'], 'SMALL')           # big but not 3 SE
        self.assertEqual(sr.reading(result([0.0, -0.02, 0.0]))['verdict'], 'STATE HURTS')
        self.assertEqual(sr.reading(result([0.5, 0.003, -0.003]))['verdict'], 'NO STATE CONTEXT')         # [512,1024) does not count
        self.assertEqual(sr.reading(result([0.0, 0.005, 0.0]))['verdict'], 'SMALL')
        self.assertEqual(sr.reading(result([0.0, 0.0, 0.0], control=2e-3))['verdict'], 'CONTROL FAILED')


if __name__ == '__main__':
    unittest.main()
