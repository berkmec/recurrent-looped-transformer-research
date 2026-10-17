"""I1-c code annex: --warmup / --adam-eps are opt-in additions; the defaults are the previous behaviour, bit for bit."""
import unittest
from unittest import mock

import torch

from horizon import depth_run as dr


class Warmup(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_ramp_boundaries(self):
        # the ramp is lr * step / warmup and ends exactly at lr at step == warmup
        _, info = dr.train('perm3', 'gru', 16, seed=1, steps=5, batch=8, curriculum=[4], lr=0.01, warmup=10)
        self.assertAlmostEqual(info['lr_last'], 0.005)
        _, info = dr.train('perm3', 'gru', 16, seed=1, steps=4, batch=8, curriculum=[4], lr=0.01, warmup=4)
        self.assertEqual(info['lr_last'], 0.01)
        _, info = dr.train('perm3', 'gru', 16, seed=1, steps=10, batch=8, curriculum=[4], lr=0.01, warmup=4)
        self.assertEqual(info['lr_last'], 0.01)

    def test_decay_min_rule(self):
        # with a decay the rate is the lower of the ramp and the decay curve
        _, info = dr.train('perm3', 'gru', 16, seed=1, steps=20, batch=8, curriculum=[4], lr=0.01, warmup=4, lr_decay_fraction=0.5)
        self.assertAlmostEqual(info['lr_last'], 0.01 * 1 / 10)   # decay at the last step: (20 - 20 + 1) / 10; the ramp is back at lr
        _, info = dr.train('perm3', 'gru', 16, seed=1, steps=10, batch=8, curriculum=[4], lr=0.01, warmup=100)
        self.assertAlmostEqual(info['lr_last'], 0.001)           # a run shorter than the ramp ends mid-ramp

    def test_warmup_zero_is_previous_behaviour(self):
        _, info = dr.train('perm3', 'gru', 16, seed=1, steps=10, batch=8, curriculum=[4], lr=0.01)
        self.assertEqual((info['lr_last'], info['warmup']), (0.01, 0))
        _, info = dr.train('perm3', 'gru', 16, seed=1, steps=10, batch=8, curriculum=[4], lr=0.01, lr_decay_fraction=0.5)
        self.assertAlmostEqual(info['lr_last'], 0.01 * 1 / 5)    # test_depth_run.py's decay values, unchanged
        self.assertEqual(info['warmup'], 0)

    def test_default_run_is_deterministic(self):
        a, ia = dr.train('perm3', 'gru', 16, seed=1, steps=8, batch=8, curriculum=[4], lr=0.01, log_every=1)
        b, ib = dr.train('perm3', 'gru', 16, seed=1, steps=8, batch=8, curriculum=[4], lr=0.01, log_every=1)
        self.assertEqual(ia['history'], ib['history'])
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())))


class AdamEps(unittest.TestCase):
    def test_eps_reaches_the_optimizer(self):
        # the smoke proof of the default path lives in the annex evidence; here the flag must reach AdamW as-is
        captured = []
        real = torch.optim.AdamW

        def spy(*args, **kwargs):
            captured.append(kwargs.get('eps'))
            return real(*args, **kwargs)

        with mock.patch.object(dr.torch.optim, 'AdamW', spy):
            dr.train('perm3', 'gru', 8, seed=1, steps=1, batch=4, curriculum=[3], adam_eps=1e-6)
            dr.train('perm3', 'gru', 8, seed=1, steps=1, batch=4, curriculum=[3])
        self.assertEqual(captured, [1e-6, 1e-8])
