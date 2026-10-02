"""E-B state-noise harness: sigma = 0 is the plain arm, the noise is relative and starts at the second step, parameters are shared, the margin maths inverts."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from core.model import Core
from horizon import state_noise as sn
from horizon.depth_models import make_arm


def tokens(task='perm4', batch=3, length=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    from horizon import depth_tasks as dt
    return torch.randint(0, dt.vocab(task), (batch, length), generator=g)


class ZeroNoiseIsThePlainArm(unittest.TestCase):
    def test_gru_and_lstm_unrolled_cells_match_the_library_layers(self):
        for arm in ('gru', 'lstm'):
            model = make_arm(arm, 'perm4', 32, 1).eval()
            x = tokens()
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(x), sn.noisy(copy.deepcopy(model), 0.0, 1).eval()(x), atol=1e-5), arm)

    def test_core_is_unchanged_at_sigma_zero(self):
        model = make_arm('core', 'perm4', 32, 1).eval()
        x = tokens()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), sn.noisy(copy.deepcopy(model), 0.0, 1)(x)))


class NoiseProperties(unittest.TestCase):
    def test_noise_changes_the_logits_but_not_the_first_step_and_is_seeded(self):
        for arm in ('core', 'gru', 'lstm'):
            model = make_arm(arm, 'perm4', 32, 1).eval()
            x = tokens()
            with torch.no_grad():
                clean = model(x)
                a = sn.noisy(copy.deepcopy(model), 0.2, 7).eval()(x)
                b = sn.noisy(copy.deepcopy(model), 0.2, 7).eval()(x)
                c = sn.noisy(copy.deepcopy(model), 0.2, 8).eval()(x)
            self.assertTrue(torch.allclose(a[:, 0], clean[:, 0], atol=1e-5), arm)       # no noise on the initial state
            self.assertFalse(torch.allclose(a[:, 1:], clean[:, 1:], atol=1e-4), arm)
            self.assertTrue(torch.equal(a, b), arm)                                      # same seed, same noise
            self.assertFalse(torch.equal(a, c), arm)

    def test_noise_is_relative_to_the_state_scale(self):
        h = torch.randn(4, 16)
        one = sn.kick(h, 0.1, torch.Generator().manual_seed(3)) - h
        ten = sn.kick(10 * h, 0.1, torch.Generator().manual_seed(3)) - 10 * h
        self.assertTrue(torch.allclose(ten, 10 * one, atol=1e-5))
        self.assertTrue(torch.equal(sn.kick(h, 0.0, None), h))

    def test_noise_size_matches_sigma_times_rms(self):
        h = torch.randn(512, 64)
        d = sn.kick(h, 0.05, torch.Generator().manual_seed(1)) - h
        ratio = d.square().mean(-1).sqrt() / h.square().mean(-1).sqrt()
        self.assertAlmostEqual(float(ratio.mean()), 0.05, delta=0.003)

    def test_the_wrapper_shares_parameters_so_it_can_train(self):
        model = make_arm('lstm', 'perm4', 32, 1)
        wrapper = sn.noisy(model, 0.1, 1)
        wrapper(tokens()).square().mean().backward()
        self.assertIsNotNone(model.rnn.weight_hh_l0.grad)
        self.assertGreater(float(model.rnn.weight_hh_l0.grad.abs().sum()), 0)
        self.assertEqual({id(p) for p in model.parameters()}, {id(p) for p in wrapper.parameters()})

    def test_only_core_gru_lstm_are_supported(self):
        with self.assertRaises(ValueError):
            sn.noisy(make_arm('tfm', 'perm4', 32, 1), 0.1, 1)


class StoredCheck(unittest.TestCase):
    def test_stored_early_sums_the_windows_up_to_64_over_the_given_points(self):
        with tempfile.TemporaryDirectory() as d:
            weights = Path(d) / 'm.pt'
            (Path(d) / 'm.json').write_text(json.dumps(dict(curve={
                '16': dict(hazard=[dict(start=0, stop=16, at_risk=100, first_errors=1)]),
                '64': dict(hazard=[dict(start=0, stop=16, at_risk=100, first_errors=0), dict(start=16, stop=32, at_risk=90, first_errors=2),
                                   dict(start=32, stop=64, at_risk=88, first_errors=1)]),
                '256': dict(hazard=[dict(start=0, stop=16, at_risk=100, first_errors=0), dict(start=64, stop=128, at_risk=50, first_errors=5)])})))
            self.assertEqual(sn.stored_early(weights, [16, 64]), (4, 1600 + 1600 + 1440 + 2816))
            self.assertEqual(sn.stored_early(weights, [16, 64, 256]), (4, 1600 + 1600 + 1440 + 2816 + 1600))      # [64, 128) is not early
            self.assertIsNone(sn.stored_early(weights, [16, 999]))
        self.assertIsNone(sn.stored_early(Path('does-not-exist.pt'), [16]))

    def test_reproduces_is_exact_for_the_core_and_tolerant_for_unrolled_cells(self):
        self.assertTrue(sn.reproduces((5, 1000), (5, 1000), True))
        self.assertFalse(sn.reproduces((5, 1000), (6, 1000), True))
        self.assertTrue(sn.reproduces((5, 1000), (7, 1005), False))
        self.assertFalse(sn.reproduces((5, 1000), (8, 1000), False))
        self.assertFalse(sn.reproduces((5, 1000), (5, 1020), False))
        self.assertIsNone(sn.reproduces(None, (1, 1), True))


class MarginMaths(unittest.TestCase):
    def test_margin_inverts_the_gaussian_tail(self):
        for m in (3.04, 3.29, 4.89, 6.11):
            self.assertAlmostEqual(sn.margin_of(math.erfc(m / math.sqrt(2))), m, places=6)

    def test_implied_radius(self):
        self.assertAlmostEqual(sn.implied_radius(0.1, math.erfc(3.0 / math.sqrt(2))), 0.3, places=6)
        for bad in (0.0, None, 0.5, 0.7):
            self.assertIsNone(sn.implied_radius(0.1, bad))
        self.assertIsNone(sn.implied_radius(0.0, 0.01))


if __name__ == '__main__':
    unittest.main()
