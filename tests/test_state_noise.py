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


class TrainingHook(unittest.TestCase):
    def test_noisy_training_returns_the_plain_arm_and_a_clean_deterministic_evaluation(self):
        from horizon import depth_run as dr
        for arm, width in (('core', 32), ('gru', 16), ('lstm', 16)):
            model, trained = dr.train('perm3', arm, width, seed=1, steps=3, batch=8, curriculum=[4], state_noise=0.1)
            self.assertEqual(trained['state_noise'], 0.1)
            self.assertNotIsInstance(model, sn.NoisyCore)
            self.assertNotIn('sigma', model.__dict__)
            x = tokens('perm3', 2, 6)
            model.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x), model(x)), arm)

    def test_noise_changes_what_is_learned_and_zero_noise_is_the_previous_behaviour(self):
        from horizon import depth_run as dr
        plain, _ = dr.train('perm3', 'gru', 16, seed=1, steps=5, batch=8, curriculum=[4])
        zero, trained = dr.train('perm3', 'gru', 16, seed=1, steps=5, batch=8, curriculum=[4], state_noise=0.0)
        noisy_, _ = dr.train('perm3', 'gru', 16, seed=1, steps=5, batch=8, curriculum=[4], state_noise=0.2)
        self.assertEqual(trained['state_noise'], 0.0)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(plain.parameters(), zero.parameters())))
        self.assertFalse(all(torch.equal(p, q) for p, q in zip(plain.parameters(), noisy_.parameters())))


class EndToEnd(unittest.TestCase):
    def test_a_saved_arm_reproduces_its_stored_result_at_sigma_zero_and_reports_windows(self):
        from horizon import depth_run as dr
        with tempfile.TemporaryDirectory() as d:
            for arm, width, exact in (('core', 32, True), ('gru', 16, False), ('lstm', 16, False)):
                out = Path(d) / f'{arm}.json'
                dr.run('perm3', arm, width, 1, steps=3, points=[16, 32], n_eval=8, out=out, batch=8, curriculum=[4])
                r = sn.margin_curve(out.with_suffix('.pt'), [0.0, 0.2], [16, 32], n=8, seed=1, batch=50)
                check = r['sigma0_vs_stored']
                self.assertTrue(check['reproduces'], (arm, check))
                if exact:
                    self.assertEqual(tuple(check['stored']), tuple(check['run']))
                for sigma, row in r['results'].items():
                    self.assertEqual(set(row['windows']), {'0-16', '16-32'}, (arm, sigma))
                    self.assertEqual(sum(c[0] for c in row['windows'].values()), row['all_events'])
                self.assertIn('by window', sn.describe(r))


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


def fake_model(task, arm, width, hazards, floor=None, reproduces=True):
    """A per-model result with the given early hazard per sigma (radii computed the way margin_curve does)."""
    rows = {}
    for sigma, h in hazards.items():
        rows[sigma] = dict(hazard_early=h, hazard_all=(floor if sigma == 0.0 else h), implied_radius_early=sn.implied_radius(sigma, h))
    return dict(task=task, arm=arm, width=width, results=rows, sigma0_vs_stored=dict(reproduces=reproduces))


class ReadingRule(unittest.TestCase):
    GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)

    def gaussian(self, radius):
        """Early hazards that follow erfc(radius / (sigma sqrt 2)) exactly."""
        return {s: (0.0 if s == 0.0 else math.erfc(radius / (s * math.sqrt(2)))) for s in self.GRID}

    def test_sigma_tol_and_spearman_and_radius_spread(self):
        rows = {s: dict(hazard_early=h, implied_radius_early=sn.implied_radius(s, h)) for s, h in self.gaussian(0.12).items()}
        self.assertEqual(sn.sigma_tol(rows), 0.05)                         # erfc(0.12 / (0.05 sqrt 2)) = erfc(1.70) ~ 0.0161 >= 1e-3, 0.02 gives ~ 1e-9
        self.assertEqual(sn.sigma_tol({s: dict(hazard_early=0.0) for s in self.GRID}), math.inf)
        self.assertAlmostEqual(sn.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertTrue(math.isnan(sn.spearman([1, 1, 1], [1, 2, 3])))
        # the standard grid puts at most two sigmas of a Gaussian world inside (1e-4, 0.3), so the spread is None there (the first version asserted `None or 0.0 == 0.0`:
        # vacuous, review finding F7); on a finer grid with three or more sigmas in range a Gaussian-basin model has one radius
        self.assertIsNone(sn.radius_spread(rows))
        fine = {s: dict(hazard_early=math.erfc(0.12 / (s * math.sqrt(2)))) for s in (0.04, 0.05, 0.06, 0.08)}
        for s, row in fine.items():
            row['implied_radius_early'] = sn.implied_radius(s, row['hazard_early'])
        self.assertIsNotNone(sn.radius_spread(fine))
        self.assertAlmostEqual(sn.radius_spread(fine), 0.0, places=6)

    def results(self, s4_radius, s5_radius, floors):
        out = [fake_model('perm4', 'core', 64, self.gaussian(s4_radius), floor=0.0),
               fake_model('perm5', 'core', 128, self.gaussian(s5_radius), floor=2.4e-3)]
        for (arm, width), (radius, floor) in zip((('gru', 64), ('lstm', 64), ('gru', 128), ('lstm', 112)), floors):
            out.append(fake_model('perm4', arm, width, self.gaussian(radius), floor=floor))
        return out

    def test_the_four_predictions_hold_for_a_consistent_gaussian_world(self):
        # S4 Core has a wide basin (O1: hazard at sigma 0.1 below 1e-3), S5 Core a narrower one (O2), the four RNNs order inversely to their floors (O4)
        floors = [(0.06, 7.6e-4), (0.08, 5.3e-4), (0.09, 4.5e-4), (0.12, 1.15e-4)]
        v = sn.verdict(self.results(0.5, 0.2, floors))
        self.assertTrue(v['O1'])
        self.assertTrue(v['O2'])
        self.assertTrue(v['O4'])
        self.assertLess(v['O4_spearman'], -0.6)
        # on the standard grid no Gaussian world has three sigmas inside (1e-4, 0.3): O3 is None for all (review finding F7: `is not False` passed vacuously)
        self.assertTrue(all(x is None for x in v['O3'].values()))

    def test_a_model_that_does_not_reproduce_is_not_read_and_a_bad_world_fails(self):
        floors = [(0.12, 7.6e-4), (0.09, 5.3e-4), (0.08, 4.5e-4), (0.06, 1.15e-4)]            # the wrong way round
        bad = self.results(0.1, 0.2, floors)                                                  # S4 Core narrower than S5 Core, hazard at 0.1 is high
        v = sn.verdict(bad)
        self.assertFalse(v['O1'])
        self.assertFalse(v['O2'])
        self.assertFalse(v['O4'])
        bad[0]['sigma0_vs_stored']['reproduces'] = False
        v = sn.verdict(bad)
        self.assertIsNone(v['O1'])
        self.assertIsNone(v['O2'])
        self.assertFalse(v['models']['perm4-core-w64']['valid'])


class MarginMaths(unittest.TestCase):
    def test_margin_inverts_the_gaussian_tail(self):
        for m in (3.04, 3.29, 4.89, 6.11):
            self.assertAlmostEqual(sn.margin_of(math.erfc(m / math.sqrt(2))), m, places=6)

    def test_implied_radius(self):
        self.assertAlmostEqual(sn.implied_radius(0.1, math.erfc(3.0 / math.sqrt(2))), 0.3, places=6)
        for bad in (0.0, None, 0.5, 0.7):
            self.assertIsNone(sn.implied_radius(0.1, bad))
        self.assertIsNone(sn.implied_radius(0.0, 0.01))


class HazardEstimator(unittest.TestCase):
    """Blinded review finding F1: events / (at_risk x width) saturates at 1 / width; the MLE of a constant per-step hazard does not."""

    @staticmethod
    def planted(h, points=(16, 64, 256, 1024), n=100, seed=0):
        """Windows from planted first-error times with a constant per-step hazard h, built through the production path (first_error_hazard -> pooled_hazards)."""
        import numpy as np
        from horizon import depth_run as dr
        rng = np.random.default_rng(seed)
        curve = {}
        for p in points:
            first = (rng.geometric(h, size=n) - 1) if h > 0 else np.full(n, 10 ** 9)
            target = torch.zeros(n, p, dtype=torch.long)
            pred = target.clone()
            for i, f in enumerate(first):
                if f < p:
                    pred[i, int(f)] = 1
            curve[p] = dict(hazard=dr.first_error_hazard(pred, target, dr.HAZARD_EDGES))
        (early, _, _), _, cells = sn.pooled_hazards(curve)
        return early, dict(windows={f'{a}-{b}': list(c) for (a, b), c in cells.items()})

    def test_the_old_estimator_is_capped_at_one_over_the_first_window_width_and_the_mle_is_not(self):
        old, row = self.planted(1.0)                                    # every sequence fails at position 0
        self.assertAlmostEqual(old, 1 / 16, places=3)                   # the cap
        self.assertGreater(sn.early_hazard_mle(row), 0.9)               # the true hazard is 1
        old, row = self.planted(0.3)
        self.assertLess(old, 0.07)
        self.assertAlmostEqual(sn.early_hazard_mle(row), 0.3, delta=0.04)

    def test_the_mle_recovers_small_hazards_too_and_returns_none_or_zero_at_the_edges(self):
        for h in (0.0005, 0.004):
            old, row = self.planted(h, n=2000, seed=1)
            self.assertAlmostEqual(sn.early_hazard_mle(row), h, delta=0.25 * h)
        self.assertEqual(sn.early_hazard_mle(self.planted(0.0)[1]), 0.0)
        self.assertIsNone(sn.hazard_mle([]))
        self.assertIsNone(sn.hazard_mle([(0, 0, 16)]))

    def world(self, r, sigmas, n, seed):
        rows = {}
        for i, s in enumerate(sigmas):
            h = 0.0 if s == 0 else math.erfc(r / (s * math.sqrt(2)))
            early, row = self.planted(h, n=n, seed=seed + i)
            rows[s] = dict(hazard_early=early, implied_radius_early=sn.implied_radius(s, early), windows=row['windows'])
        return rows

    def test_the_capped_estimator_manufactures_a_rejection_where_the_mle_says_not_evaluable(self):
        # review reproduction: a perfect Gaussian-basin world (r = 0.1) on the E-B grid has only ONE sigma inside (1e-4, 0.3) (sigma 0.05: 0.045; sigma 0.1 and 0.2 have
        # true hazards 0.32 and 0.62, outside). The capped estimator flattens those two at ~1/16 = 0.0625, which brings them INTO the selection with radii that grow
        # with sigma: CV 0.5, a spurious O3 rejection. The MLE keeps them outside: the model is simply not evaluable on this grid.
        rows = self.world(0.1, sn.SIGMAS, 100, 10)
        self.assertGreater(sn.radius_spread(rows), 0.35)
        self.assertIsNone(sn.radius_spread_mle(rows))

    def test_on_a_finer_grid_a_gaussian_basin_has_one_radius_with_the_mle(self):
        rows = self.world(0.1, (0.0, 0.03, 0.04, 0.05, 0.06, 0.07), 2000, 30)       # true hazards 8e-4 ... 0.15: all inside the selection
        self.assertLess(sn.radius_spread_mle(rows), 0.25)

    def test_verdict_records_both_spreads_and_the_primary_o3_keeps_the_pre_registered_estimator(self):
        rows = {s: dict(hazard_early=h, hazard_all=h, implied_radius_early=sn.implied_radius(s, h), windows={'0-16': [0, 1600]})
                for s, h in {0.0: 0.0, 0.05: 0.0, 0.1: 0.0}.items()}
        v = sn.verdict([dict(task='perm4', arm='gru', width=64, results=rows, sigma0_vs_stored=dict(reproduces=True))])
        self.assertIn('radius_cv_mle', v['models']['perm4-gru-w64'])
        self.assertIn('O3_mle', v)
        self.assertIsNone(v['O3']['perm4-gru-w64'])

if __name__ == '__main__':
    unittest.main()
