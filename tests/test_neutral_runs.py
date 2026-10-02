"""E-N neutral runs: identity padding is exactly neutral for the oracle, the window statistics count the right things, the reading rule separates work from time."""
import tempfile
import unittest
from pathlib import Path

import torch

from horizon import depth_tasks as dt
from horizon import neutral_runs as nr


class TokensAndOracle(unittest.TestCase):
    def test_token_zero_is_the_identity_and_padding_leaves_the_label_unchanged(self):
        x = nr.neutral_tokens('perm4', 40, 60, 0.9, seed=1)
        y = dt.oracle('perm4', x)
        self.assertTrue(((x == 0).float().mean() > 0.85) and ((x == 0).float().mean() < 0.95))
        same = (y[:, 1:] == y[:, :-1])
        self.assertTrue(bool(same[x[:, 1:] == 0].all()))                   # an identity token never changes the running product
        self.assertTrue(bool((dt.oracle('perm4', torch.zeros(3, 9, dtype=torch.long)) == 0).all()))

    def test_p_zero_is_close_to_uniform_and_p_one_is_all_identity(self):
        self.assertAlmostEqual(float((nr.neutral_tokens('perm5', 200, 80, 0.0, 1) == 0).float().mean()), 1 / 120, delta=0.004)
        self.assertTrue(bool((nr.neutral_tokens('perm5', 5, 10, 1.0, 1) == 0).all()))

    def test_draws_are_reproducible_and_differ_by_p(self):
        a, b = nr.neutral_tokens('perm4', 5, 20, 0.5, 3), nr.neutral_tokens('perm4', 5, 20, 0.5, 3)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, nr.neutral_tokens('perm4', 5, 20, 0.9, 3)))


class WindowStats(unittest.TestCase):
    def test_events_exposure_and_updates(self):
        x = torch.tensor([[1] * 8 + [0] * 8, [2] * 16])           # 8 real then 8 identity tokens; 16 real tokens
        target = torch.zeros(2, 16, dtype=torch.long)
        pred = target.clone()
        pred[0, 10] = 1                                           # sequence 0: first error at position 10
        stats = nr.window_stats(pred, target, x, edges=(0, 8, 12, 16))
        self.assertEqual(stats[(0, 8)], (0, 2 * 8, 8 + 8))
        self.assertEqual(stats[(8, 12)], (1, 2 * 4, 0 + 4))       # both at risk; sequence 0 has 0 real tokens there, sequence 1 has 4
        self.assertEqual(stats[(12, 16)], (0, 1 * 4, 4))          # only sequence 1 is still error-free


def fake(rows):
    """rows: {p_id: (late first errors, step exposure, update exposure)}."""
    return dict(results={p: dict(late_events=e, late_step_exposure=s, late_update_exposure=u, rate_per_step=e / s, rate_per_update=e / u if u else None)
                         for p, (e, s, u) in rows.items()})


class ReadingRule(unittest.TestCase):
    def test_errors_that_scale_with_updates_are_work(self):
        # base: 240 errors in 100,000 steps (2.4e-3 per step and per update); at p_id 0.5 half the steps are updates, at 0.9 a tenth
        r = fake({0.0: (240, 100_000, 100_000), 0.5: (120, 100_000, 50_000), 0.9: (24 * 10, 1_000_000, 100_000)})
        self.assertEqual(nr.classify(r)['verdict'], 'WORK')

    def test_errors_that_ignore_the_updates_are_time(self):
        r = fake({0.0: (240, 100_000, 100_000), 0.5: (230, 100_000, 50_000), 0.9: (260, 100_000, 10_000)})
        self.assertEqual(nr.classify(r)['verdict'], 'TIME')

    def test_a_blow_up_under_neutral_runs_is_a_failure_and_missing_baselines_are_flagged(self):
        r = fake({0.0: (240, 100_000, 100_000), 0.5: (900, 100_000, 50_000), 0.9: (1500, 100_000, 10_000)})
        self.assertEqual(nr.classify(r)['verdict'], 'NEUTRAL FAILURE')
        zero = fake({0.0: (0, 100_000, 100_000), 0.5: (0, 100_000, 50_000)})
        zero['results'][0.0]['rate_per_step'] = 0.0
        self.assertEqual(nr.classify(zero)['verdict'], 'NO BASELINE ERRORS')
        few = fake({0.0: (240, 100_000, 100_000), 0.5: (3, 100_000, 50_000), 0.9: (1, 100_000, 10_000)})
        self.assertEqual(nr.classify(few)['verdict'], 'TOO FEW EVENTS')


class EndToEnd(unittest.TestCase):
    def test_measure_runs_on_a_saved_arm_and_the_json_keys_survive_a_round_trip(self):
        from horizon import depth_run as dr
        with tempfile.TemporaryDirectory() as d:
            for arm, width in (('core', 32), ('gru', 16)):
                out = Path(d) / f'{arm}.json'
                dr.run('perm3', arm, width, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
                r = nr.measure(out.with_suffix('.pt'), p_ids=(0.0, 0.9), length=160, n=6, batch=6)
                self.assertEqual(set(r['results']), {0.0, 0.9})
                self.assertIn('late_events', r['results'][0.0])
                self.assertIn('reading', nr.describe(r))


if __name__ == '__main__':
    unittest.main()
