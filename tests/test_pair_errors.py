"""E-Q pair errors: the at-risk bookkeeping, the concentration statistic against its null, the split-half correlation and the reading rule."""
import unittest

import numpy as np
import torch

from horizon import pair_errors as pe


class Bookkeeping(unittest.TestCase):
    def test_visits_stop_at_the_first_error_and_pairs_use_the_true_previous_state(self):
        vocab = 3
        x = torch.tensor([[0, 1, 2, 1, 0]])
        target = torch.tensor([[0, 1, 2, 2, 0]])             # true states
        pred = target.clone()
        pred[0, 3] = 0                                        # first error at position 3
        visits, errors = pe.pair_counts(x, target, pred, vocab, start=1)
        # at-risk positions 1, 2, 3: pairs (target[0], x[1]) = (0,1) -> 1, (target[1], x[2]) = (1,2) -> 5, (target[2], x[3]) = (2,1) -> 7
        self.assertEqual(int(visits.sum()), 3)
        self.assertEqual((int(visits[1]), int(visits[5]), int(visits[7])), (1, 1, 1))
        self.assertEqual(int(errors.sum()), 1)
        self.assertEqual(int(errors[7]), 1)

    def test_error_free_sequences_are_visited_to_the_end_and_windows_cut(self):
        vocab = 3
        x = torch.randint(0, vocab, (2, 10))
        target = torch.randint(0, vocab, (2, 10))
        visits, errors = pe.pair_counts(x, target, target.clone(), vocab, start=1, stop=6)
        self.assertEqual(int(visits.sum()), 2 * 5)            # positions 1..5 of both sequences
        self.assertEqual(int(errors.sum()), 0)


class Statistics(unittest.TestCase):
    def test_concentration_is_about_one_for_uniform_errors_and_large_for_a_few_pairs(self):
        rng = np.random.default_rng(0)
        uniform = np.bincount(rng.integers(0, 576, size=150), minlength=576)
        self.assertTrue(0.7 < pe.concentration(uniform) < 1.6, pe.concentration(uniform))
        planted = np.zeros(576, dtype=np.int64)
        planted[[3, 50, 200]] = 50
        self.assertGreater(pe.concentration(planted), 20)
        self.assertIsNone(pe.concentration(np.zeros(576, dtype=np.int64)))

    def test_correlation_of_halves(self):
        a = np.zeros(100)
        a[[3, 9, 40]] = [5, 4, 6]
        self.assertAlmostEqual(pe.correlation(a, a * 2 + 0.0), 1.0, places=9)
        self.assertIsNone(pe.correlation(a, np.zeros(100)))
        rng = np.random.default_rng(1)
        self.assertLess(abs(pe.correlation(rng.poisson(0.3, 5000).astype(float), rng.poisson(0.3, 5000).astype(float))), 0.1)

    def test_reading_rule(self):
        self.assertEqual(pe.reading(8.0, 0.6, 100, 90), 'DETERMINISTIC')
        self.assertEqual(pe.reading(1.1, 0.02, 100, 90), 'NOISE-LIKE')
        self.assertEqual(pe.reading(2.0, 0.2, 100, 90), 'MIXED')
        self.assertEqual(pe.reading(8.0, 0.6, 20, 90), 'TOO FEW ERRORS')
        self.assertEqual(pe.reading(None, 0.6, 100, 90), 'TOO FEW ERRORS')

    def test_marginals_sum_correctly(self):
        errors = np.zeros(9, dtype=np.int64)
        errors[[0, 1, 4]] = [3, 3, 4]
        m = pe.marginals(errors, 3, top=1)
        self.assertAlmostEqual(m['top_pairs_share'], 0.4)
        self.assertAlmostEqual(m['top_states_share'], 0.6)          # state 0 holds the pairs (0,0) and (0,1): 3 + 3 of 10
        self.assertAlmostEqual(m['top_tokens_share'], 0.7)          # token 1 holds the pairs (0,1) and (1,1): 3 + 4 of 10


class EndToEnd(unittest.TestCase):
    def test_analyse_runs_on_a_saved_arm(self):
        import tempfile
        from pathlib import Path
        from horizon import depth_run as dr
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'gru', 16, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            r = pe.analyse(out.with_suffix('.pt'), n=40, length=16, windows=((1, 16),), batch=20)
            w = r['windows']['1-16']
            self.assertEqual(w['pairs'], 36)
            self.assertGreater(w['visits'], 0)
            self.assertIn('reading', w)
            self.assertIn('hazard', pe.describe(r))


if __name__ == '__main__':
    unittest.main()
