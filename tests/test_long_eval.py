"""E-X long evaluation: the reading rule on planted hazards, the exactness horizon, and a smoke run on a tiny saved arm."""
import tempfile
import unittest
from pathlib import Path

from horizon import depth_run as dr
from horizon import long_eval as le


def result(windows, error_free=None):
    """A result with one evaluation point per window start; `windows` = {(start, stop): (at_risk, first_errors)}."""
    curve = {}
    for i, ((a, b), (at_risk, events)) in enumerate(windows.items()):
        curve[str(b)] = dict(length=b, n=100, error_free=(100 if error_free is None else error_free), hazard=[dict(start=a, stop=b, at_risk=at_risk, first_errors=events, rate=None)])
    return dict(arm='gru', task='perm4', width=64, n=100, lengths=[b for (_, b) in windows], curve=curve, train=dict(parameters=1))


class Reading(unittest.TestCase):
    def test_clean_needs_enough_exposure_and_short_otherwise(self):
        clean = result({(1024, 2048): (100, 0), (2048, 4096): (100, 0), (4096, 8192): (100, 0)})            # 100 x (1024 + 2048 + 4096) = 716,800 < 1e6
        self.assertEqual(le.reading(clean)['verdict'], 'SHORT')
        clean = result({(1024, 2048): (100, 0), (2048, 4096): (100, 0), (4096, 8192): (100, 0), (8192, 16384): (100, 0)})   # + 819,200 -> 1,536,000
        self.assertEqual(le.reading(clean)['verdict'], 'CLEAN')

    def test_floor_break_and_sparse(self):
        floor = result({(1024, 2048): (100, 8), (2048, 4096): (92, 16), (4096, 8192): (76, 30)})          # rates 7.8e-5, 8.5e-5, 9.6e-5: no 10x rise
        self.assertEqual(le.reading(floor)['verdict'], 'FLOOR')
        rising = result({(1024, 2048): (100, 1), (2048, 4096): (99, 20), (4096, 8192): (79, 79)})          # 1e-5 -> 1e-4 -> 2.4e-4 per step
        self.assertEqual(le.reading(rising)['verdict'], 'BREAK')
        from_zero = result({(1024, 2048): (100, 0), (2048, 4096): (100, 3), (4096, 8192): (97, 40)})        # no events in the first window, 40 in the last
        self.assertEqual(le.reading(from_zero)['verdict'], 'BREAK')
        sparse = result({(1024, 2048): (100, 0), (2048, 4096): (100, 2), (4096, 8192): (98, 1)})
        self.assertEqual(le.reading(sparse)['verdict'], 'SPARSE')

    def test_windows_below_the_beyond_limit_are_ignored(self):
        r = result({(256, 1024): (100, 50), (1024, 2048): (100, 0), (2048, 4096): (100, 0), (4096, 8192): (100, 0), (8192, 16384): (100, 0)})
        self.assertEqual(le.reading(r)['verdict'], 'CLEAN')                                                   # the 50 early errors do not count: they are inside 1,024

    def test_exactness_horizon_is_the_longest_length_with_every_sequence_error_free(self):
        r = result({(1024, 2048): (100, 0), (2048, 4096): (100, 0)})
        r['curve']['2048']['error_free'] = 100
        r['curve']['4096']['error_free'] = 99
        self.assertEqual(le.exactness_horizon(r), 2048)
        r['curve']['2048']['error_free'] = 90
        self.assertIsNone(le.exactness_horizon(r))


class Smoke(unittest.TestCase):
    def test_measure_runs_on_a_saved_arm_and_the_hazard_windows_reach_the_requested_lengths(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'gru', 16, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            r = le.measure(out.with_suffix('.pt'), [64, 256], 6, batch=3)
            self.assertEqual(sorted(int(k) for k in r['curve']), [64, 256])
            top = max(w['stop'] for w in r['curve']['256']['hazard'])
            self.assertEqual(top, 256)                                                                         # the last window is clipped to the sequence length
            self.assertIn('exactness horizon', le.describe(r))


if __name__ == '__main__':
    unittest.main()
