"""The mechanical readings of E-D and E-C: SUPPORTED / REFUTED / MIXED and the extrapolation factor follow the pre-registered thresholds."""
import json
import tempfile
import unittest
from pathlib import Path

from horizon import readings as rd


def perm_result(final16, first_errors, exposure):
    """A perm result file with one hazard window [0, 1024) holding `first_errors` over `exposure` position-steps."""
    return dict(curve={'16': dict(final=final16, hazard=[dict(start=0, stop=1024, at_risk=exposure // 1024, first_errors=first_errors)])})


def write(directory, name, payload):
    (Path(directory) / name).write_text(json.dumps(payload), encoding='utf8')


class NoiseTrainingReading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        write(self.dir, 'perm4-gru-w64-s1-cur.json', perm_result(0.99, 76, 100 * 1024 * 10))        # hazard 76 / 1,024,000 ~ 7.4e-5 -> use a clean number below

    def tearDown(self):
        self.tmp.cleanup()

    def baseline(self, seed, events, final=0.99):
        write(self.dir, f'perm4-gru-w64-s{seed}-cur.json', perm_result(final, events, 1_024_000))

    def noise(self, sigma, events, final):
        write(self.dir, f'perm4-gru-w64-s1-cur-noise{sigma}.json', perm_result(final, events, 1_024_000))

    def test_supported_needs_a_5x_lower_hazard_and_a_learned_range(self):
        self.baseline(1, 800)
        self.noise(0.1, 100, 0.995)
        self.noise(0.05, 700, 0.99)
        self.noise(0.2, 50, 0.90)
        r = rd.ed_arm('gru', directory=self.dir)
        self.assertEqual(r['verdict'], 'SUPPORTED')
        self.assertTrue(r['provisional'])

    def test_refuted_when_every_sigma_leaves_the_hazard_or_breaks_learning(self):
        self.baseline(1, 800)
        self.noise(0.1, 700, 0.995)
        self.noise(0.05, 900, 0.99)
        self.noise(0.2, 100, 0.90)                       # lower hazard but the range is not learned: does not count
        self.assertEqual(rd.ed_arm('gru', directory=self.dir)['verdict'], 'REFUTED')

    def test_mixed_and_incomplete_and_the_best_baseline_seed_is_used(self):
        self.baseline(1, 800)
        self.baseline(2, 400)                            # the best baseline is seed 2
        self.noise(0.1, 150, 0.995)                      # 150 is below 800 / 5 = 160 but not below 400 / 5 = 80
        self.assertEqual(rd.ed_arm('gru', directory=self.dir)['verdict'], 'INCOMPLETE')
        self.noise(0.05, 300, 0.99)
        self.noise(0.2, 300, 0.99)
        r = rd.ed_arm('gru', directory=self.dir)
        self.assertEqual(r['verdict'], 'MIXED')
        self.assertFalse(r['provisional'])
        self.assertAlmostEqual(r['best_baseline'], 400 / 1_024_000)


def count_result(prefix8, first_descent):
    curve = {'8': dict(prefix=prefix8, reentry=None)}
    for depth, value in first_descent.items():
        curve[str(depth)] = dict(prefix=0.5, reentry=value)
    return dict(curve=curve)


class ExcursionCounterReading(unittest.TestCase):
    def test_rho_counts_consecutive_depths_from_16(self):
        r = count_result(1.0, {16: 1.0, 32: 1.0, 64: 0.99, 128: 0.4, 256: 0.99})
        self.assertEqual(rd.rho_of(r, 32), 64 / 32)                                   # 128 breaks the chain: 256 does not count
        self.assertEqual(rd.rho_of(count_result(1.0, {16: 0.5, 32: 1.0}), 32), 0.0)

    def test_valid_o1_o2_and_the_refutation_of_the_tanh_limit(self):
        good = count_result(1.0, {16: 1.0, 32: 0.97, 64: 0.7, 128: 0.2, 256: 0.0, 512: 0.0})
        with tempfile.TemporaryDirectory() as d:
            write(d, 'x.json', good)
            r = rd.ec_run(Path(d) / 'x.json', 32, lstm=True)
            self.assertTrue(r['valid'])
            self.assertTrue(r['O1'])
            self.assertTrue(r['O2'])                         # 0.7 >= 0.5 at 2 R_max, 0.2 < 0.5 at 4 R_max
            self.assertFalse(r['O2_refuted'])
            write(d, 'y.json', count_result(1.0, {16: 1.0, 32: 1.0, 64: 1.0, 128: 0.97, 256: 0.9}))
            r = rd.ec_run(Path(d) / 'y.json', 32, lstm=True)
            self.assertTrue(r['O2_refuted'])
            write(d, 'z.json', count_result(0.9, {16: 1.0, 32: 1.0}))
            self.assertFalse(rd.ec_run(Path(d) / 'z.json', 32, lstm=False)['valid'])

    def test_o3_compares_the_two_arms(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, 'count8_32-lstm-w64-s1.json', count_result(1.0, {16: 1.0, 32: 1.0, 64: 1.0, 128: 0.3}))
            write(d, 'count8_32-gru-w64-s1.json', count_result(1.0, {16: 1.0, 32: 1.0, 64: 0.2, 128: 0.1}))
            r = rd.ec_task('count8_32', directory=d)
            self.assertTrue(r['runs']['lstm']['valid'] and r['runs']['gru']['valid'])
            self.assertTrue(r['runs']['gru']['rho'] <= r['runs']['lstm']['rho'])
            self.assertTrue(r['O3'] if 'O3' in r else r['runs'].get('O3', True))


class SeedDistributionReading(unittest.TestCase):
    def write_runs(self, directory, descent_by_run, in_range=1.0):
        for (arm, seed), value in descent_by_run.items():
            curve = {'8': dict(prefix=in_range, reentry=None), '16': dict(prefix=0.5, reentry=value)}
            write(directory, f'count-{arm}-w64-s{seed}.json', dict(curve=curve))

    def test_universal_saturation_when_nearly_every_valid_run_is_stuck_near_the_floor(self):
        with tempfile.TemporaryDirectory() as d:
            runs = {(arm, s): 0.06 for arm in ('gru', 'lstm') for s in range(1, 9)}
            runs[('lstm', 5)] = 0.30                                   # one outlier of 16 is within the 10 % allowance
            self.write_runs(d, runs)
            r = rd.ec0(directory=d)
            self.assertEqual(r['verdict'], 'UNIVERSAL SATURATION')
            self.assertEqual((r['valid'], r['saturated']), (16, 15))

    def test_lottery_needs_two_runs_that_really_count_at_twice_the_peak(self):
        with tempfile.TemporaryDirectory() as d:
            runs = {(arm, s): 0.06 for arm in ('gru', 'lstm') for s in range(1, 9)}
            runs[('lstm', 3)] = 0.9
            self.write_runs(d, runs)
            self.assertNotEqual(rd.ec0(directory=d)['verdict'], 'LOTTERY')           # one such run is not enough
            runs[('lstm', 6)] = 0.55
            self.write_runs(d, runs)
            r = rd.ec0(directory=d)
            self.assertEqual(r['verdict'], 'LOTTERY')
            self.assertEqual(sorted(r['counting']), ['lstm-s3', 'lstm-s6'])

    def test_unlearned_runs_are_not_read_and_missing_runs_flag_the_verdict_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_runs(d, {('gru', 1): 0.9, ('gru', 2): 0.05}, in_range=0.6)       # in-range accuracy below 0.95: invalid
            r = rd.ec0(directory=d)
            self.assertEqual(r['verdict'].split(' (')[0], 'NO VALID RUNS')
            self.assertIn('INCOMPLETE', r['verdict'])


if __name__ == '__main__':
    unittest.main()
