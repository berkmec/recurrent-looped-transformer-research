"""The mechanical readings of E-D and E-C: SUPPORTED / REFUTED / MIXED and the extrapolation factor follow the pre-registered thresholds."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

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


class LearningRateDecayReading(unittest.TestCase):
    """E-P reuses the E-D thresholds (SUPPORTED = 5x lower hazard with the range learned; REFUTED = no halving) on the `-lrdecay0.3` result files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        write(self.dir, 'perm4-lstm-w64-s1-cur.json', perm_result(0.99, 600, 1_024_000))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_tag_selects_the_decay_files_and_the_noise_files_are_ignored(self):
        write(self.dir, 'perm4-lstm-w64-s1-cur-noise0.3.json', perm_result(0.99, 0, 1_024_000))       # would be SUPPORTED if it were read
        self.assertEqual(rd.ed_arm('lstm', sigmas=(0.3,), directory=self.dir, tag='lrdecay')['verdict'], 'NO RUNS')
        write(self.dir, 'perm4-lstm-w64-s1-cur-lrdecay0.3.json', perm_result(0.995, 100, 1_024_000))
        r = rd.ed_arm('lstm', sigmas=(0.3,), directory=self.dir, tag='lrdecay')
        self.assertEqual(r['verdict'], 'SUPPORTED')
        self.assertEqual((r['runs'][0.3]['events'], r['runs'][0.3]['exposure']), (100, 1_024_000))

    def test_no_effect_and_the_exact_rate_ratio_p_value(self):
        write(self.dir, 'perm4-lstm-w64-s1-cur-lrdecay0.3.json', perm_result(0.995, 580, 1_024_000))       # 580 against 600 events: no halving, and not distinguishable
        r = rd.ed_arm('lstm', sigmas=(0.3,), directory=self.dir, tag='lrdecay')
        self.assertEqual(r['verdict'], 'REFUTED')
        self.assertGreater(r['runs'][0.3]['p_lower'], 0.05)
        # 12 baseline events against 2 run events over equal exposure: P(X <= 2 | n = 14, p = 1/2) = (1 + 14 + 91) / 2^14
        self.assertAlmostEqual(rd.rate_ratio_p(12, 20608, 2, 20608), 106 / 16384)
        self.assertEqual(rd.rate_ratio_p(0, 100, 0, 100), 1.0)
        self.assertLess(rd.rate_ratio_p(50, 1000, 0, 1000), 1e-12)

    def test_s5_files_and_per_seed_replicates_are_read_from_their_own_names(self):
        # S5 Core w128 seed 1 against its own stored baseline
        write(self.dir, 'perm5-core-w128-s1-cur.json', perm_result(1.0, 800, 1_024_000))
        write(self.dir, 'perm5-core-w128-s1-cur-lrdecay0.3.json', perm_result(1.0, 100, 1_024_000))
        r = rd.ed_arm('core', sigmas=(0.3,), directory=self.dir, tag='lrdecay', task='perm5', width=128, baseline_seeds=(1,))
        self.assertEqual(r['verdict'], 'SUPPORTED')
        self.assertEqual(r['runs'][0.3]['events'], 100)
        # seed 2 of the S4 LSTM: compared with ITS seed's baseline (900 events), not the seed-1 baseline (600)
        write(self.dir, 'perm4-lstm-w64-s2-cur.json', perm_result(0.99, 900, 1_024_000))
        write(self.dir, 'perm4-lstm-w64-s2-cur-lrdecay0.3.json', perm_result(0.995, 150, 1_024_000))     # 150 <= 900 / 5 = 180 but not <= 600 / 5 = 120
        r2 = rd.ed_arm('lstm', sigmas=(0.3,), directory=self.dir, tag='lrdecay', baseline_seeds=(2,), run_seed=2)
        self.assertEqual(r2['verdict'], 'SUPPORTED')
        self.assertAlmostEqual(r2['best_baseline'], 900 / 1_024_000)
        # without the matching seed-2 run file nothing is read
        self.assertEqual(rd.ed_arm('lstm', sigmas=(0.3,), directory=self.dir, tag='lrdecay', baseline_seeds=(3,), run_seed=3)['verdict'], 'NO BASELINE')

    def test_annealing_plus_state_noise_files_of_the_s5_core_are_read_against_the_constant_lr_baseline(self):
        # E-D5: `perm5-core-w128-s1-cur-lrdecay0.3-noise{sigma}.json` against `perm5-core-w128-s1-cur.json`; the annealed-only file is NOT the baseline
        write(self.dir, 'perm5-core-w128-s1-cur.json', perm_result(1.0, 800, 1_024_000))
        write(self.dir, 'perm5-core-w128-s1-cur-lrdecay0.3.json', perm_result(1.0, 900, 1_024_000))
        write(self.dir, 'perm5-core-w128-s1-cur-lrdecay0.3-noise0.1.json', perm_result(1.0, 700, 1_024_000))
        r = rd.ed_arm('core', sigmas=(0.1, 0.2), tag='lrdecay0.3-noise', task='perm5', width=128, baseline_seeds=(1,), directory=self.dir)
        self.assertEqual(r['verdict'], 'INCOMPLETE')                                         # sigma 0.2 has not run yet
        self.assertAlmostEqual(r['best_baseline'], 800 / 1_024_000)
        write(self.dir, 'perm5-core-w128-s1-cur-lrdecay0.3-noise0.2.json', perm_result(1.0, 150, 1_024_000))
        r = rd.ed_arm('core', sigmas=(0.1, 0.2), tag='lrdecay0.3-noise', task='perm5', width=128, baseline_seeds=(1,), directory=self.dir)
        self.assertEqual(r['verdict'], 'SUPPORTED')                                          # 150 <= 800 / 5 = 160 with the range learned
        self.assertEqual((r['runs'][0.1]['events'], r['runs'][0.2]['events']), (700, 150))

    def test_the_width_reading_compares_annealed_wider_cores_with_the_constant_lr_w128_baseline(self):
        # E-K5: `perm5-core-w{256,192}-s1-cur-lrdecay0.3.json` against `perm5-core-w128-s1-cur.json` (800 events / 1,024,000)
        self.assertEqual(rd.ek5(directory=self.dir)['verdict'], 'NO BASELINE')
        write(self.dir, 'perm5-core-w128-s1-cur.json', perm_result(1.0, 800, 1_024_000))
        self.assertEqual(rd.ek5(directory=self.dir)['verdict'], 'NO RUNS')
        write(self.dir, 'perm5-core-w256-s1-cur-lrdecay0.3.json', perm_result(1.0, 120, 1_024_000))
        r = rd.ek5(directory=self.dir)
        self.assertEqual(r['verdict'], 'SUPPORTED')                                       # 120 <= 800 / 5 = 160 with the range learned: enough, even before w192 has run
        self.assertFalse(r['complete'])
        write(self.dir, 'perm5-core-w192-s1-cur-lrdecay0.3.json', perm_result(1.0, 700, 1_024_000))
        r = rd.ek5(directory=self.dir)
        self.assertEqual((r['verdict'], r['complete']), ('SUPPORTED', True))
        self.assertEqual((r['runs'][256]['events'], r['runs'][192]['events']), (120, 700))
        # nothing halves the hazard, or the wide run did not learn the range: REFUTED once both ran; one run short: INCOMPLETE
        write(self.dir, 'perm5-core-w256-s1-cur-lrdecay0.3.json', perm_result(0.9, 10, 1_024_000))                 # a tiny hazard but the range is not learned
        write(self.dir, 'perm5-core-w192-s1-cur-lrdecay0.3.json', perm_result(1.0, 700, 1_024_000))
        self.assertEqual(rd.ek5(directory=self.dir)['verdict'], 'REFUTED')
        write(self.dir, 'perm5-core-w192-s1-cur-lrdecay0.3.json', perm_result(1.0, 300, 1_024_000))                 # 300 < 800 / 2 but > 800 / 5: neither
        self.assertEqual(rd.ek5(directory=self.dir)['verdict'], 'MIXED')
        (Path(self.dir) / 'perm5-core-w192-s1-cur-lrdecay0.3.json').unlink()
        self.assertEqual(rd.ek5(directory=self.dir)['verdict'], 'INCOMPLETE')

    def test_the_s4_core_width_reading_follows_the_margin_verdicts_and_the_hazards(self):
        # E-K4: perm4-core-w{32,48}-s1-cur.json + margin-profile-s4core-width.json (a list of profiles with `width` and `erosion`)
        def put(w, events, in_range, erosion_verdict, ratio=1.0):
            write(self.dir, f'perm4-core-w{w}-s1-cur.json', perm_result(in_range, events, 1_024_000))
            return dict(width=w, erosion=dict(verdict=erosion_verdict, ratio=ratio))

        def profiles(*items):
            (Path(self.dir) / 'margin-profile-s4core-width.json').write_text(json.dumps(list(items)), encoding='utf8')

        self.assertEqual(rd.ek4(directory=self.dir)['verdict'], 'INCOMPLETE')
        a, b = put(32, 900, 1.0, 'EROSION', 0.5), put(48, 0, 1.0, 'FLAT')
        profiles(a, b)
        self.assertEqual(rd.ek4(directory=self.dir)['verdict'], 'RATIO SUPPORTED')                     # the narrow width erodes, the wide one does not
        a, b = put(32, 0, 1.0, 'FLAT'), put(48, 0, 1.0, 'FLAT')
        profiles(a, b)
        self.assertEqual(rd.ek4(directory=self.dir)['verdict'], 'RATIO NOT SUPPORTED (neither erodes)')
        a, b = put(32, 900, 1.0, 'FLAT'), put(48, 0, 1.0, 'FLAT')                                       # hazard alone (>= 5e-4: 900 / 1,024,000 = 8.8e-4) makes a width erode
        profiles(a, b)
        self.assertEqual(rd.ek4(directory=self.dir)['verdict'], 'RATIO SUPPORTED')
        a, b = put(32, 0, 1.0, 'EROSION', 0.4), put(48, 0, 1.0, 'PARTIAL', 0.8)
        profiles(a, b)
        self.assertEqual(rd.ek4(directory=self.dir)['verdict'], 'RATIO NOT SUPPORTED (both erode)')
        a, b = put(32, 0, 1.0, 'FLAT'), put(48, 0, 1.0, 'EROSION', 0.5)
        profiles(a, b)
        self.assertEqual(rd.ek4(directory=self.dir)['verdict'], 'MIXED')                               # the inverse order
        a, b = put(32, 0, 0.9, 'FLAT'), put(48, 0, 1.0, 'FLAT')
        profiles(a, b)
        self.assertEqual(rd.ek4(directory=self.dir)['verdict'], 'NOT LEARNED')
        a, b = put(32, 900, 1.0, 'EROSION', 0.5), put(48, 0, 0.97, 'FLAT')                              # the wide width must itself reach 0.99 in range for SUPPORTED
        profiles(a, b)
        self.assertEqual(rd.ek4(directory=self.dir)['verdict'], 'MIXED')

    def test_the_dose_response_names_the_shallowest_annealing_that_meets_the_ep_rule(self):
        # baseline 600 events: the E-P rule needs <= 120; hazards by floor 0.3 / 0.1 / 0.03 / 0.01 = 400 / 130 / 100 / 5
        for floor, events in ((0.3, 400), (0.1, 130), (0.03, 100), (0.01, 5)):
            write(self.dir, f'perm4-lstm-w64-s1-cur-lrfloor{floor}.json', perm_result(0.995, events, 1_024_000))
        r = rd.ep3_arm('lstm', directory=self.dir)
        self.assertEqual(r['sufficient_floor'], 0.03)                        # 0.1 gives 130 > 120; 0.03 is the largest floor that meets it
        self.assertEqual([f for f in r['curve']], [0.3, 0.1, 0.03, 0.01])    # reported from the shallowest annealing to the deepest
        self.assertEqual(r['curve'][0.01][1], 5)
        # a floor whose range is not learned never counts, however low its hazard
        write(self.dir, 'perm4-lstm-w64-s1-cur-lrfloor0.03.json', perm_result(0.9, 0, 1_024_000))
        self.assertEqual(rd.ep3_arm('lstm', directory=self.dir)['sufficient_floor'], 0.01)
        # nothing sufficient -> None
        write(self.dir, 'perm4-lstm-w64-s1-cur-lrfloor0.01.json', perm_result(0.995, 500, 1_024_000))
        self.assertIsNone(rd.ep3_arm('lstm', directory=self.dir)['sufficient_floor'])


class SeedLotteryReading(unittest.TestCase):
    """E-P4: per arm over seeds 1-5, constant lr vs annealed: a run 'has a floor' when its pooled hazard is >= 2e-4 and its range is learned."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def plant(self, constant, decayed, final=0.99):
        for seed, events in enumerate(constant, start=1):
            write(self.dir, f'perm4-gru-w64-s{seed}-cur.json', perm_result(final, events, 203_200))
        for seed, events in enumerate(decayed, start=1):
            write(self.dir, f'perm4-gru-w64-s{seed}-cur-lrdecay0.3.json', perm_result(final, events, 203_200))

    def read(self):
        return rd.ep4(directory=self.dir, arms=('gru',))['gru']

    def test_artefact_confirmed_when_the_floor_is_typical_at_constant_lr_and_gone_when_annealed(self):
        self.plant([150, 90, 3, 120, 60], [0, 1, 0, 2, 0])            # floors in 4 of 5 constant runs (>= 41 events), none annealed
        r = self.read()
        self.assertEqual((r['verdict'], r['constant_with_floor'], r['decayed_with_floor']), ('ARTEFACT CONFIRMED', 4, 0))

    def test_lottery_when_the_constant_lr_floor_is_the_exception(self):
        self.plant([150, 2, 3, 0, 60], [0, 0, 0, 0, 0])               # floors in 2 of 5: seed 1 was unlucky
        self.assertEqual(self.read()['verdict'], 'LOTTERY')

    def test_not_removed_when_two_annealed_runs_keep_the_floor_and_incomplete_is_flagged(self):
        self.plant([150, 90, 80, 120, 60], [0, 70, 0, 90, 0])
        self.assertEqual(self.read()['verdict'], 'NOT REMOVED')
        (Path(self.dir) / 'perm4-gru-w64-s5-cur-lrdecay0.3.json').unlink()
        self.assertTrue(self.read()['verdict'].endswith('(INCOMPLETE)'))

    def test_a_run_whose_range_was_not_learned_is_not_counted_either_way(self):
        self.plant([150, 90, 80, 120, 60], [0, 0, 0, 0, 0])
        write(self.dir, 'perm4-gru-w64-s2-cur.json', perm_result(0.5, 150, 203_200))                  # not learned: no floor counted
        r = self.read()
        self.assertEqual(r['constant_with_floor'], 4)
        self.assertFalse(r['runs']['constant'][2]['valid'])


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
            self.assertIn('O3', r['runs'])                        # the first version fell back to True when the key was missing (review finding F7)
            self.assertTrue(r['runs']['O3'])
            write(d, 'count8_32-gru-w64-s1.json', count_result(1.0, {16: 1.0, 32: 1.0, 64: 1.0, 128: 1.0}))        # the GRU now extrapolates further than the LSTM
            self.assertFalse(rd.ec_task('count8_32', directory=d)['runs']['O3'])

    def test_float32_means_at_an_exact_threshold_count_as_meeting_it(self):
        # review finding F6: n = 100 means are stored as float32, 90/100 as 0.89999998 and 10/100 as 0.10000000149
        f90 = float(torch.tensor([1.0] * 90 + [0.0] * 10).mean())
        f95 = float(torch.tensor([1.0] * 95 + [0.0] * 5).mean())
        f10 = float(torch.tensor([1.0] * 10 + [0.0] * 90).mean())
        self.assertLess(f90, 0.9)
        self.assertLess(f95, 0.95)
        self.assertGreater(f10, 0.10)
        self.assertTrue(rd.ge(f90, 0.9) and rd.ge(f95, 0.95) and rd.le(f10, 0.10))
        self.assertFalse(rd.ge(0.8999, 0.9) or rd.le(0.1001, 0.10))                 # the tolerance covers float32 rounding only
        with tempfile.TemporaryDirectory() as d:
            write(d, 'x.json', count_result(1.0, {16: f90, 32: f90, 64: f95, 128: 0.4}))
            r = rd.ec_run(Path(d) / 'x.json', 32, lstm=True)
            self.assertTrue(r['O1'])                                                 # exactly 0.90 at every depth up to R_max
            self.assertTrue(r['O2'])                                                 # 0.95 >= 0.5 at 2 R_max, 0.4 < 0.5 at 4 R_max
            write(d, 'y.json', count_result(1.0, {16: 1.0, 32: 1.0, 64: 1.0, 128: f95}))
            self.assertTrue(rd.ec_run(Path(d) / 'y.json', 32, lstm=True)['O2_refuted'])      # exactly 0.95 at 4 R_max refutes the tanh limit

    def test_range_not_learned_uses_the_tolerance_too(self):
        f95 = float(torch.tensor([1.0] * 95 + [0.0] * 5).mean())
        with tempfile.TemporaryDirectory() as d:
            write(d, 'perm4-gru-w64-s1-cur.json', perm_result(0.99, 800, 1_024_000))
            write(d, 'perm4-gru-w64-s1-cur-noise0.1.json', perm_result(f95, 700, 1_024_000))      # exactly 95/100 in range, no hazard drop
            r = rd.ed_arm('gru', sigmas=(0.1,), directory=d)
            self.assertEqual(r['verdict'], 'REFUTED')                                    # hazard 700 >= 800 / 2 and in range 0.95 counts as learned: refuted by the hazard alone
            write(d, 'perm4-gru-w64-s1-cur-noise0.1.json', perm_result(f95, 100, 1_024_000))     # a 8x drop with exactly 95/100: below the 0.99 support bar
            self.assertNotEqual(rd.ed_arm('gru', sigmas=(0.1,), directory=d)['verdict'], 'SUPPORTED')


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
