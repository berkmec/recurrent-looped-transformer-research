"""E-A9k: the renewal prediction is the injection hazard averaged over the ages 0..k-1, the consistency test is the registered one, and the reading is mechanical."""
import json
import tempfile
import unittest
from pathlib import Path

from horizon import snap_interval as si


def table(h64_events=15, h64_person=38145):
    return {'0-16': (0, 9600), '16-32': (0, 9600), '32-64': (0, 19200), '64-128': (h64_events, h64_person)}


class Renewal(unittest.TestCase):
    def test_zero_below_the_age_wall_and_a_linear_share_of_the_bin_above_it(self):
        t = table()
        h64 = 15 / 38145
        self.assertEqual(si.renewal_hazard(32, t), 0.0)
        self.assertEqual(si.renewal_hazard(64, t), 0.0)
        self.assertAlmostEqual(si.renewal_hazard(96, t), (32 / 96) * h64, places=12)
        self.assertAlmostEqual(si.renewal_hazard(128, t), (64 / 128) * h64, places=12)
        self.assertIsNone(si.renewal_hazard(256, t))                                                       # E-A5 followed 128 steps only

    def test_the_age_table_pools_the_three_injection_positions(self):
        cell = lambda e, p: {'events': e, 'person_steps': p, 'at_risk': 200}
        injection = {t0: {'state+cache': {'since': {'0-16': cell(0, 3200), '16-32': cell(0, 3200), '32-64': cell(0, 6400), '64-128': cell(e, 12700)}}} for t0, e in (('128', 3), ('256', 4), ('512', 8))}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'e.json'
            path.write_text(json.dumps([{'weights': 'results/Z1-pilot/m.pt', 'injection': injection}]), encoding='utf8')
            got = si.age_hazard(path, 'x/m.pt')
            self.assertEqual(got['64-128'], (15, 38100))
            self.assertEqual(got['0-16'], (0, 9600))
            with self.assertRaises(ValueError):
                si.age_hazard(path, 'x/other.pt')


class Consistency(unittest.TestCase):
    def test_the_poisson_interval(self):
        lo, hi = si.poisson_interval(0)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 3.689, places=2)                                                       # the exact 95 % upper limit for zero events (two-sided)
        lo, hi = si.poisson_interval(10)
        self.assertTrue(4.7 < lo < 4.9 and 18.3 < hi < 18.5)

    def test_cells_k_32_and_64_need_no_breakdown_and_k_96_128_a_hazard_in_the_band(self):
        self.assertTrue(si.consistent(0, 300000, 0.0, 64))
        self.assertTrue(si.consistent(10, 300000, 0.0, 64))                                              # 3.3e-5 <= 5e-5
        self.assertFalse(si.consistent(30, 300000, 0.0, 64))                                             # 1e-4 > 5e-5: a breakdown
        predicted = 1.3e-4
        self.assertTrue(si.consistent(40, 300000, predicted, 96))                                         # 1.3e-4 measured = predicted
        self.assertFalse(si.consistent(0, 300000, predicted, 96))                                         # 0 events: the interval's upper limit 3.7 / 3e5 = 1.2e-5 < 0.4 x predicted
        self.assertFalse(si.consistent(400, 300000, predicted, 96))                                       # ten times the prediction


def model(events, predicted, person=300000, error_free=40, weights='x/perm5-a.pt'):
    conditions = {f'class-{k}': dict(events=events.get(k, 0), person_steps=person, error_free=error_free) for k in si.INTERVALS}
    return dict(weights=weights, task='perm5', width=128, n=40, length=8192, conditions=conditions, predicted={str(k): predicted.get(k) for k in si.INTERVALS})


class Reading(unittest.TestCase):
    pred = {32: 0.0, 64: 0.0, 96: 1.3e-4, 128: 2.0e-4, 256: None}

    def test_the_registered_verdicts(self):
        pred = self.pred
        ok = dict(events={96: 40, 128: 60}, predicted=pred)
        a, b = model(**ok), model(**ok, weights='x/perm5-b.pt')
        self.assertEqual(si.reading([a, b])['verdict'], 'RENEWAL HOLDS')
        worse = dict(events={64: 60, 96: 40, 128: 60}, predicted=pred)                                    # errors already at k = 64
        self.assertEqual(si.reading([model(**worse), b])['verdict'], 'SNAP BREAKS EARLIER THAN A RESET')
        too_many = dict(events={96: 40, 128: 600}, predicted=pred)                                        # 10x the prediction at k = 128
        self.assertEqual(si.reading([a, model(**too_many, weights='x/perm5-b.pt')])['verdict'], 'SNAP BREAKS EARLIER THAN A RESET')
        better = dict(events={96: 40, 128: 5}, predicted=pred)
        self.assertEqual(si.reading([model(**better), model(**better, weights='x/perm5-b.pt')])['verdict'], 'SNAP HOLDS LONGER THAN A RESET')
        self.assertEqual(si.reading([a, model(**better, weights='x/perm5-b.pt')])['verdict'], 'MIXED')    # one model consistent, the other below the band, none broken
        self.assertEqual(si.reading([a])['verdict'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
