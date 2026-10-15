"""M-3b reading (scripts/m3b_reading.py): the 8857f99 decision rule, checked on synthetic grids and, when present, on the 9 recorded cells."""
import math
import unittest
from pathlib import Path

from scripts import m3b_reading as m


def _result(events, exposure, in16=1.0):
    # one hazard window carrying the whole exposure: pooled_windows sums first_errors and at_risk * (stop - start)
    return {'curve': {'16': {'final': in16, 'hazard': [{'start': 0, 'stop': 1, 'at_risk': exposure, 'first_errors': events}]}}}


def _grid(y, in16=None):
    """Synthetic cells whose log10 hazard equals y[w, d] exactly (exposure 10**k, events 10**(k + y))."""
    cells = {}
    for (w, d), v in y.items():
        k = 9
        cells[w, d] = _result(round(10 ** (k + v)), 10 ** k, (in16 or {}).get((w, d), 1.0))
    return cells


def _additive(row, col):
    return {(w, d): row[i] + col[j] for i, w in enumerate(m.WSEEDS) for j, d in enumerate(m.DSEEDS)}


class M3bReadingTests(unittest.TestCase):
    def test_anova_hand_computed(self):
        # y = row effect only (rows -2, -3, -4; columns equal): MS_res = 0, MS_w = 3 * 2 / 2 = 3, s2w = 1, s2d = 0
        a = m.anova(_additive((-2, -3, -4), (0, 0, 0)))
        self.assertAlmostEqual(a['MS_w'], 3.0)
        self.assertAlmostEqual(a['MS_d'], 0.0)
        self.assertAlmostEqual(a['MS_res'], 0.0)
        self.assertAlmostEqual(a['s2w'], 1.0)
        self.assertAlmostEqual(a['s2d'], 0.0)

    def test_verdict_bands(self):
        self.assertEqual(m.verdict(1.0, 1.0), m.REFUTED)        # s2d >= s2w (equality refutes)
        self.assertEqual(m.verdict(1.0, 0.2499), m.SUPPORTED)   # s2d < s2w / 4
        self.assertEqual(m.verdict(1.0, 0.25), m.PARTIAL)       # s2w / 4 itself is grey
        self.assertEqual(m.verdict(1.0, 0.5), m.PARTIAL)
        self.assertEqual(m.verdict(0.0, 0.0), m.REFUTED)        # both zero: s2d >= s2w

    def test_reading_labels(self):
        self.assertEqual(m.reading(_grid(_additive((-2, -3, -4), (0, 0, 0))))['verdict'], m.SUPPORTED)
        self.assertEqual(m.reading(_grid(_additive((0, 0, 0), (-2, -3, -4))))['verdict'], m.REFUTED)

    def test_negative_component_clamped_and_noted(self):
        y = _additive((-2, -3, -4), (0, 0, 0))
        for (w, d), e in {(1, 101): 0.3, (1, 102): -0.3, (2, 101): -0.3, (2, 102): 0.3}.items():
            y[w, d] += e   # pure interaction: row and column means unchanged -> MS_d = 0, s2d raw = -MS_res / 3 < 0
        out = m.reading(_grid(y))
        self.assertLess(out['anova']['s2d_raw'], 0)
        self.assertEqual(out['anova']['s2d'], 0.0)
        self.assertTrue(any(n.startswith('s2d negative') for n in out['notes']))

    def test_zero_events_continuity(self):
        y, ev, ex, _ = m.cell_metric(_result(0, 400000))
        self.assertAlmostEqual(y, math.log10(0.5 / 400000))
        self.assertEqual(ev, 0)

    def test_exclusion_tolerance_and_counts(self):
        base = _additive((-2, -3, -4), (0, 0, 0))
        self.assertEqual(m.reading(_grid(base, {(2, 102): 0.98999999}))['excluded'], [])     # float32 198/200 rounding: kept (F6 tolerance)
        one = m.reading(_grid(base, {(2, 102): 0.985}))
        self.assertEqual((one['excluded'], one['verdict']), (['w2d102'], None))              # rule gap reported, not invented
        three = m.reading(_grid(base, {(1, 101): 0.9, (2, 102): 0.9, (3, 103): 0.9}))
        self.assertEqual(three['verdict'], m.INCOMPLETE)

    def test_missing_cell_incomplete(self):
        cells = _grid(_additive((-2, -3, -4), (0, 0, 0)))
        cells[3, 103] = None
        self.assertEqual(m.reading(cells)['verdict'], m.INCOMPLETE)

    @unittest.skipUnless(all(m.cell_path(w, d).exists() for w in m.WSEEDS for d in m.DSEEDS), 'M-3b results not on disk')
    def test_recorded_cells_reproduce_t8_verdict(self):
        # T8 verdict (jim-mv8403zi, 2026-10-14 19:06Z): DESTEKLENİR, s2d = 0, s2w = 0.61
        out = m.reading(m.load_cells())
        self.assertEqual(out['verdict'], m.SUPPORTED)
        self.assertEqual(out['excluded'], [])
        self.assertAlmostEqual(out['anova']['s2w'], 0.6072, places=3)
        self.assertEqual(out['anova']['s2d'], 0.0)


if __name__ == '__main__':
    unittest.main()
