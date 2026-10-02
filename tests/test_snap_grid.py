"""E-A9d: k_max is the largest grid value up to which every smaller value is error-free; a measure on tiny trained arms has every part."""
import tempfile
import unittest
from pathlib import Path

from horizon import depth_run as dr
from horizon import snap_grid as sg


def row(events):
    return {str(k): dict(events=events.get(k, 0), person_steps=1000, error_free=10) for k in sg.GRID}


class KMax(unittest.TestCase):
    def test_contiguous_from_k_equal_one(self):
        self.assertEqual(sg.k_max(row({})), 128)
        self.assertEqual(sg.k_max(row({16: 3})), 8)
        self.assertEqual(sg.k_max(row({1: 1})), 0)
        self.assertEqual(sg.k_max(row({64: 2, 128: 5})), 32)
        self.assertEqual(sg.k_max(row({4: 1, 8: 0, 16: 0})), 2)                                         # a later clean cell does not count: the run must be clean from k = 1 on


class EndToEnd(unittest.TestCase):
    def test_measure_covers_the_core_and_a_recurrent_arm(self):
        for arm, width in (('core', 32), ('gru', 16)):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / 'm.json'
                dr.run('perm3', arm, width, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
                r = sg.measure(out.with_suffix('.pt'), n=6, length=64, batch=3, log=lambda m: None)
            self.assertEqual((r['arm'], r['task']), (arm, 'perm3'))
            self.assertEqual(set(r['grid']), {str(k) for k in sg.GRID})
            self.assertTrue(all(c['person_steps'] >= 0 and c['events'] >= 0 for c in r['grid'].values()))                # (an untrained model errs before position 16, so no exposure is asserted)
            self.assertIn('k_max', sg.describe(r))


if __name__ == '__main__':
    unittest.main()
