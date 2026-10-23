import json
import os
import shutil
import tempfile
import unittest

from horizon import sk_compute as sc


def _runs(**cells):
    out = {}
    for label, values in cells.items():
        cell = next(c for c in sc.CELLS if sc._label(c) == label)
        out[cell] = [(v, v) for v in values]
    return out


ALL_FAIL = {'gru-w128-u150000': [0.15], 'lstm-w112-u165000': [0.12], 'core_nofb-w64-u75000': [0.18], 'tfm_pe-w64-u60000': [0.2]}


class ComputeReadingTests(unittest.TestCase):
    def test_no_cell_learns(self):
        r = sc.reading(_runs(**ALL_FAIL))
        self.assertEqual(r['verdict'], 'ADVANTAGE HOLDS AT EQUAL WALL-CLOCK')
        self.assertEqual(r['cells']['gru-w128-u150000']['budget'], 10.0)

    def test_the_twin_that_learns_names_itself_and_asks_for_seed_2(self):
        r = sc.reading(_runs(**dict(ALL_FAIL, **{'core_nofb-w64-u75000': [0.99]})))
        self.assertEqual(r['verdict'], 'COMPUTE EXPLAINS IT: core_nofb-w64-u75000 (seed 2 needed before the verdict)')
        r = sc.reading(_runs(**dict(ALL_FAIL, **{'core_nofb-w64-u75000': [0.99, 0.97]})))
        self.assertEqual(r['verdict'], 'COMPUTE EXPLAINS IT: core_nofb-w64-u75000')

    def test_incomplete_and_partial(self):
        self.assertEqual(sc.reading(_runs(**{'gru-w128-u150000': [0.2]}))['verdict'], 'INCOMPLETE')
        r = sc.reading(_runs(**dict(ALL_FAIL, **{'lstm-w112-u165000': [0.7]})))
        self.assertEqual(r['verdict'], 'ADVANTAGE HOLDS AT EQUAL WALL-CLOCK; PARTIAL: lstm-w112-u165000')

    def test_every_budget_covers_the_measured_wall_clock_ratio(self):
        for (arm, width, updates), ratio in sc.CELLS.items():
            self.assertGreaterEqual(updates / sc.CORE_UPDATES, ratio, arm)

    def test_load_parses_names_and_skips_manifests(self):
        d = tempfile.mkdtemp()
        try:
            for name, ir in (('sk4-gru-w128-s1-u150000.json', 0.2), ('sk4-core_nofb-w64-s1-u75000.json', 0.3)):
                with open(os.path.join(d, name), 'w', encoding='utf8') as f:
                    json.dump(dict(in_range=ir, extrapolation=ir), f)
            with open(os.path.join(d, 'sk4-gru-w128-s1-u150000.manifest.json'), 'w', encoding='utf8') as f:
                json.dump(dict(other=1), f)
            runs = sc.load(sorted(os.path.join(d, n) for n in os.listdir(d)))
            self.assertEqual(sorted(runs), [('core_nofb', 64, 75000), ('gru', 128, 150000)])
        finally:
            shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
