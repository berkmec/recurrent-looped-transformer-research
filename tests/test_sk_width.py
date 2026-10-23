import json
import os
import shutil
import tempfile
import unittest

from horizon import sk_run as sr
from horizon import sk_width as sw


def _runs(**cells):
    """cells: label -> list of in_range values (extrapolation is set equal)."""
    out = {}
    for label, values in cells.items():
        cell = next(c for c in sw.CELLS if sw._label(c) == label)
        out[cell] = [(v, v) for v in values]
    return out


ALL_FAIL = dict(**{'gru-w256-lr0.003': [0.15, 0.2], 'lstm-w224-lr0.003': [0.12, 0.13], 'gru-w512-lr0.001': [0.2],
                   'lstm-w448-lr0.001': [0.18], 'tfm_pe-w128-lr0.001': [0.17]})


class WidthReadingTests(unittest.TestCase):
    def test_no_cell_learns(self):
        self.assertEqual(sw.reading(_runs(**ALL_FAIL))['verdict'], 'ADVANTAGE HOLDS AT 16x PARAMETERS')

    def test_a_4x_cell_that_learns_decides(self):
        r = sw.reading(_runs(**dict(ALL_FAIL, **{'gru-w256-lr0.003': [0.99, 0.97]})))
        self.assertEqual(r['verdict'], 'CAPACITY EXPLAINS IT AT 4x')
        self.assertEqual(r['cells']['gru-w256-lr0.003']['status'], 'LEARNS')

    def test_a_wide_single_seed_cell_asks_for_seed_2(self):
        r = sw.reading(_runs(**dict(ALL_FAIL, **{'gru-w512-lr0.001': [0.98]})))
        self.assertEqual(r['verdict'], 'CAPACITY EXPLAINS IT AT 8-16x (seed 2 needed before the verdict)')

    def test_pending_cells_and_partial(self):
        small = {k: v for k, v in ALL_FAIL.items() if k in ('gru-w256-lr0.003', 'lstm-w224-lr0.003')}
        r = sw.reading(_runs(**small))
        self.assertEqual(r['verdict'], 'ADVANTAGE HOLDS AT 4x (larger cells pending)')
        self.assertEqual(len(r['missing']), 3)
        self.assertEqual(sw.reading(_runs(**{'gru-w256-lr0.003': [0.2]}))['verdict'], 'INCOMPLETE')
        r = sw.reading(_runs(**dict(ALL_FAIL, **{'lstm-w224-lr0.003': [0.9, 0.3]})))
        self.assertEqual(r['verdict'], 'ADVANTAGE HOLDS AT 16x PARAMETERS; PARTIAL: lstm-w224-lr0.003')

    def test_load_parses_names_and_skips_manifests(self):
        d = tempfile.mkdtemp()
        try:
            for name, ir in (('sk4-gru-w256-s1-lr0.003.json', 0.2), ('sk4-gru-w256-s2-lr0.003.json', 0.4),
                             ('sk4-tfm_pe-w128-s1-lr0.001.json', 0.1)):
                with open(os.path.join(d, name), 'w', encoding='utf8') as f:
                    json.dump(dict(in_range=ir, extrapolation=ir / 2), f)
            with open(os.path.join(d, 'sk4-gru-w256-s1-lr0.003.manifest.json'), 'w', encoding='utf8') as f:
                json.dump(dict(other=1), f)
            runs = sw.load(sorted(os.path.join(d, n) for n in os.listdir(d)))
            self.assertEqual(sorted(runs), [('gru', 256, 0.003), ('tfm_pe', 128, 0.001)])
            self.assertEqual(sorted(runs[('gru', 256, 0.003)]), [(0.2, 0.1), (0.4, 0.2)])
            self.assertIn(('gru', 256, 0.003), sw.CELLS)                       # parsed lr matches the table's key
        finally:
            shutil.rmtree(d)

    def test_the_parameter_table_matches_the_arms(self):
        self.assertEqual(sum(p.numel() for p in sr.make_arm('core', 'sk4', 64, 1).parameters()), sw.CORE_PARAMS)
        for (arm, width, _), n in sw.CELLS.items():
            built = sr.make_arm('tfm' if arm == 'tfm_pe' else arm, 'sk4', width, 1)   # sinusoidal positions add no parameters
            self.assertEqual(sum(p.numel() for p in built.parameters()), n, (arm, width))


if __name__ == '__main__':
    unittest.main()
