"""L12 Stage-E: nested profiles (P0 ⊂ P50 ⊂ P90 ⊂ P97), paired L12-kayip, section 3b labels, section 3c combination,
block layout, and a tiny end-to-end session on a freshly trained checkpoint.

Written before the L12-E run (eval-only); the registered rule is tested here, not fitted to results.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class L12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        from horizon import l12_eval as l12
        return l12

    def test_contract_and_p0(self):
        l12 = self.api()
        pos = l12.contract(np.array([0, 1500, 3173]), 102)
        self.assertEqual(pos.tolist(), [0, 48, 101])
        # feasibility repair: two neighbours landing on the same slot stay strictly increasing inside the band
        pos2 = l12.contract(np.array([3000, 3001, 3173]), 204)
        self.assertTrue((np.diff(pos2) > 0).all() and pos2[-1] <= 203)
        p0 = l12.contract(np.arange(0, 3174, 31)[:102], 102)
        self.assertEqual(p0.tolist(), list(range(102)))

    def test_draw_nesting_and_determinism(self):
        l12 = self.api()
        d1 = l12.draw('perm4', 1, 6)
        d2 = l12.draw('perm4', 1, 6)
        for name, (_, nz, length) in ((n, p) for p in l12.PROFILES for n in [p[0]]):
            x1, x2 = d1['profiles'][name]['x'], d2['profiles'][name]['x']
            self.assertTrue(torch.equal(x1, x2))
            self.assertEqual(tuple(x1.shape), (6, length))
            for s in range(6):
                tokens = x1[s][x1[s] != 0]
                self.assertEqual(len(tokens), 102)                     # k fixed across profiles
                self.assertEqual(int((x1[s] == 0).sum()), length - 102)  # neutral positions are exactly the gaps
                if name == 'P0':
                    base = [x1[t][x1[t] != 0].clone() for t in range(6)]
                else:
                    self.assertTrue(torch.equal(tokens, base[s]))       # same info sequence, same order, per sequence
        from horizon import depth_tasks as dt
        x = d1['profiles']['P0']['x']
        self.assertTrue(torch.equal(d1['profiles']['P0']['y'], dt.oracle('perm4', x)))

    def test_blocks_layout(self):
        l12 = self.api()
        d = l12.draw('perm4', 1, 2, layout='blocks')
        x = d['profiles']['P90']['x']
        self.assertEqual(tuple(x.shape), (2, 1024))
        for s in range(2):
            tokens = x[s][x[s] != 0]
            self.assertEqual(len(tokens), 102)
        zeros = (x[0] == 0).numpy().astype(int)
        runs, cur = [], 0
        for v in zeros:
            if v:
                cur += 1
            else:
                runs.append(cur)
                cur = 0
        self.assertGreaterEqual(max(runs), 100)                          # ~8 long blocks, not uniform gaps
        gaps = l12.draw('perm4', 1, 2)['profiles']['P90']['x']
        self.assertFalse(torch.equal(x, gaps))

    def cell(self, cid, task, verdict_hold=0.7, loss=3.0, sign=True, valid=True, arm='hold'):
        l12 = self.api()
        c = dict(id=cid, task=task, arm=arm, valid=valid,
                 profiles={'P0': dict(acc=0.99), 'P90': dict(hold_neutral=verdict_hold, l12_loss_points=loss, sign_ok=sign)})
        c['reading'] = l12.cell_reading(c)
        return c

    def test_reading_branches(self):
        l12 = self.api()
        self.assertEqual(self.cell('c', 'perm4')['reading']['verdict'], 'DESTEKLENİR')
        self.assertEqual(self.cell('c', 'perm4', loss=25.0)['reading']['verdict'], 'ÇÜRÜTÜLDÜ')
        self.assertEqual(self.cell('c', 'perm4', verdict_hold=0.3)['reading']['verdict'], 'MANIPULATION_NOT_APPLIED')
        self.assertEqual(self.cell('c', 'perm4', loss=10.0)['reading']['verdict'], 'KISMEN')
        self.assertEqual(self.cell('c', 'perm4', sign=False)['reading']['verdict'], 'KISMEN')
        self.assertEqual(self.cell('c', 'perm4', verdict_hold=None)['reading']['verdict'], 'INCOMPLETE')
        rd = self.cell('c', 'perm4')
        rd['profiles']['P90']['loss_sinirda'] = True
        self.assertIn('SINIRDA', l12.cell_reading(rd)['flags'])

    def test_combination(self):
        l12 = self.api()
        cells = [self.cell('s1', 'perm4'), self.cell('s2', 'perm4', valid=False), self.cell('s3', 'perm4'),
                 self.cell('n1', 'perm4', arm='hold_nc'), self.cell('p1', 'perm5', valid=False), self.cell('p2', 'perm5', valid=False)]
        c = l12.combine(cells)
        self.assertEqual(c['perm4'], 'DESTEKLENİR')
        self.assertEqual(c['perm5'], 'INCOMPLETE')
        self.assertEqual(c['overall'], 'INCOMPLETE')
        cells2 = [self.cell('s1', 'perm4'), self.cell('s3', 'perm4'), self.cell('p1', 'perm5'), self.cell('p2', 'perm5')]
        self.assertEqual(l12.combine(cells2)['overall'], 'DESTEKLENİR')
        cells3 = [self.cell('s1', 'perm4', loss=25.0), self.cell('s3', 'perm4'), self.cell('p1', 'perm5'), self.cell('p2', 'perm5')]
        c3 = l12.combine(cells3)
        self.assertEqual(c3['perm4'], 'KISMEN')                          # split: one refuted, one support
        self.assertEqual(c3['overall'], 'KISMEN')
        cells4 = [self.cell('s1', 'perm4'), self.cell('s3', 'perm4'), self.cell('p1', 'perm5'), self.cell('p2', 'perm5', valid=False)]
        self.assertEqual(l12.combine(cells4)['perm5'], 'INCOMPLETE')     # 1 valid < 2

    def test_tiny_end_to_end(self):
        l12 = self.api()
        from horizon import depth_run as dr
        with tempfile.TemporaryDirectory() as tmp:
            model, trained = dr.train('perm4', 'hold', 32, 3, 20, batch=4, curriculum=[2], neutral_p=0.5, log_every=0)
            path = Path(tmp) / 'tiny-hold.pt'
            dr.save_model(model, path, 'perm4', 'hold', 32, 3, trained)
            session = l12.run_session([str(path)], n=4, seed=1, batch=2, profiles=('P0', 'P90'), log=lambda m: None)
        cell = session['cells'][0]
        self.assertTrue(Path(cell['weights']).name == 'tiny-hold.pt')
        labels = {'DESTEKLENİR', 'ÇÜRÜTÜLDÜ', 'MANIPULATION_NOT_APPLIED', 'KISMEN', 'INCOMPLETE'}
        self.assertIn(cell['reading']['verdict'], labels)
        p90 = cell['profiles']['P90']
        self.assertIn('l12_loss_points', p90)
        self.assertIn('loss_ci', p90)
        self.assertIsNotNone(p90['hold_neutral'])
        self.assertIsNone(cell['profiles']['P0']['hold_neutral'])
        self.assertIn(cell['validity_source'], ('sibling-json', 'computed-dev'))
        self.assertEqual(session['k'], 102)
        self.assertIn('combination', session)
        self.assertEqual(len(session['profile_table']), 4)
        self.assertIn('base_seed', session['draws']['perm4'])
        json.dumps(session)                                              # EG-4: the session must be JSON-serializable as is


if __name__ == '__main__':
    unittest.main()
