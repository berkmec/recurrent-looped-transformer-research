"""M-3b seed split (`--wseed/--dseed/--eval-seed`): default bit-identical, split streams separate.

Fast acceptance (tiny GRU runs); the full-recipe base-hash acceptance is the
recorded `perm4-gru-w64-s1-cur.json` comparison (scripts/run_m3b.sh step 0).
"""
import os
import tempfile
import unittest

import torch


def _tiny(**kw):
    from horizon import depth_run as dr
    base = dict(task='perm3', arm='gru', width=16, seed=5, steps=30, batch=8,
                n_eval=8, out=None, log_every=0)
    base.update(kw)
    return dr.run(**base)


def _weights(model):
    return [p.detach().clone() for p in model.parameters()]


class SeedSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_split_equal_to_seed_is_bit_identical(self):
        r0 = _tiny()
        r1 = _tiny(wseed=5, dseed=5, eval_seed=5)
        self.assertEqual(r0['curve'], r1['curve'])
        self.assertEqual(r0['train']['history'], r1['train']['history'])
        self.assertEqual(r0['train']['advances'], r1['train']['advances'])

    def test_split_streams_separate(self):
        a = _tiny(wseed=5, dseed=101, eval_seed=999)
        b = _tiny(wseed=5, dseed=102, eval_seed=999)
        c = _tiny(wseed=6, dseed=101, eval_seed=999)
        d = _tiny(wseed=5, dseed=101, eval_seed=999)
        self.assertEqual(a['curve'], d['curve'])   # same streams: identical
        self.assertNotEqual(a['curve'], b['curve'])  # data differs
        self.assertNotEqual(a['curve'], c['curve'])  # init differs

    def test_eval_seed_only_moves_eval(self):
        from horizon import depth_run as dr
        r0 = _tiny(eval_seed=999)
        r1 = _tiny(eval_seed=1000)
        self.assertEqual(r0['train']['history'], r1['train']['history'])  # train untouched
        self.assertNotEqual(r0['curve'], r1['curve'])  # eval draws differ

    def test_meta_records_streams(self):
        from horizon import depth_run as dr
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'm.json')
            r = _tiny(wseed=5, dseed=101, eval_seed=999, out=path)
            raw = torch.load(os.path.join(tmp, 'm.pt'), weights_only=True)
            self.assertEqual(raw['meta']['seed'], 5)       # init seed: loader-compatible
            self.assertEqual(raw['meta']['wseed'], 5)
            self.assertEqual(raw['meta']['dseed'], 101)
            self.assertEqual(raw['meta']['eval_seed'], 999)
            self.assertEqual(r['wseed'], 5)
            self.assertEqual(r['dseed'], 101)
            self.assertEqual(r['eval_seed'], 999)
            back = dr.load_model(os.path.join(tmp, 'm.pt'))
            self.assertEqual([p.shape for p in back.parameters()],
                             [p.shape for p in dr.make_arm('gru', 'perm3', 16, 5).parameters()])


if __name__ == '__main__':
    unittest.main()
