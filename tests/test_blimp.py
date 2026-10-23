"""E-BLIMP: a sentence's score is the summed loss of its own bytes after the document separator, batching and right padding change nothing,
the model is built with the arm of its own run, and the registered reading is mechanical."""
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from mline import blimp as bl
from mline import data as md
from mline import lm


def tiny(arm='plain', seed=5):
    return lm.build(arm, width=32, heads=2, window=4, alpha=0.1, seed=seed).eval()


SENTENCES = ['The cat sleeps.', 'Dogs bark loudly at night.', 'A', 'Who did you see?', 'These books are heavy, but that one is light.']


class Scoring(unittest.TestCase):
    def test_batching_and_padding_change_nothing_in_either_arm(self):
        for arm in ('plain', 'a0'):
            model = tiny(arm)
            together = bl.score(model, SENTENCES, batch=8)
            alone = np.array([bl.score(model, [s], batch=1)[0] for s in SENTENCES])
            self.assertTrue(np.allclose(together, alone, atol=1e-4), (arm, together - alone))

    def test_the_score_is_the_loss_of_the_sentence_bytes_after_the_separator(self):
        model, s = tiny(), 'Who did you see?'
        seq = torch.tensor([list(md.SEPARATOR + s.encode('utf8'))])
        with torch.no_grad():
            loss = F.cross_entropy(model(seq[:, :-1]).transpose(1, 2), seq[:, 1:], reduction='none')[0] / bl.LN2
        manual = float(loss[len(md.SEPARATOR) - 1:].sum())
        self.assertAlmostEqual(float(bl.score(model, [s])[0]), manual, places=4)
        self.assertEqual(len(loss[len(md.SEPARATOR) - 1:]), len(s.encode('utf8')))


class Loading(unittest.TestCase):
    def test_the_arm_comes_from_the_run(self):
        for arm, feedback in (('plain', True), ('a0', False)):
            model = tiny(arm, seed=9)
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, 'probe.json'), 'w', encoding='utf8') as f:
                    json.dump(dict(arm=arm, width=32, heads=2), f)
                torch.save({'format': 1, 'step': 77, 'meta': {}, 'model': model.state_dict(), 'optimizer': {}}, os.path.join(d, 'latest.pt'))
                loaded, step, got = bl.load(os.path.join(d, 'latest.pt'))
            self.assertEqual((step, got, loaded.feedback), (77, arm, feedback))
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(loaded.state_dict()[k], v), k)

    def test_pairs_are_read_and_counted(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            for name in ('alpha', 'beta'):
                os.makedirs(os.path.join(d, name))
                rows = dict(sentence_good=['a b.', 'c d.', 'e f.'], sentence_bad=['b a.', 'd c.', 'f e.'], field=['syntax'] * 3,
                            linguistics_term=['x'] * 3, UID=[name] * 3, pair_id=[0, 1, 2])
                pq.write_table(pa.table(rows), os.path.join(d, name, 'train-00000-of-00001.parquet'))
            pairs, digests = bl.load_pairs(d, paradigms=2, pairs=3)
            self.assertEqual(sorted(pairs), ['alpha', 'beta'])
            self.assertEqual(pairs['beta'][1], dict(good='c d.', bad='d c.', field='syntax', term='x', pair_id=1))
            self.assertEqual(len(digests['alpha']), 64)
            with self.assertRaises(ValueError):
                bl.load_pairs(d, paradigms=2, pairs=4)


def result(rate, n=100, paradigms=67, seed=0):
    """A fake scored result whose pairs are right with probability `rate` (good 0 or 2 bits, bad 1 bit)."""
    rng = np.random.default_rng(seed)
    scores = {f'p{i:02d}': dict(good=np.where(rng.random(n) < rate, 0.0, 2.0).tolist(), bad=[1.0] * n) for i in range(paradigms)}
    fields = {p: 'syntax' for p in scores}
    return dict(scores=scores, summary=bl.summary(scores, fields))


class Reading(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(bl.reading(result(0.80), result(0.70, seed=1), draws=300)['verdict'], 'FEEDBACK HELPS GRAMMAR')
        self.assertEqual(bl.reading(result(0.70), result(0.80, seed=1), draws=300)['verdict'], 'FEEDBACK HURTS GRAMMAR')
        same = result(0.75)
        r = bl.reading(same, same, draws=300)
        self.assertEqual((r['verdict'], r['delta'], r['ci']), ('NO DIFFERENCE', 0.0, [0.0, 0.0]))
        self.assertEqual(bl.reading(result(0.76, n=10), result(0.745, n=10, seed=1), draws=300)['verdict'], 'INCONCLUSIVE')

    def test_ties_are_wrong_and_counted(self):
        scores = {'p': dict(good=[1.0, 0.0, 2.0], bad=[1.0, 1.0, 1.0])}
        s = bl.summary(scores, {'p': 'syntax'})
        self.assertEqual(s['ties'], 1)
        self.assertAlmostEqual(s['accuracy'], 100 / 3)


if __name__ == '__main__':
    unittest.main()
