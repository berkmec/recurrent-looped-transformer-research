"""Before believing a zero, prove the scorer can report a one.

The M0 probe returned exact_match 0.000 in all 76 of its needle measurements while validation bits-per-byte fell
from 2.267 to 1.660, so the model was plainly learning language. A flat zero from a working measurement and a flat
zero from a broken one look identical, and this campaign has already been caught once by the second kind: the
first needle probe scored a format the model was never taught and read five below-chance checkpoints as a result.

So the scorer is pinned against two references it must be able to reach:

  a perfect oracle            exact_match 1.000, byte_accuracy 1.000
  an oracle off by one byte   exact_match 0.000, byte_accuracy 0.750   (three of four answer bytes)

The second matters as much as the first: it shows exact_match is strict over the WHOLE answer while byte_accuracy
degrades smoothly, which is why the probe's byte column can carry a signal the exact column cannot.
"""
import unittest

import torch


class ScorerReachesItsEndpointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def items(self, placement='far'):
        from mline import needle as nd
        return nd.make(16, n_filler=512, kind='repeat', seed=5, placement=placement)

    def oracle(self, items, corrupt=None):
        """Teacher-forced perfect prediction. The batch offset is tracked: a scorer is called batch by batch, and
        an oracle that ignores that silently answers for the wrong rows - which is exactly how this check first
        went wrong, reporting 0.25 where 1.0 was correct."""
        y, at = items['y'], {'i': 0}

        def forward(xb):
            n = xb.shape[0]
            idx = y[at['i']:at['i'] + n].clone()
            at['i'] += n
            if corrupt is not None:
                idx[:, corrupt] = (idx[:, corrupt] + 1) % 256
            return torch.zeros(n, xb.shape[1], 256).scatter(2, idx.unsqueeze(-1), 10.)
        return forward

    def test_a_perfect_prediction_scores_one(self):
        from mline import needle as nd
        for placement in ('far', 'near'):
            items = self.items(placement)
            s = nd.score(self.oracle(items), items, batch=4)
            self.assertEqual(s['exact_match'], 1., placement)
            self.assertEqual(s['byte_accuracy'], 1., placement)

    def test_one_wrong_byte_loses_the_exact_match_but_not_the_byte_score(self):
        from mline import needle as nd
        items = self.items()
        s = nd.score(self.oracle(items, corrupt=-2), items, batch=4)
        self.assertEqual(s['exact_match'], 0.)
        self.assertAlmostEqual(s['byte_accuracy'], .75, places=6)

    def test_a_silent_model_scores_zero(self):
        from mline import needle as nd
        items = self.items()
        s = nd.score(lambda xb: torch.zeros(xb.shape[0], xb.shape[1], 256), items, batch=4)
        self.assertEqual(s['exact_match'], 0.)
        self.assertEqual(s['byte_accuracy'], 0.)

    def test_the_batch_size_does_not_change_the_score(self):
        """A scorer that mixes rows across batches can report anything; this would catch it."""
        from mline import needle as nd
        items = self.items()
        scores = [nd.score(self.oracle(items), items, batch=b)['exact_match'] for b in (1, 4, 8, 16)]
        self.assertEqual(scores, [1., 1., 1., 1.])


if __name__ == '__main__':
    unittest.main()
