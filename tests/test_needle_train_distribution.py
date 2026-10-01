"""Score the needle on the TRAINING distribution too, or a zero cannot be diagnosed.

The M0 probe ended with exact_match 0.000 in all 76 of its needle measurements while validation bits-per-byte fell
from 2.267 to 1.655. The scorer is sound (tests/test_needle_scorer.py pins its endpoints), so the zeros are real.
But a zero on the EVALUATION distribution has two very different causes, and the probe as built cannot tell them
apart:

  the format was never learned at all       -> a budget/capacity result; more steps or an easier task
  the format was learned on TRAIN_KEYS but  -> a generalisation result; the interesting one, and the one the
  does not survive unseen keys or distance     gate is supposed to help with

The evaluation sets draw keys from EVAL_KEYS, disjoint from the TRAIN_KEYS mixed into training, which is what
makes memorisation impossible - and also what makes this ambiguity unavoidable without a second measurement. So
`make` takes the alphabet, and a set built from TRAIN_KEYS at a filler the training mix actually saw becomes a
direct read on whether the format exists in the model at all.

This matters for M1, not only for the probe: if all three arms score zero, the train-distribution number decides
whether the answer is NOT_MEASURABLE_AT_THIS_BUDGET or GATE_DOES_NOT_TRANSFER.
"""
import unittest

import torch


class AlphabetIsSelectableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def nd(self):
        from mline import needle as nd
        return nd

    def test_the_default_is_unchanged(self):
        """The evaluation sets must keep drawing from EVAL_KEYS, bit for bit."""
        nd = self.nd()
        a = nd.make(8, n_filler=128, kind='repeat', seed=11)
        b = nd.make(8, n_filler=128, kind='repeat', seed=11, alphabet=nd.EVAL_KEYS)
        self.assertTrue(torch.equal(a['x'], b['x']))
        self.assertTrue(torch.equal(a['y'], b['y']))

    def test_train_keys_produce_train_keys(self):
        nd = self.nd()
        items = nd.make(12, n_filler=32, kind='repeat', seed=13, alphabet=nd.TRAIN_KEYS)
        allowed = set(nd.TRAIN_KEYS.decode('ascii'))
        for key in items['answers_keys']:
            self.assertTrue(set(key) <= allowed, f'{key} is not drawn from TRAIN_KEYS')
        self.assertEqual(items['alphabet'], 'TRAIN_KEYS')

    def test_the_two_alphabets_never_collide(self):
        nd = self.nd()
        train = nd.make(24, n_filler=0, kind='repeat', seed=17, alphabet=nd.TRAIN_KEYS)
        evaluation = nd.make(24, n_filler=0, kind='repeat', seed=17)
        self.assertEqual(set(train['answers_keys']) & set(evaluation['answers_keys']), set(),
                         'the disjointness that makes memorisation impossible must survive this change')

    def test_the_layout_is_identical_so_the_two_are_comparable(self):
        nd = self.nd()
        a = nd.make(6, n_filler=64, kind='repeat', seed=19, alphabet=nd.TRAIN_KEYS)
        b = nd.make(6, n_filler=64, kind='repeat', seed=19)
        self.assertEqual(a['width'], b['width'])
        self.assertTrue(torch.equal(a['mask'], b['mask']))

    def test_an_unknown_alphabet_is_refused(self):
        nd = self.nd()
        with self.assertRaises(ValueError):
            nd.make(4, n_filler=0, kind='repeat', seed=1, alphabet=b'')


if __name__ == '__main__':
    unittest.main()
