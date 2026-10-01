"""Induction rows: a random string, some filler, then the SAME string again.

Predicting the second copy needs the circuit key-addressed retrieval needs - match what came just before against an earlier
occurrence, then copy what followed it - and nothing in these rows can be solved by position: the copy offset is random and the
string is fresh each time. If that circuit is what a model lacks, teaching it here, with no key/value template to shortcut on,
should carry over to the facts.
"""
import unittest

import numpy as np
import torch

from mline import induction as ind


class ExampleTests(unittest.TestCase):
    def test_the_second_copy_is_the_first_and_sits_where_copy_start_says(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            ex = ind.example(rng, seg=(6, 20), gap=(0, 40), lead=(0, 9))
            text, start, n = ex['text'], ex['copy_start'], ex['length']
            self.assertEqual(text[start:start + n], ex['segment'])
            self.assertEqual(text[start - ex['gap'] - n:start - ex['gap']], ex['segment'])
            self.assertEqual(len(text), start + n)

    def test_the_draws_stay_inside_the_requested_ranges_and_the_alphabet(self):
        rng = np.random.default_rng(2)
        lengths, gaps = set(), set()
        for _ in range(200):
            ex = ind.example(rng, seg=(6, 12), gap=(0, 8), lead=(0, 4), alphabet=b'XYZ')
            lengths.add(ex['length'])
            gaps.add(ex['gap'])
            self.assertTrue(set(ex['segment']) <= set(b'XYZ'))
        self.assertEqual(lengths, set(range(6, 13)))
        self.assertEqual(gaps, set(range(0, 9)))

    def test_the_default_alphabet_is_wide_enough_that_no_letter_set_can_be_memorised(self):
        self.assertGreaterEqual(len(ind.ALPHABET), 62)
        self.assertTrue(set(b'ABCDEFGHJKLMNPQRSTUVWXYZ') <= set(ind.ALPHABET))        # covers every needle key letter


class ItemsTests(unittest.TestCase):
    KW = dict(seg=(6, 14), gap=(0, 30), lead=(0, 6), skip=2)

    def test_rows_are_the_padded_width_next_byte_pairs(self):
        x, y, mask = ind.items(5, seed=3, pad_to=160, **self.KW)
        self.assertEqual(tuple(x.shape), (5, 160))
        self.assertEqual(tuple(mask.shape), (5, 160))
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))

    def test_the_mask_marks_exactly_the_second_copy_bytes_after_the_first_skip_of_them(self):
        x, y, mask = ind.items(6, seed=4, pad_to=200, **self.KW)
        for i in range(6):
            row = bytes(x[i].tolist())
            marked = [j for j in range(200) if mask[i, j]]
            self.assertGreater(len(marked), 0)
            for j in marked:
                predicted = row[j + 1]
                # the byte predicted at j must be a byte of a repeated string: it occurred earlier in the row, and so did
                # the `skip` bytes before it (what the model has just seen of the copy)
                context = row[j + 1 - self.KW['skip']:j + 1]
                self.assertIn(context, row[:j + 1 - self.KW['skip']])
                self.assertIsInstance(predicted, int)

    def test_the_mask_leaves_out_the_first_copy_and_the_first_skip_bytes_of_every_copy(self):
        x, y, mask = ind.items(1, seed=5, pad_to=120, seg=(10, 10), gap=(5, 5), lead=(0, 0), skip=3)
        # one example of 3 + ... : with seg 10 and skip 3 the copy contributes 10 - 3 = 7 marked positions per example
        per_example = 10 - 3
        self.assertEqual(int(mask.sum()) % per_example, 0)
        self.assertGreaterEqual(int(mask.sum()), per_example)

    def test_the_mask_positions_are_exactly_those_computed_by_hand_for_a_single_example_row(self):
        kw = dict(seg=(9, 9), gap=(4, 4), lead=(3, 3))
        ex = ind.example(np.random.default_rng(6 + 424_243), **kw)           # the first example items() draws for seed 6
        pad_to = len(ex['text']) + 3                                          # too narrow for a second example
        x, y, mask = ind.items(1, seed=6, pad_to=pad_to, skip=2, **kw)
        pad = pad_to - len(ex['text'])
        expected = sorted(pad + ex['copy_start'] + m - 1 for m in range(2, ex['length']))
        self.assertEqual([j for j in range(pad_to) if mask[0, j]], expected)
        row = bytes(x[0].tolist())
        for j in expected:
            self.assertEqual(row[j + 1], row[j + 1 - ex['gap'] - ex['length']])   # the predicted byte repeats the first copy's

    def test_it_is_deterministic_and_a_different_seed_differs(self):
        a = ind.items(4, seed=7, pad_to=120, **self.KW)
        b = ind.items(4, seed=7, pad_to=120, **self.KW)
        c = ind.items(4, seed=8, pad_to=120, **self.KW)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a, b)))
        self.assertFalse(torch.equal(a[0], c[0]))

    def test_an_example_that_cannot_fit_the_row_is_refused(self):
        with self.assertRaises(ValueError):
            ind.items(2, seed=1, pad_to=30, seg=(20, 20), gap=(20, 20), lead=(0, 0), skip=2)


class BatchFnTests(unittest.TestCase):
    def test_the_batch_depends_only_on_seed_and_step(self):
        fn = ind.batch_fn(4, seed=9, pad_to=120, seg=(6, 10), gap=(0, 20), lead=(0, 4), skip=2)
        x1, y1 = fn(3)
        x2, y2 = fn(3)
        x3, _ = fn(4)
        self.assertTrue(torch.equal(x1, x2) and torch.equal(y1, y2))
        self.assertFalse(torch.equal(x1, x3))


class CopyAccuracyTests(unittest.TestCase):
    @staticmethod
    def oracle():
        class Shifted(torch.nn.Module):
            def forward(self, x):
                out = torch.zeros(x.shape[0], x.shape[1], 256)
                out.scatter_(2, torch.cat([x[:, 1:], x[:, -1:]], 1)[..., None], 1.0)
                return out
        return Shifted()

    @staticmethod
    def constant():
        class Const(torch.nn.Module):
            def forward(self, x):
                out = torch.zeros(x.shape[0], x.shape[1], 256)
                out[..., 0] = 1.0
                return out
        return Const()

    def test_a_model_that_knows_the_next_byte_scores_one_and_a_constant_model_scores_zero(self):
        kw = dict(n_items=8, batch=4, seed=11, pad_to=160, seg=(8, 16), gap=(0, 30), lead=(0, 6), skip=2)
        self.assertEqual(ind.copy_accuracy(self.oracle(), **kw), 1.0)
        self.assertEqual(ind.copy_accuracy(self.constant(), **kw), 0.0)

    def test_the_model_is_left_in_the_mode_it_was_in(self):
        model = self.oracle()
        model.train()
        ind.copy_accuracy(model, n_items=4, batch=4, seed=1, pad_to=120, seg=(6, 8), gap=(0, 10), lead=(0, 2), skip=2)
        self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
