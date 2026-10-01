"""The decoy-aware training mix: several facts per example, the ASKED one chosen at random.

The original mix has one fact per example and always asks it, so 'copy the nearest fact's value' solves it and key matching is
never needed (mline/decoy_probe.py measured exactly that shortcut). With several facts and a random asked one the shortcut
fails and the key has to be used. Two things must be true before any model is trained on it: with ONE fact it is the original
generator byte for byte (otherwise the comparison with the earlier fine-tunes is not like for like), and with several facts
the asked fact is genuinely random.
"""
import unittest

import numpy as np
import torch

from mline import decoy_train as dt
from mline import needle as nd


class OneFactIsTheOriginalGenerator(unittest.TestCase):
    def test_rows_targets_and_distances_match_byte_for_byte(self):
        for seed in (0, 3, 11):
            a = nd.training_items(10, filler_range=(0, 64), seed=seed, pad_to=512)
            b = dt.decoy_items(10, filler_range=(0, 64), seed=seed, pad_to=512, n_facts=1)
            self.assertTrue(torch.equal(a['x'], b['x']), seed)
            self.assertTrue(torch.equal(a['y'], b['y']), seed)
            self.assertEqual(a['fillers'], b['fillers'], seed)

    def test_it_also_matches_unpacked(self):
        a = nd.training_items(6, filler_range=(0, 64), seed=5, pad_to=512, pack=False)
        b = dt.decoy_items(6, filler_range=(0, 64), seed=5, pad_to=512, pack=False, n_facts=1)
        self.assertTrue(torch.equal(a['x'], b['x']))


class ExampleTests(unittest.TestCase):
    def draw(self, n_facts, count=1, lo=0, hi=64, seed=1):
        rng = np.random.default_rng(seed)
        return [dt.example(rng, n_facts, lo, hi, 'repeat', None) for _ in range(count)]

    def test_an_example_has_n_distinct_key_facts_and_asks_one_of_them_with_its_own_value(self):
        for ex in self.draw(3, count=40):
            text = ex['text'].decode('ascii')
            self.assertEqual(len(set(ex['keys'])), 3)
            self.assertEqual(text.count('The code for '), 3 + 1)               # three facts and one question
            asked = ex['asked']
            tail = '\nThe code for %s is %s' % (ex['keys'][asked], ex['values'][asked])
            self.assertTrue(text.endswith(tail))
            for key, value in zip(ex['keys'], ex['values']):
                self.assertIn('The code for %s is %s.\n' % (key, value), text)

    def test_the_length_is_the_facts_plus_the_filler_plus_the_question(self):
        for ex in self.draw(3, count=30):
            self.assertEqual(len(ex['text']), 27 * 3 + ex['total'] + 26)
            self.assertTrue(0 <= ex['total'] <= 64)

    def test_the_asked_fact_is_uniform_over_the_facts_not_always_the_nearest(self):
        counts = np.bincount([ex['asked'] for ex in self.draw(3, count=300)], minlength=3)
        self.assertTrue((counts > 60).all(), counts)

    def test_facts_appear_in_the_order_they_are_numbered_and_filler_follows_each(self):
        ex = self.draw(2, count=1, lo=10, hi=10)[0]
        text = ex['text'].decode('ascii')
        self.assertTrue(text.index(ex['keys'][0]) < text.index(ex['keys'][1]))


class RowTests(unittest.TestCase):
    def test_packed_rows_are_pad_to_wide_with_next_byte_targets(self):
        items = dt.decoy_items(5, filler_range=(0, 64), seed=2, pad_to=512, n_facts=3)
        self.assertEqual(tuple(items['x'].shape), (5, 512))
        self.assertTrue(torch.equal(items['y'][:, :-1], items['x'][:, 1:]))
        self.assertTrue(all(n >= 1 for n in items['examples_per_row']))

    def test_an_example_too_wide_for_the_row_is_refused(self):
        with self.assertRaises(ValueError):
            dt.decoy_items(2, filler_range=(450, 450), seed=1, pad_to=512, n_facts=3)

    def test_it_is_deterministic(self):
        a = dt.decoy_items(4, filler_range=(0, 64), seed=9, pad_to=512, n_facts=3)
        b = dt.decoy_items(4, filler_range=(0, 64), seed=9, pad_to=512, n_facts=3)
        self.assertTrue(torch.equal(a['x'], b['x']))


class MultiQueryTests(unittest.TestCase):
    """MQAR-style examples: n facts, then several questions, each about a DIFFERENT fact, in random order.

    Several answers per example is far more key->value signal for the same bytes, and 'answer the first fact' - the shortcut D1
    collapsed onto - is wrong for most of the questions."""

    def draw(self, n_facts, n_queries, count=1, lo=0, hi=64, seed=1):
        rng = np.random.default_rng(seed)
        return [dt.example(rng, n_facts, lo, hi, 'repeat', None, n_queries=n_queries) for _ in range(count)]

    def test_each_question_asks_a_different_fact_and_carries_its_own_value(self):
        for ex in self.draw(3, 3, count=30):
            text = ex['text'].decode('ascii')
            self.assertEqual(len(set(ex['asked_all'])), 3)
            tail = text[len(text) - 26 * 3:]
            for slot, j in enumerate(ex['asked_all']):
                piece = tail[26 * slot:26 * (slot + 1)]
                self.assertEqual(piece, '\nThe code for %s is %s' % (ex['keys'][j], ex['values'][j]))

    def test_the_length_adds_one_question_per_query(self):
        for ex in self.draw(4, 2, count=20):
            self.assertEqual(len(ex['text']), 27 * 4 + ex['total'] + 26 * 2)

    def test_the_first_asked_fact_is_random(self):
        counts = np.bincount([ex['asked'] for ex in self.draw(3, 3, count=240)], minlength=3)
        self.assertTrue((counts > 40).all(), counts)

    def test_more_questions_than_facts_is_refused(self):
        rng = np.random.default_rng(1)
        with self.assertRaises(ValueError):
            dt.example(rng, 2, 0, 10, 'repeat', None, n_queries=3)

    def test_one_query_still_exposes_the_old_fields(self):
        ex = self.draw(3, 1)[0]
        self.assertEqual(ex['asked_all'], [ex['asked']])

    def test_rows_pack_and_patch_with_several_queries(self):
        with dt.patched(3, n_queries=3):
            items = nd.training_items(4, filler_range=(0, 32), seed=3, pad_to=512)
        self.assertEqual(tuple(items['x'].shape), (4, 512))
        self.assertTrue(torch.equal(items['y'][:, :-1], items['x'][:, 1:]))


class FillerKindTests(unittest.TestCase):
    """Natural-text filler never generalised from a repeat-only mix, so the filler TYPE has to be in the training mix too."""
    CORPUS = np.frombuffer(bytes((i * 7 + 3) % 200 + 32 for i in range(5000)), dtype=np.uint8)

    def test_text_filler_is_a_slice_of_the_corpus(self):
        rng = np.random.default_rng(4)
        for _ in range(10):
            ex = dt.example(rng, 1, 40, 40, 'text', self.CORPUS)
            filler = ex['text'][27:-26]                         # one fact, one question: what lies between them
            self.assertEqual(len(filler), 40)
            self.assertGreaterEqual(bytes(self.CORPUS).find(filler), 0)

    def test_mix_draws_both_kinds(self):
        rng = np.random.default_rng(5)
        repeat = 0
        for _ in range(80):
            filler = dt.example(rng, 1, 30, 30, 'mix', self.CORPUS)['text'][27:-26]
            repeat += filler in (nd.FILLER_PHRASE * 3)
        self.assertTrue(0.25 < repeat / 80 < 0.75, repeat)

    def test_text_or_mix_without_a_corpus_is_refused(self):
        for kind in ('text', 'mix'):
            with self.assertRaises(ValueError, msg=kind):
                with dt.patched(3, kind=kind):
                    pass

    def test_the_patch_applies_the_kind_and_the_corpus(self):
        with dt.patched(1, kind='text', corpus=self.CORPUS):
            items = nd.training_items(3, filler_range=(20, 20), seed=1, pad_to=512)
        self.assertFalse(any('and so it went on' in t for t in items['texts']))


class PatchTests(unittest.TestCase):
    def test_the_patch_replaces_the_generator_and_always_restores_it(self):
        original = nd.training_items
        with dt.patched(3):
            self.assertIsNot(nd.training_items, original)
            items = nd.training_items(2, filler_range=(0, 32), seed=1, pad_to=512)
            self.assertEqual(tuple(items['x'].shape), (2, 512))
        self.assertIs(nd.training_items, original)
        with self.assertRaises(RuntimeError):
            with dt.patched(3):
                raise RuntimeError('boom')
        self.assertIs(nd.training_items, original)


if __name__ == '__main__':
    unittest.main()
