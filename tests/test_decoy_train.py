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


class PangramFillerTests(unittest.TestCase):
    """A small model trained only on MQAR rows never sees U-Z (or most of T-Z) anywhere, so their embeddings stay at their random
    initial values and 'does matching generalise to T-Z' is answered by construction. A pangram as filler puts every letter in the
    text - never in a key - so what is tested is the matching, not an untrained embedding."""

    def test_the_pangram_contains_every_upper_case_letter(self):
        self.assertTrue(set(b'ABCDEFGHIJKLMNOPQRSTUVWXYZ') <= set(dt.PANGRAM))

    def test_the_pangram_filler_has_the_asked_length_and_is_the_pangram_repeated(self):
        rng = np.random.default_rng(1)
        for n in (0, 1, 30, 200):
            got = dt._filler('pangram', n, rng, None)
            self.assertEqual(len(got), n)
            self.assertEqual(got, (dt.PANGRAM * 10)[:n])

    def test_the_other_kinds_are_unchanged(self):
        rng = np.random.default_rng(1)
        self.assertEqual(dt._filler('repeat', 50, rng, None), nd.filler_bytes('repeat', 50, rng))

    def test_an_example_with_pangram_filler_contains_the_pangram_between_the_facts(self):
        rng = np.random.default_rng(2)
        ex = dt.example(rng, 3, 60, 60, 'pangram', None, n_queries=2)
        self.assertEqual(ex['total'], 60)
        self.assertIn(dt.PANGRAM[:20], ex['text'])

    def test_the_default_filler_is_still_the_repeated_phrase(self):
        rng = np.random.default_rng(3)
        ex = dt.example(rng, 3, 60, 60, 'repeat', None, n_queries=2)
        self.assertNotIn(dt.PANGRAM[:20], ex['text'])


class WithReplacementTests(unittest.TestCase):
    """Without replacement every question asks a different fact, so the last question is answered by elimination (the one fact
    not asked yet) - T1b scored 1.00 on it with no key use at all. Asking with replacement removes that shortcut."""

    def test_the_default_still_asks_each_fact_at_most_once(self):
        rng = np.random.default_rng(1)
        for _ in range(60):
            ex = dt.example(rng, 4, 0, 10, 'repeat', None, n_queries=4)
            self.assertEqual(sorted(ex['asked_all']), [0, 1, 2, 3])

    def test_with_replacement_a_fact_can_be_asked_twice_and_every_fact_can_be_asked(self):
        rng = np.random.default_rng(2)
        repeats, seen = 0, set()
        for _ in range(200):
            ex = dt.example(rng, 4, 0, 10, 'repeat', None, n_queries=4, with_replacement=True)
            self.assertEqual(len(ex['asked_all']), 4)
            repeats += len(set(ex['asked_all'])) < 4
            seen |= set(ex['asked_all'])
        self.assertGreater(repeats, 100)                    # 1 - 4!/4^4 = 90 % of examples repeat a fact
        self.assertEqual(seen, {0, 1, 2, 3})

    def test_with_replacement_the_asked_facts_are_uniform_and_independent_of_the_position_in_the_row(self):
        rng = np.random.default_rng(3)
        counts = np.zeros((4, 4))
        for _ in range(2000):
            ex = dt.example(rng, 4, 0, 10, 'repeat', None, n_queries=4, with_replacement=True)
            for q, fact in enumerate(ex['asked_all']):
                counts[q, fact] += 1
        self.assertTrue(np.all(np.abs(counts / 2000 - .25) < .04))

    def test_with_replacement_the_text_still_ends_each_question_with_its_facts_key_and_value(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            ex = dt.example(rng, 3, 0, 10, 'repeat', None, n_queries=4, with_replacement=True, key_length=(2, 2))
            for j, start in zip(ex['asked_all'], ex['value_starts']):
                self.assertEqual(ex['text'][start:start + 4].decode('ascii'), ex['values'][j])
                self.assertEqual(ex['text'][start - 4 - 2:start - 4].decode('ascii'), ex['keys'][j])

    def test_more_questions_than_facts_is_allowed_only_with_replacement(self):
        rng = np.random.default_rng(5)
        with self.assertRaises(ValueError):
            dt.example(rng, 2, 0, 10, 'repeat', None, n_queries=4)
        self.assertEqual(len(dt.example(rng, 2, 0, 10, 'repeat', None, n_queries=4, with_replacement=True)['asked_all']), 4)

    def test_the_patch_passes_it_through_and_the_default_batch_is_unchanged(self):
        with dt.patched(4, n_queries=4):
            plain = nd.training_items(4, filler_range=(0, 10), seed=5, pad_to=400)
        with dt.patched(4, n_queries=4, with_replacement=False):
            same = nd.training_items(4, filler_range=(0, 10), seed=5, pad_to=400)
        with dt.patched(4, n_queries=4, with_replacement=True):
            other = nd.training_items(4, filler_range=(0, 10), seed=5, pad_to=400)
        self.assertTrue(torch.equal(plain['x'], same['x']))
        self.assertFalse(torch.equal(plain['x'], other['x']))


class AnswerStartTests(unittest.TestCase):
    """Where each question's answer begins, so an MQAR model can be scored on the distribution it was trained on."""

    def test_an_example_reports_where_each_answer_starts_in_its_text(self):
        rng = np.random.default_rng(3)
        for facts, queries in ((1, 1), (3, 2), (4, 4)):
            ex = dt.example(rng, facts, 0, 20, 'repeat', None, n_queries=queries, key_length=(1, 3))
            self.assertEqual(len(ex['value_starts']), queries)
            for j, start in zip(ex['asked_all'], ex['value_starts']):
                self.assertEqual(ex['text'][start:start + 4].decode('ascii'), ex['values'][j])

    def test_the_rows_report_the_index_of_every_answers_first_digit_and_the_query_it_belongs_to(self):
        items = dt.decoy_items(6, filler_range=(0, 15), seed=4, pad_to=400, pack=True, n_facts=3, n_queries=3,
                               key_length=(1, 2), alphabet=b'ABCDEFG')
        self.assertEqual(len(items['answer_starts']), 6)
        for i, starts in enumerate(items['answer_starts']):
            row = bytes(items['x'][i].tolist())
            self.assertGreater(len(starts), 0)
            for position, query_index in starts:
                self.assertTrue(0 <= query_index < 3)
                self.assertTrue(row[position:position + 4].isdigit())
                self.assertEqual(row[position - 4:position], b' is ')            # the answer follows the question's ' is '
            # three queries per example, in order, for every packed example
            self.assertEqual([q for _, q in starts], [k % 3 for k in range(len(starts))])

    def test_adding_the_positions_changes_nothing_else_about_the_batch(self):
        a = dt.decoy_items(5, filler_range=(0, 15), seed=9, pad_to=400, pack=True, n_facts=3, n_queries=2)
        self.assertEqual(a['x'].shape, (5, 400))
        again = dt.decoy_items(5, filler_range=(0, 15), seed=9, pad_to=400, pack=True, n_facts=3, n_queries=2)
        self.assertTrue(torch.equal(a['x'], again['x']))


class InductionMixTests(unittest.TestCase):
    """A share of the needle rows can be induction rows (a random string repeated later) instead of MQAR facts: the rows that
    demand the match-then-copy circuit without a key/value template to shortcut on."""
    SHAPE = dict(seg=(6, 10), gap=(0, 20), lead=(0, 4), skip=2)

    def rows(self, n, share, seed=1):
        with dt.patched(3, n_queries=3, induction_share=share, induction_shape=self.SHAPE):
            items = nd.training_items(n, filler_range=(0, 10), seed=seed, pad_to=256)
        return items

    def test_the_share_of_rows_that_are_induction_rows_follows_induction_share(self):
        for share, n_induction in ((0.0, 0), (0.25, 2), (0.5, 4), (1.0, 8)):
            with self.subTest(share=share):
                items = self.rows(8, share)
                self.assertEqual(tuple(items['x'].shape), (8, 256))
                facts = sum('The code for' in t for t in items['texts'])
                self.assertEqual(facts, 8 - n_induction)

    def test_the_induction_rows_come_after_the_mqar_rows_and_the_batch_is_deterministic(self):
        a, b = self.rows(8, 0.5, seed=3), self.rows(8, 0.5, seed=3)
        self.assertTrue(torch.equal(a['x'], b['x']))
        self.assertFalse(torch.equal(a['x'], self.rows(8, 0.5, seed=4)['x']))
        self.assertTrue(all('The code for' in t for t in a['texts'][:4]))
        self.assertFalse(any('The code for' in t for t in a['texts'][4:]))

    def test_next_byte_targets_hold_for_the_whole_mixed_batch(self):
        items = self.rows(8, 0.5)
        self.assertTrue(torch.equal(items['y'][:, :-1], items['x'][:, 1:]))

    def test_no_induction_is_the_plain_mqar_generator_byte_for_byte(self):
        with dt.patched(3, n_queries=3):
            plain = nd.training_items(4, filler_range=(0, 10), seed=5, pad_to=256)
        mixed = self.rows(4, 0.0, seed=5)
        self.assertTrue(torch.equal(plain['x'], mixed['x']))

    def test_a_share_outside_zero_to_one_is_refused(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                with dt.patched(3, induction_share=bad, induction_shape=self.SHAPE):
                    pass

    def test_a_share_without_a_shape_is_refused(self):
        with self.assertRaises(ValueError):
            with dt.patched(3, induction_share=0.5):
                pass


class AlphabetTests(unittest.TestCase):
    """Key matching generalises only if the key LETTERS in training are varied enough that equality, not a particular letter's
    routing, is what gets learned. The wide training alphabet must stay disjoint from the letters the probe holds out."""

    def test_every_key_letter_comes_from_the_given_alphabet(self):
        rng = np.random.default_rng(6)
        for _ in range(40):
            ex = dt.example(rng, 3, 0, 10, 'repeat', None, key_length=(1, 3), alphabet=b'XYZ')
            self.assertTrue(all(set(k) <= set('XYZ') for k in ex['keys']))

    def test_the_default_alphabet_is_the_original_training_alphabet(self):
        self.assertEqual(dt.ALPHABETS['default'], nd.TRAIN_KEYS)

    def test_the_wide_alphabet_extends_the_training_alphabet_and_leaves_the_held_out_letters_unseen(self):
        from mline import decoy_probe as dp
        wide = set(dt.ALPHABETS['wide'])
        self.assertTrue(set(nd.TRAIN_KEYS) <= wide)
        self.assertTrue(wide.isdisjoint(set(dp.ALPHABETS['HELD_KEYS'])))
        self.assertGreater(len(wide), len(nd.TRAIN_KEYS))

    def test_the_broad_alphabet_holds_out_only_the_probes_letters_and_has_many_symbols(self):
        from mline import decoy_probe as dp
        broad = set(dt.ALPHABETS['broad'])
        self.assertTrue(set(dt.ALPHABETS['wide']) <= broad)
        self.assertTrue(broad.isdisjoint(set(dp.ALPHABETS['HELD_KEYS'])))
        self.assertGreaterEqual(len(broad), 50)
        self.assertTrue(set(b'abcdefghijklmnopqrstuvwxyz0123456789') <= broad)

    def test_too_few_letters_for_distinct_keys_is_refused(self):
        rng = np.random.default_rng(1)
        with self.assertRaises(ValueError):
            dt.example(rng, 4, 0, 5, 'repeat', None, key_length=(1, 1), alphabet=b'AB')

    def test_the_patch_applies_the_alphabet(self):
        import re
        with dt.patched(2, alphabet=b'QR'):
            items = nd.training_items(4, filler_range=(0, 10), seed=1, pad_to=512)
        keys = [k for t in items['texts'] for k in re.findall(r'The code for ([A-Z]+) is', t)]
        self.assertGreater(len(keys), 8)
        self.assertTrue(all(set(k) <= set('QR') for k in keys))


class KeyLengthTests(unittest.TestCase):
    """Matching a 4-letter key is a harder circuit to find than matching a 1-letter one, so the key length can be mixed:
    one length per example, drawn from a range, the same for every fact and question of that example."""

    def test_a_range_draws_one_length_per_example_and_uses_it_everywhere(self):
        rng = np.random.default_rng(2)
        seen = set()
        for _ in range(80):
            ex = dt.example(rng, 3, 0, 20, 'repeat', None, key_length=(1, 4))
            seen.add(ex['key_length'])
            self.assertTrue(all(len(k) == ex['key_length'] for k in ex['keys']))
            text = ex['text'].decode('ascii')
            self.assertTrue(text.endswith('\nThe code for %s is %s' % (ex['keys'][ex['asked']], ex['values'][ex['asked']])))
        self.assertEqual(seen, {1, 2, 3, 4})

    def test_the_length_follows_the_key_length(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            ex = dt.example(rng, 3, 0, 30, 'repeat', None, n_queries=2, key_length=(2, 2))
            self.assertEqual(len(ex['text']), 3 * (23 + 2) + ex['total'] + 2 * (22 + 2))

    def test_keys_stay_distinct_even_with_one_letter_keys(self):
        rng = np.random.default_rng(4)
        for _ in range(50):
            ex = dt.example(rng, 4, 0, 10, 'repeat', None, key_length=(1, 1))
            self.assertEqual(len(set(ex['keys'])), 4)

    def test_the_default_is_the_four_letter_key(self):
        rng = np.random.default_rng(5)
        self.assertEqual(dt.example(rng, 2, 0, 10)['key_length'], 4)

    def test_the_range_parses_and_bad_ranges_are_refused(self):
        self.assertEqual(dt.parse_key_length('1-4'), (1, 4))
        self.assertEqual(dt.parse_key_length('2'), (2, 2))
        for bad in ('0-3', '4-2', '1-9', 'x'):
            with self.assertRaises(ValueError, msg=bad):
                dt.parse_key_length(bad)

    def test_the_patch_applies_the_key_length(self):
        with dt.patched(3, key_length=(1, 1)):
            items = nd.training_items(3, filler_range=(0, 10), seed=1, pad_to=512)
        self.assertTrue(any('The code for A is' in t or 'The code for B is' in t or 'The code for C is' in t
                            or 'The code for D is' in t or 'The code for E is' in t or 'The code for F is' in t
                            or 'The code for G is' in t or 'The code for H is' in t or 'The code for J is' in t
                            or 'The code for K is' in t or 'The code for L is' in t or 'The code for M is' in t
                            for t in items['texts']))


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
