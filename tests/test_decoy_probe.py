"""The decoy probe puts a SECOND fact (different key, different value) between the first fact and the question, then asks for
one of them. Whether the answer follows the asked key or the nearer fact says whether the retrieval is content-addressed or
position-locked, so the layout must be exactly what it claims: both facts, the stated filler lengths, the asked key in the
question, the asked value at the end.
"""
import unittest

import torch

from mline import decoy_probe as dp
from mline import needle as nd


def decode(row):
    return bytes(row.tolist()).decode('ascii')


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.items = dp.make(6, a=5, b=7, ask='first', seed=11, alphabet=nd.TRAIN_KEYS)

    def test_every_row_has_the_two_facts_the_stated_fillers_and_the_asked_question(self):
        for i in range(6):
            text = decode(self.items['x'][i])
            k1, k2 = self.items['keys'][i], self.items['decoy_keys'][i]
            v1, v2 = self.items['answers'][i], self.items['decoy_answers'][i]
            fact1 = 'The code for %s is %s.\n' % (k1, v1)
            fact2 = 'The code for %s is %s.\n' % (k2, v2)
            filler_a = (nd.FILLER_PHRASE * 2)[:5].decode('ascii')
            filler_b = (nd.FILLER_PHRASE * 2)[:7].decode('ascii')
            expected = fact1 + filler_a + fact2 + filler_b + '\nThe code for %s is %s' % (k1, v1)
            self.assertEqual(text, expected)

    def test_the_two_keys_and_values_differ(self):
        for i in range(6):
            self.assertNotEqual(self.items['keys'][i], self.items['decoy_keys'][i])
            self.assertNotEqual(self.items['answers'][i][0], self.items['decoy_answers'][i][0])   # first digit tells them apart

    def test_asking_for_the_decoy_swaps_the_roles(self):
        items = dp.make(6, a=5, b=7, ask='decoy', seed=11, alphabet=nd.TRAIN_KEYS)
        for i in range(6):
            text = decode(items['x'][i])
            self.assertTrue(text.endswith('\nThe code for %s is %s' % (items['decoy_keys'][i], items['decoy_answers'][i])))

    def test_targets_are_next_bytes_and_the_mask_marks_the_four_answer_positions(self):
        x, y, mask = self.items['x'], self.items['y'], self.items['mask']
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))
        self.assertEqual(int(mask.sum(1).min()), 4)
        self.assertEqual(int(mask.sum(1).max()), 4)
        first = self.items['first']
        self.assertTrue(mask[:, first].all())
        self.assertTrue(mask[:, first + 3].all())
        self.assertFalse(mask[:, first + 4].any())

    def test_it_is_deterministic_and_rows_share_a_width(self):
        again = dp.make(6, a=5, b=7, ask='first', seed=11, alphabet=nd.TRAIN_KEYS)
        self.assertTrue(torch.equal(self.items['x'], again['x']))
        self.assertEqual(self.items['x'].shape[1], self.items['width'])

    def test_a_bad_ask_is_refused(self):
        with self.assertRaises(ValueError):
            dp.make(2, a=1, b=1, ask='third', seed=1)

    def test_zero_length_fillers_put_the_facts_back_to_back(self):
        items = dp.make(3, a=0, b=0, ask='first', seed=2, alphabet=nd.TRAIN_KEYS)
        text = decode(items['x'][0])
        k1, k2 = items['keys'][0], items['decoy_keys'][0]
        self.assertIn('%s.\nThe code for %s' % (items['answers'][0], k2), text)


class KeyLengthTests(unittest.TestCase):
    """A model trained on mixed key lengths has to be probed at every length it saw, not only the longest."""

    def test_every_key_has_the_asked_length_and_the_row_is_laid_out_around_it(self):
        items = dp.make(8, a=3, b=4, ask='decoy', seed=5, alphabet=nd.TRAIN_KEYS, key_length=2)
        for i in range(8):
            self.assertEqual(len(items['keys'][i]), 2)
            self.assertEqual(len(items['decoy_keys'][i]), 2)
            text = decode(items['x'][i])
            k1, k2 = items['keys'][i], items['decoy_keys'][i]
            v1, v2 = items['answers'][i], items['decoy_answers'][i]
            expected = ('The code for %s is %s.\n' % (k1, v1) + (nd.FILLER_PHRASE * 2)[:3].decode('ascii')
                        + 'The code for %s is %s.\n' % (k2, v2) + (nd.FILLER_PHRASE * 2)[:4].decode('ascii')
                        + '\nThe code for %s is %s' % (k2, v2))
            self.assertEqual(text, expected)

    def test_the_default_is_still_the_four_letter_key(self):
        items = dp.make(4, a=1, b=1, seed=3)
        self.assertTrue(all(len(k) == 4 for k in items['keys']))

    def test_one_letter_keys_stay_distinct(self):
        items = dp.make(60, a=0, b=0, seed=9, alphabet=nd.TRAIN_KEYS, key_length=1)
        for k1, k2 in zip(items['keys'], items['decoy_keys']):
            self.assertNotEqual(k1, k2)

    def test_an_alphabet_too_small_for_two_distinct_keys_is_refused_not_looped_on(self):
        with self.assertRaises(ValueError):
            dp.make(2, a=0, b=0, seed=1, alphabet=b'A', key_length=1)

    def test_the_fact_size_follows_the_key_length(self):
        self.assertEqual(dp.fact_bytes(4), 27)
        self.assertEqual(dp.fact_bytes(1), 24)
        items = dp.make(2, a=0, b=0, seed=1, key_length=3)
        self.assertEqual(items['x'].shape[1], 2 * dp.fact_bytes(3) + len(b'\nThe code for ') + 3 + len(b' is ') + 4)


class TableTests(unittest.TestCase):
    ROWS = [dict(alphabet=al, key_length=kl, a=a, b=b, ask=ask, gap_to_question=a + dp.fact_bytes(kl) + b,
                 decoy_to_question=b, exact_match=.5, usable=10, asked=.25 if ask == 'first' else .75, decoy=.5,
                 neither=.25)
            for al in ('TRAIN_KEYS', 'HELD_KEYS') for kl in (1, 4) for a, b in ((0, 0), (10, 0)) for ask in dp.ASK]

    def test_the_table_has_one_block_per_alphabet_and_key_length_and_shows_both_asks(self):
        text = dp.format_table(dict(rows=self.ROWS, items=64))
        for al in ('TRAIN_KEYS', 'HELD_KEYS'):
            for kl in (1, 4):
                self.assertIn('%s, key length %d' % (al, kl), text)
        self.assertIn('0.75', text)
        self.assertIn('10:0', text)

    def test_a_report_from_before_key_lengths_existed_still_prints_as_length_four(self):
        old = [{k: v for k, v in r.items() if k != 'key_length'} for r in self.ROWS if r['key_length'] == 4]
        self.assertIn('key length 4', dp.format_table(dict(rows=old, items=128)))


class ProbeModelTests(unittest.TestCase):
    """The probe loop, on a model that is already built - so it can score a model of any size, in the middle of training."""

    @staticmethod
    def shifted_model():
        """A model that always predicts the next byte of its own input: it 'knows' the asked value, so it must score 1."""
        class Shifted(torch.nn.Module):
            def forward(self, x):
                out = torch.zeros(x.shape[0], x.shape[1], 256)
                out.scatter_(2, torch.cat([x[:, 1:], x[:, -1:]], 1)[..., None], 1.0)
                return out
        return Shifted()

    def test_one_row_per_alphabet_key_length_layout_and_ask(self):
        rows = dp.probe_model(self.shifted_model(), [(0, 0), (5, 3)], alphabets=('TRAIN_KEYS', 'HELD_KEYS'),
                              key_lengths=(1, 4), items=6, batch=3, verbose=False)
        self.assertEqual(len(rows), 2 * 2 * 2 * 2)
        for r in rows:
            for field in ('alphabet', 'key_length', 'a', 'b', 'ask', 'gap_to_question', 'exact_match', 'asked', 'decoy', 'neither'):
                self.assertIn(field, r)
            self.assertEqual(r['gap_to_question'], r['a'] + dp.fact_bytes(r['key_length']) + r['b'])

    def test_a_model_that_knows_the_asked_value_scores_one_on_both_asks(self):
        rows = dp.probe_model(self.shifted_model(), [(0, 0), (10, 10)], items=8, batch=4, verbose=False)
        for r in rows:
            self.assertEqual(r['exact_match'], 1.0)
            self.assertEqual(r['asked'], 1.0)
            self.assertEqual(r['decoy'], 0.0)

    def test_the_model_is_left_in_the_mode_it_was_in(self):
        model = self.shifted_model()
        model.train()
        dp.probe_model(model, [(0, 0)], items=4, batch=4, verbose=False)
        self.assertTrue(model.training)
        model.eval()
        dp.probe_model(model, [(0, 0)], items=4, batch=4, verbose=False)
        self.assertFalse(model.training)


class AttributionTests(unittest.TestCase):
    def test_a_prediction_is_attributed_to_the_asked_fact_the_decoy_or_neither(self):
        got = torch.tensor([1, 9, 5, 0])
        asked = torch.tensor([1, 2, 3, 4])
        decoy = torch.tensor([7, 9, 8, 6])
        r = dp.attribute(got, asked, decoy)
        self.assertEqual(r['usable'], 4)
        self.assertAlmostEqual(r['asked'], 1 / 4)
        self.assertAlmostEqual(r['decoy'], 1 / 4)
        self.assertAlmostEqual(r['neither'], 2 / 4)


if __name__ == '__main__':
    unittest.main()
