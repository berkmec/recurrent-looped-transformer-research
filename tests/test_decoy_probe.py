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
