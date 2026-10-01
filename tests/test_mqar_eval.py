"""Scoring an MQAR model on the distribution it was trained on.

The decoy probe lays out two facts and one question, which is not what an MQAR model saw (four facts, four questions, the filler
split between them); a model can learn the training task and still read as chance there. This scores the answers of
training-shaped rows, teacher-forced, per question number - and on key letters the model never saw.
"""
import unittest

import torch

from mline import mqar_eval as me


class Shifted(torch.nn.Module):
    """Predicts, at every position, the byte that actually comes next: it knows every answer."""

    def forward(self, x):
        out = torch.zeros(x.shape[0], x.shape[1], 256)
        out.scatter_(2, torch.cat([x[:, 1:], x[:, -1:]], 1)[..., None], 1.0)
        return out


class Constant(torch.nn.Module):
    def forward(self, x):
        out = torch.zeros(x.shape[0], x.shape[1], 256)
        out[..., 0] = 1.0
        return out


KW = dict(alphabet=b'ABCDEFGHJKLM', key_length=(1, 3), facts=3, queries=3, filler=(0, 12), pad_to=300, n_items=8, batch=4,
          seed=5)


class AccuracyTests(unittest.TestCase):
    def test_a_model_that_knows_the_next_byte_scores_one_on_every_question(self):
        r = me.mqar_accuracy(Shifted(), **KW)
        self.assertEqual(r['first'], [1.0, 1.0, 1.0])
        self.assertEqual(r['exact'], [1.0, 1.0, 1.0])
        self.assertEqual(r['first_all'], 1.0)
        self.assertEqual(r['exact_all'], 1.0)

    def test_a_constant_model_scores_zero(self):
        r = me.mqar_accuracy(Constant(), **KW)
        self.assertEqual(r['first_all'], 0.0)
        self.assertEqual(r['exact_all'], 0.0)

    def test_there_is_one_number_per_question_and_the_count_is_reported(self):
        r = me.mqar_accuracy(Shifted(), **KW)
        self.assertEqual(len(r['first']), 3)
        self.assertGreaterEqual(r['answers'], 8 * 3)             # at least one example per row, three answers each

    def test_a_model_that_gets_only_the_first_digit_right_has_first_above_exact(self):
        class FirstDigitOnly(Shifted):
            def forward(self, x):
                out = super().forward(x)
                after_space = (x == 32) & (torch.roll(x, 1, 1) == ord('s'))       # the position that predicts the first digit
                wrong = torch.zeros_like(out)
                wrong[..., 0] = 1.0
                return torch.where(after_space[..., None], out, wrong)
        r = me.mqar_accuracy(FirstDigitOnly(), **KW)
        self.assertEqual(r['first_all'], 1.0)
        self.assertEqual(r['exact_all'], 0.0)

    def test_it_is_deterministic_and_leaves_the_mode_alone(self):
        model = Shifted()
        model.train()
        a = me.mqar_accuracy(model, **KW)
        self.assertTrue(model.training)
        self.assertEqual(a, me.mqar_accuracy(model, **KW))


class WithReplacementTests(unittest.TestCase):
    def test_more_questions_than_facts_can_be_scored_with_replacement_and_an_oracle_gets_them_all(self):
        kw = dict(KW, facts=2, queries=4, with_replacement=True)
        r = me.mqar_accuracy(Shifted(), **kw)
        self.assertEqual(len(r['first']), 4)
        self.assertEqual(r['first_all'], 1.0)

    def test_the_default_still_refuses_more_questions_than_facts(self):
        with self.assertRaises(ValueError):
            me.mqar_accuracy(Shifted(), **dict(KW, facts=2, queries=4))


class NaturalKeyTests(unittest.TestCase):
    CORPUS = (b'The Prime Minister Theresa May said on Monday that Great Britain would leave. Officials in Washington said the '
              b'president had spoken with Angela Merkel about trade agreements and international negotiations. Local residents '
              b'described destruction across neighborhoods. Marie Curie discovered radium. ') * 20

    def test_natural_keys_are_deterministic_distinct_and_look_like_words_and_names(self):
        a = me.natural_keys(self.CORPUS, n=12, seed=1)
        b = me.natural_keys(self.CORPUS, n=12, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), len(a))
        self.assertTrue(all(isinstance(k, bytes) and 4 <= len(k) <= 24 for k in a))
        self.assertTrue(any(k[:1].isupper() for k in a) and any(k[:1].islower() for k in a))
        self.assertNotEqual(a, me.natural_keys(self.CORPUS, n=12, seed=2))

    def test_an_oracle_scores_one_on_natural_keys_and_the_source_is_what_the_rows_use(self):
        keys = me.natural_keys(self.CORPUS, n=12, seed=1)
        r = me.mqar_accuracy(Shifted(), **dict(KW, key_source=keys, alphabet=None))
        self.assertEqual(r['first_all'], 1.0)


class NaturalKeySetsTests(unittest.TestCase):
    TRAIN = (b'The Prime Minister Theresa May said that Great Britain would leave. Officials in Washington said the president '
             b'had spoken about trade agreements and negotiations. Residents described destruction across neighborhoods. ') * 10
    VAL = (b'The Prime Minister said that Great Britain would leave. Scientists in Geneva described experiments about particle '
           b'physics and detectors. Marie Curie discovered radium. Mountaineers reached Everest during summer. '
           b'The president described trade agreements. ') * 10

    def test_the_validation_keys_exclude_every_training_key_and_both_lists_are_deterministic(self):
        train, val = me.natural_key_sets(self.TRAIN, self.VAL, n_train=10, n_val=10, seed=1)
        self.assertTrue(set(train).isdisjoint(set(val)))
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        again = me.natural_key_sets(self.TRAIN, self.VAL, n_train=10, n_val=10, seed=1)
        self.assertEqual((train, val), again)

    def test_the_validation_keys_really_come_from_the_validation_text(self):
        _, val = me.natural_key_sets(self.TRAIN, self.VAL, n_train=10, n_val=10, seed=1)
        text = self.VAL.decode('ascii')
        self.assertTrue(all(k.decode('ascii') in text for k in val))


class FillerKindTests(unittest.TestCase):
    def test_an_oracle_scores_one_on_pangram_rows_too_and_the_rows_differ_from_the_repeat_ones(self):
        r = me.mqar_accuracy(Shifted(), **dict(KW, filler_kind='pangram'))
        self.assertEqual(r['first_all'], 1.0)
        from mline import decoy_train as dt
        a = dt.decoy_items(4, (10, 20), 1, 'repeat', None, 300, True, 3, 3, (1, 2), b'ABCDEFG')['x']
        b = dt.decoy_items(4, (10, 20), 1, 'pangram', None, 300, True, 3, 3, (1, 2), b'ABCDEFG')['x']
        self.assertFalse(torch.equal(a, b))


class ExtraEvalTests(unittest.TestCase):
    def test_the_hook_scores_the_training_letters_and_the_held_out_letters_at_each_key_length(self):
        fn = me.extra_eval(alphabet=b'ABCDEFGHJKLM', facts=3, queries=3, filler=(0, 12), pad_to=300, n_items=8,
                           key_lengths=(1, 2))
        out = fn(Shifted())
        self.assertEqual(set(out), {'train_k1_first', 'train_k1_exact', 'train_k2_first', 'train_k2_exact',
                                    'held_k1_first', 'held_k1_exact', 'held_k2_first', 'held_k2_exact'})
        self.assertTrue(all(v == 1.0 for v in out.values()))
        self.assertTrue(all(isinstance(v, float) for v in out.values()))

    def test_the_held_out_letters_are_not_in_the_training_alphabet(self):
        from mline import decoy_probe as dp
        self.assertTrue(set(dp.ALPHABETS['HELD_KEYS']).isdisjoint(set(b'ABCDEFGHJKLMNPQRS')))


if __name__ == '__main__':
    unittest.main()
