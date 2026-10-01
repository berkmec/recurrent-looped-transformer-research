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
