"""Z1-G: the sealed U1 protocol applied to a plain GRU (mechanics only, not learning outcomes)."""
import tempfile
import unittest
from pathlib import Path

import torch

from horizon import gru_control as gc
from horizon import sweep as sw
from horizon import task_families as tf


class Protocol(unittest.TestCase):
    def test_the_numbers_are_the_sealed_u1_numbers_not_copies(self):
        p = gc.protocol()
        u1 = sw.protocol()
        for key in ('steps', 'batch', 'train_length', 'lr', 'weight_decay', 'clip', 'primary_seeds', 'validation_every', 'validation_n',
                    'test_n', 'test_lengths', 'neutral_slots', 'neutral_placements', 'source_seeds', 'tasks'):
            self.assertEqual(p[key], u1[key], key)

    def test_conditions_cover_two_widths_two_tasks_three_seeds(self):
        cs = gc.conditions()
        self.assertEqual(len(cs), 12)
        self.assertEqual({c['width'] for c in cs}, {32, 64})
        self.assertEqual({c['task'] for c in cs}, {'five', 'stack'})
        self.assertEqual({c['seed'] for c in cs}, {117, 143, 191})
        self.assertEqual(len({gc.ident(c) for c in cs}), 12)

    def test_width_64_is_about_parameter_matched_to_the_core_arm(self):
        small = gc.parameter_count(gc.make_model(dict(task='five', width=32, seed=1)))
        matched = gc.parameter_count(gc.make_model(dict(task='five', width=64, seed=1)))
        self.assertLess(small, 10_000)
        self.assertTrue(20_000 < matched < 30_000, matched)       # the Core arm has about 24k parameters

    def test_model_shapes_follow_the_task_vocabulary_and_classes(self):
        for task in ('five', 'stack'):
            m = gc.make_model(dict(task=task, width=16, seed=1))
            x, _ = tf.generate(task, 3, 10, seed=1)
            self.assertEqual(tuple(m(x).shape), (3, 10, tf.CLASSES[task]))


class Training(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.c = dict(arm='gru16', task='five', width=16, seed=117)

    def test_training_uses_the_u1_batches_and_is_deterministic(self):
        p = gc.protocol()
        a, ra = gc.train(self.c, p, steps=8, validation_every=4, validation_n=16)
        b, rb = gc.train(self.c, p, steps=8, validation_every=4, validation_n=16)
        self.assertEqual(ra['selected_step'], rb['selected_step'])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())))
        # the Core arms of U1 saw exactly these batches
        batch = tf.train_batch(p['source_seeds'], 'five', 117, 1, p['batch'], p['train_length'])
        self.assertEqual(tuple(batch['x'].shape), (p['batch'], p['train_length']))

    def test_selection_is_the_minimum_validation_final_loss_with_earliest_step_on_ties(self):
        p = gc.protocol()
        _, r = gc.train(self.c, p, steps=12, validation_every=4, validation_n=16)
        losses = {h['step']: h['final_loss'] for h in r['history']}
        self.assertEqual(sorted(losses), [4, 8, 12])
        best = min(losses.values())
        self.assertEqual(r['selected_step'], min(s for s, v in losses.items() if v == best))
        self.assertEqual(r['guard']['skipped'], 0)


class Evaluation(unittest.TestCase):
    def test_evaluation_reports_clean_lengths_and_neutral_rows_with_the_u1_keys(self):
        torch.set_num_threads(1)
        p = gc.protocol()
        c = dict(arm='gru16', task='five', width=16, seed=117)
        model = gc.make_model(c)
        row = gc.evaluate(model, c, p, n=8, clean_lengths=[16, 32], neutral_slots=[0, 8], neutral_placements=['prefix', 'suffix'], train_length=16)
        self.assertEqual(sorted(row['clean']), ['16', '32'])
        for r in row['clean'].values():
            self.assertTrue(0.0 <= r['final_accuracy'] <= 1.0 and 0.0 <= r['prefix_accuracy'] <= 1.0)
        self.assertEqual(sorted(row['neutral']), ['0-base', '8-prefix', '8-suffix'])
        self.assertAlmostEqual(row['neutral']['8-suffix']['loss_pp'], 100 * (row['neutral']['0-base']['final_accuracy'] - row['neutral']['8-suffix']['final_accuracy']))
        self.assertEqual(row['neutral']['8-suffix']['n'], 8)

    def test_a_model_that_ignores_its_input_loses_nothing_to_neutral_insertion_it_was_already_wrong_about(self):
        # the loss is base minus inserted accuracy; for a constant predictor it is exactly zero (same answer with or without neutrals)
        class Constant(torch.nn.Module):
            def forward(self, x):
                out = torch.zeros(x.shape[0], x.shape[1], 5)
                out[..., 3] = 1.0
                return out
        torch.set_num_threads(1)
        p = gc.protocol()
        row = gc.evaluate(Constant(), dict(task='five', seed=117), p, n=8, clean_lengths=[16], neutral_slots=[0, 8], neutral_placements=['middle'], train_length=16)
        self.assertEqual(row['neutral']['8-middle']['loss_pp'], 0.0)


class Runner(unittest.TestCase):
    def test_run_one_writes_json_and_weights_and_is_idempotent(self):
        torch.set_num_threads(1)
        c = dict(arm='gru16', task='five', width=16, seed=117)
        with tempfile.TemporaryDirectory() as d:
            small = dict(steps=4, validation_every=2, validation_n=8, n=8, clean_lengths=[16], neutral_slots=[0, 8], neutral_placements=['suffix'], train_length=16)
            first = gc.run_one(d, c, **small)
            self.assertTrue((Path(d) / f'{gc.ident(c)}.json').exists())
            weights = Path(d) / f'{gc.ident(c)}.pt'
            self.assertTrue(weights.exists())
            raw = torch.load(weights, weights_only=True)
            self.assertEqual(raw['meta']['task'], 'five')
            second = gc.run_one(d, c, **small)
            self.assertEqual(first['train']['selected_step'], second['train']['selected_step'])      # reused, not re-run


if __name__ == '__main__':
    unittest.main()
