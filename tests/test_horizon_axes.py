"""U1 (t axis) behavioral tests, written before the implementation."""
import hashlib
import tempfile
from pathlib import Path
import unittest

import torch


class HorizonAxesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import task_families as tf, sweep as sw, metrics as mt
        except ImportError as e:
            self.fail('U1 horizon modules are not implemented: ' + str(e))
        return tf, sw, mt

    def test_stack_oracle_hand_computed(self):
        tf, _, _ = self.api()
        # tokens: 0 push a, 1 push b, 2 neutral, 3 pop (pop on empty is a no-op); label: top symbol a=0, b=1, empty=2
        x = torch.tensor([[0, 1, 2, 3, 3, 3, 1, 2, 0, 3]])
        expected = torch.tensor([[0, 1, 1, 0, 2, 2, 1, 1, 0, 1]])
        self.assertTrue(torch.equal(tf.oracle_targets(x, 'stack'), expected))
        self.assertEqual(tf.VOCAB['stack'], 4)
        self.assertEqual(tf.CLASSES['stack'], 3)
        self.assertEqual((tf.VOCAB['five'], tf.CLASSES['five']), (3, 5))

    def test_stack_memory_grows_with_length_and_five_is_finite_state(self):
        tf, _, _ = self.api()
        short = tf.generate('stack', 64, 128, 5)[0]
        long = tf.generate('stack', 64, 1024, 5)[0]
        d_short = tf.max_depth(short).float().mean()
        d_long = tf.max_depth(long).float().mean()
        self.assertGreater(float(d_long), float(d_short) * 2)
        self.assertGreater(float(d_long), 16.)
        x, y = tf.generate('five', 8, 300, 3)
        self.assertTrue(torch.equal(y, tf.oracle_targets(x, 'five')))
        self.assertEqual(int(y.max()), 4)

    def test_neutral_insertion_preserves_final_target_for_both_tasks(self):
        tf, _, _ = self.api()
        for task in ('stack', 'five'):
            x, y = tf.generate(task, 6, 64, 11)
            for placement in ('prefix', 'middle', 'suffix'):
                xx, yy = tf.insert_neutrals(x, task, 32, placement)
                self.assertEqual(xx.shape[1], 96)
                self.assertTrue(torch.equal(yy[:, -1], y[:, -1]))
                self.assertEqual(int((xx == 2).sum()) - int((x == 2).sum()), 6 * 32)

    def test_effective_horizon_and_flip_metrics(self):
        _, _, mt = self.api()
        self.assertEqual(mt.effective_horizon({128: .99, 256: .96, 512: .94, 1024: .97}, .95), 256)
        self.assertEqual(mt.effective_horizon({128: .90}, .95), 0)
        self.assertEqual(mt.effective_horizon({128: 1., 256: 1.}, .95), 256)

    def test_conditions_models_and_core_immutability(self):
        tf, sw, _ = self.api()
        p = sw.protocol()
        cs = sw.conditions(p)
        self.assertEqual(len(cs), 6)
        self.assertEqual({c['task'] for c in cs}, {'five', 'stack'})
        stack = sw.make_model(next(c for c in cs if c['task'] == 'stack'), p)
        five = sw.make_model(next(c for c in cs if c['task'] == 'five'), p)
        self.assertEqual(stack.embedding.num_embeddings, 4)
        self.assertEqual(five.embedding.num_embeddings, 3)
        self.assertEqual(stack.read.out_features, 3)
        a = sw.state_hash(sw.make_model(cs[0], p).state_dict()); b = sw.state_hash(sw.make_model(cs[0], p).state_dict())
        self.assertEqual(a, b)
        core = Path(__file__).resolve().parents[1] / 'core' / 'model.py'
        self.assertEqual(hashlib.sha256(core.read_bytes()).hexdigest(), '121f11fcb3c1308b7f67d898f2aa09c37226d3b0db27dd4e5bf1ebeaea281f43')

    def test_tiny_end_to_end_run_and_frozen_refusal(self):
        _, sw, _ = self.api()
        p = sw.protocol(); p.update(steps=3, validation_every=2, validation_n=8, train_length=16, test_lengths=[16, 32], neutral_slots=[0, 4], test_n=6, eval_batch=8)
        with tempfile.TemporaryDirectory() as d:
            sw.freeze(d, p)
            c = next(c for c in sw.conditions(p) if c['task'] == 'stack')
            receipt = sw.run_one(d, c)
            self.assertEqual(receipt['updates'], 3)
            self.assertNotEqual(receipt['initial_state_hash'], receipt['last_state_hash'])
            evaluation = sw.evaluate_one(d, c)
            self.assertEqual([r['length'] for r in evaluation['lengths']], [16, 32])
            self.assertEqual(len(evaluation['neutral']), 4)
            for row in evaluation['lengths'] + evaluation['neutral']:
                self.assertTrue((Path(d) / row['raw_file']).exists())
                self.assertIn('final_accuracy', row)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                sw.freeze(d, dict(p, steps=4))


if __name__ == '__main__':
    unittest.main()
