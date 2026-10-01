"""Z1 arms and runner mechanics (not learning outcomes)."""
import unittest

import torch

from horizon import depth_run as dr
from horizon import depth_tasks as dt
from horizon.depth_models import ARMS, make_arm, parameter_count


class Arms(unittest.TestCase):
    def test_shapes_for_every_arm_and_task(self):
        for task in ('perm3', 'count'):
            x, _ = dt.generate(task, 3, 10, seed=1)
            for arm in ARMS:
                m = make_arm(arm, task, 32, seed=1)
                self.assertEqual(tuple(m(x).shape), (3, 10, dt.classes(task)), (task, arm))

    def test_core_embedding_matches_the_task_vocabulary(self):
        m = make_arm('core', 'perm5', 32, seed=1)
        self.assertEqual(m.embedding.num_embeddings, 120)
        self.assertTrue(m.feedback)
        self.assertFalse(make_arm('core_nofb', 'perm5', 32, seed=1).feedback)

    def test_same_seed_same_weights_different_seed_different(self):
        a = make_arm('core', 'perm3', 32, seed=4)
        b = make_arm('core', 'perm3', 32, seed=4)
        c = make_arm('core', 'perm3', 32, seed=5)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())))
        self.assertFalse(all(torch.equal(p, q) for p, q in zip(a.parameters(), c.parameters())))
        self.assertGreater(parameter_count(a), 10000)

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            make_arm('lstm', 'perm3', 32, seed=1)


class Runner(unittest.TestCase):
    def test_training_is_deterministic_and_counts_guard_steps(self):
        torch.set_num_threads(1)
        m1, r1 = dr.train('perm3', 'gru', 16, seed=2, steps=4, batch=8)
        m2, r2 = dr.train('perm3', 'gru', 16, seed=2, steps=4, batch=8)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(m1.parameters(), m2.parameters())))
        self.assertEqual(r1['guard']['steps'], 4)
        self.assertEqual(r1['guard']['skipped'], 0)

    def test_evaluate_reports_accuracy_in_range_and_uses_the_requested_lengths(self):
        m = make_arm('gru', 'perm3', 16, seed=1)
        curve = dr.evaluate(m, 'perm3', [16, 32], n=10, seed=1)
        self.assertEqual(sorted(curve), [16, 32])
        for row in curve.values():
            self.assertTrue(0.0 <= row['final'] <= 1.0 and 0.0 <= row['prefix'] <= 1.0)
        self.assertIsNone(curve[16]['beyond_horizon'] if dt.TRAIN_LENGTH >= 16 else 0)   # length 16: nothing beyond the trained horizon
        self.assertIsNotNone(curve[32]['beyond_horizon'])

    def test_count_evaluation_reports_unseen_depth_and_peak(self):
        m = make_arm('gru', 'count', 16, seed=1)
        curve = dr.evaluate(m, 'count', [8, 16], n=10, seed=1)
        self.assertIsNone(curve[8]['unseen_depth'])             # peak 8 = the trained range: no position is deeper than 8
        self.assertIsNotNone(curve[16]['unseen_depth'])
        self.assertIsNotNone(curve[16]['at_peak'])

    def test_eval_split_is_separate_from_training_data(self):
        b = dt.train_batch(dr.NAMESPACE, 'perm4', 1, 1, 8)
        x, _, _ = dt.heldout(dr.NAMESPACE, 'perm4', 1, 'dev', 8, dt.TRAIN_LENGTH)
        self.assertFalse(torch.equal(b['x'], x))

    def test_effective_horizon(self):
        curve = {16: dict(final=1.0), 32: dict(final=0.97), 64: dict(final=0.90), 128: dict(final=1.0)}
        self.assertEqual(dr.effective_horizon(curve, 'final'), 32)       # 64 breaks the run; the 128 recovery does not count
        self.assertEqual(dr.effective_horizon({16: dict(final=0.5)}, 'final'), 0)
        self.assertEqual(dr.effective_horizon({8: dict(unseen_depth=None)}, 'unseen_depth'), 0)


if __name__ == '__main__':
    unittest.main()
