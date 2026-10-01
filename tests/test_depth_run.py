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

    def test_curriculum_advances_only_after_the_accuracy_threshold_and_never_trains_past_the_last_length(self):
        torch.set_num_threads(1)
        _, r = dr.train('perm3', 'gru', 32, seed=1, steps=500, batch=32, curriculum=[2, 4], advance_at=0.9, min_stage_steps=50, window=20)
        self.assertEqual(r['curriculum'], [2, 4])
        self.assertGreaterEqual(len(r['advances']), 1)
        self.assertEqual(r['advances'][0]['from_length'], 2)
        self.assertEqual(r['advances'][0]['to_length'], 4)
        self.assertGreaterEqual(r['advances'][0]['step'], 50)               # never before the minimum stage length
        self.assertEqual(r['final_stage_length'], 4)
        # an impossible threshold keeps the first stage for the whole run
        _, r = dr.train('perm3', 'gru', 16, seed=1, steps=60, batch=8, curriculum=[2, 4], advance_at=1.01, min_stage_steps=5, window=5)
        self.assertEqual(r['advances'], [])
        self.assertEqual(r['final_stage_length'], 2)

    def test_evaluate_reports_accuracy_in_range_and_uses_the_requested_lengths(self):
        m = make_arm('gru', 'perm3', 16, seed=1)
        curve = dr.evaluate(m, 'perm3', [16, 32], n=10, seed=1)
        self.assertEqual(sorted(curve), [16, 32])
        for row in curve.values():
            self.assertTrue(0.0 <= row['final'] <= 1.0 and 0.0 <= row['prefix'] <= 1.0)
        self.assertIsNone(curve[16]['beyond_horizon'] if dt.TRAIN_LENGTH >= 16 else 0)   # length 16: nothing beyond the trained horizon
        self.assertIsNotNone(curve[32]['beyond_horizon'])

    def test_count_evaluation_reports_deep_and_reentry_accuracy(self):
        m = make_arm('gru', 'count', 16, seed=1)
        curve = dr.evaluate(m, 'count', [8, 16], n=10, seed=1)
        self.assertIsNone(curve[8]['deep'])                     # peak 8 = the trained range: no position is deeper than 8
        self.assertIsNone(curve[8]['reentry'])
        self.assertIsNotNone(curve[16]['deep'])
        self.assertIsNotNone(curve[16]['reentry'])              # climbs to 16, then comes back down to <= 8

    def test_effective_horizon_skips_points_where_a_metric_does_not_apply(self):
        curve = {8: dict(reentry=None), 16: dict(reentry=1.0), 32: dict(reentry=0.99), 64: dict(reentry=0.4)}
        self.assertEqual(dr.effective_horizon(curve, 'reentry'), 32)

    def test_length_control_uses_the_training_distribution_at_long_lengths(self):
        m = make_arm('gru', 'count', 16, seed=1)
        control = dr.evaluate_length_control(m, [64, 128], n=6, seed=1)
        self.assertEqual(sorted(control), [64, 128])
        for row in control.values():
            self.assertTrue(0.0 <= row['final'] <= 1.0 and 0.0 <= row['prefix'] <= 1.0)

    def test_an_untrained_model_is_flagged_as_not_learned_and_gets_no_horizon(self):
        torch.set_num_threads(1)
        r = dr.run('count', 'gru', 16, seed=1, steps=0, points=[8, 16], n_eval=6)
        self.assertFalse(r['learned_in_range'])
        self.assertEqual(r['effective_horizon'], 0)
        self.assertEqual(sorted(r['length_control']), [64, 256, 1024])

    def test_counter_trains_on_long_sequences_so_only_depth_is_extrapolated(self):
        self.assertEqual(dt.train_length('count'), dt.COUNT_TRAIN_LENGTH)
        self.assertEqual(dt.train_length('perm5'), dt.TRAIN_LENGTH)
        b = dt.train_batch(dr.NAMESPACE, 'count', 1, 1, 8)
        self.assertEqual(b['x'].shape[1], dt.COUNT_TRAIN_LENGTH)
        self.assertLessEqual(int(b['y'].max()), dt.TRAIN_DEPTH)
        # the shallowest evaluation point (peak = the trained peak) must use a sequence length that was trained on
        m = make_arm('gru', 'count', 16, seed=1)
        curve = dr.evaluate(m, 'count', [8], n=4, seed=1)
        self.assertEqual(curve[8]['length'], dt.COUNT_TRAIN_LENGTH)

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
