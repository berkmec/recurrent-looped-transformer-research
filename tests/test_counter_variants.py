"""The counter with a larger trained peak (count16 / count32 / count64): the default counter is untouched and the variants scale with P."""
import hashlib
import unittest

import torch

from horizon import depth_run as dr
from horizon import depth_tasks as dt
from horizon.depth_models import make_arm


def digest(*tensors):
    m = hashlib.sha256()
    for t in tensors:
        m.update(t.numpy().tobytes())
    return m.hexdigest()[:20]


class DefaultCounterIsUnchanged(unittest.TestCase):
    """Fingerprints taken from the code before the variants existed: every recorded counter pilot used these draws."""

    def test_training_and_held_out_draws_are_bit_identical(self):
        b = dt.train_batch(dr.NAMESPACE, 'count', 1, 1, 8)
        self.assertEqual(digest(b['x'], b['y']), '2bcca3e3fe329f47ddd8')
        b = dt.train_batch(dr.NAMESPACE, 'count', 3, 17, 16)
        self.assertEqual(digest(b['x'], b['y']), 'c89a49a7d95e40e06806')
        for depth, expected in ((16, '45b62e558d453fad54ca'), (64, 'b69e82a9aa091c03faf1')):
            x, y, _ = dt.heldout(dr.NAMESPACE, 'count', 1, 'dev', 20, dt.count_eval_length(depth), depth=depth)
            self.assertEqual(digest(x, y), expected)
        x, y, _ = dt.heldout(dr.NAMESPACE, 'count', 1, 'dev', 20, 256)
        self.assertEqual(digest(x, y), 'c05c679918fefc3a4431')

    def test_sizes_and_default_points(self):
        self.assertEqual((dt.peak('count'), dt.classes('count'), dt.train_length('count'), dt.count_eval_length(16)), (8, 9, 64, 96))
        self.assertEqual(dr.default_points('count'), [8, 16, 32, 64, 128, 256])
        self.assertEqual(dr.default_points('perm4'), [16, 32, 64, 128, 256, 512, 1024])


class Variants(unittest.TestCase):
    def test_peak_classes_and_lengths_scale_with_the_trained_peak(self):
        for task, p in (('count16', 16), ('count32', 32), ('count64', 64)):
            self.assertEqual((dt.peak(task), dt.classes(task), dt.train_length(task), dt.vocab(task)), (p, p + 1, 8 * p, 2))
            self.assertEqual(dt.count_eval_length(p, task), 8 * p)            # 6 p < 8 p: the training length is the floor
            self.assertEqual(dt.count_eval_length(8 * p, task), 48 * p)
        with self.assertRaises(ValueError):
            dt.peak('perm4')
        self.assertFalse(dt.is_count('perm4'))
        self.assertTrue(dt.is_count('count32'))

    def test_default_points_are_multiples_of_the_peak_capped_at_512(self):
        self.assertEqual(dr.default_points('count16'), [16, 32, 64, 128, 256, 512])
        self.assertEqual(dr.default_points('count32'), [32, 64, 128, 256, 512])
        self.assertEqual(dr.default_points('count64'), [64, 128, 256, 512])

    def test_training_depth_stays_within_the_variant_peak_and_uses_the_whole_range(self):
        for task, p in (('count16', 16), ('count32', 32)):
            x, y = dt.generate(task, 300, dt.train_length(task), seed=2)
            traj = dt.depth_trajectory(x)
            self.assertLessEqual(int(traj.max()), p)
            self.assertGreater(int(traj.max()), p // 2)                       # the walk really visits the upper half of the range
            self.assertEqual(set(y.flatten().tolist()), set(range(p + 1)))    # every output class occurs in training

    def test_a_label_saturates_at_the_variant_peak_not_at_eight(self):
        x = torch.zeros(1, 40, dtype=torch.long)                              # 40 pushes
        self.assertEqual(int(dt.oracle('count32', x).max()), 32)
        self.assertEqual(int(dt.oracle('count', x).max()), 8)

    def test_masks_follow_the_peak_and_partition_the_positions_after_the_first_excursion(self):
        traj = torch.tensor([[0, 5, 9, 12, 20, 12, 9, 8, 7, 3, 0, 1, 14, 9]])
        for peak in (8, 11, 16):
            deep, reentry, resync = traj > peak, dt.reentry_mask(traj, peak), dt.resync_mask(traj, peak)
            after = traj.cummax(dim=1).values > peak
            self.assertTrue(torch.equal(deep.int() + reentry.int() + resync.int(), after.int()), peak)    # each position in exactly one
        self.assertNotEqual(dt.reentry_mask(traj, 8).tolist(), dt.reentry_mask(traj, 16).tolist())

    def test_a_test_sequence_climbs_to_the_requested_depth_and_comes_back_down(self):
        task, depth = 'count32', 64
        x, _ = dt.generate(task, 20, dt.count_eval_length(depth, task), seed=4, depth=depth)
        traj = dt.depth_trajectory(x)
        self.assertTrue((traj.max(1).values == depth).all())
        self.assertTrue(bool(dt.reentry_mask(traj, dt.peak(task)).any(1).all()))

    def test_variants_draw_from_their_own_seeds(self):
        a = dt.train_batch(dr.NAMESPACE, 'count', 1, 1, 4, length=64)['x']
        b = dt.train_batch(dr.NAMESPACE, 'count16', 1, 1, 4, length=64)['x']
        self.assertFalse(torch.equal(a, b))

    def test_an_arm_has_the_variant_output_classes_and_a_run_reports_the_variant_ranges(self):
        x, _ = dt.generate('count32', 3, 10, seed=1)
        self.assertEqual(tuple(make_arm('gru', 'count32', 8, seed=1)(x).shape), (3, 10, 33))
        r = dr.run('count16', 'gru', 8, 1, steps=2, points=[16, 32], n_eval=4, batch=4)
        self.assertEqual(r['trained_range'], 16)
        self.assertEqual(sorted(r['curve']), [16, 32])
        self.assertEqual(sorted(r['length_control']), [128, 512, 2048])      # the trained length (8 x 16), 4 x and 16 x it
        self.assertEqual(r['horizon_metric'], 'reentry')


if __name__ == '__main__':
    unittest.main()
