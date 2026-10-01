"""Findings of the independent review of the Z1 code (swarm-bridge task 001, reviewer: jcode, 2026-10-02). Every finding was reproduced
with the reviewer's own test first (11 failing + 4 pinning tests, all as reported), judged on its merits, and the accepted ones are pinned
here in their FIXED form. B-numbers are the reviewer's. B11 is a design choice that is documented and pinned, not changed.

  B1  learned_in_range failed open when the trained-range point was not evaluated
  B2  resync_mask docstring said "everything after the first floor visit": deeper post-reset positions belong to `deep` (partition test)
  B3  gru_control skipped a non-finite step where U1 aborts (the "EXACTLY the U1 protocol" claim was false for that event)
  B4  the "sealed numbers" test compared a pass-through with itself: the literal numbers are now pinned and drift is refused
  B5  train_batch's source_seed did not name the depth it generated with
  B6  heldout(split='train') could reproduce a training draw
  B7  `deep` measures emission of the saturated label, not counting (pinned)
  B8  losses_from_gru read back a stored loss_pp without checking it against the accuracies
  B9  a NaN validation loss locked model selection
  B10 gru_control.train(steps=0) raised an undefined TypeError
  B11 heldout draws of different sizes share one seed (by design; perm draws nest)
  B12 the "256x" scale claim: the default evaluation points reach 64x (permutation length) and 32x (counter peak)
"""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from horizon import depth_run as dr
from horizon import depth_tasks as dt
from horizon import gru_control as gc
from horizon import sweep as sw

NS = dict(train='z1-train', dev='z1-dev', test='z1-test')


class B1TrainedRangeIsAlwaysMeasured(unittest.TestCase):
    def test_custom_points_without_the_trained_range_point_still_measure_it(self):
        torch.set_num_threads(1)
        r = dr.run('count', 'gru', 16, seed=1, steps=0, points=[16, 32], n_eval=6)
        self.assertIn(dt.TRAIN_DEPTH, r['curve'])                  # added automatically
        self.assertIsNotNone(r['in_range_accuracy'])
        self.assertFalse(r['learned_in_range'])                     # an untrained model did not learn anything
        self.assertEqual(r['effective_horizon'], 0)


class B2PostExceedancePartition(unittest.TestCase):
    def test_every_position_after_the_first_exceedance_is_in_exactly_one_of_deep_reentry_resync(self):
        x = torch.tensor([[0] * 9 + [1] * 9 + [0] * 10 + [1] * 2])      # up to 9, down to the floor, up to 10, down to 8
        traj = dt.depth_trajectory(x)
        exceeded = traj.cummax(dim=1).values > dt.TRAIN_DEPTH
        parts = torch.stack([traj > dt.TRAIN_DEPTH, dt.reentry_mask(traj), dt.resync_mask(traj)]).int().sum(0)
        self.assertTrue(bool((parts[exceeded] == 1).all()))
        self.assertTrue(bool((parts[~exceeded] == 0).all()))

    def test_the_docstring_no_longer_claims_everything_after_the_floor_visit(self):
        self.assertNotIn('Everything after that first floor visit', dt.resync_mask.__doc__)
        self.assertIn('deep', dt.resync_mask.__doc__)


class B3NonFiniteAborts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        real = torch.nn.functional.cross_entropy
        self.calls = {'n': 0}

        def fake(*a, **k):
            self.calls['n'] += 1
            return torch.tensor(float('nan')) if self.calls['n'] == 1 else real(*a, **k)
        self.fake = fake
        self.c = dict(arm='gru8', task='five', width=8, seed=117)

    def test_a_nonfinite_training_loss_aborts_like_u1(self):
        p = {**gc.protocol(), 'batch': 2, 'train_length': 8}
        with mock.patch('torch.nn.functional.cross_entropy', side_effect=self.fake):
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                gc.train(self.c, p, steps=2, validation_every=2, validation_n=2)

    def test_run_one_preserves_the_failure_like_u1(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch('torch.nn.functional.cross_entropy', side_effect=self.fake):
                with self.assertRaises(ValueError):
                    gc.run_one(d, self.c, steps=2, validation_every=2, validation_n=2, n=4, clean_lengths=[16], neutral_slots=[0], neutral_placements=['suffix'])
            self.assertTrue((Path(d) / f'{gc.ident(self.c)}-failure.json').exists())
            self.assertFalse((Path(d) / f'{gc.ident(self.c)}.json').exists())            # no result is written for a failed run


class B4SealedProtocol(unittest.TestCase):
    SEAL = dict(steps=2048, batch=32, train_length=128, lr=.003, weight_decay=.01, clip=1., validation_every=128, validation_n=256, test_n=1000,
                primary_seeds=[117, 143, 191], source_seeds=dict(train=91010000, validation=92010000, test=93010000, neutral=93020000))

    def test_the_protocol_matches_the_literal_sealed_numbers(self):
        p = gc.protocol()
        for key, value in self.SEAL.items():
            self.assertEqual(p[key], value, key)

    def test_a_drifting_sweep_protocol_is_refused(self):
        drifted = dict(sw.protocol(), steps=9999)
        with mock.patch('horizon.sweep.protocol', return_value=drifted):
            with self.assertRaises(ValueError):
                gc.protocol()


class B5TrainSeedNamesTheDepth(unittest.TestCase):
    def test_different_depths_get_different_source_seeds(self):
        a = dt.train_batch(NS, 'count', 1, 5, 8, length=64, depth=0)
        b = dt.train_batch(NS, 'count', 1, 5, 8, length=64, depth=5)
        self.assertNotEqual(a['source_seed'], b['source_seed'])

    def test_the_standard_training_seed_is_unchanged_so_every_recorded_pilot_stays_reproducible(self):
        old = dt.source_seed(NS, 'count', 1, 'train', step=5, length=64)              # the derivation before the fix (depth = 0)
        self.assertEqual(dt.train_batch(NS, 'count', 1, 5, 8, length=64)['source_seed'], old)
        self.assertEqual(old, int.from_bytes(hashlib.sha256(b'z1-train|count|1|5|64|0').digest()[:8], 'little'))


class B6TrainSplitIsNotHeldout(unittest.TestCase):
    def test_heldout_refuses_the_train_split(self):
        with self.assertRaises(ValueError):
            dt.heldout(NS, 'count', 1, 'train', 8, 64)

    def test_the_evaluation_splits_still_work(self):
        for split in ('dev', 'test'):
            x, y, seed = dt.heldout(NS, 'count', 1, split, 4, 16)
            self.assertEqual(tuple(x.shape), (4, 16))


class B7DeepMeasuresTheSaturatedLabelNotCounting(unittest.TestCase):
    class ConstantCap:
        def eval(self):
            return self

        def __call__(self, x):
            out = torch.zeros(x.shape[0], x.shape[1], dt.classes('count'))
            out[:, :, dt.LABEL_CAP] = 1.0
            return out

    class PerfectCounter:
        def eval(self):
            return self

        def __call__(self, x):
            return torch.nn.functional.one_hot(dt.oracle('count', x), dt.classes('count')).float()

    def test_a_model_that_always_says_the_cap_ties_a_perfect_counter_on_deep_and_only_reentry_separates_them(self):
        constant = dr.evaluate(self.ConstantCap(), 'count', [16], n=6, seed=1)[16]
        perfect = dr.evaluate(self.PerfectCounter(), 'count', [16], n=6, seed=1)[16]
        self.assertEqual(constant['deep'], 1.0)
        self.assertEqual(constant['deep'], perfect['deep'])
        self.assertGreater(perfect['reentry'], constant['reentry'])


class B8InconsistentRowIsRefused(unittest.TestCase):
    def test_a_row_whose_stored_loss_contradicts_its_accuracy_is_refused(self):
        row = dict(neutral={'0-base': dict(final_accuracy=1.0)})
        for cell in gc.CELLS_ALL:
            row['neutral'][cell] = dict(final_accuracy=0.7, loss_pp=1.0)          # the accuracies say 30 points
        with self.assertRaises(ValueError):
            gc.losses_from_gru(row)


class B9NanValidationIsNotSelected(unittest.TestCase):
    def test_a_nan_validation_candidate_never_blocks_a_finite_one(self):
        torch.set_num_threads(1)
        calls = {'n': 0}

        def fake_predict(model, x, y, batch=32):
            calls['n'] += 1
            loss = float('nan') if calls['n'] == 1 else 1.0                          # step 2: NaN, step 4: finite
            return dict(final_loss=loss, final_accuracy=0.5, prefix_accuracy=0.5), None
        p = {**gc.protocol(), 'batch': 2, 'train_length': 8}
        c = dict(arm='gru8', task='five', width=8, seed=117)
        with mock.patch('horizon.gru_control.predict', side_effect=fake_predict):
            _, info = gc.train(c, p, steps=4, validation_every=2, validation_n=2)
        self.assertEqual(info['selected_step'], 4)

    def test_no_finite_candidate_at_all_is_an_error(self):
        torch.set_num_threads(1)

        def fake_predict(model, x, y, batch=32):
            return dict(final_loss=float('nan'), final_accuracy=0.5, prefix_accuracy=0.5), None
        p = {**gc.protocol(), 'batch': 2, 'train_length': 8}
        with mock.patch('horizon.gru_control.predict', side_effect=fake_predict):
            with self.assertRaises(ValueError):
                gc.train(dict(arm='gru8', task='five', width=8, seed=117), p, steps=2, validation_every=2, validation_n=2)


class B10ZeroStepsIsACleanError(unittest.TestCase):
    def test_steps_zero_is_a_value_error(self):
        p = {**gc.protocol(), 'batch': 2, 'train_length': 8}
        with self.assertRaises(ValueError):
            gc.train(dict(arm='gru8', task='five', width=8, seed=117), p, steps=0, validation_every=2, validation_n=2)


class B11DrawSizeIsNotInTheSeedByDesign(unittest.TestCase):
    def test_permutation_draws_of_different_sizes_share_a_seed_and_nest(self):
        a = dt.heldout(NS, 'perm4', 1, 'dev', 4, 12)
        b = dt.heldout(NS, 'perm4', 1, 'dev', 8, 12)
        self.assertEqual(a[2], b[2])
        self.assertTrue(torch.equal(a[0], b[0][:4]))               # documented: use ONE n per evaluation point


class B12DefaultScales(unittest.TestCase):
    def test_default_points_reach_64x_for_permutation_length_and_32x_for_the_counter_peak(self):
        self.assertEqual(max(dr.default_points('perm5')) // dt.TRAIN_LENGTH, 64)
        self.assertEqual(max(dr.default_points('count')) // dt.TRAIN_DEPTH, 32)

    def test_the_module_does_not_claim_a_scale_the_defaults_do_not_reach(self):
        self.assertNotIn('256x', dr.__doc__)


class Q6GeneratorPins(unittest.TestCase):
    """The reviewer's independent counts of the counter generator (question 6), kept as regression pins."""

    def test_training_never_pushes_at_the_cap_depth(self):
        x, _ = dt.generate('count', 300, 80, seed=2026)
        traj = dt.depth_trajectory(x)
        before = torch.cat([torch.zeros_like(traj[:, :1]), traj[:, :-1]], dim=1)
        self.assertEqual(int(((x == 0) & (before == dt.TRAIN_DEPTH)).sum()), 0)
        self.assertLessEqual(int(traj.max()), dt.TRAIN_DEPTH)

    def test_test_sequences_reach_exactly_the_peak_at_every_protocol_depth(self):
        for depth in (8, 16, 32, 64, 128, 256):
            x, _ = dt.generate('count', 30, dt.count_eval_length(depth), seed=5, depth=depth)
            self.assertTrue(bool((dt.depth_trajectory(x).max(dim=1).values == depth).all()), depth)


if __name__ == '__main__':
    unittest.main()
