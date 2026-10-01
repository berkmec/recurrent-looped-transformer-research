"""Z1 depth-extrapolation tasks: the oracle is checked against an independent, explicit computation."""
import itertools
import unittest

import numpy as np
import torch

from horizon import depth_tasks as dt


def explicit_product(n, tokens):
    """Independent re-computation: multiply the permutations named by the tokens one by one."""
    perms = list(itertools.permutations(range(n)))
    state = tuple(range(n))
    out = []
    for g in tokens:
        b = perms[int(g)]
        state = tuple(state[b[k]] for k in range(n))      # state o g
        out.append(perms.index(state))
    return out


def explicit_depth(tokens):
    """True (uncapped) depth after each token."""
    depth, out = 0, []
    for t in tokens:
        depth = depth + 1 if int(t) == 0 else max(depth - 1, 0)
        out.append(depth)
    return out


class PermutationTasks(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual((dt.vocab('perm3'), dt.classes('perm3')), (6, 6))
        self.assertEqual((dt.vocab('perm4'), dt.classes('perm4')), (24, 24))
        self.assertEqual((dt.vocab('perm5'), dt.classes('perm5')), (120, 120))

    def test_oracle_matches_explicit_product(self):
        for task, n in (('perm3', 3), ('perm4', 4), ('perm5', 5)):
            x, y = dt.generate(task, 5, 40, seed=11)
            self.assertEqual(tuple(x.shape), (5, 40))
            self.assertEqual(tuple(y.shape), (5, 40))
            for row in range(5):
                self.assertEqual(y[row].tolist(), explicit_product(n, x[row].tolist()), task)

    def test_identity_token_keeps_the_state_and_start_is_identity(self):
        # token 0 is the identity permutation (itertools order), so it never changes the running product
        x = torch.zeros(1, 6, dtype=torch.long)
        self.assertEqual(dt.oracle('perm5', x)[0].tolist(), [0] * 6)

    def test_groups_are_closed_and_s5_is_not_abelian(self):
        _, _, table = dt.group(5)
        self.assertEqual(table.shape, (120, 120))
        self.assertTrue(((table >= 0) & (table < 120)).all())
        self.assertTrue((table != table.T).any())              # non-commutative: order matters, a counter cannot do it

    def test_labels_are_not_trivial(self):
        x, y = dt.generate('perm5', 64, 32, seed=3)
        self.assertGreater(len(set(y[:, -1].tolist())), 40)     # the running product spreads over many of the 120 states


class CounterTask(unittest.TestCase):
    def test_every_output_class_occurs_in_training(self):
        # a classifier cannot emit a class it never saw: the label is the depth capped at the TRAINED peak, so the test of counting
        # past the trained depth is about holding the count, not about unseen output classes
        self.assertEqual((dt.vocab('count'), dt.classes('count')), (2, dt.TRAIN_DEPTH + 1))
        self.assertEqual(dt.LABEL_CAP, dt.TRAIN_DEPTH)
        x, y = dt.generate('count', 400, 64, seed=2)
        self.assertEqual(set(y.flatten().tolist()), set(range(dt.TRAIN_DEPTH + 1)))

    def test_oracle_label_is_the_capped_true_depth_and_floors_at_zero(self):
        x = torch.tensor([[1, 1, 0, 0, 0, 1, 1, 1, 1]])
        self.assertEqual(dt.oracle('count', x)[0].tolist(), [0, 0, 1, 2, 3, 2, 1, 0, 0])
        rng = np.random.default_rng(5)
        x = torch.from_numpy(rng.integers(0, 2, (4, 60)))
        for row in range(4):
            depth = explicit_depth(x[row].tolist())
            self.assertEqual(dt.depth_trajectory(x)[row].tolist(), depth)
            self.assertEqual(dt.oracle('count', x)[row].tolist(), [min(d, dt.LABEL_CAP) for d in depth])

    def test_label_is_capped_but_the_depth_trajectory_is_not(self):
        x = torch.zeros(1, 30, dtype=torch.long)
        self.assertEqual(int(dt.oracle('count', x).max()), dt.LABEL_CAP)
        self.assertEqual(int(dt.depth_trajectory(x).max()), 30)

    def test_training_depth_never_exceeds_the_training_peak(self):
        x, _ = dt.generate('count', 300, 80, seed=2)           # default: the training distribution, peak 8
        self.assertLessEqual(int(dt.depth_trajectory(x).max()), dt.TRAIN_DEPTH)
        self.assertEqual(dt.TRAIN_DEPTH, 8)

    def test_controlled_peak_is_reached_and_not_exceeded(self):
        for peak in (16, 64, 128):
            x, _ = dt.generate('count', 20, 5 * peak, seed=4, depth=peak)
            traj = dt.depth_trajectory(x)
            self.assertEqual(int(traj.max()), peak)
            self.assertTrue((traj.max(1).values == peak).all())    # every sequence climbs to the target first

    def test_test_sequences_come_back_down_so_every_row_has_reentry_positions(self):
        x, _ = dt.generate('count', 20, 5 * 64, seed=4, depth=64)
        mask = dt.reentry_mask(dt.depth_trajectory(x))
        self.assertTrue(mask.any(1).all())

    def test_reentry_mask(self):
        traj = torch.tensor([[0, 5, 9, 12, 9, 8, 7, 3, 9, 8]])
        expected = [False, False, False, False, False, True, True, True, False, True]
        self.assertEqual(dt.reentry_mask(traj)[0].tolist(), expected)
        self.assertFalse(bool(dt.reentry_mask(torch.tensor([[0, 3, 8, 2, 8]])).any()))     # never deeper than the trained peak: no re-entry

    def test_first_descent_ends_at_the_first_return_to_the_floor_and_resync_is_what_follows(self):
        # after the first return to depth 0 the floor resets the count: those positions measure re-synchronisation, not counting
        traj = torch.tensor([[0, 5, 9, 8, 4, 0, 1, 2, 1, 0]])
        self.assertEqual(dt.reentry_mask(traj)[0].tolist(), [False, False, False, True, True, True, False, False, False, False])
        self.assertEqual(dt.resync_mask(traj)[0].tolist(),  [False, False, False, False, False, False, True, True, True, True])
        # the two masks never overlap and together cover every position that is back in range after having been deeper
        both = dt.reentry_mask(traj) | dt.resync_mask(traj)
        back_in_range = (traj <= dt.TRAIN_DEPTH) & (traj.cummax(dim=1).values > dt.TRAIN_DEPTH)
        self.assertTrue(torch.equal(both, back_in_range))
        self.assertFalse(bool((dt.reentry_mask(traj) & dt.resync_mask(traj)).any()))

    def test_a_zero_before_ever_exceeding_the_trained_depth_does_not_end_the_first_descent(self):
        traj = torch.tensor([[0, 1, 0, 4, 9, 12, 9, 7, 2, 0, 3]])
        self.assertEqual(dt.reentry_mask(traj)[0].tolist(), [False] * 7 + [True, True, True, False])
        self.assertEqual(dt.resync_mask(traj)[0].tolist(), [False] * 10 + [True])

    def test_every_generated_test_row_has_a_first_descent_of_useful_length(self):
        x, _ = dt.generate('count', 50, dt.count_eval_length(32), seed=5, depth=32)
        mask = dt.reentry_mask(dt.depth_trajectory(x))
        self.assertTrue(bool(mask.any(1).all()))
        self.assertGreater(float(mask.sum(1).float().mean()), 8.0)       # roughly the climb down from 8 to 0 (about 16 steps)


class Seeds(unittest.TestCase):
    def test_deterministic_and_split_separated(self):
        a = dt.generate('perm4', 8, 12, seed=9)
        b = dt.generate('perm4', 8, 12, seed=9)
        c = dt.generate('perm4', 8, 12, seed=10)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertFalse(torch.equal(a[0], c[0]))
        ns = dict(train='z1-train', dev='z1-dev', test='z1-test')
        seeds = {s: dt.source_seed(ns, 'perm4', 1, s) for s in ns}
        self.assertEqual(len(set(seeds.values())), 3)
        self.assertEqual(dt.source_seed(ns, 'perm4', 1, 'train', step=5), dt.source_seed(ns, 'perm4', 1, 'train', step=5))
        self.assertNotEqual(dt.source_seed(ns, 'perm4', 1, 'train', step=5), dt.source_seed(ns, 'perm4', 1, 'train', step=6))

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            dt.generate('perm3', 0, 5, seed=1)
        with self.assertRaises(ValueError):
            dt.generate('nope', 2, 5, seed=1)


if __name__ == '__main__':
    unittest.main()
