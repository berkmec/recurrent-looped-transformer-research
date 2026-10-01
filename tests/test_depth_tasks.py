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
    depth, out = 0, []
    for t in tokens:
        depth = depth + 1 if int(t) == 0 else max(depth - 1, 0)
        out.append(min(depth, dt.CAP))
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
    def test_sizes(self):
        self.assertEqual((dt.vocab('count'), dt.classes('count')), (2, dt.CAP + 1))

    def test_oracle_matches_explicit_depth_and_floors_at_zero(self):
        x = torch.tensor([[1, 1, 0, 0, 0, 1, 1, 1, 1]])
        self.assertEqual(dt.oracle('count', x)[0].tolist(), [0, 0, 1, 2, 3, 2, 1, 0, 0])
        rng = np.random.default_rng(5)
        x = torch.from_numpy(rng.integers(0, 2, (4, 60)))
        for row in range(4):
            self.assertEqual(dt.oracle('count', x)[row].tolist(), explicit_depth(x[row].tolist()))

    def test_oracle_caps_the_label(self):
        x = torch.zeros(1, dt.CAP + 40, dtype=torch.long)
        y = dt.oracle('count', x)[0]
        self.assertEqual(int(y.max()), dt.CAP)
        self.assertEqual(int(y[dt.CAP + 10]), dt.CAP)

    def test_training_depth_never_exceeds_the_training_peak(self):
        x, y = dt.generate('count', 200, 80, seed=2)           # default: the training distribution, peak 8
        self.assertLessEqual(int(y.max()), dt.TRAIN_DEPTH)
        self.assertEqual(dt.TRAIN_DEPTH, 8)

    def test_controlled_peak_is_reached_and_not_exceeded(self):
        for peak in (16, 64, 128):
            x, y = dt.generate('count', 20, 4 * peak, seed=4, depth=peak)
            self.assertEqual(int(y.max()), peak)
            self.assertTrue((y.max(1).values == peak).all())    # every sequence climbs to the target first


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
