"""Transplant: from one step on, every row receives another row's (or the batch mean's) whole feedback stream.

A single-step swap is a weak test, because the carried state is rewritten at every step and the swapped content dilutes
within a few bytes. The transplant replaces the feedback channel for the REST of the sequence, so anything the answer
reads from the recurrent channel has to come from the donor. It is only evidence if, with the row's own clean states as
the source, it is bitwise the ordinary forward pass, and if it cannot reach backwards in time.
"""
import unittest

import torch
from torch import nn

from core.model import Core
from mline import state_swap as ss


def tiny_core(seed=0):
    torch.manual_seed(seed)
    core = Core(width=32, heads=2, window=4, classes=256, feedback=True, alpha=.1)
    core.embedding = nn.Embedding(256, 32)
    return core.eval()


class TransplantTests(unittest.TestCase):
    def setUp(self):
        self.core = tiny_core()
        torch.manual_seed(3)
        self.x = torch.randint(0, 256, (4, 16))
        with torch.no_grad():
            self.clean = ss.clean_states(self.core, self.x)
            self.base = self.core(self.x)

    def test_clean_states_have_the_expected_shape(self):
        self.assertEqual(tuple(self.clean.shape), (4, 16, 32))

    def test_a_rows_own_clean_states_change_nothing(self):
        with torch.no_grad():
            out = ss.forward_with_transplant(self.core, self.x, 8, self.clean)
        self.assertTrue(torch.equal(self.base, out))

    def test_it_cannot_reach_backwards_in_time(self):
        source = self.clean[torch.tensor([1, 2, 3, 0])]
        with torch.no_grad():
            out = ss.forward_with_transplant(self.core, self.x, 8, source)
        self.assertTrue(torch.equal(self.base[:, :8], out[:, :8]))

    def test_another_rows_stream_changes_what_follows(self):
        source = self.clean[torch.tensor([1, 2, 3, 0])]
        with torch.no_grad():
            out = ss.forward_with_transplant(self.core, self.x, 8, source)
        self.assertFalse(torch.equal(self.base[:, 8:], out[:, 8:]))

    def test_the_batch_mean_stream_changes_what_follows_too(self):
        source = self.clean.mean(0, keepdim=True).expand_as(self.clean)
        with torch.no_grad():
            out = ss.forward_with_transplant(self.core, self.x, 8, source)
        self.assertFalse(torch.equal(self.base[:, 8:], out[:, 8:]))


class ConditionTests(unittest.TestCase):
    def test_conditions_parse_into_a_kind_and_a_point(self):
        self.assertEqual(ss.parse_condition('none'), ('none', None))
        self.assertEqual(ss.parse_condition('donor-question'), ('donor', 'question'))
        self.assertEqual(ss.parse_condition('mean-fact'), ('mean', 'fact'))
        with self.assertRaises(ValueError):
            ss.parse_condition('donor')
        with self.assertRaises(ValueError):
            ss.parse_condition('lobotomy-fact')


if __name__ == '__main__':
    unittest.main()
