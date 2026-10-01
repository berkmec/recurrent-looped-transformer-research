"""Time-shifted transplant: feed each row its OWN feedback stream, delta steps stale.

If the recurrent feedback is a generic clock (how long since the fact) rather than content, a stream that is delta steps
out of phase should break retrieval near the trained maximum, while a stream with the right phase and different content
(the donor transplant) does not. The intervention is only evidence if delta = 0 is bitwise the ordinary pass and the
shifted source is exactly the clean stream read at an earlier index.
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


class ShiftedSourceTests(unittest.TestCase):
    def setUp(self):
        self.core = tiny_core()
        torch.manual_seed(3)
        self.x = torch.randint(0, 256, (3, 20))
        with torch.no_grad():
            self.clean = ss.clean_states(self.core, self.x)
            self.base = self.core(self.x)

    def test_delta_zero_is_the_clean_stream(self):
        self.assertTrue(torch.equal(ss.shifted_source(self.clean, 0), self.clean))

    def test_a_positive_delta_reads_an_earlier_index_and_clamps_at_the_start(self):
        shifted = ss.shifted_source(self.clean, 5)
        self.assertTrue(torch.equal(shifted[:, 10], self.clean[:, 5]))
        self.assertTrue(torch.equal(shifted[:, 2], self.clean[:, 0]))      # nothing earlier than the first state

    def test_delta_zero_transplant_is_bitwise_the_ordinary_pass(self):
        with torch.no_grad():
            out = ss.forward_with_transplant(self.core, self.x, 8, ss.shifted_source(self.clean, 0))
        self.assertTrue(torch.equal(self.base, out))

    def test_a_shifted_stream_changes_what_follows(self):
        with torch.no_grad():
            out = ss.forward_with_transplant(self.core, self.x, 8, ss.shifted_source(self.clean, 6))
        self.assertFalse(torch.equal(self.base[:, 8:], out[:, 8:]))
        self.assertTrue(torch.equal(self.base[:, :8], out[:, :8]))          # and cannot reach backwards in time

    def test_a_negative_delta_is_refused_because_it_would_read_the_future(self):
        with self.assertRaises(ValueError):
            ss.shifted_source(self.clean, -3)


class AlignedSourceTests(unittest.TestCase):
    """A graft aligns another row's stream by the QUESTION's start, so the offset can have either sign and the target can
    be a different width from the source."""

    def setUp(self):
        torch.manual_seed(5)
        self.clean = torch.randn(2, 12, 4)          # a stand-in stream: [batch, steps, width]

    def test_offset_zero_at_the_same_width_is_the_stream_itself(self):
        self.assertTrue(torch.equal(ss.aligned_source(self.clean, 0, 12), self.clean))

    def test_a_positive_offset_reads_earlier_states(self):
        aligned = ss.aligned_source(self.clean, 3, 12)
        self.assertTrue(torch.equal(aligned[:, 7], self.clean[:, 4]))

    def test_a_negative_offset_reads_later_states_and_clamps_at_the_last(self):
        aligned = ss.aligned_source(self.clean, -3, 12)
        self.assertTrue(torch.equal(aligned[:, 4], self.clean[:, 7]))
        self.assertTrue(torch.equal(aligned[:, 11], self.clean[:, 11]))      # 11 + 3 = 14 > 11: clamped

    def test_the_target_width_decides_the_length(self):
        self.assertEqual(ss.aligned_source(self.clean, 2, 30).shape, (2, 30, 4))

    def test_shifted_source_still_refuses_a_negative_delta(self):
        with self.assertRaises(ValueError):
            ss.shifted_source(self.clean, -1)

    def test_shifted_source_is_the_aligned_source_at_the_same_width(self):
        self.assertTrue(torch.equal(ss.shifted_source(self.clean, 4), ss.aligned_source(self.clean, 4, 12)))


class ConditionTests(unittest.TestCase):
    def test_shift_conditions_parse(self):
        self.assertEqual(ss.parse_condition('shift32-question'), ('shift32', 'question'))
        self.assertEqual(ss.parse_condition('shift8-fact'), ('shift8', 'fact'))

    def test_a_malformed_shift_is_refused(self):
        for bad in ('shift-question', 'shiftx-fact', 'shift-3-fact', 'shift32'):
            with self.assertRaises(ValueError, msg=bad):
                ss.parse_condition(bad)


if __name__ == '__main__':
    unittest.main()
