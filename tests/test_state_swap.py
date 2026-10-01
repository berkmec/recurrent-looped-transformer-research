"""State swap: replace the recurrent state of each row with another row's at one step, then see whose answer comes out.

If the fact is carried in the recurrent state, the answer follows the DONOR; if attention carries it, the answer stays
the recipient's. The intervention is only evidence if, without a swap, it is bitwise the sealed forward pass, and if it
cannot reach backwards in time.
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


class SwapTests(unittest.TestCase):
    def setUp(self):
        self.core = tiny_core()
        torch.manual_seed(3)
        self.x = torch.randint(0, 256, (4, 16))
        self.donor = torch.tensor([1, 2, 3, 0])

    def test_without_a_swap_it_is_bitwise_the_sealed_forward(self):
        with torch.no_grad():
            self.assertTrue(torch.equal(self.core(self.x), ss.forward_with_swap(self.core, self.x)))

    def test_an_identity_donor_changes_nothing(self):
        identity = torch.arange(4)
        with torch.no_grad():
            base = ss.forward_with_swap(self.core, self.x)
            swapped = ss.forward_with_swap(self.core, self.x, swap_at=8, donor=identity)
        self.assertTrue(torch.equal(base, swapped))

    def test_it_cannot_reach_backwards_in_time(self):
        with torch.no_grad():
            base = ss.forward_with_swap(self.core, self.x)
            swapped = ss.forward_with_swap(self.core, self.x, swap_at=8, donor=self.donor)
        self.assertTrue(torch.equal(base[:, :8], swapped[:, :8]))

    def test_a_real_swap_changes_what_follows(self):
        with torch.no_grad():
            base = ss.forward_with_swap(self.core, self.x)
            swapped = ss.forward_with_swap(self.core, self.x, swap_at=8, donor=self.donor)
        self.assertFalse(torch.equal(base[:, 8:], swapped[:, 8:]))

    def test_a_swap_step_without_a_donor_is_refused(self):
        with self.assertRaises(ValueError):
            ss.forward_with_swap(self.core, self.x, swap_at=8)


class ClassifyTests(unittest.TestCase):
    def test_answers_are_attributed_to_donor_recipient_or_neither(self):
        own = torch.tensor([1, 2, 3, 4, 5])
        don = torch.tensor([9, 8, 3, 7, 6])       # row 2 has the same digit: it cannot tell the two apart
        got = torch.tensor([9, 2, 3, 0, 6])
        r = ss.attribute(got, own, don)
        self.assertEqual(r['usable'], 4)           # row 2 excluded
        self.assertAlmostEqual(r['donor'], 2 / 4)  # rows 0 and 4
        self.assertAlmostEqual(r['recipient'], 1 / 4)  # row 1
        self.assertAlmostEqual(r['neither'], 1 / 4)    # row 3


if __name__ == '__main__':
    unittest.main()
