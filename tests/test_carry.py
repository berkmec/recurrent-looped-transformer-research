"""mline.carry: one window is the sealed Core itself; carry off is the Core on every window alone; carry on lets window 0 change
window 1 through the state and the gradient of window 1's loss reach window 0 - for the feedback Core only, never for its A0 twin."""
import unittest

import torch

from mline import carry
from mline import lm as ml


def _x(seed, n, w):
    return torch.randint(0, 256, (2, n, w), generator=torch.Generator().manual_seed(seed))


class Carry(unittest.TestCase):
    def setUp(self):
        self.core = ml.build('plain', width=32, heads=2, window=4, seed=3).eval()
        self.a0 = ml.build('a0', width=32, heads=2, window=4, seed=3).eval()

    def test_one_window_is_the_sealed_core(self):
        x = _x(0, 1, 24)
        with torch.no_grad():
            self.assertLess(float((carry.CarryLM(self.core)(x)[:, 0] - self.core(x[:, 0])).abs().max()), 1e-5)
            self.assertLess(float((carry.CarryLM(self.a0)(x)[:, 0] - self.a0(x[:, 0])).abs().max()), 1e-5)

    def test_carry_off_is_the_core_on_each_window_alone(self):
        x = _x(1, 3, 16)
        with torch.no_grad():
            out = carry.CarryLM(self.core)(x, read=False)
            for j in range(3):
                self.assertLess(float((out[:, j] - self.core(x[:, j])).abs().max()), 1e-5)

    def test_only_the_state_links_the_windows(self):
        x = _x(2, 2, 16)
        y = x.clone()
        y[:, 0] = _x(9, 1, 16)[:, 0]                                     # a different first window, the same second one
        with torch.no_grad():
            on = (carry.CarryLM(self.core)(x)[:, 1] - carry.CarryLM(self.core)(y)[:, 1]).abs().max()
            off = (carry.CarryLM(self.core)(x, read=False)[:, 1] - carry.CarryLM(self.core)(y, read=False)[:, 1]).abs().max()
            a0 = (carry.CarryLM(self.a0)(x)[:, 1] - carry.CarryLM(self.a0)(y)[:, 1]).abs().max()
        self.assertGreater(float(on), 1e-4)
        self.assertEqual(float(off), 0.0)
        self.assertEqual(float(a0), 0.0)

    def test_the_gradient_reaches_the_earlier_window(self):
        model = carry.CarryLM(self.core.train())
        x = _x(4, 2, 12)
        e0 = {}

        def keep(module, inputs, output):                                  # the first window's embedding output
            if 'first' not in e0:
                output.retain_grad()
                e0['first'] = output
        hook = self.core.embedding.register_forward_hook(keep)
        model(x)[:, 1].sum().backward()
        hook.remove()
        self.assertGreater(float(e0['first'].grad.abs().max()), 0.0)


if __name__ == '__main__':
    unittest.main()
