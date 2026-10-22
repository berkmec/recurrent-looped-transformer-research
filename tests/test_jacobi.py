import unittest

import torch
import torch.nn.functional as F

from mline import jacobi as mj
from mline import lm as ml


def small(seed=3, width=32, heads=2):
    core = ml.build('plain', width=width, heads=heads, seed=seed).double()
    with torch.no_grad():                       # a strong merge, so the test does not pass only because feedback is weak
        core.merge.weight.mul_(8.0)
    return core


class JacobiTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.core = small()
        self.x = torch.randint(0, 256, (3, 24))

    def test_as_many_passes_as_positions_reproduce_the_loop(self):
        with torch.no_grad():
            exact = self.core(self.x)
            par = mj.jacobi(self.core, self.x, self.x.shape[1])
        self.assertLess(float((par - exact).abs().max()), 1e-10)

    def test_k_passes_make_the_first_k_positions_exact(self):
        with torch.no_grad():
            exact = self.core(self.x)
            for k in (1, 2, 5, 9):
                err = (mj.jacobi(self.core, self.x, k) - exact).abs().amax(dim=(0, 2))
                self.assertLess(float(err[:k].max()), 1e-10, k)

    def test_gradients_through_enough_passes_equal_the_loop_gradients(self):
        y = torch.randint(0, 256, self.x.shape)
        _, g_loop = mj._grads(self.core, self.core(self.x), y)
        _, g_par = mj._grads(self.core, mj.jacobi(self.core, self.x, self.x.shape[1]), y)
        cmp = mj._compare(g_par, g_loop)
        self.assertLess(cmp['relative'], 1e-9)
        self.assertGreater(cmp['cosine'], 1 - 1e-12)

    def test_feedback_off_is_refused(self):
        core = ml.build('a0', width=32, heads=2, seed=3)
        with self.assertRaises(ValueError):
            mj.jacobi(core, self.x, 2)

    def test_reading_rule(self):
        r = dict(floor_logits=1e-6, floor_grad=dict(relative=1e-6),
                 forward={str(k): dict(max=10.0 ** -k) for k in mj.ITERS},
                 gradient={str(k): dict(relative=10.0 ** -k) for k in mj.GRAD_ITERS})
        out = mj.reading(r)
        self.assertEqual(out['passes_forward'], 6)                  # first pass count with error <= max(10 * 1e-6, 1e-5)
        self.assertEqual(out['passes_gradient'], 4)
        self.assertEqual(out['verdict'], 'EXACT-FAST')
        r['forward'] = {str(k): dict(max=1.0) for k in mj.ITERS}
        self.assertEqual(mj.reading(r)['verdict'], 'NO CONVERGENCE')


if __name__ == '__main__':
    unittest.main()
