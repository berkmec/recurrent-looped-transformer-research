"""U4-sep2 contract (written before the implementation).

U4-sep optimised gap / within-class spread and failed twice over: the quantity does not separate robust from fragile
when conditions are grouped by measured repair, and a gated model can satisfy it by closing the gate, which collapses
the spread and destroys the task. The corrected target is gap / drift - the quantity that does separate - and the
decisive property asserted here is that collapsing the state now makes the loss WORSE, not better.
"""
import unittest

import torch


class GeomLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import geom_loss as gl
        except ImportError as e:
            self.fail('horizon.geom_loss is not implemented: ' + str(e))
        return gl

    def test_min_centre_distance_is_the_u2d_gap(self):
        gl = self.api()
        z = torch.tensor([[0., 0.], [2., 0.], [3., 4.], [5., 4.]])
        y = torch.tensor([0, 0, 1, 1])                       # centres (1,0) and (4,4) -> distance 5
        self.assertAlmostEqual(float(gl.min_centre_distance(z, y, 2)), 5.0, places=4)
        self.assertEqual(float(gl.min_centre_distance(z, torch.zeros(4, dtype=torch.long), 2)), 0.)   # one class -> 0

    def test_drift_is_displacement_across_the_run(self):
        gl = self.api()
        states = torch.zeros(2, 10, 3)
        states[:, 8] = torch.tensor([3., 4., 0.])            # position 8 sits 5 away from position 3
        self.assertAlmostEqual(float(gl.run_drift(states, start=4, end=8)), 5.0, places=4)

    def test_collapse_is_penalised_not_rewarded(self):
        """The failure that killed U4-sep: a frozen state made the old loss happy. It must make this one unhappy."""
        gl = self.api()
        y = torch.tensor([0, 0, 1, 1])
        collapsed = torch.full((4, 2), 0.5)                  # every state identical: gap 0 and drift 0
        flat_states = collapsed.unsqueeze(1).repeat(1, 10, 1)
        ratio = gl.gap_over_drift(collapsed, y, 2, flat_states, start=4, end=8)
        self.assertLess(float(ratio), 1e-3)
        self.assertAlmostEqual(float(gl.hinge(ratio, target=1.5)), 1.5, places=3)   # maximal penalty

        spread_out = torch.tensor([[0., 0.], [0., 0.], [10., 0.], [10., 0.]])       # gap 10
        moving = torch.zeros(4, 10, 2); moving[:, 8] = torch.tensor([2., 0.])       # drift 2 -> ratio 5
        good = gl.gap_over_drift(spread_out, y, 2, moving, start=4, end=8)
        self.assertAlmostEqual(float(good), 5.0, places=3)
        self.assertEqual(float(gl.hinge(good, target=1.5)), 0.)                     # past the target: stop pushing

    def test_hinge_stops_at_the_target_and_carries_gradient_below_it(self):
        gl = self.api()
        below = torch.tensor(0.5, requires_grad=True)
        loss = gl.hinge(below, target=1.5)
        self.assertAlmostEqual(float(loss), 1.0, places=5)
        loss.backward()
        self.assertLess(float(below.grad), 0.)               # pushing the ratio UP reduces the loss
        above = torch.tensor(9.0, requires_grad=True)
        self.assertEqual(float(gl.hinge(above, target=1.5)), 0.)

    def test_ratio_is_scale_invariant_but_not_collapse_invariant(self):
        """Scaling everything changes nothing (gap and drift scale together); collapsing changes everything."""
        gl = self.api()
        y = torch.tensor([0, 0, 1, 1])
        z = torch.tensor([[0., 0.], [0., 0.], [10., 0.], [10., 0.]])
        s = torch.zeros(4, 10, 2); s[:, 8] = torch.tensor([2., 0.])
        a = gl.gap_over_drift(z, y, 2, s, start=4, end=8)
        b = gl.gap_over_drift(z * 7, y, 2, s * 7, start=4, end=8)
        self.assertAlmostEqual(float(a), float(b), places=3)
        c = gl.gap_over_drift(z * 0, y, 2, s, start=4, end=8)
        self.assertLess(float(c), float(a))

    def test_end_to_end_term_is_differentiable_through_a_model_state(self):
        gl = self.api()
        z = torch.randn(40, 6, requires_grad=True)
        y = torch.randint(0, 5, (40,))
        s = torch.randn(40, 12, 6, requires_grad=True)
        loss = gl.hinge(gl.gap_over_drift(z, y, 5, s, start=4, end=10), target=1.5)
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(float(z.grad.abs().sum()) + float(s.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
