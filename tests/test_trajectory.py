"""U2 measurement layer tests (written before the implementation): anchors, displacement, repair curve."""
import unittest

import numpy as np
import torch


class TrajectoryTests(unittest.TestCase):
    def api(self):
        try:
            from horizon import trajectory as tj
        except ImportError as e:
            self.fail('U2 trajectory module is not implemented: ' + str(e))
        return tj

    def test_centroids_and_nearest_hand_computed(self):
        tj = self.api()
        z = np.array([[0., 0.], [0., 2.], [10., 10.], [10., 12.], [5., 5.]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1, 2])
        anchors = tj.centroids(z, labels, classes=3)
        np.testing.assert_allclose(anchors['centers'], [[0., 1.], [10., 11.], [5., 5.]])
        self.assertEqual(anchors['counts'].tolist(), [2, 2, 1])
        nearest = tj.nearest(np.array([[0., 0.9], [9., 11.], [6., 4.]]), anchors['centers'])
        self.assertEqual(nearest.tolist(), [0, 1, 2])
        self.assertGreater(anchors['spread'][0], 0.)

    def test_displacement_and_repair_curve(self):
        tj = self.api()
        # trajectory: oracle class 1 throughout; the model's nearest anchor drifts to 0 during a neutral run then returns
        oracle = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        nearest = np.array([1, 1, 0, 0, 0, 1, 1, 1])
        run = (2, 5)  # neutral positions 2..4
        d = tj.displacement(nearest, oracle, run)
        self.assertEqual(d['displaced_in_run'].tolist(), [1, 1, 1])  # nearest anchor is 0 at all three neutral positions
        self.assertEqual(d['displaced_at_run_end'], 1)
        curve = tj.repair_curve(nearest, oracle, run, horizon=3)
        self.assertEqual(curve.tolist(), [0, 0, 0])  # repaired from the first real op after the run
        curve2 = tj.repair_curve(np.array([1, 1, 0, 0, 0, 0, 0, 1]), oracle, run, horizon=3)
        self.assertEqual(curve2.tolist(), [1, 1, 0])

    def test_states_from_core_match_forward_and_are_per_position(self):
        tj = self.api()
        from core.model import Core, generate
        torch.manual_seed(3)
        core = Core(width=8, heads=1, window=4, classes=5)
        x, y = generate('five', 4, 12, 9)
        logits, states = tj.run_states(core, x)
        self.assertEqual(tuple(states.shape), (4, 12, 8))
        torch.testing.assert_close(logits, core(x), rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
