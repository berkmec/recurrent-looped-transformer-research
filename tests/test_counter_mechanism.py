"""The first-descent mechanism report: a saturated counter is right only at the floor arrival, a real counter everywhere."""
import unittest

import torch

from horizon import depth_tasks as dt
from horizon.counter_mechanism import window_by_depth


def excursion():
    """Climbs to depth 10 (above the trained peak 8), comes back down to 0 and stays on the floor for two more pops."""
    x = torch.tensor([[0] * 10 + [1] * 12])
    trajectory = dt.depth_trajectory(x)
    return trajectory, torch.clamp(trajectory, max=dt.LABEL_CAP)


class WindowByDepth(unittest.TestCase):
    def test_window_is_the_first_descent_depth_8_down_to_the_first_zero(self):
        trajectory, target = excursion()
        by_depth, total = window_by_depth(target.clone(), target, trajectory)
        self.assertEqual(total, [9, 9])
        self.assertEqual(sorted(by_depth), list(range(9)))
        self.assertTrue(all(v == [1, 1] for v in by_depth.values()))

    def test_saturated_counter_is_right_only_at_the_floor_arrival(self):
        trajectory, target = excursion()
        pred = torch.zeros_like(target)           # the model's own count hit the floor early and stayed there
        by_depth, total = window_by_depth(pred, target, trajectory)
        self.assertEqual(total, [1, 9])
        self.assertEqual(by_depth[0], [1, 1])
        self.assertTrue(all(by_depth[d] == [0, 1] for d in range(1, 9)))

    def test_positions_after_the_first_floor_visit_are_not_in_the_window(self):
        trajectory, target = excursion()
        _, total = window_by_depth(torch.full_like(target, 99), target, trajectory)    # nothing is right: only the window size matters
        self.assertEqual(total, [0, 9])           # 22 positions, 2 of them after the first floor visit, 3 deeper than the peak (depth 9, 10, 9)


if __name__ == '__main__':
    unittest.main()
