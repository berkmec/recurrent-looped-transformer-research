"""Margin profile: the gap is the true logit minus the largest other logit, survivors stop at the first error, the window statistics are what they say."""
import tempfile
import unittest
from pathlib import Path

import torch

from horizon import depth_run as dr
from horizon import margin_profile as mp


class Gap(unittest.TestCase):
    def test_gap_is_the_true_logit_minus_the_largest_other_logit(self):
        logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.5, 0.2, 0.9]]])
        target = torch.tensor([[0, 1, 0]])
        self.assertTrue(torch.allclose(mp.gaps(logits, target), torch.tensor([[1.0, 2.0, -0.4]])))

    def test_survivors_stop_at_the_first_error_and_keep_error_free_sequences_whole(self):
        gap = torch.tensor([[1.0, 2.0, -1.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0, 1.0], [-1.0, 5.0, 5.0, 5.0, 5.0], [2.0, 2.0, 2.0, 2.0, 0.0]])
        alive = mp.survivors(gap)
        self.assertEqual(alive.tolist(), [[True, True, False, False, False], [True] * 5, [False] * 5, [True, True, True, True, False]])


class Windows(unittest.TestCase):
    def test_statistics_use_survivors_only_and_drop_windows_past_the_length(self):
        gap = torch.stack([torch.arange(1.0, 11.0), torch.arange(1.0, 11.0)])           # gaps 1..10 at positions 0..9 in two sequences
        alive = torch.ones_like(gap, dtype=torch.bool)
        alive[1, 5:] = False                                                           # the second sequence died at position 5
        out = mp.window_stats(gap, alive, windows=((0, 5), (5, 10), (10, 20)))
        self.assertEqual(set(out), {'0-5', '5-10'})
        self.assertEqual(out['0-5']['n'], 10)
        self.assertAlmostEqual(out['0-5']['mean'], 3.0)
        self.assertEqual(out['5-10']['n'], 5)                                          # only the first sequence survives there
        self.assertAlmostEqual(out['5-10']['mean'], 8.0)
        self.assertAlmostEqual(out['5-10']['p50'], 8.0)
        self.assertAlmostEqual(out['0-5']['below']['2.0'], 2 / 10)                      # the gaps 1 and 1 of the two sequences
        self.assertAlmostEqual(out['0-5']['below']['4.0'], 6 / 10)                      # 1, 2, 3 in both
        self.assertEqual(mp.window_stats(gap, torch.zeros_like(alive), windows=((0, 5),))['0-5'], dict(n=0))


class EndToEnd(unittest.TestCase):
    def test_profile_of_a_saved_arm_and_its_description(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'gru', 16, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            r = mp.profile(out.with_suffix('.pt'), length=40, n=8, batch=4)
            self.assertEqual((r['arm'], r['task'], r['n'], r['length']), ('gru', 'perm3', 8, 40))
            self.assertEqual(list(r['windows']), ['0-16', '16-40'])                          # WINDOWS clipped to the evaluated length, later windows dropped
            first = r['windows']['0-16']
            self.assertLessEqual(first['n'], 8 * 16)
            self.assertIn('0-16', mp.describe(r))


if __name__ == '__main__':
    unittest.main()
