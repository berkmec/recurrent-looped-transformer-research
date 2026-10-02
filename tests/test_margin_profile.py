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


class Erosion(unittest.TestCase):
    def windows(self, base_mean, late_mean, late_n=1000, late_key='512-1024', base_below=0.0, late_below=0.05):
        return {'16-64': dict(n=4800, mean=base_mean, below={'2.0': base_below}), late_key: dict(n=late_n, mean=late_mean, below={'2.0': late_below})}

    def test_erosion_flat_partial_growth_and_no_data(self):
        r = mp.erosion(self.windows(13.6, 5.8))
        self.assertEqual((r['verdict'], r['late_window']), ('EROSION', '512-1024'))
        self.assertAlmostEqual(r['ratio'], 5.8 / 13.6)
        self.assertEqual((r['near_miss_base'], r['near_miss_late']), (0.0, 0.05))
        self.assertEqual(mp.erosion(self.windows(15.2, 15.1))['verdict'], 'FLAT')
        self.assertEqual(mp.erosion(self.windows(10.0, 7.0))['verdict'], 'EROSION')            # exactly the 0.7 limit counts
        self.assertEqual(mp.erosion(self.windows(10.0, 7.8))['verdict'], 'PARTIAL')
        self.assertEqual(mp.erosion(self.windows(10.0, 11.5))['verdict'], 'FLAT')              # exactly +15 %
        self.assertEqual(mp.erosion(self.windows(10.0, 12.0))['verdict'], 'GROWTH')
        # a late window with too few survivors is skipped in favour of the previous one, and no usable window means NO DATA
        w = self.windows(10.0, 5.0, late_n=100)
        w['256-512'] = dict(n=900, mean=9.9, below={'2.0': 0.0})
        r = mp.erosion(w)
        self.assertEqual((r['verdict'], r['late_window']), ('FLAT', '256-512'))
        self.assertEqual(mp.erosion(self.windows(10.0, 5.0, late_n=100))['verdict'], 'NO DATA')
        self.assertEqual(mp.erosion({})['verdict'], 'NO DATA')


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
