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
        self.assertEqual(mp.window_stats(gap, torch.zeros_like(alive), windows=((0, 5),))['0-5'], dict(n=0, sequences=0))


class ReviewFixes(unittest.TestCase):
    def test_survivors_follow_the_error_mask_when_one_is_given(self):
        # blind review R2-14: an exact tie with the target at the lower index is an error by the gap (gap = 0) but correct by argmax (depth_run's definition)
        logits = torch.tensor([[[1.0, 1.0], [0.0, 2.0], [2.0, 0.0]]])
        target = torch.tensor([[0, 1, 0]])
        gap = mp.gaps(logits, target)
        self.assertEqual(gap.tolist(), [[0.0, 2.0, 2.0]])
        self.assertEqual(mp.survivors(gap).tolist(), [[False, False, False]])                                    # gap <= 0 at position 0
        wrong = logits.argmax(-1) != target
        self.assertEqual(mp.survivors(gap, wrong).tolist(), [[True, True, True]])                                 # argmax picks index 0 = the target: no error
        # a NaN logit elsewhere: argmax != target is the mask depth_run would use, whatever the gap says
        nan_logits = torch.tensor([[[1.0, float('nan')], [3.0, 0.0]]])
        nan_target = torch.tensor([[0, 0]])
        self.assertEqual(mp.survivors(mp.gaps(nan_logits, nan_target), nan_logits.argmax(-1) != nan_target).tolist(), [[False, False]])

    def test_window_statistics_count_the_contributing_sequences_and_the_late_window_needs_enough_of_them(self):
        gap = torch.ones(30, 1024)
        alive = torch.zeros(30, 1024, dtype=torch.bool)
        alive[:, :64] = True
        alive[0, 64:] = True                                         # one single survivor to the end
        w = mp.window_stats(gap, alive)
        self.assertEqual((w['512-1024']['n'], w['512-1024']['sequences']), (512, 1))
        self.assertEqual(w['16-64']['sequences'], 30)
        self.assertEqual(mp.erosion(w)['verdict'], 'NO DATA')       # 512 positions but from ONE sequence (the first version read GROWTH or EROSION from it)
        w['512-1024']['sequences'] = mp.MIN_LATE_SEQUENCES
        self.assertEqual(mp.erosion(w)['late_window'], '512-1024')

    def test_counter_tasks_are_refused(self):
        import tempfile
        from pathlib import Path
        from horizon import depth_run as dr
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'c.json'
            dr.run('count', 'gru', 16, 1, steps=2, points=[8], n_eval=4, out=out, batch=4)
            with self.assertRaises(ValueError):
                mp.profile(out.with_suffix('.pt'), length=20, n=4, batch=4)


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
            # an independent recount (blind review R2-22: the old assertion n <= 8 x 16 could not fail): the positions before each sequence's first error, from the model's own argmax
            from horizon import depth_tasks as dt
            model = dr.load_model(out.with_suffix('.pt')).eval()
            x, y, _ = dt.heldout(dr.NAMESPACE, 'perm3', 1, 'dev', 8, 40)
            with torch.no_grad():
                wrong = model(x).argmax(-1) != y
            first_error = [int(w.int().argmax()) if bool(w.any()) else 40 for w in wrong]
            self.assertEqual(first['n'], sum(min(f, 16) for f in first_error))
            self.assertEqual(r['windows']['16-40']['n'], sum(max(0, min(f, 40) - 16) for f in first_error))
            self.assertIn('0-16', mp.describe(r))


if __name__ == '__main__':
    unittest.main()
