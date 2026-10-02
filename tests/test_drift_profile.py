"""E-A5x: the deviation split is what it says (systematic = the class-wide offset, unbiased by the even/odd estimate), the geometry scale is the mean nearest-centroid distance, a profile of a tiny
trained Core has every window."""
import tempfile
import unittest
from pathlib import Path

import torch

from horizon import depth_run as dr
from horizon import drift_profile as dp


class Decompose(unittest.TestCase):
    def test_a_class_wide_offset_is_systematic_and_independent_noise_is_not(self):
        g = torch.Generator().manual_seed(0)
        classes, per, dims = 6, 2000, 16
        offsets = torch.randn(classes, dims, generator=g)
        offsets = 2.0 * offsets / offsets.norm(dim=1, keepdim=True)                          # |offset| = 2: squared 4
        lab = torch.arange(classes).repeat_interleave(per)
        parity = torch.arange(classes * per) % 2 == 0
        noise = 0.5 * torch.randn(classes * per, dims, generator=g)                          # squared norm 16 * 0.25 = 4
        total, systematic = dp.decompose(offsets[lab] + noise, lab, parity)
        self.assertAlmostEqual(total, 8.0, delta=0.3)
        self.assertAlmostEqual(systematic, 4.0, delta=0.25)
        total0, systematic0 = dp.decompose(noise, lab, parity)
        self.assertAlmostEqual(total0, 4.0, delta=0.2)
        self.assertLess(abs(systematic0), 0.05)                                              # the even / odd product is unbiased: no inflation from sampling noise

    def test_nearest_centroid_distance_is_the_mean_over_classes(self):
        c = torch.tensor([[0.0], [1.0], [3.0]])
        self.assertAlmostEqual(dp.nearest_centroid_distance(c), (1 + 1 + 2) / 3, places=6)


class EndToEnd(unittest.TestCase):
    def test_a_profile_of_a_tiny_core_has_every_window_and_a_description(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'core', 32, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            r = dp.profile(out.with_suffix('.pt'), t0s=(8, 32), span=32, n=40, batch=20, log=lambda m: None)
        self.assertEqual((r['task'], r['width'], r['n'], r['span']), ('perm3', 32, 40, 32))
        self.assertEqual(set(r['profile']), {'injected@8', 'injected@32', 'untouched@32'})
        self.assertEqual(list(r['profile']['injected@32']), ['0-1', '1-4', '4-16', '16-32'])        # windows past the span are dropped, the last is clipped
        for rows in r['profile'].values():
            for v in rows.values():
                self.assertGreaterEqual(v['total'], v['systematic'] - 1e-9)
                self.assertGreaterEqual(v['random'], 0.0)
        self.assertIn('injected@32', dp.describe(r))


if __name__ == '__main__':
    unittest.main()
