"""E-W weight noise: the perturbation is relative to each tensor's rms, reproducible, leaves the original untouched; eps_tol reads the grid; a saved arm runs end to end."""
import math
import tempfile
import unittest
from pathlib import Path

import torch

from horizon import depth_run as dr
from horizon import weight_noise as wn
from horizon.depth_models import make_arm


class Perturbation(unittest.TestCase):
    def test_zero_is_an_exact_copy_and_the_original_is_never_touched(self):
        model = make_arm('gru', 'perm4', 16, 1)
        before = [p.clone() for p in model.parameters()]
        copy0 = wn.perturb(model, 0.0, seed=1)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(copy0.parameters(), model.parameters())))
        self.assertIsNot(copy0, model)
        wn.perturb(model, 0.1, seed=1)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_the_noise_is_relative_to_each_tensors_rms_and_reproducible(self):
        model = make_arm('lstm', 'perm4', 32, 1)
        a, b, c = wn.perturb(model, 0.05, seed=3), wn.perturb(model, 0.05, seed=3), wn.perturb(model, 0.05, seed=4)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters())))
        self.assertFalse(all(torch.equal(x, y) for x, y in zip(a.parameters(), c.parameters())))
        for p0, p1 in zip(model.parameters(), a.parameters()):
            rms = float(p0.detach().square().mean().sqrt())
            if p0.numel() > 500 and rms > 0:                                      # large tensors: the sample std of the added noise matches eps x rms
                noise = (p1 - p0).detach()
                self.assertAlmostEqual(float(noise.std()) / (0.05 * rms), 1.0, delta=0.15)

    def test_zero_rms_tensors_stay_zero_and_a_core_runs(self):
        model = make_arm('core', 'perm4', 32, 1)
        with torch.no_grad():
            model.initial.zero_()
        out = wn.perturb(model, 0.1, seed=1)
        self.assertTrue(bool((out.initial == 0).all()))
        x = torch.randint(0, 24, (2, 10))
        self.assertEqual(tuple(out(x).shape), (2, 10, 24))


class Reading(unittest.TestCase):
    def test_eps_tol_is_the_first_grid_value_at_or_above_the_threshold(self):
        rows = {0.0: dict(hazard=2e-4), 0.01: dict(hazard=3e-4), 0.02: dict(hazard=2e-3), 0.05: dict(hazard=0.1)}
        self.assertEqual(wn.eps_tol(rows), 0.02)
        self.assertEqual(wn.eps_tol({0.0: dict(hazard=0.5), 0.01: dict(hazard=1e-5)}), math.inf)      # eps = 0 is not a tolerance
        self.assertEqual(wn.eps_tol({0.5: dict(hazard=None)}), math.inf)


class EndToEnd(unittest.TestCase):
    def test_measure_runs_on_a_saved_arm(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'gru', 16, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            r = wn.measure(out.with_suffix('.pt'), eps_grid=(0.0, 0.05), draws=2, points=(16, 32), n=8, batch=4)
            self.assertEqual(sorted(r['results']), [0.0, 0.05])
            self.assertEqual(r['results'][0.0]['draws'], 1)                                        # eps = 0 needs a single draw
            self.assertEqual(r['results'][0.05]['draws'], 2)
            self.assertGreater(r['results'][0.0]['exposure'], 0)
            self.assertIn('eps_tol', wn.describe(r))


if __name__ == '__main__':
    unittest.main()
