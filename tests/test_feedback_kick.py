"""E-L1K: a one-step kick changes only the steps from the kick on, the paired differences and their bins are what they say, the reading rule is mechanical."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from core.model import Core
from mline import feedback_kick as fk
from mline import lm as ml


def small_model():
    return ml.build('plain', width=64, heads=2, window=4, alpha=0.1, seed=3).eval()


def batches(n=2, batch=4, length=24, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = rng.integers(0, 256, size=(batch, length + 1))
        out.append((x[:, :-1].astype(np.int64), x[:, 1:].astype(np.int64)))
    return out


class Kick(unittest.TestCase):
    def test_a_kick_changes_only_the_steps_from_the_kick_on(self):
        model, x = small_model(), torch.from_numpy(batches(1, 3, 20)[0][0])
        with torch.no_grad():
            base = copy.deepcopy(model)(x)
        for mode in fk.MODES:
            kicked = fk.configure(copy.deepcopy(model), mode, at=7)
            with torch.no_grad():
                out = kicked(x)
            self.assertTrue(torch.allclose(out[:, :7], base[:, :7], atol=1e-6), mode)         # causal: nothing before the kick moves
            self.assertFalse(torch.allclose(out[:, 7], base[:, 7], atol=1e-4), mode)            # the kicked step itself
            self.assertFalse(torch.allclose(out[:, 8], base[:, 8], atol=1e-6), mode)            # the next step: the kicked output was fed back

    def test_no_mode_is_the_untouched_core_and_invalid_kicks_are_refused(self):
        model, x = small_model(), torch.from_numpy(batches(1, 3, 16)[0][0])
        with torch.no_grad():
            base = copy.deepcopy(model)(x)
            same = fk.configure(copy.deepcopy(model))(x)
        self.assertTrue(torch.equal(base, same))
        self.assertRaises(ValueError, fk.configure, model, 'other', 0)           # the first step has no fed-back state
        self.assertRaises(ValueError, fk.configure, model, 'other', None)
        self.assertRaises(ValueError, fk.configure, model, 'nonsense', 3)

    def test_the_substitution_is_the_one_a_hand_written_decode_loop_makes(self):
        # independent of the subclass: the sealed Core's own encode / decode in a loop, the fed-back state replaced at the kick step only
        model, x = small_model(), torch.from_numpy(batches(1, 3, 14)[0][0])

        def manual(mode, at):
            with torch.no_grad():
                e, _, _ = model.encode(x)
                n = model.mn(e)
                mk, mv = model.rope(model.split(model.mk(n))), model.split(model.mv(n))
                s, out = None, []
                for t in range(x.shape[1]):
                    if t == at:
                        s = dict(s)
                        s['s'] = s['s'].roll(1, dims=0) if mode == 'other' else torch.zeros_like(s['s'])
                    y, s = model.decode(e[:, t], s, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
                    out.append(y)
                return torch.stack(out, 1)

        for mode in fk.MODES:
            kicked = fk.configure(copy.deepcopy(model), mode, at=5)
            with torch.no_grad():
                out = kicked(x)
            self.assertTrue(torch.allclose(out, manual(mode, 5), atol=1e-6), mode)

    def test_zeroing_the_state_is_cutting_the_channel_for_that_step_only(self):
        # RMSNorm(0) = 0 and the merge has no bias, so a zero state feeds nothing: the same function as alpha = 0 at that one step (review finding F5, one step wide)
        model, x = small_model(), torch.from_numpy(batches(1, 3, 14)[0][0])

        class AlphaZeroAt(Core):
            at = 5

            def decode(self, e, s, mk, mv, t):
                keep = self.alpha
                if t == self.at:
                    self.alpha = 0.0
                try:
                    return super().decode(e, s, mk, mv, t)
                finally:
                    self.alpha = keep

        cut = copy.deepcopy(model)
        cut.__class__ = AlphaZeroAt
        zero = fk.configure(copy.deepcopy(model), 'zero', at=5)
        with torch.no_grad():
            self.assertTrue(torch.allclose(cut(x), zero(x), atol=1e-6))


class Differences(unittest.TestCase):
    def test_paired_differences_have_the_expected_shape_and_a_zero_control(self):
        model, pairs = small_model(), batches(2, 4, 30)
        diff, control = fk.kick_differences(model, pairs, at=6, mode='other', kmax=5)
        self.assertEqual(diff.shape, (8, 6))
        self.assertEqual(control, 0.0)
        self.assertGreater(float(np.abs(diff[:, 0]).mean()), 1e-4)                  # the kicked step moves
        reference = [fk.bits(fk.configure(model), x, y) for x, y in pairs]        # a longer untouched reference is sliced, not recomputed
        diff2, control2 = fk.kick_differences(model, pairs, at=6, mode='other', kmax=5, reference=reference)
        self.assertLess(control2, 1e-4)
        self.assertTrue(np.allclose(diff, diff2, atol=1e-4))

    def test_bins_are_per_sequence_means_with_a_standard_error(self):
        d = np.zeros((10, 129))
        d[:, 0], d[:, 8:16], d[:, 32:64] = 2.0, 0.1, 0.05
        d[:5, 1] = 1.0                                       # half of the sequences hurt at step 1: mean 0.5, per-sequence std 0.527
        bins = fk.binned(d)
        self.assertAlmostEqual(bins['0-0']['mean'], 2.0)
        self.assertAlmostEqual(bins['0-0']['se'], 0.0)
        self.assertAlmostEqual(bins['1-1']['mean'], 0.5)
        self.assertAlmostEqual(bins['1-1']['se'], 0.5270463 / np.sqrt(10), places=5)
        self.assertAlmostEqual(bins['8-15']['mean'], 0.1)
        self.assertAlmostEqual(bins['32-63']['mean'], 0.05)
        self.assertAlmostEqual(bins['64-128']['mean'], 0.0)
        self.assertEqual(bins['64-128']['n'], 10)
        self.assertNotIn('64-128', fk.binned(d[:, :40]))                                # bins past the available steps are dropped
        self.assertIn('32-39', fk.binned(d[:, :40]))                                    # the last bin is clipped


class ReadingRule(unittest.TestCase):
    def bins(self, d0, d8, d32, se32=0.01):
        return {'0-0': dict(mean=d0, se=0.01, n=100), '8-15': dict(mean=d8, se=0.01, n=100), '32-63': dict(mean=d32, se=se32, n=100)}

    def test_local_persistent_intermediate_no_bite_invalid_and_incomplete(self):
        self.assertEqual(fk.reading(self.bins(2.0, 0.1, 0.02))['verdict'], 'LOCAL')
        self.assertEqual(fk.reading(self.bins(2.0, 0.2, 0.1))['verdict'], 'LOCAL')                     # exactly at the two limits
        self.assertEqual(fk.reading(self.bins(2.0, 0.5, 0.2))['verdict'], 'INTERMEDIATE')
        self.assertEqual(fk.reading(self.bins(2.0, 1.5, 1.0, se32=0.1))['verdict'], 'PERSISTENT')
        self.assertEqual(fk.reading(self.bins(2.0, 1.5, 1.0, se32=0.6))['verdict'], 'INTERMEDIATE')    # 1.0 < 2 x 0.6: not distinguishable from noise
        self.assertEqual(fk.reading(self.bins(0.01, 0.0, 0.0))['verdict'], 'NO BITE')
        self.assertTrue(fk.reading(self.bins(2.0, 0.1, 0.02), control=0.01)['verdict'].startswith('INVALID'))
        self.assertTrue(fk.reading({'0-0': dict(mean=2.0, se=0.1, n=5)})['verdict'].startswith('INCOMPLETE'))

    def test_decay_per_step_is_the_log_ratio_of_the_4_7_and_32_63_bins(self):
        bins = {'4-7': dict(mean=1.0, se=0.1, n=10), '32-63': dict(mean=0.25, se=0.1, n=10)}
        self.assertAlmostEqual(fk.decay_per_step(bins), np.log(0.25) / 42.0)
        self.assertIsNone(fk.decay_per_step({'4-7': dict(mean=1.0, se=0.1, n=10), '32-63': dict(mean=-0.1, se=0.1, n=10)}))
        self.assertIsNone(fk.decay_per_step({}))


class EndToEnd(unittest.TestCase):
    def test_run_on_a_tiny_snapshot_and_a_fake_validation_file(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'validation.bin').write_bytes(np.random.default_rng(1).integers(0, 256, size=20000, dtype=np.uint8).tobytes())
            snap = Path(d) / 'step001000.pt'
            torch.save({k: v.detach().cpu() for k, v in small_model().state_dict().items()}, snap)
            r = fk.run(snap, directory=d, batches=2, batch=4, length=64, ats=(5, 9), kmax=40, width=64, heads=2, log=lambda m: None)
            self.assertEqual(set(r['modes']), set(fk.MODES))
            for mode in fk.MODES:
                m = r['modes'][mode]
                self.assertEqual(set(m['per_at']), {'5', '9'})
                self.assertLess(m['control'], 1e-3)
                self.assertIn('32-40', m['pooled'])
                self.assertIn('verdict', m['reading'])
                self.assertEqual(len(m['per_at']['5']['mean_by_step']), 41)
            with self.assertRaises(ValueError):
                fk.run(snap, directory=d, batches=1, batch=2, length=16, ats=(15,), kmax=5, width=64, heads=2, log=lambda m: None)


if __name__ == '__main__':
    unittest.main()
