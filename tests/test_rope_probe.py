"""The RoPE probe re-scales the sealed Core's rotary positions at inference time, without editing the sealed file.

Its readings are only worth having if its identity setting is EXACTLY the sealed behaviour: any difference at
factor 1 would be the probe's own artefact rather than a property of the model. That is pinned first.
"""
import unittest

import torch
from torch import nn

from core.model import Core
from mline import rope_probe as rp


def tiny_core():
    torch.manual_seed(0)
    core = Core(width=32, heads=2, window=4, classes=256, feedback=True, alpha=.1)
    core.embedding = nn.Embedding(256, 32)
    return core.eval()


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.core = tiny_core()
        self.x = torch.randint(0, 256, (2, 14))

    def logits(self):
        with torch.no_grad():
            return self.core(self.x)

    def test_identity_is_bitwise_the_sealed_behaviour(self):
        base = self.logits()
        with rp.rescaled(self.core, 'identity'):
            self.assertTrue(torch.equal(base, self.logits()))

    def test_factor_one_of_both_schemes_is_identity(self):
        base = self.logits()
        for mode in ('pi', 'ntk'):
            with rp.rescaled(self.core, mode, 1.0):
                self.assertTrue(torch.equal(base, self.logits()), mode)

    def test_a_real_factor_changes_the_output(self):
        base = self.logits()
        for mode, factor in (('pi', .5), ('ntk', 4.)):
            with rp.rescaled(self.core, mode, factor):
                self.assertFalse(torch.equal(base, self.logits()), mode)

    def test_the_patch_is_removed_on_exit_even_after_an_error(self):
        base = self.logits()
        with self.assertRaises(RuntimeError):
            with rp.rescaled(self.core, 'pi', .5):
                raise RuntimeError('boom')
        self.assertNotIn('rope', self.core.__dict__)
        self.assertTrue(torch.equal(base, self.logits()))

    def test_an_unknown_scheme_is_refused(self):
        with self.assertRaises(ValueError):
            with rp.rescaled(self.core, 'yarn', 2.):
                pass


class RelativityTests(unittest.TestCase):
    """RoPE must stay relative after rescaling: moving query and key together must change nothing."""

    def test_dot_products_depend_only_on_the_offset(self):
        core = tiny_core()
        torch.manual_seed(1)
        q = torch.randn(1, 2, 1, 16)
        k = torch.randn(1, 2, 1, 16)
        for mode, factor in (('identity', 1.), ('pi', .5), ('ntk', 8.)):
            with rp.rescaled(core, mode, factor):
                a = (core.rope(q, 7) * core.rope(k, 3)).sum()
                b = (core.rope(q, 40) * core.rope(k, 36)).sum()
            self.assertAlmostEqual(float(a), float(b), places=3, msg=mode)

    def test_position_interpolation_reads_a_distance_as_its_fraction(self):
        """PI at factor 1/2 rotates by the same angles as the sealed rope at half the distance."""
        core = tiny_core()
        torch.manual_seed(2)
        q = torch.randn(1, 2, 1, 16)
        k = torch.randn(1, 2, 1, 16)
        sealed = (core.rope(q, 10) * core.rope(k, 0)).sum()      # distance 10
        with rp.rescaled(core, 'pi', .5):
            halved = (core.rope(q, 20) * core.rope(k, 0)).sum()  # distance 20 read as 10
        self.assertAlmostEqual(float(sealed), float(halved), places=3)


class AdapterTests(unittest.TestCase):
    def test_a_gated_wrapper_is_patched_through_its_core(self):
        from mline import lm as ml
        model = ml.build('hard_st', width=32, heads=2, window=4, alpha=.1, seed=1, gate_k=2).eval()
        x = torch.randint(0, 256, (2, 10))
        with torch.no_grad():
            base = ml.forward_logits(model, x)
            with rp.rescaled(model, 'pi', .5):
                changed = ml.forward_logits(model, x)
            after = ml.forward_logits(model, x)
        self.assertFalse(torch.equal(base, changed))
        self.assertTrue(torch.equal(base, after))


if __name__ == '__main__':
    unittest.main()
