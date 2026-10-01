"""U4-inject contract (written before the implementation).

The sealed Core feeds the previous state back additively: z = e + alpha * sigmoid(gate([e,r])) * merge(r). The Fully
Looped Transformer paper (arXiv 2605.18797) identifies exactly this shape as the source of residual explosion and
replaces it with attention injection: the previous state becomes a Query, and its contribution arrives softmax-
normalised rather than as an unbounded residual addition.

`core/model.py` is hashed inside every sealed manifest, so the variant must live beside it, not inside it.
"""
import unittest

import torch


class InjectCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import inject as ij
        except ImportError as e:
            self.fail('horizon.inject is not implemented: ' + str(e))
        return ij

    def core(self, feedback=True):
        from core.model import Core
        torch.manual_seed(0)
        return Core(width=16, heads=2, window=4, classes=5, feedback=feedback, alpha=.1)

    def batch(self, n=3, t=9):
        torch.manual_seed(1)
        return torch.randint(0, 3, (n, t))

    def test_modes_are_the_three_arms(self):
        ij = self.api()
        self.assertEqual(tuple(ij.MODES), ('additive', 'inject', 'inject_mix'))
        with self.assertRaises(ValueError):
            ij.InjectCore(self.core(), mode='nonsense')

    def test_additive_arm_is_bitwise_identical_to_the_sealed_core(self):
        """The control arm must BE the sealed model, or it cannot serve as a control."""
        ij = self.api()
        core = self.core(feedback=True)
        model = ij.InjectCore(core, mode='additive')
        x = self.batch()
        with torch.no_grad():
            sealed = core(x)
            mine, states = model(x)
        self.assertTrue(torch.equal(sealed, mine))
        self.assertEqual(states.shape, (x.shape[0], x.shape[1], core.width))

    def test_injection_arms_run_and_carry_gradients(self):
        ij = self.api()
        x = self.batch()
        for mode in ('inject', 'inject_mix'):
            model = ij.InjectCore(self.core(feedback=False), mode=mode)
            logits, states = model(x)
            self.assertEqual(logits.shape, (x.shape[0], x.shape[1], 5))
            self.assertTrue(torch.isfinite(logits).all(), mode)
            self.assertTrue(states.requires_grad, mode)
            logits.sum().backward()
            grads = [p.grad for p in model.injection_parameters() if p.grad is not None]
            self.assertTrue(grads and any(float(g.abs().sum()) > 0 for g in grads), mode)

    def test_inject_mix_is_a_true_convex_combination(self):
        """The anti-explosion property, asserted directly: the mixed state lies on the segment between its two inputs.

        This is what makes the arm principled rather than decorative - the state cannot grow by accumulating residuals,
        because each step returns a weighted average, not a sum.
        """
        ij = self.api()
        model = ij.InjectCore(self.core(feedback=False), mode='inject_mix')
        x = self.batch()
        with torch.no_grad():
            model(x)
        rec = model.last_mix            # (weights, e, injected) captured at the last step
        w, e, a = rec['weights'], rec['e'], rec['injected']
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(w.shape[0]), atol=1e-6))
        self.assertTrue((w >= 0).all())
        mixed = w[:, :1] * e + w[:, 1:] * a
        self.assertTrue(torch.allclose(mixed, rec['mixed'], atol=1e-6))
        bound = torch.maximum(e.norm(dim=-1), a.norm(dim=-1))
        self.assertTrue((mixed.norm(dim=-1) <= bound + 1e-5).all())   # never outside the convex hull

    def test_injection_reuses_the_sealed_core_parameters(self):
        """The variant must not silently clone the Core: same tensors, plus a small, countable injection head."""
        ij = self.api()
        core = self.core(feedback=False)
        model = ij.InjectCore(core, mode='inject')
        ids = {id(p) for p in core.parameters()}
        shared = [p for p in model.parameters() if id(p) in ids]
        self.assertEqual(len(shared), len(list(core.parameters())))
        extra = [p for p in model.parameters() if id(p) not in ids]
        self.assertEqual(sum(p.numel() for p in extra), sum(p.numel() for p in model.injection_parameters()))

    def test_sealed_core_source_is_untouched(self):
        import json
        from pathlib import Path
        from horizon import sweep as sw
        man = json.loads(Path('results/U2c-v2/frozen.json').read_text(encoding='utf8'))
        self.assertEqual(sw.sha(Path('core/model.py')), man['sources']['core/model.py'])


if __name__ == '__main__':
    unittest.main()
