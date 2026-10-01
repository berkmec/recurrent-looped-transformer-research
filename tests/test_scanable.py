"""U5: is the gate's dependence on the state load-bearing, or could the gate read the input alone?

WHY THIS QUESTION. Measured: 2068 bytes/s for a 30M model, because the decoder loop is sequential over 512
timesteps and nothing about it parallelises. Mamba and S4 face the same shape of limit and defeat it with a
parallel scan, which needs the recurrence in the form

    s_t = a_t * s_{t-1} + b_t      with a_t and b_t INDEPENDENT of s

Reading our code, neither coefficient qualifies. The gate is `sigmoid(head([e_t, s_{t-1}]))` and the candidate
carries the Core's additive feedback from `s_{t-1}`. So there are TWO state dependences, and a scan needs both
gone.

THIS CYCLE TESTS THE CHEAPER ONE FIRST, and the framing matters: an input-only gate is NECESSARY for a scan and
nowhere near SUFFICIENT. If exact preservation - the campaign's one replicated positive, across 3 seeds x 2 task
families x 2 learning rates - does not survive the gate losing its view of the state, the second condition is not
worth testing and the sequential cost stands. If it does survive, removing the candidate's feedback becomes the
next question, and that one is a much larger change to the architecture.

`horizon/gated.py` is hashed inside sealed manifests, so the variant lives beside it and must be proven
bit-identical in the mode that reproduces it.
"""
import unittest

import torch


def core(width=16, heads=2, window=4, classes=5, feedback=True, alpha=.1, seed=0):
    from core.model import Core
    torch.manual_seed(seed)
    return Core(width=width, heads=heads, window=window, classes=classes, feedback=feedback, alpha=alpha)


class EquivalenceTests(unittest.TestCase):
    """The variant must reproduce the sealed model exactly in the mode that stands for it."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pair(self, mode='hard_st', seed=0):
        from horizon import gated as gd
        from horizon import scanable as sc
        sealed = gd.GatedCore(core(seed=seed), mode=mode)
        variant = sc.ScanGatedCore(core(seed=seed), mode=mode, gate_input='state')
        variant.head.load_state_dict(sealed.head.state_dict())
        variant.core.load_state_dict(sealed.core.state_dict())
        return sealed, variant

    def batch(self, n=3, t=9):
        torch.manual_seed(1)
        return torch.randint(0, 3, (n, t))

    def test_state_mode_is_bit_identical_to_the_sealed_gate(self):
        x = self.batch()
        for mode in ('soft', 'hard_st'):
            sealed, variant = self.pair(mode)
            for a, b in zip(sealed(x), variant(x)):
                self.assertTrue(torch.equal(a, b), f'{mode}: the variant must reproduce the sealed model exactly')

    def test_the_equivalence_test_exercises_holds(self):
        """A bias of 2.0 opens every gate, so an equivalence check at the default init proves nothing about HOLD."""
        from horizon import scanable as sc
        x = self.batch()
        variant = sc.ScanGatedCore(core(), mode='hard_st', gate_input='state')
        with torch.no_grad():
            variant.head.bias.fill_(-2.)
        _, gates, _ = variant(x)
        self.assertGreater(float((gates <= .5).float().mean()), .5, 'the check must observe HOLD, not only WRITE')

    def test_the_variant_reproduces_the_sealed_model_under_holds_too(self):
        from horizon import gated as gd
        from horizon import scanable as sc
        x = self.batch()
        for bias in (2., -2., 0.):
            sealed = gd.GatedCore(core(), mode='hard_st')
            variant = sc.ScanGatedCore(core(), mode='hard_st', gate_input='state')
            with torch.no_grad():
                sealed.head.bias.fill_(bias)
            variant.head.load_state_dict(sealed.head.state_dict())
            variant.core.load_state_dict(sealed.core.state_dict())
            for a, b in zip(sealed(x), variant(x)):
                self.assertTrue(torch.equal(a, b), f'bias {bias}: divergence under a gate that actually holds')


class InputOnlyGateTests(unittest.TestCase):
    """The arm under test: the gate sees e_t and nothing else."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_the_head_is_narrower_because_it_reads_one_thing(self):
        from horizon import scanable as sc
        state = sc.ScanGatedCore(core(), gate_input='state')
        inputs = sc.ScanGatedCore(core(), gate_input='input')
        self.assertEqual(state.head.in_features, 2 * 16)
        self.assertEqual(inputs.head.in_features, 16)

    def test_the_gate_no_longer_depends_on_the_state(self):
        """Measured, not asserted: perturb the committed state and see whether the gate logits move."""
        from horizon import scanable as sc
        torch.manual_seed(4)
        x = torch.randint(0, 3, (3, 9))
        moved = {}
        for kind in ('state', 'input'):
            model = sc.ScanGatedCore(core(), gate_input=kind)
            with torch.no_grad():
                model.head.bias.fill_(0.)
            base = model(x)[2]
            with torch.no_grad():
                model.core.initial.add_(1.5)            # a different starting state, same inputs
            after = model(x)[2]
            moved[kind] = float((base - after).abs().max())
        self.assertGreater(moved['state'], 1e-4, 'the state-reading gate must react to the state')
        self.assertLess(moved['input'], 1e-6, f'the input-only gate must not: moved {moved["input"]}')

    def test_both_arms_keep_the_same_write_prior(self):
        """The arms must differ in what the gate READS, not in where it starts."""
        from horizon import scanable as sc
        for kind in ('state', 'input'):
            model = sc.ScanGatedCore(core(), gate_input=kind)
            self.assertAlmostEqual(float(model.head.bias), 2., places=6)

    def test_an_input_only_gate_is_not_by_itself_scannable(self):
        """Guards the claim this cycle is allowed to make: the candidate still carries feedback from the state."""
        from horizon import scanable as sc
        model = sc.ScanGatedCore(core(feedback=True), gate_input='input')
        self.assertTrue(model.core.feedback,
                        'the Core still feeds the previous state into the candidate, so a scan needs a second change')
        self.assertIn('NECESSARY', sc.__doc__)
        self.assertIn('not sufficient', sc.__doc__.lower())


if __name__ == '__main__':
    unittest.main()
