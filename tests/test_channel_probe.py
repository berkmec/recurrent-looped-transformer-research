"""The channel probe switches parts of a saved model off at inference, to see which one the needle depends on.

An intervention is only evidence if it does exactly what its name says and nothing else, so each switch is pinned
against an independently built equivalent model, and each is shown to change the output at all (a switch that does
nothing would make every later 'no effect' reading meaningless).
"""
import unittest

import torch
from torch import nn

from core.model import Core
from horizon import gated as gd
from mline import channel_probe as cp
from mline import lm as ml


def tiny_core(alpha=.1, seed=0):
    torch.manual_seed(seed)
    core = Core(width=32, heads=2, window=4, classes=256, feedback=True, alpha=alpha)
    core.embedding = nn.Embedding(256, 32)
    return core.eval()


class FeedbackOffTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.randint(0, 256, (2, 14))

    def test_it_equals_a_core_built_with_zero_feedback_scale(self):
        a, b = tiny_core(.1), tiny_core(.1)
        b.load_state_dict(a.state_dict())
        b.alpha = 0.
        with torch.no_grad():
            with cp.feedback_off(a):
                off = a(self.x)
            reference = b(self.x)
        self.assertTrue(torch.equal(off, reference))

    def test_it_changes_the_output_at_all(self):
        core = tiny_core(.1)
        with torch.no_grad():
            base = core(self.x)
            with cp.feedback_off(core):
                off = core(self.x)
        self.assertFalse(torch.equal(base, off))

    def test_the_scale_is_restored_even_after_an_error(self):
        core = tiny_core(.1)
        with self.assertRaises(RuntimeError):
            with cp.feedback_off(core):
                raise RuntimeError('boom')
        self.assertEqual(core.alpha, .1)

    def test_it_works_through_a_gated_wrapper(self):
        model = ml.build('hard_st', width=32, heads=2, window=4, alpha=.1, seed=1, gate_k=2).eval()
        with torch.no_grad():
            base = ml.forward_logits(model, self.x)
            with cp.feedback_off(model):
                off = ml.forward_logits(model, self.x)
        self.assertFalse(torch.equal(base, off))
        self.assertEqual(model.core.alpha, .1)


class GateOpenTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.randint(0, 256, (2, 14))

    def test_an_open_gate_is_the_plain_core_with_the_same_weights(self):
        core = tiny_core(.1)
        gated = gd.GatedCore(core, mode='hard_st', k=2).eval()
        with torch.no_grad():
            plain = core(self.x)
            with cp.gate_open(gated):
                opened = ml.forward_logits(gated, self.x)
        self.assertTrue(torch.allclose(plain, opened, atol=1e-6), float((plain - opened).abs().max()))

    def test_it_changes_the_output_of_a_gate_that_ever_holds(self):
        core = tiny_core(.1)
        gated = gd.GatedCore(core, mode='hard_st', k=2).eval()
        gated.close_all()
        with torch.no_grad():
            closed = ml.forward_logits(gated, self.x)
            with cp.gate_open(gated):
                opened = ml.forward_logits(gated, self.x)
        self.assertFalse(torch.equal(closed, opened))

    def test_the_gate_head_is_restored_on_exit(self):
        gated = gd.GatedCore(tiny_core(.1), mode='hard_st', k=2).eval()
        before = {k: v.clone() for k, v in gated.head.state_dict().items()}
        with cp.gate_open(gated):
            self.assertTrue(float(gated.head.bias.detach()) > 10)
        after = gated.head.state_dict()
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_a_model_without_a_gate_is_refused(self):
        with self.assertRaises(ValueError):
            with cp.gate_open(tiny_core(.1)):
                pass


class ModeTests(unittest.TestCase):
    def test_modes_compose_and_unknown_modes_are_refused(self):
        core = tiny_core(.1)
        with cp.apply(core, 'full'):
            pass
        with cp.apply(core, 'nofeedback'):
            self.assertEqual(core.alpha, 0.)
        self.assertEqual(core.alpha, .1)
        with self.assertRaises(ValueError):
            with cp.apply(core, 'lobotomy'):
                pass


if __name__ == '__main__':
    unittest.main()
