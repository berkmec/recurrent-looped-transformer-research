"""E-A inference-time attention interventions on the sealed Core: the base row must be the plain Core, windows and temperatures must do what they say."""
import copy
import math
import unittest

import torch

from core.model import Core
from horizon import attention_intervention as ai
from horizon.depth_models import make_arm, parameter_count


def tiny(width=32, heads=2, window=4):
    torch.manual_seed(0)
    return Core(width=width, heads=heads, window=window, classes=5)


def configured(model, **iv):
    model.__class__ = ai.IntervenedCore
    model.iv = dict(window=0, temp=False, enc=False, mem=False)
    model.iv.update(iv)
    return model


def manual(q, k, v, mask, scale=None):
    """softmax(scale * q k^T / sqrt(d)) v with a boolean mask (True = may attend); q, k, v are (batch, heads, length, d)."""
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if scale is not None:
        scores = scores * scale
    return (torch.softmax(scores.masked_fill(~mask, float('-inf')), -1) @ v).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


class BaseIsThePlainCore(unittest.TestCase):
    def test_base_logits_are_identical(self):
        model = make_arm('core', 'perm4', 32, 1)
        x = torch.randint(0, 24, (2, 20))
        model.eval()
        with torch.no_grad():
            plain = model(x)
            same = ai.intervene(copy.deepcopy(model), 'base')
            self.assertTrue(torch.equal(plain, same(x)))
        self.assertEqual(parameter_count(same), parameter_count(model))

    def test_a_window_longer_than_the_sequence_changes_nothing(self):
        model = make_arm('core', 'perm4', 32, 1)
        x = torch.randint(0, 24, (2, 20))
        model.eval()
        with torch.no_grad():
            plain = model(x)
            windowed = ai.intervene(copy.deepcopy(model), 'w64')(x)
            tempered = ai.intervene(copy.deepcopy(model), 'temp')(x[:, :16])       # up to the trained context the temperature is 1
        self.assertTrue(torch.allclose(plain, windowed, atol=1e-5))
        self.assertTrue(torch.allclose(plain[:, :16], tempered, atol=1e-5))

    def test_the_streaming_step_path_is_refused_rather_than_silently_misclassified(self):
        model = ai.intervene(make_arm('core', 'perm4', 32, 1), 'w16')
        with self.assertRaises(NotImplementedError):
            model.step(torch.zeros(2, dtype=torch.long))

    def test_unknown_names_and_other_arms_are_refused(self):
        with self.assertRaises(ValueError):
            ai.intervene(tiny(), 'w7')
        with self.assertRaises(ValueError):
            ai.intervene(make_arm('gru', 'perm4', 32, 1), 'w16')


class WindowsAndTemperatures(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.q = torch.randn(1, 2, 6, 16)
        self.k = torch.randn(1, 2, 6, 16)
        self.v = torch.randn(1, 2, 6, 16)

    def test_encoder_window_is_a_band_of_the_last_w_positions(self):
        model = configured(tiny(), window=3, enc=True)
        idx = torch.arange(6)
        delta = idx[:, None] - idx[None, :]
        expected = manual(self.q, self.k, self.v, (delta >= 0) & (delta < 3))
        self.assertTrue(torch.allclose(model.attend(self.q, self.k, self.v, True), expected, atol=1e-5))

    def test_encoder_temperature_scales_each_row_by_its_own_key_count(self):
        torch.manual_seed(2)
        q, k, v = torch.randn(1, 2, 40, 16), torch.randn(1, 2, 40, 16), torch.randn(1, 2, 40, 16)
        model = configured(tiny(), temp=True, enc=True)
        got = model.attend(q, k, v, True)
        idx = torch.arange(40)
        mask = idx[:, None] >= idx[None, :]
        for row in (3, 15, 39):     # n_keys = row + 1: factor 1 up to 16 keys, log(40) / log(16) at the last row
            factor = max(1.0, math.log(row + 1) / math.log(ai.N_REF))
            expected = manual(q[:, :, row:row + 1], k[:, :, :row + 1], v[:, :, :row + 1], mask[row:row + 1, :row + 1], scale=factor)
            self.assertTrue(torch.allclose(got[:, row:row + 1], expected, atol=1e-5), row)

    def test_memory_window_keeps_the_last_w_keys_and_leaves_short_reads_alone(self):
        torch.manual_seed(3)
        q, k, v = torch.randn(1, 2, 1, 16), torch.randn(1, 2, 60, 16), torch.randn(1, 2, 60, 16)
        model = configured(tiny(), window=16, mem=True)
        expected = manual(q, k[:, :, -16:], v[:, :, -16:], torch.ones(1, 16, dtype=torch.bool))
        self.assertTrue(torch.allclose(model.attend(q, k, v), expected, atol=1e-5))
        short = model.attend(q, k[:, :, :10], v[:, :, :10])
        self.assertTrue(torch.allclose(short, manual(q, k[:, :, :10], v[:, :, :10], torch.ones(1, 10, dtype=torch.bool)), atol=1e-5))

    def test_memory_temperature_doubles_the_logits_at_the_square_of_the_trained_context(self):
        torch.manual_seed(4)
        q, k, v = torch.randn(1, 2, 1, 16), torch.randn(1, 2, 256, 16), torch.randn(1, 2, 256, 16)
        model = configured(tiny(), temp=True, mem=True)
        expected = manual(q, k, v, torch.ones(1, 256, dtype=torch.bool), scale=2.0)            # log(256) / log(16) = 2
        self.assertTrue(torch.allclose(model.attend(q, k, v), expected, atol=1e-5))

    def test_the_decoder_window_attention_is_never_touched(self):
        torch.manual_seed(5)
        q, k, v = torch.randn(1, 2, 1, 16), torch.randn(1, 2, 4, 16), torch.randn(1, 2, 4, 16)       # at most `window` keys
        base = tiny()
        model = configured(tiny(), window=2, temp=True, enc=True, mem=True)
        self.assertTrue(torch.equal(model.attend(q, k, v), base.attend(q, k, v)))


if __name__ == '__main__':
    unittest.main()
