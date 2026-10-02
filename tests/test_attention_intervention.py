"""E-A inference-time attention interventions on the sealed Core: the base row must be the plain Core, windows and temperatures must do what they say."""
import copy
import math
import unittest

import torch

from core.model import Core
from horizon import attention_intervention as ai
from horizon.depth_models import make_arm, parameter_count


def tokens_for(length, batch=3, vocab=24, seed=0):
    return torch.randint(0, vocab, (batch, length), generator=torch.Generator().manual_seed(seed))


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

    def test_a_window_changes_the_logits_beyond_the_window_and_only_there(self):
        # guards against the intervention being a silent no-op: with random weights every attention matters
        model = make_arm('core', 'perm4', 32, 1).eval()
        x = tokens_for(40)
        with torch.no_grad():
            plain = model(x)
            for name in ('w16', 'w16_enc', 'w16_mem', 'temp', 'temp_enc', 'temp_mem'):
                changed = ai.intervene(copy.deepcopy(model), name)(x)
                self.assertTrue(torch.allclose(plain[:, :16], changed[:, :16], atol=1e-5), name)          # inside the trained context: untouched
                self.assertFalse(torch.allclose(plain[:, 20:], changed[:, 20:], atol=1e-4), name)         # beyond it: the intervention does something

    def test_the_streaming_step_path_is_refused_rather_than_silently_misclassified(self):
        model = ai.intervene(make_arm('core', 'perm4', 32, 1), 'w16')
        with self.assertRaises(NotImplementedError):
            model.step(torch.zeros(2, dtype=torch.long))

    def test_unknown_names_and_other_arms_are_refused(self):
        with self.assertRaises(ValueError):
            ai.intervene(tiny(), 'w7')
        with self.assertRaises(ValueError):
            ai.intervene(make_arm('gru', 'perm4', 32, 1), 'w16')


def fake_result(late_by_name, early=0.0):
    """A result dict whose [128, 256) window has the given late rate (exposure 1e6) and whose early windows have the given early rate."""
    rows = {}
    for name, rate in late_by_name.items():
        e = early[name] if isinstance(early, dict) else early
        rows[name] = dict(windows={'0-16': [int(e * 1e6), int(1e6)], '16-64': [0, 0], '128-256': [int(rate * 1e6), int(1e6)]})
    return dict(results=rows)


class ReadingRule(unittest.TestCase):
    NAMES = ('base', 'w16', 'w32', 'w64', 'w16_enc', 'w16_mem', 'temp', 'temp_enc', 'temp_mem')

    def result(self, **over):
        rates = {n: 2.0e-3 for n in self.NAMES}
        rates.update(over)
        return fake_result(rates)

    def test_no_intervention_moves_the_rate_means_state(self):
        v = ai.verdict(self.result(w16=1.9e-3, temp=2.1e-3))
        self.assertEqual(v['verdict'], 'STATE (H2)')

    def test_a_large_drop_with_clean_early_windows_means_attention_and_names_the_path(self):
        v = ai.verdict(self.result(temp=2.0e-4, temp_enc=1.0e-3, temp_mem=2.5e-4))
        self.assertEqual(v['verdict'], 'ATTENTION (H1)')
        self.assertEqual(v['best_of_w16_w32_temp'], 'temp')
        self.assertEqual(v['localised_to'], ['temp_mem'])

    def test_a_drop_that_breaks_the_early_windows_is_not_attention(self):
        rates = {n: 2.0e-3 for n in self.NAMES}
        rates['w16'] = 1.0e-4
        result = fake_result(rates, early={**{n: 0.0 for n in self.NAMES}, 'w16': 5e-4})
        self.assertNotEqual(ai.verdict(result)['verdict'], 'ATTENTION (H1)')

    def test_in_between_is_mixed(self):
        self.assertEqual(ai.verdict(self.result(w16=1.0e-3))['verdict'], 'MIXED')

    def test_a_result_whose_baseline_does_not_reproduce_is_not_read(self):
        # blinded review finding F8: the first verdict() ignored the recorded reproduction check and labelled a non-reproducing run
        bad = self.result()
        bad['baseline_max_abs_difference'] = 0.01
        self.assertTrue(ai.verdict(bad)['verdict'].startswith('INVALID'))
        cells = self.result()
        cells['baseline_max_abs_difference'] = 0.0
        cells['baseline_cell_differences'] = 2
        self.assertTrue(ai.verdict(cells)['verdict'].startswith('INVALID'))
        fine = self.result()
        fine['baseline_max_abs_difference'] = 0.0
        fine['baseline_cell_differences'] = 0
        self.assertEqual(ai.verdict(fine)['verdict'], 'STATE (H2)')


class BaselineReproduction(unittest.TestCase):
    def files(self, directory, flipped=False):
        import json
        from pathlib import Path
        row = lambda first_errors: [dict(start=0, stop=16, at_risk=100, first_errors=first_errors, rate=0.0)]
        stored = dict(task='perm5', seed=1, curve={'16': dict(final=1.0, hazard=row(0)), '32': dict(final=0.98, hazard=row(2))})
        path = Path(directory) / 'm.json'
        path.write_text(json.dumps(stored), encoding='utf8')
        draws = {'16': dict(final=1.0, hazard=row(0)), '32': dict(final=0.98, hazard=row(3 if flipped else 2))}      # the same final accuracy, one more first error in a window
        result = dict(task='perm5', seeds=[1, 2], points=[16, 32], results={'base': dict(per_seed={'1': draws, '2': draws})})
        return result, path

    def test_final_accuracies_can_agree_while_a_hazard_cell_differs_and_the_cell_check_sees_it(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            result, path = self.files(d, flipped=False)
            self.assertEqual(ai.baseline_reproduces(result, path, [16, 32]), 0.0)
            self.assertEqual(ai.baseline_cell_differences(result, path), 0)
            result, path = self.files(d, flipped=True)
            self.assertEqual(ai.baseline_reproduces(result, path, [16, 32]), 0.0)          # the final-accuracy gate alone would pass
            self.assertEqual(ai.baseline_cell_differences(result, path), 1)                # the cell gate does not
            result['results']['base']['per_seed'] = {1: result['results']['base']['per_seed']['1']}      # int keys, as in memory before the JSON round trip
            self.assertEqual(ai.baseline_cell_differences(result, path), 1)

    def test_a_task_mismatch_and_a_missing_model_seed_are_not_comparable(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            result, path = self.files(d)
            result['task'] = 'perm4'
            self.assertIsNone(ai.baseline_reproduces(result, path, [16, 32]))               # the first version's task check compared the curve dict with itself
            result['task'] = 'perm5'
            result['seeds'] = [2, 3]
            self.assertIsNone(ai.baseline_reproduces(result, path, [16, 32]))
            self.assertIsNone(ai.baseline_cell_differences(result, path))


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
