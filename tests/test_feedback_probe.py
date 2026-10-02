"""E-L1P: the channel conditions do what they say on a small model, the recorded validation is read from the L1 log, the reading rule is mechanical."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mline import feedback_probe as fp
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


class Conditions(unittest.TestCase):
    def test_normal_is_the_untouched_core_and_the_other_conditions_change_the_loss(self):
        model, data = small_model(), batches()
        plain = fp.evaluate(copy.deepcopy(model), data, conditions=('normal',), log=lambda m: None)['normal']['bits_per_byte']
        with torch.no_grad():
            ce = [fp.per_position_loss(model, torch.from_numpy(x), torch.from_numpy(y)).mean() for x, y in data]
        self.assertAlmostEqual(plain, float(sum(ce) / len(ce)) / np.log(2), places=5)
        results = fp.evaluate(copy.deepcopy(model), data, log=lambda m: None)
        for name in ('a0', 'a005', 'a02', 'zero_state', 'other_doc'):
            self.assertNotEqual(round(results[name]['bits_per_byte'], 6), round(results['normal']['bits_per_byte'], 6), name)

    def test_alpha_zero_equals_the_core_without_feedback(self):
        model = small_model()
        x = torch.from_numpy(batches(1, 3, 20)[0][0])
        fp.configure(model, 'a0')
        with torch.no_grad():
            cut = model(x)
        nofb = copy.deepcopy(model)
        nofb.__class__ = type(nofb).__mro__[1]                  # back to the sealed Core
        nofb.feedback = False
        with torch.no_grad():
            self.assertTrue(torch.allclose(cut, nofb(x), atol=1e-4))

    def test_zero_state_and_other_doc_substitute_the_state_from_the_second_step_on(self):
        model, x = small_model(), torch.from_numpy(batches(1, 3, 12)[0][0])
        with torch.no_grad():
            base = copy.deepcopy(model).eval()(x)
        for name in ('zero_state', 'other_doc'):
            changed = fp.configure(copy.deepcopy(model), name)
            with torch.no_grad():
                out = changed(x)
            self.assertTrue(torch.allclose(out[:, 0], base[:, 0], atol=1e-5), name)      # nothing is fed back into the first step
            self.assertFalse(torch.allclose(out[:, 3:], base[:, 3:], atol=1e-4), name)
        self.assertRaises(ValueError, fp.configure, model, 'nonsense')

    def test_slices_are_means_of_the_per_position_loss(self):
        data = batches(1, 4, 40)
        r = fp.evaluate(small_model(), data, conditions=('normal',), slices=((0, 10), (10, 40)), log=lambda m: None)['normal']
        mean_all = (10 * r['slices']['0-10'] + 30 * r['slices']['10-40']) / 40
        self.assertAlmostEqual(mean_all, r['bits_per_byte'], places=6)


class SnapshotAndLog(unittest.TestCase):
    def test_snapshot_round_trip_and_recorded_bits_per_byte_from_the_log(self):
        with tempfile.TemporaryDirectory() as d:
            model = small_model()
            (Path(d) / 'snapshots').mkdir()
            path = Path(d) / 'snapshots' / 'step002000.pt'
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
            loaded = fp.load_model(path, width=64, heads=2, seed=3)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values())))
            (Path(d) / 'probe.jsonl').write_text(
                json.dumps(dict(step=1000, eval=dict(bits_per_byte=2.0))) + '\n' + json.dumps(dict(step=2000, eval=dict(bits_per_byte=1.85))) + '\n'
                + json.dumps(dict(step=2001)) + '\n', encoding='utf8')
            self.assertEqual(fp.recorded_bits_per_byte(path, Path(d) / 'probe.jsonl'), 1.85)
            self.assertIsNone(fp.recorded_bits_per_byte(Path(d) / 'snapshots' / 'step009000.pt', Path(d) / 'probe.jsonl'))
            self.assertIsNone(fp.recorded_bits_per_byte(path, Path(d) / 'missing.jsonl'))


class ReadingRule(unittest.TestCase):
    def results(self, normal, **others):
        return {**{'normal': dict(bits_per_byte=normal)}, **{k: dict(bits_per_byte=normal + v) for k, v in others.items()}}

    def test_used_not_used_in_between_and_invalid(self):
        self.assertEqual(fp.reading(self.results(1.8, a0=0.05, other_doc=0.02), recorded=1.8)['verdict'], 'CHANNEL USED FOR LANGUAGE')
        self.assertEqual(fp.reading(self.results(1.8, a0=0.001, other_doc=0.0005), recorded=1.8)['verdict'], 'NOT USED')
        self.assertEqual(fp.reading(self.results(1.8, a0=0.005, other_doc=0.0005), recorded=1.8)['verdict'], 'IN BETWEEN')
        self.assertTrue(fp.reading(self.results(1.8, a0=0.05, other_doc=0.02), recorded=1.9)['verdict'].startswith('INVALID'))
        self.assertIsNone(fp.reading(self.results(1.8, a0=0.05, other_doc=0.02), recorded=None)['normal_reproduces_recorded'])


class EndToEnd(unittest.TestCase):
    def test_run_on_a_tiny_snapshot_and_a_fake_validation_file(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'validation.bin').write_bytes(np.random.default_rng(1).integers(0, 256, size=20000, dtype=np.uint8).tobytes())
            model = small_model()
            snap = Path(d) / 'run' / 'snapshots' / 'step001000.pt'
            snap.parent.mkdir(parents=True)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, snap)
            r = fp.run(snap, directory=d, batches=2, batch=4, length=32, width=64, heads=2, log=lambda m: None)
            self.assertEqual(set(r['results']), set(fp.CONDITIONS))
            self.assertIn('verdict', r['reading'])


if __name__ == '__main__':
    unittest.main()
