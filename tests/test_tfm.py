import json
import tempfile
import unittest
from pathlib import Path

import torch

from core.model import Core
from mline import lm as ml
from mline import tfm as tf
from tests.test_longrun import tiny


class TransformerBaselineTests(unittest.TestCase):
    def test_the_parameter_budget_matches_l1_within_half_a_percent(self):
        t = sum(p.numel() for p in tf.build('tfm6', width=528, heads=8).parameters())
        l1 = sum(p.numel() for p in ml.build('plain', width=928, heads=8, seed=401).parameters())
        self.assertEqual((t, l1), (20365744, 20297472))
        self.assertLess(abs(t - l1) / l1, 0.005)

    def test_it_is_causal(self):
        m = tf.build('tfm6', width=32, heads=2).eval()
        x = torch.randint(0, 256, (2, 20))
        y = x.clone(); y[:, 11:] = torch.randint(0, 256, (2, 9))
        with torch.no_grad():
            a, b = m(x), m(y)
        self.assertEqual(tuple(a.shape), (2, 20, 256))
        self.assertTrue(torch.equal(a[:, :11], b[:, :11]))
        self.assertGreater(float((a[:, 11:] - b[:, 11:]).abs().max()), 1e-4)

    def test_the_rotary_embedding_is_the_cores(self):
        x = torch.randn(2, 4, 9, 8)
        core = Core(width=32, heads=4, window=4, classes=256)
        for start in (0, 5):
            self.assertTrue(torch.allclose(tf.rope(x, start), core.rope(x, start), atol=1e-6))

    def test_gpt2_initialisation_and_the_seed(self):
        m = tf.build('tfm6', width=64, heads=4, seed=401)
        self.assertAlmostEqual(float(m.blocks[0].q.weight.std()), 0.02, delta=0.002)
        self.assertAlmostEqual(float(m.blocks[0].o.weight.std()), 0.02 / 12 ** 0.5, delta=0.001)
        self.assertEqual(float(m.blocks[0].ff[0].bias.abs().max()), 0.0)
        self.assertTrue(torch.equal(m.outn.weight, torch.ones(64)))
        again = tf.build('tfm6', width=64, heads=4, seed=401)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(m.parameters(), again.parameters())))
        with self.assertRaises(ValueError):
            tf.build('tfm6', width=30, heads=4)
        with self.assertRaises(ValueError):
            tf.build('plain')

    def test_the_long_run_trainer_trains_it(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=4, arm='tfm6', width=16, heads=2)
            probe = json.loads(Path(d, 'probe.json').read_text(encoding='utf8'))
            self.assertEqual((probe['arm'], probe['parameters']), ('tfm6', sum(p.numel() for p in tf.build('tfm6', width=16, heads=2).parameters())))
            rows = [json.loads(l) for l in Path(d, 'probe.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([r['step'] for r in rows], [1, 2, 3, 4])
            self.assertTrue(all(r['applied'] for r in rows))
            self.assertTrue(Path(d, 'complete.json').exists())

    def test_std1_telemetry_is_refused_for_the_transformer(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                tiny(d, steps=2, arm='tfm6', width=16, heads=2, telemetry='std1')


if __name__ == '__main__':
    unittest.main()
