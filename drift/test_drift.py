"""Behavioral tests for the drift mini-experiment (written before the implementation)."""
import copy
import json
import tempfile
from pathlib import Path
import unittest

import torch
from core.model import Core, generate


class DriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from drift import model as m
            from drift import run as r
        except ImportError as e:
            self.fail('drift runner is not implemented: ' + str(e))
        return m, r

    def test_decode_loop_matches_core_forward(self):
        m, _ = self.api()
        torch.manual_seed(5)
        core = Core(width=8, heads=1, window=4, classes=5).double()
        x, _ = generate('five', 3, 19, 11)
        logits, states = m.run_sequence(core, x, arm='base')
        self.assertEqual(tuple(logits.shape), (3, 19, 5))
        self.assertEqual(tuple(states.shape), (3, 19, 8))
        torch.testing.assert_close(logits, core(x), rtol=1e-9, atol=1e-11)

    def test_reread_is_identity_at_init_and_learns(self):
        m, _ = self.api()
        torch.manual_seed(7)
        core = Core(width=8, heads=1, window=4, classes=2).double()
        wrapped = m.RereadCore(core, seed=99).double()
        x, y = generate('parity', 2, 12, 3)
        base_logits, _ = m.run_sequence(core, x, arm='base')
        logits, _ = m.run_sequence(wrapped, x, arm='reread')
        torch.testing.assert_close(logits, base_logits, rtol=1e-9, atol=1e-11)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        self.assertGreater(float(wrapped.reread_o.weight.grad.abs().max()), 0.)
        with torch.no_grad():
            wrapped.reread_o.weight.add_(0.5)
        logits2, _ = m.run_sequence(wrapped, x, arm='reread')
        self.assertGreater(float((logits2 - base_logits).abs().max()), 1e-6)

    def test_invariance_loss_only_counts_neutral_positions(self):
        m, _ = self.api()
        states = torch.zeros(2, 5, 4)
        states[:, 1] = 1.; states[:, 3] = 2.; states[:, 4] = 2.
        x = torch.tensor([[0, 2, 1, 2, 2], [1, 1, 1, 1, 1]])
        loss = m.neutral_invariance_loss(states, x)
        self.assertGreater(float(loss), 0.)
        self.assertEqual(float(m.neutral_invariance_loss(states, torch.ones_like(x))), 0.)
        same = torch.ones(2, 5, 4)
        self.assertEqual(float(m.neutral_invariance_loss(same, x)), 0.)

    def test_conditions_and_pairing(self):
        m, r = self.api()
        p = r.protocol()
        cs = r.conditions(p)
        self.assertEqual(len(cs), 24)
        self.assertEqual({c['arm'] for c in cs}, {'base', 'win16', 'idinv', 'reread'})
        for task in ('five', 'parity'):
            for seed in p['primary_seeds']:
                hashes = {c['arm']: r.core_initial_hash(c, p) for c in cs if c['task'] == task and c['seed'] == seed}
                self.assertEqual(len(set(hashes.values())), 1, hashes)
                batches = {c['arm']: r.train_batch(c, p, 7)['input_hash'] for c in cs if c['task'] == task and c['seed'] == seed}
                self.assertEqual(len(set(batches.values())), 1)
        self.assertEqual(r.make_model(next(c for c in cs if c['arm'] == 'win16'), p).core.window if hasattr(r.make_model(next(c for c in cs if c['arm'] == 'win16'), p), 'core') else r.make_model(next(c for c in cs if c['arm'] == 'win16'), p).window, 16)

    def test_tiny_end_to_end_run_writes_receipts_and_changes_parameters(self):
        _, r = self.api()
        p = r.protocol(); p.update(steps=3, validation_every=2, validation_n=8, test_n=6)
        with tempfile.TemporaryDirectory() as d:
            frozen = r.freeze(d, p)
            c = next(c for c in r.conditions(p) if c['arm'] == 'idinv' and c['task'] == 'five')
            receipt = r.run_one(d, c)
            self.assertEqual(receipt['updates'], 3)
            self.assertNotEqual(receipt['initial_state_hash'], receipt['last_state_hash'])
            self.assertTrue(all(torch.isfinite(torch.tensor(row['loss'])) for row in receipt['history']))
            self.assertEqual(len(receipt['history']), 3)
            self.assertTrue((Path(d) / r.ident(c) / 'selected.pt').exists())
            evaluation = r.evaluate_one(d, c)
            self.assertEqual(len(evaluation['lengths']), 3)
            self.assertEqual(len(evaluation['neutral']), 10)
            for row in evaluation['lengths'] + evaluation['neutral']:
                self.assertTrue((Path(d) / row['raw_file']).exists())
            with self.assertRaisesRegex(ValueError, 'protocol|drift'):
                p2 = dict(p, steps=4); r.freeze(d, p2)


if __name__ == '__main__':
    unittest.main()
