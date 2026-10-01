"""U2c runner contract (written before the implementation): freeze/verify, train one gated condition with the U1 loop
under a tiny protocol, select by validation, evaluate on the U2b endpoints with gate diagnostics, refuse protocol drift."""
import json
import tempfile
import unittest
from pathlib import Path

import torch


class GatedSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import gated_sweep as gs
        except ImportError as e:
            self.fail('horizon.gated_sweep is not implemented: ' + str(e))
        return gs

    def tiny(self, gs):
        p = gs.protocol()
        p.update(steps=6, batch=4, train_length=12, validation_every=3, validation_n=8, n_neutral=8, n_clean=8, neutral_slots=[4], clean_lengths=[12, 20],
                 width=8, heads=1, arms=['hard_st', 'supervised'], primary_seeds=[117], tasks=['five'], u1=None)
        return p

    def test_freeze_train_select_evaluate(self):
        gs = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'U2c'
            m = gs.freeze(out, self.tiny(gs))
            self.assertEqual(m['protocol_hash'], gs.verify_frozen(out)['protocol_hash'])
            cs = gs.conditions(m['protocol'])
            self.assertEqual([gs.ident(c) for c in cs], ['u2c-hard_st-five-w8-s117', 'u2c-supervised-five-w8-s117'])
            r = gs.run_one(out, cs[1])
            d = out / gs.ident(cs[1])
            self.assertTrue((d / 'selected.pt').exists() and (d / 'complete.json').exists() and (d / 'updates.jsonl').exists())
            self.assertEqual(r['steps'], 6); self.assertIn('selected_step', r); self.assertIn('guard', r)
            self.assertEqual(r['guard']['steps'], 6)
            self.assertEqual(len((d / 'updates.jsonl').read_text(encoding='utf8').strip().splitlines()), 6)
            self.assertEqual(gs.run_one(out, cs[1])['selected_step'], r['selected_step'])  # idempotent reuse of a complete condition
            e = gs.evaluate_one(out, cs[1])
            self.assertEqual(set(e['neutral']), {'0-base', '4-prefix', '4-middle', '4-suffix'})
            self.assertEqual(set(e['clean']), {'12', '20'})
            for key, v in e['neutral'].items():
                self.assertIn('final_accuracy', v); self.assertIn('gate', v)
                if key != '0-base':
                    self.assertIn('false_write', v['gate']); self.assertIn('false_hold', v['gate'])
            self.assertEqual(e['protocol_hash'], m['protocol_hash'])
            with self.assertRaises(ValueError):
                gs.freeze(out, dict(self.tiny(gs), steps=7))

    def test_supervised_arm_uses_gate_loss_and_hard_st_does_not(self):
        gs = self.api()
        p = self.tiny(gs)
        self.assertGreater(gs.gate_weight('supervised', p), 0.)
        self.assertEqual(gs.gate_weight('hard_st', p), 0.)
        self.assertEqual(gs.gate_weight('soft', p), 0.)


if __name__ == '__main__':
    unittest.main()
