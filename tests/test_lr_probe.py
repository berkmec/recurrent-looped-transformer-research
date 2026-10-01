"""U2c-lr development probe contract (written before the implementation).

Purpose: decide whether the hard_st instability is an artifact of the learning rate inherited from U1 (0.003).
It is a DEVELOPMENT probe: training stability and validation only — it must never touch the test or neutral namespaces,
and it must not modify anything inside the frozen U2c source set.
"""
import json
import tempfile
import unittest
from pathlib import Path

import torch


class LrProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import lr_probe as lp
        except ImportError as e:
            self.fail('horizon.lr_probe is not implemented: ' + str(e))
        return lp

    def tiny(self, lp):
        p = lp.protocol()
        p.update(steps=4, batch=4, train_length=12, validation_every=2, validation_n=8, width=8, heads=1,
                 arms=['hard_st'], lrs=[0.003, 0.0003], seeds=[117], tasks=['five'])
        return p

    def test_probe_is_development_only(self):
        lp = self.api()
        p = lp.protocol()
        self.assertEqual(p['role'], 'development probe: training stability and validation only; never test or neutral')
        # no endpoint configuration may exist: the probe cannot evaluate on test/neutral even by accident
        for key in p:
            self.assertNotIn('test', key, f'protocol key {key} configures a test endpoint')
            self.assertNotIn('neutral', key, f'protocol key {key} configures a neutral endpoint')
        for forbidden in ('n_neutral', 'neutral_slots', 'placements', 'clean_lengths', 'test_lengths', 'test_n'):
            self.assertNotIn(forbidden, p)
        self.assertEqual(p['source_seeds'].keys() & {'train', 'validation'}, {'train', 'validation'})
        import inspect
        from horizon import lr_probe as lp
        src = inspect.getsource(lp)
        self.assertNotIn("'test'", src); self.assertNotIn("'neutral'", src)   # no heldout(... 'test'/'neutral' ...) call

    def test_freeze_run_and_report(self):
        lp = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'U2c-lr'
            m = lp.freeze(out, self.tiny(lp))
            self.assertEqual(lp.verify_frozen(out)['protocol_hash'], m['protocol_hash'])
            cs = lp.conditions(m['protocol'])
            self.assertEqual([lp.ident(c) for c in cs], ['lr-hard_st-five-s117-lr0.003', 'lr-hard_st-five-s117-lr0.0003'])
            r = lp.run_one(out, cs[0])
            self.assertEqual(r['steps'], 4)
            for key in ('guard', 'max_loss_jump', 'final_train_loss', 'best_validation_accuracy', 'hold_rate_last', 'unstable'):
                self.assertIn(key, r)
            self.assertIn('skipped', r['guard'])
            self.assertTrue((out / lp.ident(cs[0]) / 'updates.jsonl').exists())
            self.assertEqual(lp.run_one(out, cs[0])['final_train_loss'], r['final_train_loss'])  # cached
            summary = lp.analyze(out)
            self.assertEqual(set(summary['conditions']), {lp.ident(cs[0])})   # only what has been run
            self.assertIn('recommendation', summary)
            with self.assertRaises(ValueError):
                lp.freeze(out, dict(self.tiny(lp), steps=5))

    def test_frozen_u2c_sources_are_untouched(self):
        """The probe may import the gated modules but must not be part of their frozen set, and must not edit them."""
        lp = self.api()
        from horizon import gated_sweep as gs
        self.assertNotIn('horizon/lr_probe.py', gs.SOURCES)
        for name, stored in json.loads(Path('results/U2c-v2/frozen.json').read_text(encoding='utf8'))['sources'].items():
            self.assertEqual(gs.sw.sha(Path(name)), stored, f'{name} changed while U2c-v2 is running')


if __name__ == '__main__':
    unittest.main()
