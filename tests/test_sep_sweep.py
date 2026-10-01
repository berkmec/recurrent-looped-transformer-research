"""U4-sep runner contract (written before the implementation)."""
import json
import tempfile
import unittest
from pathlib import Path

import torch


class SepSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import sep_sweep as ss
        except ImportError as e:
            self.fail('horizon.sep_sweep is not implemented: ' + str(e))
        return ss

    def tiny(self, ss):
        p = ss.protocol()
        p.update(steps=4, batch=4, train_length=12, validation_every=2, validation_n=8, width=8, heads=1,
                 n_neutral=8, n_clean=8, neutral_slots=[4], clean_lengths=[12], primary_seeds=[117], tasks=['five'], eval_batch=8)
        return p

    def test_freeze_train_evaluate_and_record_the_ratio(self):
        ss = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'U4sep'
            m = ss.freeze(out, self.tiny(ss))
            self.assertEqual(ss.verify_frozen(out)['protocol_hash'], m['protocol_hash'])
            self.assertGreater(m['protocol']['sep_weight'], 0.)
            cs = ss.conditions(m['protocol'])
            self.assertEqual([ss.ident(c) for c in cs], ['u4sep-soft_sep-five-w8-s117'])
            r = ss.run_one(out, cs[0])
            self.assertEqual(r['steps'], 4)
            for key in ('guard', 'selected_step', 'sep_ratio_last', 'sep_loss_last'):
                self.assertIn(key, r)
            e = ss.evaluate_one(out, cs[0])
            self.assertEqual(set(e['neutral']), {'0-base', '4-prefix', '4-middle', '4-suffix'})
            self.assertIn('12', e['clean'])
            self.assertIn('separation_ratio', e)
            self.assertIn('gate', e['neutral']['0-base'])
            self.assertEqual(ss.run_one(out, cs[0])['selected_step'], r['selected_step'])   # cached
            with self.assertRaises(ValueError):
                ss.freeze(out, dict(self.tiny(ss), steps=5))

    def test_forward_matches_the_sealed_soft_forward_bitwise(self):
        """The duplicated training loop must not drift from the sealed GatedCore: identical logits, gates and states."""
        ss = self.api()
        from horizon import task_families as tf
        p = self.tiny(ss)
        c = ss.conditions(p)[0]
        model = ss.make_model(c, p)
        x, _, _ = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'validation', 4, p['train_length'])
        with torch.no_grad():
            sealed_logits, sealed_gates, _ = model(x)
            mine_logits, mine_gates, mine_states = ss.forward_with_states(model, x)
        self.assertTrue(torch.equal(sealed_logits, mine_logits))
        self.assertTrue(torch.equal(sealed_gates, mine_gates))
        self.assertTrue(torch.equal(model.last_states, mine_states))
        self.assertTrue(ss.forward_with_states(model, x)[2].requires_grad)   # and mine stays on the graph

    def test_sealed_gated_sweep_sources_are_untouched(self):
        ss = self.api()
        from horizon import gated_sweep as gs
        for d in ('results/U2c-v2', 'results/U2c-lr0003'):
            man = json.loads((Path(d) / 'frozen.json').read_text(encoding='utf8'))
            for name, stored in man['sources'].items():
                self.assertEqual(gs.sw.sha(Path(name)), stored, f'{name} changed: the sealed U2c manifests would break')


if __name__ == '__main__':
    unittest.main()
