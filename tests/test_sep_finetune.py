"""U4-sep2 runner contract (written before the implementation)."""
import json
import tempfile
import unittest
from pathlib import Path

import torch


class SepFinetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import sep_finetune as sf
        except ImportError as e:
            self.fail('horizon.sep_finetune is not implemented: ' + str(e))
        return sf

    def tiny(self, sf):
        p = sf.protocol()
        p.update(steps=2, validation_every=1, validation_n=8, n_neutral=8, n_clean=8,
                 neutral_slots=[4], clean_lengths=[128], drift_slots=8, n_geometry=4,
                 checkpoints=[dict(p['checkpoints'][0])])
        return p

    def test_forward_matches_the_sealed_gated_core_in_both_modes(self):
        """The fine-tuner drives soft AND hard_st checkpoints, so the duplicated loop must match the sealed one twice."""
        sf = self.api()
        from core.model import Core
        from horizon import gated as gd
        x = torch.randint(0, 3, (3, 9))
        for mode in ('soft', 'hard_st'):
            torch.manual_seed(0)
            model = gd.GatedCore(Core(width=16, heads=2, window=4, classes=5, feedback=True, alpha=.1), mode=mode, k=2)
            with torch.no_grad():
                sealed_logits, sealed_gates, _ = model(x)
                mine_logits, mine_gates, mine_states = sf.forward_with_states(model, x)
            self.assertTrue(torch.equal(sealed_logits, mine_logits), mode)
            self.assertTrue(torch.equal(sealed_gates, mine_gates), mode)
            self.assertTrue(torch.equal(model.last_states, mine_states), mode)

    def test_conditions_pair_every_checkpoint_with_a_control(self):
        sf = self.api()
        p = sf.protocol()
        self.assertEqual(tuple(p['arms']), ('none', 'hinge'))
        ids = [sf.ident(c) for c in sf.conditions(p)]
        self.assertEqual(len(ids), 2 * len(p['checkpoints']))
        self.assertEqual(len(set(ids)), len(ids))
        self.assertIn('u4sep2-hinge-from_soft-s117', ids)
        self.assertIn('u4sep2-none-from_soft-s117', ids)      # same checkpoint, same budget, no geometry term
        for c in p['checkpoints']:
            self.assertGreaterEqual(c['u2c_validation'], .95)  # it solves the task
            self.assertLess(c['u2c_repair'], 50.)              # and it is fragile: there is something to rescue

    def test_freeze_finetune_and_evaluate(self):
        sf = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'U4sep2'
            m = sf.freeze(out, self.tiny(sf))
            self.assertEqual(sf.verify_frozen(out)['protocol_hash'], m['protocol_hash'])
            c = [x for x in sf.conditions(m['protocol']) if x['arm'] == 'hinge'][0]
            r = sf.run_one(out, c)
            for key in ('guard', 'selected_step', 'ratio_first', 'ratio_last', 'gap_last', 'drift_last', 'source_sha256'):
                self.assertIn(key, r)
            self.assertGreater(r['selected_validation']['final_accuracy'], .5)   # fine-tuning starts from a working model
            e = sf.evaluate_one(out, c)
            self.assertIn('geometry', e)
            self.assertIn('gap_over_drift', e['geometry'])
            self.assertEqual(sf.run_one(out, c)['selected_step'], r['selected_step'])   # cached
            with self.assertRaises(ValueError):
                sf.freeze(out, dict(self.tiny(sf), steps=3))

    def test_sealed_u2c_sources_are_untouched(self):
        sf = self.api()
        from horizon import sweep as sw
        self.assertTrue(sf.CHECKPOINTS)
        for d in ('results/U2c-v2', 'results/U2c-lr0003'):
            man = json.loads((Path(d) / 'frozen.json').read_text(encoding='utf8'))
            for name, stored in man['sources'].items():
                self.assertEqual(sw.sha(Path(name)), stored, f'{name} changed: the sealed U2c manifests would break')


if __name__ == '__main__':
    unittest.main()
