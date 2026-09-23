"""U4-inject runner contract (written before the implementation)."""
import tempfile
import unittest
from pathlib import Path

import torch


class InjectSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import inject_sweep as isw
        except ImportError as e:
            self.fail('horizon.inject_sweep is not implemented: ' + str(e))
        return isw

    def tiny(self, isw):
        p = isw.protocol()
        p.update(steps=4, batch=4, train_length=12, validation_every=2, validation_n=8, width=8, heads=1,
                 n_neutral=8, n_clean=8, neutral_slots=[4], clean_lengths=[12], primary_seeds=[117], tasks=['five'],
                 eval_batch=8, anchor_batches=2, norm_run_slots=4, norm_run_n=8)
        return p

    def test_three_arms_share_one_frozen_protocol(self):
        isw = self.api()
        p = isw.protocol()
        self.assertEqual(tuple(p['arms']), ('additive', 'inject', 'inject_mix'))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'U4inj'
            m = isw.freeze(out, self.tiny(isw))
            self.assertEqual(isw.verify_frozen(out)['protocol_hash'], m['protocol_hash'])
            self.assertEqual([isw.ident(c) for c in isw.conditions(m['protocol'])],
                             ['u4inj-additive-five-w8-s117', 'u4inj-inject-five-w8-s117', 'u4inj-inject_mix-five-w8-s117'])
            with self.assertRaises(ValueError):
                isw.freeze(out, dict(self.tiny(isw), steps=5))

    def test_train_and_evaluate_record_the_residual_explosion_quantities(self):
        isw = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'U4inj'
            m = isw.freeze(out, self.tiny(isw))
            c = [x for x in isw.conditions(m['protocol']) if x['arm'] == 'inject_mix'][0]
            r = isw.run_one(out, c)
            for key in ('guard', 'selected_step', 'selected_validation'):
                self.assertIn(key, r)
            self.assertIn('max_grad_norm_before_clip', r['guard'])   # the gradient-explosion side of the claim
            e = isw.evaluate_one(out, c)
            self.assertEqual(set(e['neutral']), {'0-base', '4-prefix', '4-middle', '4-suffix'})
            self.assertIn('12', e['clean'])
            self.assertIn('separation_ratio', e)
            self.assertIn('norm_growth', e)                          # the residual-explosion side of the claim
            self.assertIn('run_norm', e)
            self.assertEqual(isw.run_one(out, c)['selected_step'], r['selected_step'])   # cached

    def test_sealed_sources_are_untouched(self):
        import json
        from horizon import sweep as sw
        self.api()
        man = json.loads(Path('results/U2c-v2/frozen.json').read_text(encoding='utf8'))
        for name, stored in man['sources'].items():
            self.assertEqual(sw.sha(Path(name)), stored, f'{name} changed: the sealed U2c manifests would break')


if __name__ == '__main__':
    unittest.main()
