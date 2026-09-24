"""The static-shape rewrite must be the SAME FUNCTION as the sealed Core, not merely a faster one that looks similar."""
import unittest

import torch


class FastCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from mline import fastcore
        except ImportError as e:
            self.fail('mline.fastcore is not implemented: ' + str(e))
        return fastcore

    def core(self, width=32, heads=2, window=4, classes=5, feedback=True):
        from core.model import Core
        torch.manual_seed(0)
        return Core(width=width, heads=heads, window=window, classes=classes, feedback=feedback, alpha=.1)

    def test_matches_the_sealed_core_forward(self):
        fc = self.api()
        core = self.core()
        for T in (1, 3, 5, 17):                       # 1 and 3 exercise the window before it is full
            x = torch.randint(0, 3, (3, T))
            with torch.no_grad():
                sealed = core(x)
                fast = fc.forward_static(core, x)
            self.assertEqual(sealed.shape, fast.shape)
            diff = float((sealed - fast).abs().max())
            self.assertLess(diff, 2e-5, f'T={T}: max difference {diff:.3e}')

    def test_states_match_the_sealed_gated_core_in_both_modes(self):
        """An independent review showed the previous version of this test never executed the branch where the two
        implementations differ: GatedCore initialises its head at bias 2.0, so every gate was WRITE and the HOLD
        rollback path was never entered. The test now FORCES holds and refuses to pass unless it observes some."""
        fc = self.api()
        from horizon import gated as gd
        x = torch.randint(0, 3, (3, 11))
        for mode in ('soft', 'hard_st'):
            for bias in (2., -2., 0.):                      # WRITE-heavy, HOLD-heavy, mixed
                torch.manual_seed(0)
                model = gd.GatedCore(self.core(), mode=mode, k=2)
                with torch.no_grad():
                    model.head.bias.fill_(bias)
                    s_logits, s_gates, _ = model(x)
                    f_logits, f_gates, f_states = fc.forward_static_gated(model, x)
                label = f'{mode}/bias={bias}'
                self.assertLess(float((s_logits - f_logits).abs().max()), 2e-5, label)
                self.assertLess(float((s_gates - f_gates).abs().max()), 2e-5, label)
                self.assertLess(float((model.last_states - f_states).abs().max()), 2e-5, label)
        # the guard: a run of this test that observed no HOLD proves nothing
        torch.manual_seed(0)
        model = gd.GatedCore(self.core(), mode='hard_st', k=2)
        with torch.no_grad():
            model.head.bias.fill_(-2.)
            _, gates, _ = model(x)
        self.assertGreater(float((gates <= .5).float().mean()), .5, 'the forced-HOLD case did not actually hold')

    def test_gradients_flow_and_states_stay_on_the_graph(self):
        fc = self.api()
        from horizon import gated as gd
        x = torch.randint(0, 3, (2, 9))
        torch.manual_seed(0)
        model = gd.GatedCore(self.core(), mode='soft', k=2)
        logits, _, states = fc.forward_static_gated(model, x)
        self.assertTrue(states.requires_grad)
        logits.sum().backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters()))

    def test_shapes_are_identical_at_every_timestep(self):
        """The whole point: if any per-step tensor changes shape, a compiler still has to recompile."""
        fc = self.api()
        core = self.core()
        seen = {}
        original = torch.nn.functional.scaled_dot_product_attention

        def spy(q, k, v, *a, **kw):
            seen.setdefault('q', []).append(tuple(q.shape))
            seen.setdefault('k', []).append(tuple(k.shape))
            return original(q, k, v, *a, **kw)

        torch.nn.functional.scaled_dot_product_attention = spy
        try:
            with torch.no_grad():
                fc.forward_static(core, torch.randint(0, 3, (2, 12)))
        finally:
            torch.nn.functional.scaled_dot_product_attention = original
        # the encoder contributes one call; every decoder call must repeat the same two shapes
        decoder_k = seen['k'][1:]
        self.assertEqual(len(set(decoder_k)), 2, f'decoder key shapes vary: {sorted(set(decoder_k))}')

    def test_sealed_core_is_untouched(self):
        import json
        from pathlib import Path
        from horizon import sweep as sw
        self.api()
        man = json.loads(Path('results/U2c-v2/frozen.json').read_text(encoding='utf8'))
        self.assertEqual(sw.sha(Path('core/model.py')), man['sources']['core/model.py'])


if __name__ == '__main__':
    unittest.main()
