"""The single-step rewrite must reproduce the sealed Core exactly, including its rotary embedding."""
import unittest

import torch


class FastStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from mline import faststep
        except ImportError as e:
            self.fail('mline.faststep is not implemented: ' + str(e))
        return faststep

    def core(self, width=32, heads=2, window=4, classes=5):
        from core.model import Core
        torch.manual_seed(0)
        return Core(width=width, heads=heads, window=window, classes=classes, feedback=True, alpha=.1)

    def test_rotary_tables_match_core_rope(self):
        """The reimplemented rotary must be the sealed one, at every position and for a whole sequence."""
        fs = self.api()
        core = self.core()
        dh = core.width // core.heads
        x = torch.randn(2, core.heads, 7, dh)
        cos_t, sin_t = fs.rope_tables(dh, 7, x.device, x.dtype)
        whole = fs._rope_all(x, cos_t, sin_t)
        self.assertLess(float((core.rope(x) - whole).abs().max()), 1e-6)
        for t in (0, 1, 5):
            one = x[:, :, t:t + 1]
            mine = fs.apply_rope(one, cos_t[t].view(1, 1, 1, -1), sin_t[t].view(1, 1, 1, -1))
            self.assertLess(float((core.rope(one, t) - mine).abs().max()), 1e-6, f't={t}')

    def test_forward_matches_the_sealed_core(self):
        fs = self.api()
        core = self.core()
        for T in (1, 3, 5, 17):
            x = torch.randint(0, 3, (3, T))
            with torch.no_grad():
                diff = float((core(x) - fs.forward(core, x)).abs().max())
            self.assertLess(diff, 2e-5, f'T={T}: max difference {diff:.3e}')

    def test_the_step_function_takes_only_tensors(self):
        """No Python int may reach the step, or the compiler specialises a separate graph per timestep."""
        import inspect
        fs = self.api()
        step = fs.make_step(self.core())
        names = list(inspect.signature(step).parameters)
        self.assertNotIn('t', names)
        self.assertEqual(names, ['e_t', 'prev', 'kbuf', 'vbuf', 'mk', 'mv', 'cos', 'sin', 'wmask', 'mmask'])

    def test_every_call_of_the_step_sees_identical_shapes_and_dtypes(self):
        fs = self.api()
        core = self.core()
        seen = set()
        base = fs.make_step(core)

        def spy(*args):
            seen.add(tuple((tuple(a.shape), a.dtype) for a in args))
            return base(*args)

        with torch.no_grad():
            fs.forward(core, torch.randint(0, 3, (2, 12)), step=spy)
        self.assertEqual(len(seen), 1, f'{len(seen)} distinct step signatures; a compiler would build that many graphs')

    def test_gradients_flow(self):
        fs = self.api()
        core = self.core()
        logits = fs.forward(core, torch.randint(0, 3, (2, 9)))
        logits.sum().backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in core.parameters()))

    def test_sealed_core_is_untouched(self):
        import json
        from pathlib import Path
        from horizon import sweep as sw
        self.api()
        man = json.loads(Path('results/U2c-v2/frozen.json').read_text(encoding='utf8'))
        self.assertEqual(sw.sha(Path('core/model.py')), man['sources']['core/model.py'])


if __name__ == '__main__':
    unittest.main()
