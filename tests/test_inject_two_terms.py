"""U4-inject's arms differ in TWO terms, not one, and this file demonstrates the second one rather than asserting it.

The campaign's own method rule, learned from U4-sep2: two arms of a paired comparison must differ in exactly one
term, or a difference in outcome cannot be attributed. U4-inject breaks it. Reading the two decoders side by side:

    sealed Core.decode            z = e + alpha*sigmoid(gate([e,r]))*merge(r)   <- feedback FIRST
                                  n = dn(z); k = dk(n); v = dv(n)               <- window K/V from the POST-feedback z

    InjectCore.decode             n = dn(e); k = dk(n); v = dv(n)               <- window K/V from e, PRE-injection
                                  z = e + io(attend(iq(pn(prev)), K, V))        <- injection AFTER

So the injection arms change how the state enters AND remove the state from the sliding-window key/value cache. In
the sealed model the cached dk/dv carry state history forward as a second recurrent channel; in the injection arms
they carry only the encoder output. `horizon/inject.py` names this ("the window K/V are derived from e_t alone ...
that is the point of the variant") and calls the paths "cleanly separable", which is the opposite of what the rule
requires: it makes the comparison confounded, not clean.

The consequence for the sealed conclusion is bounded and specific. "6/6 injection conditions failed" remains true
as a fact about those six conditions. What it cannot support is "attention injection fails", because an arm that
lost a recurrent channel would be expected to do worse whatever the injection mechanism did. The arm that would
separate them does not exist yet: injection PLUS window K/V taken from the post-injection z.

These tests exist so the confound is a measured property of the code and not a paragraph someone can quietly drop.
"""
import unittest

import torch


class WindowCacheCarriesStateOnlyInTheSealedCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def core(self):
        from core.model import Core
        torch.manual_seed(0)
        return Core(width=16, heads=2, window=4, classes=5, feedback=True, alpha=.1)

    def rollout(self, model, steps=3):
        """Two states after a real rollout over DIFFERENT inputs, differing only in `s`.

        The window values must not be degenerate: if every cached v is the same vector, any query returns the same
        answer and the test would `prove` the injection arm is not recurrent, which it is. So the rollout feeds a
        different embedding at every step.
        """
        core = model.core if hasattr(model, 'core') else model
        torch.manual_seed(2)
        es = torch.randn(steps + 1, 3, 16)
        mk, mv = self.memory(core, es[0])
        state = None
        for t in range(steps):
            _, state = model.decode(es[t], state, mk, mv, t)
        a, b = dict(state), dict(state)
        torch.manual_seed(7)
        a['s'] = torch.randn_like(state['s']); b['s'] = torch.randn_like(state['s'])
        self.assertFalse(torch.equal(a['s'], b['s']))
        self.assertFalse(torch.allclose(state['dv'][:, :, 0], state['dv'][:, :, -1], atol=1e-6),
                         'the cached window values must differ, or a query cannot change the attention output')
        return es[steps], mk, mv, steps, a, b

    def memory(self, core, e):
        n = core.mn(e.unsqueeze(1))
        return core.rope(core.split(core.mk(n))), core.split(core.mv(n))

    def test_the_sealed_core_writes_the_state_into_the_window_cache(self):
        core = self.core()
        e, mk, mv, t, a, b = self.rollout(core)
        ka = core.decode(e, a, mk, mv, t)[1]['dk']
        kb = core.decode(e, b, mk, mv, t)[1]['dk']
        self.assertFalse(torch.allclose(ka, kb, atol=1e-6),
                         'the sealed cache must depend on the previous state - that is the second recurrent channel')

    def test_the_injection_arms_do_not(self):
        from horizon import inject as ij
        for mode in ('inject', 'inject_mix'):
            model = ij.InjectCore(self.core(), mode=mode)
            e, mk, mv, t, a, b = self.rollout(model)
            ka = model.decode(e, a, mk, mv, t)[1]['dk']
            kb = model.decode(e, b, mk, mv, t)[1]['dk']
            self.assertTrue(torch.allclose(ka, kb, atol=1e-7),
                            f'{mode}: the window cache still depends on the state, so this file describes the wrong flaw')

    def test_the_injected_output_does_depend_on_the_state(self):
        """Guards the obvious objection: the arms are not simply ignoring the state."""
        from horizon import inject as ij
        model = ij.InjectCore(self.core(), mode='inject')
        e, mk, mv, t, a, b = self.rollout(model)
        za = model.decode(e, a, mk, mv, t)[1]['s']
        zb = model.decode(e, b, mk, mv, t)[1]['s']
        self.assertFalse(torch.allclose(za, zb, atol=1e-6),
                         'the injection arm must still be recurrent through the query path')

class TheReportDisclosesTheConfoundTests(unittest.TestCase):
    def test_the_report_names_the_second_term_and_the_missing_arm(self):
        from pathlib import Path
        text = Path(__file__).resolve().parents[1].joinpath('notes/U4inject_RAPORU.md').read_text(encoding='utf8')
        for token in ('dk', 'dv', 'tek terim'):
            self.assertIn(token, text, f'the report must disclose the second differing term ({token})')


if __name__ == '__main__':
    unittest.main()
