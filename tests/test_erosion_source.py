"""E-A5: the windows restrict exactly the context paths they say, the injection replaces exactly the carried state it says, the counts are the hazard's definition and the registered
reading is mechanical."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from core.model import Core
from horizon import depth_models as dm
from horizon import depth_run as dr
from horizon import depth_tasks as dt
from horizon import erosion_source as es
from horizon import long_eval
from horizon.state_growth import core_trajectory


def tiny(seed=3):
    model = dm.make_arm('core', 'perm3', 32, seed)
    return model.eval()


def tokens(batch=3, length=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 6, (batch, length), generator=g)


class Windows(unittest.TestCase):
    def test_windowed_encoder_equals_the_unwindowed_encoder_on_the_last_k_tokens(self):
        # RoPE scores depend on relative offsets only and the first token carries no marker: a window of K is a trained-range prefix of K tokens
        model, x, k = tiny(), tokens(), 8
        with torch.no_grad():
            es.configure(model, enc=k)
            windowed, _, _ = model.encode(x)
            es.configure(model)
            for t in (3, 7, 8, 20, 39):
                reference, _, _ = model.encode(x[:, max(0, t - k + 1):t + 1])
                self.assertTrue(torch.allclose(windowed[:, t], reference[:, -1], atol=1e-5), t)

    def test_no_window_or_a_window_longer_than_the_sequence_is_the_sealed_computation(self):
        model, x = tiny(), tokens()
        with torch.no_grad():
            sealed = Core.forward(model, x)
            es.configure(model)
            self.assertTrue(torch.equal(model(x), sealed))
            es.configure(model, cross=100, enc=100)
            self.assertTrue(torch.equal(model(x), sealed))

    def test_the_cross_attention_sees_the_last_k_memory_positions(self):
        model, x, seen = tiny(), tokens(2, 14), []
        original = Core.decode

        def spy(self, e, s, mk, mv, t):
            seen.append((t, mk.shape[2], mv.shape[2]))
            return original(self, e, s, mk, mv, t)

        Core.decode = spy
        try:
            with torch.no_grad():
                es.configure(model, cross=5)
                model(x)
        finally:
            Core.decode = original
        self.assertEqual(seen, [(t, min(t + 1, 5), min(t + 1, 5)) for t in range(14)])

    def test_a_window_changes_nothing_before_its_first_k_positions_and_something_later(self):
        model, x = tiny(), tokens(3, 40)
        with torch.no_grad():
            base = model(x)
            for kwargs in (dict(cross=6), dict(enc=6), dict(cross=6, enc=6)):
                es.configure(model, **kwargs)
                out = model(x)
                self.assertTrue(torch.allclose(out[:, :6], base[:, :6], atol=1e-5), kwargs)
                self.assertGreater(float((out[:, 12:] - base[:, 12:]).abs().max()), 1e-4, kwargs)
        es.configure(model)

    def test_the_banded_chunked_attention_equals_the_dense_masked_attention(self):
        g = torch.Generator().manual_seed(4)
        q, k, v = (torch.randn(2, 3, 300, 8, generator=g) for _ in range(3))
        idx = torch.arange(300)
        for window in (1, 17, 64, 299):
            mask = ((idx[:, None] - idx[None, :]) >= 0) & ((idx[:, None] - idx[None, :]) < window)
            dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            banded = es.windowed_attention(q, k, v, window, chunk=64)
            self.assertTrue(torch.allclose(dense, banded, atol=1e-5), window)

    def test_the_encoder_gives_the_same_result_on_the_dense_and_on_the_banded_path(self):
        model, x = tiny(), tokens(2, 90)
        es.configure(model, enc=12)
        with torch.no_grad():
            dense, _, _ = model.encode(x)
            with mock.patch.object(es, 'DENSE_LIMIT', 20):
                banded, _, _ = model.encode(x)
        es.configure(model)
        self.assertTrue(torch.allclose(dense, banded, atol=1e-5))

    def test_configure_validates(self):
        model = tiny()
        for kwargs in (dict(cross=0), dict(enc=0), dict(at=0, state=torch.zeros(1, 32)), dict(at=3)):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                es.configure(model, **kwargs)
        with self.assertRaises(ValueError):
            es.configure(dm.make_arm('gru', 'perm3', 16, 1))


class Injection(unittest.TestCase):
    def test_injecting_the_models_own_state_changes_nothing_and_another_state_changes_only_later_steps(self):
        model, x, t0 = tiny(), tokens(3, 30), 12
        with torch.no_grad():
            states, logits = core_trajectory(model, x)
            es.configure(model, at=t0, state=states[:, t0 - 1])
            self.assertTrue(torch.allclose(model(x), logits, atol=1e-6))                          # the same state: the same run
            es.configure(model, at=t0, state=states[:, t0 - 1] + 1.0)
            moved = model(x)
            self.assertTrue(torch.allclose(moved[:, :t0], logits[:, :t0], atol=1e-6))              # causal: nothing before the injection moves
            self.assertGreater(float((moved[:, t0] - logits[:, t0]).abs().max()), 1e-4)            # the injected step itself
            es.configure(model, at=t0, state=states[:, t0 - 1], reset_cache=True)
            cleared = model(x)
            self.assertTrue(torch.allclose(cleared[:, :t0], logits[:, :t0], atol=1e-6))
            self.assertGreater(float((cleared[:, t0] - logits[:, t0]).abs().max()), 1e-6)          # an emptied cache is a different decoder input
        es.configure(model)

    def test_class_centroids_are_the_class_means_of_the_carried_state(self):
        model = tiny()
        c = es.class_centroids(model, 'perm3', 1, n=400, length=16, positions=(8, 16))
        self.assertEqual(tuple(c.shape), (6, 32))
        x, y, _ = dt.heldout(dr.NAMESPACE, 'perm3', 1, 'dev', 400, 16)
        with torch.no_grad():
            states, _ = core_trajectory(model, x)
        members = states[:, 8:16].reshape(-1, 32)[y[:, 8:16].reshape(-1) == 2]
        self.assertTrue(torch.allclose(c[2], members.mean(0), atol=1e-5))
        with self.assertRaises(ValueError):
            es.class_centroids(model, 'perm3', 1, n=2, length=16)                                  # too few states in some class


class Counts(unittest.TestCase):
    def test_window_counts_are_the_hazard_rows_definition(self):
        g = torch.Generator().manual_seed(5)
        target = torch.randint(0, 5, (60, 50), generator=g)
        pred = target.clone()
        flips = torch.rand(60, 50, generator=g) < 0.03
        pred[flips] = (pred[flips] + 1) % 5
        first = es.first_errors(pred, target)
        for row in long_eval.hazard_rows(pred, target, [0, 16, 40, 50]):
            e, p, a = es.window_counts(first, row['start'], row['stop'])
            self.assertEqual((e, p, a), (row['first_errors'], row['person_steps'], row['at_risk']))

    def test_classify_labels_and_the_zero_event_floor(self):
        self.assertEqual(es.classify((100, 10000), (10, 10000))[0], 'RESCUES')
        self.assertEqual(es.classify((100, 10000), (30, 10000))[0], 'PARTIAL')
        self.assertEqual(es.classify((100, 10000), (80, 10000))[0], 'NEUTRAL')
        self.assertEqual(es.classify((100, 10000), (300, 10000))[0], 'HARMS')
        self.assertEqual(es.classify((100, 10000), (0, 10000))[0], 'RESCUES')                      # 3 / person-steps floor: R = 33
        self.assertEqual(es.classify((0, 10000), (0, 10000))[0], 'NEUTRAL')
        self.assertEqual(es.classify((0, 10000), (20, 10000))[0], 'HARMS')
        self.assertEqual(es.classify((0, 10000), (3, 10000))[0], 'NEUTRAL')                        # fewer than 5 events never counts as harm
        self.assertEqual(es.classify((100, 0), (10, 10000))[0], 'NO DATA')
        label, details = es.classify((100, 10000), (10, 10000))
        self.assertAlmostEqual(details['ratio'], 10.0)
        self.assertLess(details['p_lower'], 1e-6)
        self.assertGreater(details['p_higher'], 0.99)
        # the rate-ratio boundary is exact: R = 5 counts as a rescue, R just below as partial
        self.assertEqual(es.classify((100, 10000), (20, 10000))[0], 'RESCUES')
        self.assertEqual(es.classify((100, 10000), (21, 10000))[0], 'PARTIAL')
        self.assertEqual(es.classify((100, 10000), (50, 10000))[0], 'PARTIAL')                    # R = 2 exactly
        self.assertEqual(es.classify((100, 10000), (51, 10000))[0], 'NEUTRAL')                    # R = 1.96
        self.assertEqual(es.classify((100, 10000), (200, 10000))[0], 'HARMS')                     # R = 0.5 exactly


def fake(task='perm5', width=128, base=(100, 10000), both16=None, inject=None, stored=True, control=1.0, s4_events=0, prefix=True):
    """A synthetic `measure` result: window events for base / both16 and the injection counts for the pooled steps."""
    both16 = both16 if both16 is not None else base
    inject = inject if inject is not None else (base, base)
    windows = {'base': dict(events=base[0], person_steps=base[1], at_risk=100, final=0.1),
               'both16': dict(events=both16[0], person_steps=both16[1], at_risk=100, final=0.1, prefix_identical=prefix)}
    injection = {str(es.CONTROL_AT): dict(span=8, untouched=dict(events=0, person_steps=800, at_risk=100),
                                          **{v: dict(events=0, person_steps=800, at_risk=100, accuracy=control) for v in ('state', 'state+cache')})}
    for t0 in es.POOLED_AT:
        injection[str(t0)] = dict(span=es.SPAN, untouched=dict(events=base[0], person_steps=base[1], at_risk=100),
                                  state=dict(events=inject[0][0] + s4_events, person_steps=inject[0][1], at_risk=100, accuracy=0.99),
                                  **{'state+cache': dict(events=inject[1][0] + s4_events, person_steps=inject[1][1], at_risk=100, accuracy=0.99)})
    return dict(weights=f'x/{task}-w{width}.pt', task=task, width=width, seed=1, n=200, length=1024, parameters=1, stored_control=stored, windows=windows, injection=injection, in_range={})


class Reading(unittest.TestCase):
    def s5(self, **kw):
        return [fake(width=128 + i, **kw) for i in range(3)]

    def test_verdicts_follow_the_registered_rule(self):
        s4 = [fake('perm4', 32, base=(0, 10000), both16=(0, 10000), inject=((0, 10000), (0, 10000)))]
        rescued = es.reading(self.s5(both16=(10, 10000), inject=((100, 10000), (100, 10000))) + s4)
        self.assertEqual(rescued['window'], 'ATTENTION PATHS CARRY THE EROSION')
        self.assertEqual(rescued['injection'], 'NOT THE CARRIED STATE')
        reset = es.reading(self.s5(both16=(100, 10000), inject=((100, 10000), (10, 10000))) + s4)
        self.assertEqual(reset['window'], 'NOT THE ATTENTION PATHS')
        self.assertEqual(reset['injection'], 'THE CARRIED STATE HISTORY CARRIES THE EROSION')
        mixed = [fake(width=128, both16=(10, 10000)), fake(width=129), fake(width=130)]
        self.assertEqual(es.reading(mixed + s4)['window'], 'MIXED')                                # one rescue of three is not enough
        partial = es.reading(self.s5(both16=(40, 10000)) + s4)
        self.assertEqual(partial['window'], 'MIXED')                                              # PARTIAL everywhere: neither carries nor excluded

    def test_a_harmful_intervention_is_inconclusive_not_a_negative(self):
        s4 = [fake('perm4', 32, base=(0, 10000), both16=(30, 10000), inject=((0, 10000), (0, 10000)))]              # the window creates errors in the working S4 model
        verdict = es.reading(self.s5(both16=(100, 10000)) + s4)
        self.assertTrue(verdict['window'].startswith('INCONCLUSIVE'))
        harmed = es.reading(self.s5(both16=(300, 10000)) + [fake('perm4', 32, base=(0, 10000), both16=(0, 10000), inject=((0, 10000), (0, 10000)))])
        self.assertTrue(harmed['window'].startswith('INCONCLUSIVE'))
        creates = es.reading(self.s5() + [fake('perm4', 32, base=(0, 10000), both16=(0, 10000), inject=((0, 10000), (0, 10000)), s4_events=1)])
        self.assertTrue(creates['injection'].startswith('INCONCLUSIVE'))                          # 3 pooled events after injection in the S4 control (> 2)

    def test_failed_controls_void_the_verdicts(self):
        s4 = [fake('perm4', 32, base=(0, 10000), both16=(0, 10000), inject=((0, 10000), (0, 10000)))]
        for kw, text in ((dict(stored=False), 'stored first-error windows'), (dict(prefix=False), 'before its first K positions'), (dict(control=0.9), 'in-range injection')):
            out = es.reading(self.s5(**kw) + s4)
            self.assertTrue(out['window'].startswith('CONTROL FAILED'), kw)
            self.assertIn(text, out['injection'], kw)
        self.assertEqual(es.reading(s4)['window'], 'NO S5 MODEL')


class EndToEnd(unittest.TestCase):
    def test_a_measure_of_a_tiny_trained_core_has_every_part_and_a_reading(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'core', 32, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            r = es.measure(out.with_suffix('.pt'), n=6, length=300, batch=3, log=lambda m: None)
        self.assertEqual((r['task'], r['width'], r['n'], r['length']), ('perm3', 32, 6, 300))
        self.assertEqual(set(r['windows']), {'base', 'both16', 'both32', 'both64', 'both128', 'cross16', 'cross64', 'enc16', 'enc64'})
        self.assertTrue(all(isinstance(v['prefix_identical'], bool) for k, v in r['windows'].items() if k != 'base'))      # the value itself is read on trained models (an untrained one has near-tied logits)
        self.assertEqual(set(r['injection']), {'8', '32', '64', '128'})                              # 256 + 128 and 512 + 128 do not fit 300 positions
        self.assertEqual(set(r['injection']['64']['state']['since']), {'0-16', '16-32', '32-64', '64-128'})
        self.assertEqual(r['injection']['8']['state']['since'], {})                                  # the in-range control follows 8 steps only
        self.assertEqual(set(r['in_range']), {'full', 'K1', 'K2', 'K4', 'K8'})
        self.assertIsNone(r['stored_control'])                                                       # no stored 1,024-position curve for a tiny run
        self.assertIn('both16', es.describe(r))
        # the in-range injection control: the true class's centroid, 8 steps in, on an in-range sequence
        self.assertIn('accuracy', r['injection']['8']['state'])


if __name__ == '__main__':
    unittest.main()
