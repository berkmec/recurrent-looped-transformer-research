"""Z6 (I3-d eval zeyli; seal Rev.2): `long_eval --reset-every m` — the E-L1X periodic state reset (`mline/long_doc.py`'s
`reset512` semantics) is opt-in; with it off (0) `measure` is the previous behaviour, bit for bit, and the flag only appends
`reset_every` to the result. The configured path is checked, without instrumentation, against the manual reset loop."""
import tempfile
import unittest
from pathlib import Path

import torch

from horizon import depth_run as dr
from horizon import depth_tasks as dt
from horizon import long_eval as le


def strip_seconds(obj):
    """The result's only nondeterministic field: the wall-clock `seconds` of each evaluation point."""
    if isinstance(obj, dict):
        return {k: strip_seconds(v) for k, v in obj.items() if k != 'seconds'}
    if isinstance(obj, list):
        return [strip_seconds(v) for v in obj]
    return obj


class Configure(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def tiny(self, tmp, arm='core'):
        out = Path(tmp) / f'm-{arm}.json'
        dr.run('perm3', arm, 64, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
        return out.with_suffix('.pt')

    def test_needs_a_positive_interval_and_a_recurrent_core(self):
        with tempfile.TemporaryDirectory() as d:
            model = dr.load_model(self.tiny(d))
            for m in (0, -3):
                with self.assertRaises(ValueError):
                    le.configure_reset(model, m)
            with self.assertRaises(ValueError):
                le.configure_reset(dr.load_model(self.tiny(d, 'gru')), 8)

    def test_configured_reset_equals_the_manual_e_l1x_loop(self):
        with tempfile.TemporaryDirectory() as d:
            pt = self.tiny(d)
            x, _, _ = dt.heldout(dr.NAMESPACE, 'perm3', 1, 'dev', 2, 12)
            m = 5
            model = dr.load_model(pt).eval()
            e, _, _ = model.encode(x)
            n = model.mn(e)
            mk = model.rope(model.split(model.mk(n)))
            mv = model.split(model.mv(n))
            outs, s = [], None
            for t in range(x.shape[1]):
                if s is not None and t > 0 and t % m == 0:
                    s = dict(s)
                    s['s'] = model.initial.expand_as(s['s'])
                y, s = model.decode(e[:, t], s, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
                outs.append(y)
            manual = torch.stack(outs, 1)
            configured = dr.load_model(pt).eval()
            le.configure_reset(configured, m)
            got = configured(x)
            self.assertTrue(torch.equal(manual, got))                   # the configured path IS the E-L1X reset, bit for bit against the manual loop
            plain = dr.load_model(pt).eval()(x)
            self.assertFalse(torch.equal(plain, got))                   # and the reset fires within 12 steps: the output changes

    def test_no_event_is_the_plain_forward(self):
        with tempfile.TemporaryDirectory() as d:
            pt = self.tiny(d)
            x, _, _ = dt.heldout(dr.NAMESPACE, 'perm3', 1, 'dev', 2, 12)
            plain = dr.load_model(pt).eval()(x)
            model = dr.load_model(pt).eval()
            le.configure_reset(model, 64)
            self.assertTrue(torch.equal(plain, model(x)))


class Measure(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def tiny(self, tmp, arm='core'):
        out = Path(tmp) / f'm-{arm}.json'
        dr.run('perm3', arm, 64, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
        return out.with_suffix('.pt')

    def test_off_path_is_bit_for_bit_and_the_flag_only_appends_meta(self):
        with tempfile.TemporaryDirectory() as d:
            pt = self.tiny(d)
            off = le.measure(pt, [20], 4, batch=2)
            off0 = le.measure(pt, [20], 4, batch=2, reset_every=0)
            self.assertEqual(strip_seconds(off), strip_seconds(off0))
            self.assertNotIn('reset_every', off)
            on = le.measure(pt, [20], 4, batch=2, reset_every=64)       # no reset within 20 steps
            self.assertEqual(on['reset_every'], 64)
            self.assertEqual(strip_seconds(off), {k: v for k, v in strip_seconds(on).items() if k != 'reset_every'})

    def test_the_flag_is_rejected_for_other_arms(self):
        with tempfile.TemporaryDirectory() as d:
            pt = self.tiny(d, 'gru')
            with self.assertRaises(ValueError):
                le.measure(pt, [20], 4, batch=2, reset_every=8)


if __name__ == '__main__':
    unittest.main()
