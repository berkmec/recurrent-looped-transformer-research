import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from horizon import depth_models as dm
from horizon import depth_rot as dr
from horizon import depth_run


def _curve(prefix8, reentry):
    return dict(curve={'8': dict(prefix=prefix8, reentry=None), **{str(d): dict(prefix=1.0, reentry=v) for d, v in reentry.items()}})


class RotaryCounterTests(unittest.TestCase):
    def test_a_closed_channel_is_the_core(self):                                   # K3
        m = dr.make_arm('core_rotq', 'count8_32', 32, 1).eval()
        m.gamma.data.zero_()
        x = torch.randint(0, 2, (3, 40))
        with torch.no_grad():
            self.assertLess(float((m(x) - m.core(x)).abs().max()), 1e-5)

    def test_rounded_steps_cancel_exactly_and_tanh_steps_only_nearly(self):
        x = torch.tensor([[0] * 50 + [1] * 50, [0, 1] * 50])                       # balanced: as many pushes as pops
        for quantize, exact in ((True, True), (False, False)):
            m = dr.make_arm('core_rotq' if quantize else 'core_rot', 'count8_32', 32, 1)
            with torch.no_grad():
                m.step.weight[0].fill_(5.0)
                m.step.weight[1].fill_(-3.0)                                           # deliberately not opposite
                end = m.angles(x)[:, -1]
            if exact:
                self.assertEqual(float(end.abs().max()), 0.0)
            else:
                self.assertGreater(float(end.abs().max()), 1e-3)

    def test_the_slowest_angle_stays_monotone_over_the_deepest_test(self):
        period = 2 * math.pi / float(dr.make_arm('core_rot', 'count8_32', 32, 1).omega.min())
        self.assertGreaterEqual(period, 2 * 512)

    def test_it_wraps_the_plain_core_of_the_same_seed(self):
        rot, plain = dr.make_arm('core_rot', 'count8_32', 64, 3), dm.make_arm('core', 'count8_32', 64, 3)
        for (n, p), (m, q) in zip(rot.core.named_parameters(), plain.named_parameters()):
            self.assertEqual(n, m)
            self.assertTrue(torch.equal(p, q), n)
        self.assertIs(type(dr.make_arm('gru', 'count8_32', 64, 3)), type(dm.make_arm('gru', 'count8_32', 64, 3)))

    def test_it_trains_through_depth_run(self):
        saved = depth_run.make_arm
        try:
            depth_run.make_arm = dr.make_arm
            with tempfile.TemporaryDirectory() as d:
                r = depth_run.run('count8_32', 'core_rotq', 32, 1, 2, points=[8, 16], n_eval=4, out=Path(d, 'r.json'))
                self.assertEqual(set(json.loads(Path(d, 'r.json').read_text(encoding='utf8'))['curve']), {'8', '16'})
        finally:
            depth_run.make_arm = saved

    def test_the_reading(self):
        with tempfile.TemporaryDirectory() as d:
            def put(arm, prefix8, reentry):
                p = Path(d, 'count8_32-%s-w64-s1.json' % arm)
                p.write_text(json.dumps(_curve(prefix8, reentry)), encoding='utf8')
                return p
            core = put('core', 0.99, {16: 0.99, 32: 0.97, 64: 0.2, 128: 0.0})
            rotq = put('core_rotq', 0.99, {16: 0.99, 32: 0.99, 64: 0.99, 128: 0.98, 256: 0.4})
            self.assertEqual(dr.reading([core, rotq])['verdict'], 'ROTARY EXTRAPOLATES: core_rotq')
            flat = put('core_rot', 0.99, {16: 0.99, 32: 0.96, 64: 0.1})
            self.assertEqual(dr.reading([core, flat])['verdict'], 'NO EXTRAPOLATION GAIN')
            self.assertTrue(dr.reading([core, put('core_rotq', 0.5, {16: 0.1})])['verdict'].startswith('VACUOUS'))
            weak = put('core', 0.6, {16: 0.3})
            good = put('core_rotq', 0.99, {16: 0.99, 32: 0.99, 64: 0.99, 128: 0.98})
            self.assertEqual(dr.reading([weak, good])['verdict'], 'ROTARY EXTRAPOLATES: core_rotq (the plain Core did not learn the range)')
            self.assertEqual(dr.reading([good])['verdict'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
