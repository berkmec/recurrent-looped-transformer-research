"""E-RM0R8: the long-window arm is E-RM0R6's shifted-key delta arm (same weights and outputs at window 32), the parallel write equals the
loop, a closed write gate starts closed, the selectivity probe indexes the fact positions, and the registered reading is mechanical."""
import unittest

import torch

from horizon import erm_r as er
from horizon import erm_r8 as r8
from horizon import erm_tasks as et


class Arm(unittest.TestCase):
    def test_window_32_is_the_e_rm0r6_arm(self):
        old = er.make('B_delta_shift', seed=1, positions=True).eval()
        new = r8.ArmDelta8(32, seed=1, ut_chunk=0).eval()
        self.assertEqual(sorted(old.state_dict()), sorted(new.state_dict()))
        for k, v in old.state_dict().items():
            self.assertTrue(torch.equal(v, new.state_dict()[k]), k)
        x = er.generate(3, seed=4)[0].reshape(3, et.N_WINDOWS, et.WINDOW)
        with torch.no_grad():
            self.assertTrue(torch.allclose(old(x), new(x), atol=1e-6))

    def test_the_parallel_write_is_the_loop_at_a_long_window(self):
        loop, ut = r8.ArmDelta8(128, seed=2, ut_chunk=0).eval(), r8.ArmDelta8(128, seed=2, ut_chunk=64).eval()
        x = er.generate(2, seed=5, window=128)[0].reshape(2, et.N_WINDOWS, 128)
        with torch.no_grad():
            self.assertLess(float((loop(x) - ut(x)).abs().max()), 1e-4)

    def test_a_closed_gate_starts_closed(self):
        closed, default = r8.make('ds128c', seed=1), r8.make('ds128', seed=1)
        x = er.generate(4, seed=6, window=128)[0].reshape(4, et.N_WINDOWS, 128)
        with torch.no_grad():
            b_closed = closed.gates(closed.enc.forward_windows(x)[:, 0])[2]
            b_default = default.gates(default.enc.forward_windows(x)[:, 0])[2]
        self.assertLess(float(b_closed.mean()), 0.05)
        self.assertGreater(float(b_default.mean()), 0.2)

    def test_selectivity_finds_the_fact_values(self):
        model = r8.make('ds128', seed=1)
        with torch.no_grad():
            model.wb.weight.zero_()
            model.wb.bias.fill_(0.0)
        s = r8.selectivity(model, 128, n=4)
        self.assertAlmostEqual(s['fact'], 0.5, places=5)
        self.assertAlmostEqual(s['other'], 0.5, places=5)


def result(acc):
    return dict(eval={'3': dict(acc=acc)}, selectivity=None)


class Reading(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(r8.reading(dict(ds32=result(1.0), ds128=result(0.2), ds128c=result(0.95)))['verdict'], 'CLOSED WRITE FIXES IT')
        self.assertEqual(r8.reading(dict(ds32=result(1.0), ds128=result(0.2), ds128c=result(0.5)))['verdict'], 'INTERFERENCE REPRODUCED')
        self.assertEqual(r8.reading(dict(ds32=result(0.99), ds128=result(0.93), ds128c=result(0.95)))['verdict'], 'NO INTERFERENCE AT 2.0')
        self.assertEqual(r8.reading(dict(ds32=result(0.6), ds128=result(0.2), ds128c=result(0.95)))['verdict'], 'INCONCLUSIVE')
        self.assertEqual(r8.reading(dict(ds32=result(1.0), ds128=result(0.5), ds128c=result(0.95)))['verdict'], 'INCONCLUSIVE')


if __name__ == '__main__':
    unittest.main()
