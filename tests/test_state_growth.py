"""E-B2 state-kick growth: the hand-run trajectories equal the models' forward, a zero kick leaves no trace, the slope recovers a known rate."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from core.model import Core
from horizon import depth_tasks as dt
from horizon import state_growth as sg
from horizon.depth_models import make_arm


def tokens(length=14, batch=3, task='perm4', seed=0):
    return torch.randint(0, dt.vocab(task), (batch, length), generator=torch.Generator().manual_seed(seed))


class TrajectoriesEqualTheForward(unittest.TestCase):
    def test_core_gru_lstm_logits_match(self):
        for arm, width in (('core', 32), ('gru', 16), ('lstm', 16)):
            model = make_arm(arm, 'perm4', width, 1).eval()
            x = tokens()
            run = sg.core_trajectory if arm == 'core' else sg.recurrent_trajectory
            with torch.no_grad():
                states, logits = run(model, x)
                self.assertTrue(torch.allclose(logits, model(x), atol=1e-5), arm)
            self.assertEqual(states.shape[:2], x.shape)
            self.assertEqual(states.shape[-1], {'core': 32, 'gru': 16, 'lstm': 32}[arm])      # the LSTM's diagnostic state is h and c together


class KicksLeaveTheRightTrace(unittest.TestCase):
    def test_zero_kick_and_positions_before_the_kick_are_untouched(self):
        for arm, width in (('core', 32), ('gru', 16), ('lstm', 16)):
            model = make_arm(arm, 'perm4', width, 1)
            x = tokens(20)
            none = sg.growth_series(model, x, t0=6, eps=0.0, seed=1)
            self.assertTrue((none['d'] == 0).all(), arm)
            kicked = sg.growth_series(model, x, t0=6, eps=0.05, seed=1)
            self.assertTrue((kicked['d'][:, :6] == 0).all(), arm)
            self.assertTrue((kicked['d'][:, 6:] > 0).all(), arm)
            self.assertFalse(bool(kicked['flip'][:, :6].any()), arm)

    def test_kick_size_is_eps_times_the_rms_of_the_clean_state(self):
        model = make_arm('core', 'perm4', 32, 1).double().eval()
        x = tokens(10)
        with torch.no_grad():
            states, _ = sg.core_trajectory(model, x)
        kick = sg.kick_for(model, states, 5, 0.01, torch.Generator().manual_seed(2))
        ratio = kick.square().mean(-1).sqrt() / states[:, 4].square().mean(-1).sqrt()
        self.assertTrue(bool(((ratio > 0.002) & (ratio < 0.05)).all()), ratio)

    def test_a_damped_run_shrinks_and_an_amplified_one_grows(self):
        # GRU with its recurrent weights scaled down is contracting; scaled up it is expanding (weights only, so the check is of the diagnostic, not of a trained model)
        base = make_arm('gru', 'perm4', 32, 1)
        x = tokens(60, batch=8)
        rates = {}
        for name, factor in (('damped', 0.1), ('amplified', 6.0)):
            m = copy.deepcopy(base)
            with torch.no_grad():
                m.rnn.weight_hh_l0.mul_(factor)
            series = sg.growth_series(m, x, t0=4, eps=1e-6, seed=3)
            rates[name] = sg.slope(series['log_d'], 6, 30)
        self.assertLess(rates['damped'], 0.0)
        self.assertGreater(rates['amplified'], rates['damped'] + 0.05)


class JacobianNorms(unittest.TestCase):
    def test_gru_with_closed_or_open_update_gate_gives_norm_zero_or_one(self):
        # recurrent weights zero: h' = (1 - z) n + z h with n independent of h, so the Jacobian is z * I: 0 when z -> 0 and 1 when z -> 1
        for bias, expected in ((-30.0, 0.0), (30.0, 1.0)):
            model = make_arm('gru', 'perm4', 16, 1)
            with torch.no_grad():
                model.rnn.weight_hh_l0.zero_()
                model.rnn.bias_hh_l0.zero_()
                model.rnn.bias_ih_l0[16:32].fill_(bias)           # the z gate rows
            norms = sg.jacobian_norms(model, tokens(8, batch=2), [3, 6])
            for t in (3, 6):
                self.assertTrue(all(abs(v - expected) < 1e-6 for v in norms[t]), (bias, norms))

    def test_core_jacobian_predicts_a_small_finite_difference_kick(self):
        model = make_arm('core', 'perm4', 32, 1).double().eval()
        x = tokens(12, batch=1)
        t = 7
        norms = sg.jacobian_norms(model, x, [t])
        self.assertTrue(np.isfinite(norms[t][0]) and norms[t][0] > 0)
        # first-order check: kicking the state at t by a small delta changes the NEXT state by about (norm-bounded) J delta
        with torch.no_grad():
            e, _, _ = model.encode(x)
            mk = model.rope(model.split(model.mk(model.mn(e))))
            mv = model.split(model.mv(model.mn(e)))
            s = None
            for step in range(t):
                _, s = model.decode(e[:, step], s, mk[:, :, :step + 1], mv[:, :, :step + 1], step)
            delta = 1e-7 * torch.randn(1, 32, dtype=torch.float64)
            base = model.decode(e[:, t], s, mk[:, :, :t + 1], mv[:, :, :t + 1], t)[1]['s']
            moved = model.decode(e[:, t], dict(s, s=s['s'] + delta), mk[:, :, :t + 1], mv[:, :, :t + 1], t)[1]['s']
        ratio = float((moved - base).norm() / delta.norm())
        self.assertLessEqual(ratio, norms[t][0] * (1 + 1e-4))          # |J delta| <= ||J||_2 |delta|


class SlopeAndSummary(unittest.TestCase):
    def test_slope_recovers_a_known_rate(self):
        t = np.arange(40, dtype=np.float64)
        values = np.stack([2.0 + 0.07 * t, -1.0 + 0.07 * t])
        self.assertAlmostEqual(sg.slope(values, 5, 35), 0.07, places=9)
        self.assertAlmostEqual(sg.slope(-values, 0, 40), -0.07, places=9)

    def test_measure_runs_end_to_end_on_a_saved_arm(self):
        from horizon import depth_run as dr
        with tempfile.TemporaryDirectory() as d:
            for arm, width in (('core', 32), ('gru', 16), ('lstm', 16)):
                out = Path(d) / f'{arm}.json'
                dr.run('perm3', arm, width, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
                r = sg.measure(out.with_suffix('.pt'), steps=60, t0=4, eps_list=[0.01], n=4, seed=1)
                row = r['results'][0.01]
                self.assertGreaterEqual(len(row['windows']), 2, arm)
                for v in row['windows'].values():
                    self.assertTrue(np.isfinite(v['growth_per_step']), arm)
                self.assertIn('growth', sg.describe(r))


if __name__ == '__main__':
    unittest.main()
