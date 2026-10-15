"""E-RM S0 ölçü-onarım zeyli (S0-Z1..Z6): G0.2 cetveli, DIVERGED, hücre-n, eşik-sınırları (float32), --kural.

Eşik DEĞERLERİ değişmedi; testler yalnız onarımı ve sınır davranışını sabitler. Kapı ölçümü (gate_g02 sonucu) burada
koşulmaz — o bir ölçümdür (yürürlük + onay god'da); burada yalnız cetvel (fd_check) bilinen fonksiyonlarda sınanır.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from horizon import erm_run as er
from horizon import erm_tasks as et


def _f32(v):
    return float(np.float32(v))   # acc'ler float32 ortalaması olarak saklanır


def _res(b3, a3=0.10, sign_ok=True, slots=4, hist=None, nonfinite=0):
    r = {'eval': {'3': {'n': 200, 'acc': b3, 'nonfinite': nonfinite}}, 'control_A': {'3': {'acc': a3}},
         'telemetry': {'sign_ok': sign_ok, 'slots_used': slots}}
    if hist is not None:
        r['train'] = {'history': hist}
    return r


class G02RulerTests(unittest.TestCase):
    def test_fd_check_matches_known_gradient_float64(self):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(5, 3), nn.Tanh(), nn.Linear(3, 1)).double()
        x = torch.randn(7, 5, dtype=torch.float64)
        loss_fn = lambda: (m(x) ** 2).mean() + 2.0
        loss_fn().backward()
        fd = er.fd_check(m, loss_fn, ['0.weight', '2.bias'])
        for v in fd.values():
            self.assertTrue(v['resolved'])
            self.assertLess(v['rel_err'], 1e-6)
            self.assertNotEqual(v['analytic'], 0.0)

    def test_fd_check_picks_largest_gradient_index(self):
        w = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        holder = nn.Module(); holder.w = w
        coef = torch.tensor([0.0, 1.0, -5.0, 2.0], dtype=torch.float64)
        loss_fn = lambda: (coef * holder.w).sum() + (holder.w ** 2).sum()
        loss_fn().backward()
        fd = er.fd_check(holder, loss_fn, ['w'])
        self.assertEqual(fd['w']['index'], [2])               # |-5| en büyük; eski kapı indeks 0'a (gradyan 0) bakıyordu
        self.assertAlmostEqual(fd['w']['analytic'], -5.0)

    def test_fd_check_reports_unresolvable_float32(self):
        # eski arıza sınıfı: float32'de L≈2 iken gradyan 1e-9 → beklenen fark yuvarlama altında; 'resolved' False olmalı
        w = nn.Parameter(torch.zeros(3))
        holder = nn.Module(); holder.w = w
        loss_fn = lambda: 1e-9 * holder.w.sum() + 2.0
        loss_fn().backward()
        self.assertFalse(er.fd_check(holder, loss_fn, ['w'])['w']['resolved'])

    def test_arm_b_runs_in_float64(self):
        # gate_g02'nin float64 kopyası için dtype uyumu (yalnız ileri geçiş; kapı ölçülmez)
        model = er.em.make_arm('B', seed=13)
        b = et.train_batch(13, 1, 2)
        with er.float64():
            model = model.double()
            out = model(b['x'].reshape(2, et.N_WINDOWS, et.WINDOW))
        self.assertEqual(out.dtype, torch.float64)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(torch.get_default_dtype(), torch.float32)   # bağlam varsayılanı geri yükler


class DivergedTests(unittest.TestCase):
    NAN_HIST = [{'step': 1000, 'loss': 2.08, 'aux_loss': 0.008}, {'step': 2000, 'loss': float('nan'), 'aux_loss': float('nan')}]

    def test_nan_history_is_diverged_in_every_rule(self):
        r = _res(0.0, hist=self.NAN_HIST)
        for rule in (er.reading, er.reading_p1, er.reading_p2, er.reading_rmt):
            self.assertEqual(rule(dict(r))['verdict'], 'DIVERGED', rule.__name__)

    def test_nonfinite_eval_logits_is_diverged(self):
        self.assertEqual(er.reading(_res(0.0, nonfinite=81))['verdict'], 'DIVERGED')

    def test_finite_runs_unchanged(self):
        ok_hist = [{'step': 1000, 'loss': 2.0, 'aux_loss': 0.5}, {'step': 6000, 'loss': 0.1, 'aux_loss': 0.1}]
        self.assertEqual(er.reading(_res(0.97, hist=ok_hist))['verdict'], 'GECTI')
        self.assertEqual(er.reading(_res(0.10, hist=ok_hist))['verdict'], 'KALDI')
        self.assertTrue(er._aux_applied({'train': {'history': ok_hist}}))

    def test_aux_applied_ignores_nonfinite(self):
        self.assertIsNone(er._aux_applied({'train': {'history': self.NAN_HIST}}))   # tek sonlu değer: karar verilemez

    @unittest.skipUnless(Path('results/ERM-S0/erm-P1.json').exists() and Path('results/ERM-S0/erm-P1b.json').exists(), 'P1 sonuçları diskte yok')
    def test_recorded_p1_p1b_reread_as_diverged(self):
        for f in ('results/ERM-S0/erm-P1.json', 'results/ERM-S0/erm-P1b.json'):
            r = json.loads(Path(f).read_text(encoding='utf8'))
            self.assertEqual(er.reading_p1(r)['verdict'], 'DIVERGED', f)


class BoundaryTests(unittest.TestCase):
    """Kayıtlı eşikler aynen: B ≥ 0,95; A ≤ 0,30; A > 0,50 bozuk; yuva ≥ 2 — float32 saklamada da."""

    def test_b_exactly_095(self):
        self.assertEqual(er.reading(_res(_f32(190 / 200)))['verdict'], 'GECTI')     # 0.949999988 eskiden KALDI'ydı
        self.assertEqual(er.reading(_res(_f32(189 / 200)))['verdict'], 'KALDI')

    def test_a_exactly_030(self):
        self.assertEqual(er.reading(_res(0.97, a3=_f32(60 / 200)))['verdict'], 'GECTI')
        self.assertEqual(er.reading(_res(0.97, a3=_f32(61 / 200)))['verdict'], 'KISMEN')

    def test_a_exactly_050_is_not_broken(self):
        self.assertEqual(er.reading(_res(0.97, a3=_f32(100 / 200)))['verdict'], 'KISMEN')   # > 0,50 değil
        self.assertEqual(er.reading(_res(0.97, a3=_f32(101 / 200)))['verdict'], 'KONTROL_BOZUK')

    def test_slots_exactly_2(self):
        self.assertEqual(er.reading(_res(0.97, slots=2))['verdict'], 'GECTI')
        self.assertEqual(er.reading(_res(0.97, slots=1))['verdict'], 'KISMEN')

    def test_acc_only_rules_share_boundaries(self):
        for rule in (er.reading_rmt, er.reading_p2):
            self.assertEqual(rule(_res(_f32(190 / 200)))['verdict'], 'GECTI')
            self.assertEqual(rule(_res(0.97, a3=_f32(101 / 200)))['verdict'], 'KONTROL_BOZUK')


class EvalCellTests(unittest.TestCase):
    def test_default_draw_is_the_old_draw(self):
        for d in (1, 2, 3):
            x, apos, tgt, meta = et.generate(200, seed=777 * 131 + d)
            keep = torch.tensor([m['d'] == d for m in meta])
            xs, aps, tgs, ms = er._cell_draw(200, d, 777)
            self.assertTrue(torch.equal(xs, x[keep]) and torch.equal(tgs, tgt[keep]) and torch.equal(aps, apos[keep]))
            self.assertLess(len(xs), 200)   # kayıtlı n=200/hücre DEĞİL (65/55/81): S0-Z3

    def test_per_cell_exact_n_with_old_prefix(self):
        for d in (1, 2, 3):
            old, _, _, _ = er._cell_draw(200, d, 777)
            new, _, _, meta = er._cell_draw(200, d, 777, per_cell=True)
            self.assertEqual(len(new), 200)
            self.assertTrue(all(m['d'] == d for m in meta))
            self.assertTrue(torch.equal(new[:len(old)], old))


class CliTests(unittest.TestCase):
    def test_unknown_kural_is_an_error(self):
        with mock.patch.object(sys, 'argv', ['erm_run', 'g03', '--b', 'x.json', '--a', 'y.json', '--kural', 'g3']):
            with self.assertRaises(SystemExit):
                er.main()


if __name__ == '__main__':
    unittest.main()
