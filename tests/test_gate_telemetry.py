"""A3 gate telemetry: row schema, sign convention, RNG/numeric neutrality (bit identity), gradient no-write,
K3 label branches, K4 bands, K5 validity and the R0 compare helper.

Written before the A3 runs (pre-registration); the registered rules are tested here, not fitted to results.
"""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


class GateTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        from horizon import gate_telemetry as gt
        return gt

    def build(self, n_steps, g_at=None, p_at=None, h_at=None, a_at=None, applied_at=None):
        rows = []
        for step in range(1, n_steps + 1):
            applied = True if applied_at is None else bool(applied_at(step))
            g = 1.0 if g_at is None else g_at(step)
            p = 0.05 if p_at is None else p_at(step)
            h = 0.01 if h_at is None else h_at(step)
            a = 0.1 if a_at is None else a_at(step)
            rows.append(dict(step=step, stage_len=16, loss=2.0, applied=applied, clipped=False,
                             G=(g if applied else None), P=(p if applied else None), A=(a if applied else None), H=h))
        return rows

    def test_pressure_sign_convention(self):
        gt = self.api()
        p, a = gt.pressure_from_bias(torch.full((8,), -1.0))   # dL/db < 0: the loss wants b larger -> gate opens
        self.assertAlmostEqual(p, 1.0, places=12)
        self.assertAlmostEqual(a, 1.0, places=12)
        p2, a2 = gt.pressure_from_bias(torch.full((4,), 0.25))
        self.assertAlmostEqual(p2, -0.25, places=12)
        self.assertAlmostEqual(a2, 0.25, places=12)

    def make_model(self):
        from horizon import depth_models
        return depth_models.make_arm('hold', 'perm4', 32, 1)

    def test_read_gate_row_values_and_no_write(self):
        gt = self.api()
        model = self.make_model()
        x = torch.randint(0, 4, (4, 6))
        y = torch.randint(0, 4, (4, 6))
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        before = {id(p): p.grad.detach().clone() for p in model.parameters() if p.grad is not None}
        row = gt.read_gate_row(model, applied=True)
        head = model.head
        manual_g = float(torch.sqrt(head.weight.grad.detach().double().square().sum() + head.bias.grad.detach().double().square().sum()))
        self.assertAlmostEqual(row['G'], manual_g, places=10)
        self.assertAlmostEqual(row['P'], -float(head.bias.grad.detach().double().mean()), places=10)
        self.assertAlmostEqual(row['A'], float(head.bias.grad.detach().abs().double().mean()), places=10)
        self.assertAlmostEqual(row['H'], float(model.last_gates.detach().mean()), places=6)
        for p in model.parameters():                      # acceptance 5: the read never writes to the gradients
            if p.grad is not None:
                self.assertTrue(torch.equal(p.grad, before[id(p)]))
        # applied=False: gradientless row, H still there
        row2 = gt.read_gate_row(model, applied=False)
        self.assertIsNone(row2['G']) and self.assertIsNone(row2['P']) and self.assertIsNone(row2['A'])
        self.assertIsNotNone(row2['H'])

    def test_flag_off_is_bit_identical_and_rows_schema(self):
        from horizon import depth_run as dr
        kw = dict(batch=8, curriculum=[2], neutral_p=0.5, log_every=6)
        model_a, trained_a = dr.train('perm4', 'hold', 32, 3, 12, **kw)
        with tempfile.TemporaryDirectory() as tmp:
            tel = Path(tmp) / 'x.tel.jsonl'
            model_b, trained_b = dr.train('perm4', 'hold', 32, 3, 12, gate_telemetry=str(tel), **kw)
            self.assertEqual(trained_a['history'], trained_b['history'])
            self.assertEqual(trained_a['guard'], trained_b['guard'])
            for (k1, v1), (k2, v2) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
                self.assertEqual(k1, k2)
                self.assertTrue(torch.equal(v1, v2), f'weights differ at {k1}')
            rows, malformed = gate_telemetry_rows(tel)
        self.assertEqual(malformed, 0)
        self.assertEqual(len(rows), 12)
        for r in rows:
            self.assertEqual(set(r.keys()), set(self.api().ROW_KEYS))
        self.assertTrue(all(r['H'] is not None for r in rows))

    def test_windows_and_sources(self):
        gt = self.api()
        rows = self.build(500)
        s = gt.summarize(rows)
        self.assertEqual(s['n_steps'], 500)
        self.assertEqual([w['w'] for w in s['windows']], [1, 2])
        self.assertAlmostEqual(s['windows'][0]['L_w'], 2.0, places=9)
        self.assertAlmostEqual(s['G0'], 1.0, places=9)
        self.assertAlmostEqual(s['G_end'], 1.0, places=9)
        self.assertAlmostEqual(s['G_ratio'], 1.0, places=9)
        self.assertAlmostEqual(s['H_init'], 0.01, places=9)
        self.assertAlmostEqual(s['H_final'], 0.01, places=9)
        self.assertFalse(s['PRESSURE_DEAD'])

    def test_label_branches(self):
        gt = self.api()
        self.assertEqual(gt.summarize(self.build(6000, g_at=lambda s: 1.0 if s <= 250 else 0.01))['label'], 'SÖNÜM')
        self.assertEqual(gt.summarize(self.build(6000, h_at=lambda s: 0.6))['label'], 'AÇILMA')
        osc = self.build(6000, p_at=lambda s: 0.05 if ((s - 1) // 250) % 2 == 0 else -0.05)
        s = gt.summarize(osc)
        self.assertEqual(s['label'], 'KISIR-DÖNGÜ')
        self.assertAlmostEqual(s['F'], 1.0, places=9)
        self.assertEqual(gt.summarize(self.build(6000, p_at=lambda s: -0.05))['label'], 'YÖNLÜ-KİLİT')
        self.assertEqual(gt.summarize(self.build(6000, h_at=lambda s: 0.2))['label'], 'KARIŞIK')
        missing = self.build(6000, applied_at=lambda s: False)
        for r in missing[:800]:                          # > 10 % rows without gradients while applied=True
            r['applied'] = True
        self.assertEqual(gt.summarize(missing)['label'], 'INCOMPLETE')
        self.assertEqual(gt.summarize(self.build(6000, g_at=lambda s: 0.0))['label'], 'INCOMPLETE')

    def test_k5_and_bands(self):
        gt = self.api()
        s = gt.summarize(self.build(6000), r3_loss=0.9)          # R3 did NOT learn: K5(c) fails -> INCOMPLETE
        self.assertFalse(s['k5']['passed'])
        self.assertEqual(s['label'], 'INCOMPLETE')
        s2 = gt.summarize(self.build(6000, applied_at=lambda s: s > 61))    # 1.02 % gradless rows: K5(d) fails
        self.assertFalse(s2['k5']['passed'])
        s3 = gt.summarize(self.build(6000, applied_at=lambda s: s > 50))    # 0.83 % gradless rows: still valid
        self.assertTrue(s3['k5']['applied_false_ok'])
        band_g = gt.summarize(self.build(6000, g_at=lambda s: 0.11 if s > 5500 else 1.0))
        self.assertIn('G_ratio', band_g['bands'])
        self.assertTrue(band_g['needs_seed2'])
        self.assertEqual(band_g['label'], 'YÖNLÜ-KİLİT')         # 0.11 >= 0.10 keeps it live; H_final 0.01, F 0 -> one-way lock
        band_h = gt.summarize(self.build(6000, h_at=lambda s: 0.45))
        self.assertIn('H_final', band_h['bands'])

    def test_compare_runs(self):
        gt = self.api()
        run = dict(train=dict(steps=300, history=[dict(step=1, loss=2.3, grad_norm=1.0)], guard=dict(steps=300, skipped=0)))
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.json', Path(tmp) / 'b.json'
            a.write_text(json.dumps(run), encoding='utf8')
            b.write_text(json.dumps(run), encoding='utf8')
            self.assertTrue(gt.compare_runs(a, b)['equal'])
            other = json.loads(json.dumps(run))
            other['train']['history'][0]['loss'] = 2.3000001
            c = Path(tmp) / 'c.json'
            c.write_text(json.dumps(other), encoding='utf8')
            self.assertFalse(gt.compare_runs(a, c)['equal'])


def gate_telemetry_rows(path):
    from horizon import gate_telemetry as gt
    return gt.read_jsonl(path)
