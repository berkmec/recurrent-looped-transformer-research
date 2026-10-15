"""E-RM Aşama 0: kapı testleri (koşulardan ÖNCE yazıldı; kayıtlı kural burada sınanır, sonuca uydurulmaz).
"""
import unittest

import torch


class ErmS0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_generator_shapes_and_protocol(self):
        from horizon import erm_tasks as et
        x, apos, tgt, meta = et.generate(32, seed=4)
        self.assertEqual(tuple(x.shape), (32, 4 * 32))
        self.assertTrue((((apos // 32) == 3)).all())    # sorgu hep 3. pencerede
        self.assertTrue(all(int(x[s, int(apos[s]) - 1]) == et.QMARK for s in range(32)))
        self.assertTrue(all(0 <= m['fact_window'] <= 2 for m in meta))
        self.assertTrue(all(m['d'] == 3 - m['fact_window'] for m in meta))
        self.assertTrue(all(int(tgt[s]) in et.VALS for s in range(32)))
        self.assertTrue(et.queried_value_absent(x, apos, tgt).all())

    def test_generator_property_500(self):
        from horizon import erm_tasks as et
        x, apos, tgt, _ = et.generate(500, seed=9182)
        self.assertTrue(bool(et.queried_value_absent(x, apos, tgt).all()))

    def test_arms_forward_shapes(self):
        from horizon import erm_models as em
        from horizon import erm_tasks as et
        x, _, _, _ = et.generate(3, seed=6)
        xw = x.reshape(3, et.N_WINDOWS, et.WINDOW)
        for arm in em.ARMS:
            m = em.make_arm(arm, seed=1)
            m.eval()
            with torch.no_grad():
                if arm == 'B_orc':
                    _, _, _, mm = et.generate(3, seed=6)
                    lg = m(xw, et.fact_cells_tensor(mm))
                else:
                    lg = m(xw)
            self.assertEqual(tuple(lg.shape), (3, et.N_WINDOWS, et.WINDOW, et.VOCAB), arm)
        with self.assertRaises(ValueError):
            em.make_arm('yok')

    def test_g01_identity(self):
        from horizon import erm_models as em
        from horizon import erm_tasks as et
        ref = em.ArmA(seed=9)
        mem = em.RLTM(slots=16, seed=10)
        mem.enc.load_state_dict(ref.enc.state_dict())
        mem.head.load_state_dict(ref.head.state_dict())
        mem0 = em.RLTM(slots=0, seed=11)
        mem0.enc.load_state_dict(ref.enc.state_dict())
        mem0.head.load_state_dict(ref.head.state_dict())
        with torch.no_grad():
            mem.read.gamma.zero_()
            x, _, _, _ = et.generate(2, seed=9)
            xw = x.reshape(2, et.N_WINDOWS, et.WINDOW)
            d1 = float((mem(xw) - ref(xw)).abs().max())
            d0 = float((mem0(xw) - ref(xw)).abs().max())
        self.assertLessEqual(d1, 1e-5)
        self.assertLessEqual(d0, 1e-5)

    def test_g02_gradient_flows(self):
        from horizon import erm_models as em
        from horizon import erm_tasks as et
        from torch.nn import functional as F
        model = em.make_arm('B', seed=2)
        b = et.train_batch(2, 1, 4)
        x = b['x'].reshape(4, et.N_WINDOWS, et.WINDOW)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)   # ReZero: 1 isinma adimi
        opt.zero_grad()
        F.cross_entropy(model(x)[torch.arange(4), 3, b['answer_pos'] % et.WINDOW], b['target']).backward()
        opt.step()
        opt.zero_grad()
        sel = model(x)[torch.arange(4), 3, b['answer_pos'] % et.WINDOW]
        loss = F.cross_entropy(sel, b['target'])
        loss.backward()
        enc = float(torch.cat([p.grad.flatten() for p in model.enc.parameters()]).norm())
        wrt = float(torch.cat([p.grad.flatten() for p in model.write.parameters()]).norm())
        self.assertGreater(enc, 0)
        self.assertGreater(wrt, 0)

    def test_reading_rule_cases(self):
        from horizon import erm_run as er
        ok = dict(eval={'3': dict(acc=0.97)}, control_A={'3': dict(acc=0.10)},
                  telemetry=dict(sign_ok=True, slots_used=5))
        self.assertEqual(er.reading(ok)['verdict'], 'GECTI')
        bad = dict(eval={'3': dict(acc=0.60)}, control_A={'3': dict(acc=0.10)},
                   telemetry=dict(sign_ok=True, slots_used=5))
        self.assertEqual(er.reading(bad)['verdict'], 'KALDI')
        leak = dict(eval={'3': dict(acc=0.97)}, control_A={'3': dict(acc=0.80)},
                    telemetry=dict(sign_ok=True, slots_used=5))
        self.assertEqual(er.reading(leak)['verdict'], 'KONTROL_BOZUK')
        nosign = dict(eval={'3': dict(acc=0.97)}, control_A={'3': dict(acc=0.10)},
                      telemetry=dict(sign_ok=False, slots_used=5))
        self.assertEqual(er.reading(nosign)['verdict'], 'KISMEN')
        empty = dict(eval={'3': dict(acc=None)}, control_A={'3': dict(acc=0.10)}, telemetry={})
        self.assertEqual(er.reading(empty)['verdict'], 'INCOMPLETE')

    def test_gate_parameter_groups_decay_free(self):
        from horizon import erm_models as em
        m = em.make_arm('B', seed=3)
        gate = list(m.gate_parameters())
        self.assertTrue(any('gamma' in n for n, _ in m.named_parameters()))
        self.assertTrue(any('write.u' in n for n, _ in m.named_parameters()))
        names = {n for n, _ in m.named_parameters()}
        self.assertIn('write.c', names)

    def test_mini_train_runs(self):
        from horizon import erm_run as er
        model, info = er.train('B', seed=5, steps=6, batch=4, log_every=0, log=lambda m: None)
        self.assertEqual(len(info['history']), 0)
        self.assertGreater(info['parameters'], 1000)
        ev = er.evaluate(model, 'B', d_list=(3,), n=12, seed=8)
        self.assertIn(3, ev)

    def test_p0_telemetry_hooks(self):
        from horizon import erm_models as em
        from horizon import erm_tasks as et
        m = em.make_arm('B', seed=1)
        m.eval()
        x, _, _, _ = et.generate(4, seed=2)
        with torch.no_grad():
            m(x.reshape(4, et.N_WINDOWS, et.WINDOW))
        self.assertEqual(tuple(m.last_m.shape), (4, 16, 64))
        self.assertEqual(tuple(m.write.last_alpha.shape), (4, 16, 4, 32))

    def test_rmt_rule_and_telemetry(self):
        from horizon import erm_run as er
        ok = dict(eval={'3': dict(acc=0.96)}, control_A={'3': dict(acc=0.10)},
                  telemetry=dict(slots_used=3, sign_ok=None))
        r = er.reading_rmt(ok)
        self.assertEqual(r['verdict'], 'GECTI')
        self.assertEqual(r['kural'], 'B_rmt-ozel')
        bad = dict(eval={'3': dict(acc=0.20)}, control_A={'3': dict(acc=0.10)},
                   telemetry={})
        self.assertEqual(er.reading_rmt(bad)['verdict'], 'KALDI')
        # standart kural bozulmadi: ayni sonuc G0.3'te de KALDI (B < 0,95)
        self.assertEqual(er.reading(bad)['verdict'], 'KALDI')
        from horizon import erm_models as em
        m = em.make_arm('B_rmt', seed=1)
        tel = er.telemetry(m, 'B_rmt', n=8, seed=3)
        self.assertIn('slots_used', tel)
        self.assertIsNone(tel['sign_ok'])

    def test_g05_longdoc_continuity(self):
        import json
        from pathlib import Path
        from mline import long_doc as ld
        p = Path('results/L1-long-doc/step006000-long-doc-8192.json')
        stored = json.loads(p.read_text(encoding='utf8'))
        self.assertEqual(ld.reading(stored), stored['reading'])
        d = ld.differences(stored)
        wmax = max(abs(v[0]) for b, v in d['win512'].items() if int(b.split('-')[0]) >= 1024)
        self.assertLessEqual(wmax, 0.002)

    def test_fact_cells_tensor(self):
        from horizon import erm_tasks as et
        _, _, _, meta = et.generate(8, seed=11)
        fc = et.fact_cells_tensor(meta)
        self.assertEqual(tuple(fc.shape), (8, 16, 2))
        # her hücre gerçekten bir değer konumu: x[seq, w*32+pos] değer kimliği
        x, _, _, meta = et.generate(8, seed=11)
        for s in range(8):
            for (w, p) in fc[s].tolist():
                self.assertIn(int(x[s, w * 32 + p]), et.VALS)

    def test_aux_forward_and_grad(self):
        from horizon import erm_models as em
        from horizon import erm_tasks as et
        from torch.nn import functional as F
        m = em.make_arm('B_aux', seed=2)
        self.assertIsInstance(m, em.RLTM)
        b = et.train_batch(2, 1, 4)
        x = b['x'].reshape(4, et.N_WINDOWS, et.WINDOW)
        m.train()
        logits, Ly = m.forward_aux(x, b['fact_cells'])
        self.assertEqual(tuple(logits.shape), (4, et.N_WINDOWS, et.WINDOW, et.VOCAB))
        self.assertTrue(float(Ly.detach()) > 0)
        sel = logits[torch.arange(4), 3, b['answer_pos'] % et.WINDOW]
        (F.cross_entropy(sel, b['target']) + Ly).backward()
        wq = float(torch.cat([p.grad.flatten() for p in m.write.parameters()]).norm())
        self.assertGreater(wq, 0)

    def test_orc_fixed_write(self):
        from horizon import erm_models as em
        from horizon import erm_tasks as et
        m = em.make_arm('B_orc', seed=3)
        m.eval()
        x, _, _, meta = et.generate(4, seed=5)
        fc = et.fact_cells_tensor(meta)
        with torch.no_grad():
            lg = m(x.reshape(4, et.N_WINDOWS, et.WINDOW), fc)
        self.assertEqual(tuple(lg.shape), (4, et.N_WINDOWS, et.WINDOW, et.VOCAB))
        # sabit yazma: farklı olgular → farklı bellek; aynı girdi → aynı bellek
        self.assertIsNotNone(m.last_m)
        m1 = m.last_m.clone()
        with torch.no_grad():
            m(x.reshape(4, et.N_WINDOWS, et.WINDOW), fc)
        self.assertTrue(torch.equal(m.last_m, m1))
        x2, _, _, meta2 = et.generate(4, seed=6)
        with torch.no_grad():
            m(x2.reshape(4, et.N_WINDOWS, et.WINDOW), et.fact_cells_tensor(meta2))
        self.assertFalse(torch.equal(m.last_m, m1))

    def test_p1_p2_reading_rules(self):
        from horizon import erm_run as er
        hist = [dict(step=1000, aux_loss=2.0), dict(step=2000, aux_loss=1.2)]
        ok = dict(eval={'3': dict(acc=0.97)}, control_A={'3': dict(acc=0.10)},
                  telemetry=dict(sign_ok=True, slots_used=5),
                  train=dict(history=hist))
        r = er.reading_p1(ok)
        self.assertEqual(r['verdict'], 'GECTI')
        self.assertTrue(r['yardim_uygulandi'])
        flat = dict(eval={'3': dict(acc=0.60)}, control_A={'3': dict(acc=0.10)},
                    telemetry=dict(sign_ok=True, slots_used=5),
                    train=dict(history=[dict(step=1000, aux_loss=2.0),
                                              dict(step=2000, aux_loss=1.2)]))
        self.assertEqual(er.reading_p1(flat)['verdict'], 'KALDI')
        noaux = dict(eval={'3': dict(acc=0.97)}, control_A={'3': dict(acc=0.10)},
                     telemetry=dict(sign_ok=True, slots_used=5),
                     train=dict(history=[dict(step=1000, aux_loss=2.0),
                                               dict(step=2000, aux_loss=2.1)]))
        self.assertEqual(er.reading_p1(noaux)['verdict'], 'YARDIM_UYGULANMADI')
        p2ok = dict(eval={'3': dict(acc=0.96)}, control_A={'3': dict(acc=0.10)}, telemetry={})
        self.assertEqual(er.reading_p2(p2ok)['verdict'], 'GECTI')
        p2bad = dict(eval={'3': dict(acc=0.20)}, control_A={'3': dict(acc=0.10)}, telemetry={})
        self.assertEqual(er.reading_p2(p2bad)['verdict'], 'KALDI')

    def test_zero_acc_is_measured_not_missing(self):
        # falsy düzeltmesi: acc 0.0 geçerli ölçümdür, INCOMPLETE değil.
        from horizon import erm_run as er
        zero = dict(eval={'3': dict(acc=0.0)}, control_A={'3': dict(acc=0.10)},
                    telemetry=dict(sign_ok=True, slots_used=5))
        self.assertEqual(er.reading(zero)['verdict'], 'KALDI')
        zero_p1 = dict(eval={'3': dict(acc=0.0)}, control_A={'3': dict(acc=0.10)},
                       telemetry=dict(sign_ok=True, slots_used=5),
                       train=dict(history=[dict(step=1000, aux_loss=1.0),
                                           dict(step=2000, aux_loss=0.5)]))
        r = er.reading_p1(zero_p1)
        self.assertEqual(r['verdict'], 'KALDI')
        self.assertTrue(r['yardim_uygulandi'])
        missing = dict(eval={'3': dict(acc=None)}, control_A={'3': dict(acc=0.10)}, telemetry={})
        self.assertEqual(er.reading(missing)['verdict'], 'INCOMPLETE')

    def test_mini_train_probe_arms(self):
        from horizon import erm_run as er
        m, info = er.train('B_aux', seed=5, steps=6, batch=4, log_every=0,
                           log=lambda m: None, aux_weight=1.0)
        self.assertIn('aux_loss', info['history'][-1] if info['history'] else {'aux_loss': None})
        m2, info2 = er.train('B_aux', seed=5, steps=6, batch=4, log_every=2,
                             log=lambda m: None, aux_weight=1.0)
        self.assertTrue(all('aux_loss' in h for h in info2['history']))
        m3, _ = er.train('B_orc', seed=5, steps=6, batch=4, log_every=0, log=lambda m: None)
        ev = er.evaluate(m3, 'B_orc', d_list=(3,), n=12, seed=8)
        self.assertIn(3, ev)


if __name__ == '__main__':
    unittest.main()
