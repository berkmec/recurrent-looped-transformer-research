import unittest

import torch
from torch.nn import functional as F

from horizon import erm_r as er
from horizon import erm_tasks as et
from mline import rltm


def _model(gamma=1.0, memory=True, seed=3):
    return rltm.RLTM(rltm.small_core(et.VOCAB, width=32, heads=2, seed=seed), d_k=16, d_v=16, gamma_init=gamma, seed=seed + 7,
                     memory=memory).eval()


def _rows(b=2, n=3, w=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, et.VOCAB, (b, n, w), generator=g)


class RLTMTests(unittest.TestCase):
    def test_closed_or_absent_memory_is_the_core_window_by_window(self):     # K3
        xw = _rows()
        for m in (_model(gamma=0.0), _model(memory=False)):
            with torch.no_grad():
                out = m(xw)
                for j in range(xw.shape[1]):
                    self.assertLess(float((out[:, j] - m.core(xw[:, j])).abs().max()), 1e-5)
        m = _model()
        with torch.no_grad():
            self.assertLess(float((m(xw, read=False)[:, 1] - m.core(xw[:, 1])).abs().max()), 1e-5)

    def test_the_memory_holds_the_shifted_writes_of_the_earlier_windows(self):
        m, xw = _model(), _rows()
        with torch.no_grad():
            e = m.core.encode(xw.reshape(-1, xw.shape[2]))[0].reshape(*xw.shape, -1)
            M = m.memories(e)
            self.assertEqual(float(M[:, 0].abs().max()), 0.0)                  # window 0 reads an empty memory
            ref = torch.zeros(xw.shape[0], m.d_v, m.d_k)
            for i in range(1, xw.shape[2]):
                k = F.normalize(m.Wk(e[:, 0, i - 1]), dim=-1)
                ref = er.delta_write(ref, k, m.Wv(e[:, 0, i]), torch.sigmoid(m.wb(e[:, 0, i])))
            self.assertTrue(torch.allclose(M[:, 1], ref, atol=1e-6))

    def test_later_windows_see_earlier_ones_and_never_the_reverse(self):
        m, xw = _model(), _rows()
        changed = xw.clone(); changed[:, 0, 2:5] = (changed[:, 0, 2:5] + 1) % et.VOCAB
        later = xw.clone(); later[:, 2] = (later[:, 2] + 1) % et.VOCAB
        with torch.no_grad():
            base = m(xw)
            self.assertGreater(float((m(changed)[:, 1] - base[:, 1]).abs().max()), 1e-4)      # window 1 reads window 0
            self.assertLess(float((m(later)[:, :2] - base[:, :2]).abs().max()), 1e-6)        # no window reads a later one
            off = _model(memory=False)
            self.assertLess(float((off(changed)[:, 1] - off(xw)[:, 1]).abs().max()), 1e-6)

    def test_the_loss_of_a_later_window_reaches_the_write(self):           # G0.2
        m, xw = _model(), _rows()
        m.train()
        logits = m(xw)
        F.cross_entropy(logits[:, 1].reshape(-1, et.VOCAB), xw[:, 1].reshape(-1)).backward()
        for name, p in (('Wk', m.Wk.weight), ('Wv', m.Wv.weight), ('wb', m.wb.weight), ('Wq', m.Wq.weight), ('Wo', m.Wo.weight),
                        ('encoder', m.core.eq.weight)):
            self.assertGreater(float(p.grad.abs().max()), 0.0, name)

    def test_the_chunked_write_gives_the_same_output_and_gradients(self):
        xw = _rows(n=3, w=11)
        grads = []
        for chunk in (0, 4):
            m = _model()
            m.write_chunk = chunk
            m.train()
            out = m(xw)
            out.square().mean().backward()
            grads.append((out.detach(), [p.grad.clone() for p in (m.Wk.weight, m.Wv.weight, m.wb.weight, m.core.eq.weight)]))
        self.assertTrue(torch.allclose(grads[0][0], grads[1][0], atol=1e-6))
        for a, b in zip(grads[0][1], grads[1][1]):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-5))

    def test_the_chunkwise_parallel_write_is_the_loop(self):
        """chunked_write against the loop: values and gradients in float64 (chunks that do not divide the length, a nonzero start, a
        long chunk), and at the language shape in float32 (511 writes, d = 256, the trainer's chunk of 64)."""
        g = torch.Generator().manual_seed(11)
        for steps, d, chunk in ((23, 8, 5), (23, 8, 64), (40, 16, 16)):
            k = F.normalize(torch.randn(3, steps, d, generator=g, dtype=torch.float64), dim=-1).requires_grad_()
            v = torch.randn(3, steps, d, generator=g, dtype=torch.float64).requires_grad_()
            beta = torch.rand(3, steps, 1, generator=g, dtype=torch.float64).requires_grad_()
            M0 = torch.randn(3, d, d, generator=g, dtype=torch.float64).requires_grad_()
            w = torch.randn(3, d, d, generator=g, dtype=torch.float64)
            out = []
            for f in (lambda: rltm._write_chunk(M0, k, v, beta), lambda: rltm.chunked_write(M0, k, v, beta, chunk)):
                M = f()
                grads = torch.autograd.grad((M * w).sum(), (M0, k, v, beta))
                out.append((M.detach(), grads))
            self.assertLess(float((out[0][0] - out[1][0]).abs().max()), 1e-10, (steps, d, chunk))
            for a, b in zip(out[0][1], out[1][1]):
                self.assertLess(float((a - b).abs().max()), 1e-9, (steps, d, chunk))
        k = F.normalize(torch.randn(2, 511, 256, generator=g), dim=-1)
        v, beta = torch.randn(2, 511, 256, generator=g), torch.rand(2, 511, 1, generator=g)
        loop, ut = rltm._write_chunk(torch.zeros(2, 256, 256), k, v, beta), rltm.chunked_write(torch.zeros(2, 256, 256), k, v, beta, 64)
        self.assertLess(float((loop - ut).abs().max() / loop.abs().max()), 1e-4)

    def test_the_parallel_write_option_gives_the_same_model(self):
        xw = _rows(n=3, w=11)
        runs = []
        for ut in (0, 4):
            m = _model()
            m.ut_chunk = ut
            m.train()
            out = m(xw)
            out.square().mean().backward()
            runs.append((out.detach(), [p.grad.clone() for p in (m.Wk.weight, m.Wv.weight, m.wb.weight, m.Wq.weight, m.core.eq.weight)]))
        self.assertTrue(torch.allclose(runs[0][0], runs[1][0], atol=1e-6))
        for a, b in zip(runs[0][1], runs[1][1]):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-5))

    def test_a_zero_output_start_is_the_core_and_still_learns_the_read(self):
        xw = _rows()
        m = rltm.RLTM(rltm.small_core(et.VOCAB, width=32, heads=2, seed=3), d_k=16, d_v=16, gamma_init=1.0, seed=10, zero_out=True)
        with torch.no_grad():
            m.eval()
            for j in range(xw.shape[1]):
                self.assertLess(float((m(xw)[:, j] - m.core(xw[:, j])).abs().max()), 1e-5)        # exactly the Core at the start
        m.train()
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        F.cross_entropy(m(xw)[:, 1].reshape(-1, et.VOCAB), xw[:, 1].reshape(-1)).backward()
        self.assertGreater(float(m.Wo.weight.grad.abs().max()), 0.0)                              # the output learns at once
        self.assertEqual(float(m.Wk.weight.grad.abs().max()), 0.0)                                # the write waits one update
        opt.step(); opt.zero_grad()
        F.cross_entropy(m(xw)[:, 1].reshape(-1, et.VOCAB), xw[:, 1].reshape(-1)).backward()
        self.assertGreater(float(m.Wk.weight.grad.abs().max()), 0.0)

    def test_the_full_row_control_is_the_core_over_the_row(self):
        core, xw = rltm.small_core(et.VOCAB, width=32, heads=2), _rows()
        with torch.no_grad():
            self.assertTrue(torch.allclose(rltm.FullRow(core)(xw), core(xw.reshape(2, -1)).reshape(2, 3, 8, -1), atol=1e-6))

    def test_the_toy_arms_train_and_the_gate_reads(self):
        for arm in er.CORE_ARMS:
            model, info = er.train(arm, 1, 2, batch=4, log=lambda *_: None)
            self.assertEqual(set(er.evaluate(model, n=10, arm=arm)), {'all', '1', '2', '3'})
        self.assertIs(type(er.make('C_ds')), rltm.RLTM)
        self.assertTrue(er.make('C_ds').memory and not er.make('C_none').memory)
        g = er.core_gate
        self.assertEqual(g({'C_ds': 0.97, 'C_none': 0.13}), 'GATE PASSED')
        self.assertEqual(g({'C_ds': 0.97, 'C_none': 0.45}), 'CONTROL BROKEN (the windows without memory are above chance: the task leaks)')
        self.assertEqual(g({'C_ds': 0.6, 'C_none': 0.13}), 'NOT PASSED (positive control not run)')
        self.assertEqual(g({'C_ds': 0.6, 'C_none': 0.13, 'C_full': 0.99}), 'NOT PASSED (memory)')
        self.assertEqual(g({'C_ds': 0.6, 'C_full': 0.5}), 'VACUOUS (the full-row Core does not learn the task)')
        self.assertEqual(er.reading({'C_ds': dict(eval={'3': dict(acc=0.99)}), 'C_none': dict(eval={'3': dict(acc=0.12)})})['core_gate'],
                         'GATE PASSED')

    def test_it_wraps_only_the_sealed_core(self):
        with self.assertRaises(TypeError):
            rltm.RLTM(torch.nn.Linear(4, 4))
        self.assertTrue(torch.equal(rltm.delta_write(torch.zeros(1, 2, 3), torch.tensor([[1., 0, 0]]), torch.tensor([[2., 3]]), torch.ones(1, 1)),
                                    er.delta_write(torch.zeros(1, 2, 3), torch.tensor([[1., 0, 0]]), torch.tensor([[2., 3]]), torch.ones(1, 1))))


if __name__ == '__main__':
    unittest.main()
