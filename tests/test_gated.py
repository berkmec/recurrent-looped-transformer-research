"""U2c (package 'kanal C'): a learned HOLD/WRITE gate over the immutable Core, written before the implementation.

Contract: the committed state s_t is either the Core's candidate (WRITE) or exactly the previous committed state (HOLD).
Modes: soft (sigmoid mix, differentiable, not exact), hard_st (hard decision, straight-through gradient to the gate head),
oracle (run-aware neutral mask, k neutrals tolerated — the U2b ceiling), and the same hard forward is used by the
'supervised' training arm whose auxiliary loss is gate_loss. Readout is from the committed state."""
import unittest

import torch


class GatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import gated
        except ImportError as e:
            self.fail('horizon.gated is not implemented: ' + str(e))
        return gated

    def make(self):
        from core.model import Core
        torch.manual_seed(21)
        return Core(width=8, heads=1, window=4, classes=5)

    def test_open_gate_matches_core_and_oracle_matches_runaware_freeze(self):
        gd = self.api(); core = self.make()
        from horizon import interventions_runaware as ivr
        x = torch.tensor([[0, 1, 2, 2, 2, 2, 1, 0, 2, 1], [2, 2, 2, 0, 1, 1, 2, 2, 2, 2]])
        g = gd.GatedCore(core, mode='soft'); g.open_all()
        logits, gates, _ = g(x)
        self.assertEqual(tuple(gates.shape), tuple(x.shape))
        self.assertTrue(bool((gates > 0.999).all()))
        torch.testing.assert_close(logits, core(x), rtol=1e-4, atol=1e-5)
        o = gd.GatedCore(core, mode='oracle', k=2)
        logits_o, gates_o, _ = o(x)
        ref, _ = ivr.run(core, x, k=2)
        torch.testing.assert_close(logits_o, ref, rtol=1e-5, atol=1e-6)
        self.assertEqual(gates_o.tolist(), (ivr.run_length(x) <= 2).float().tolist())

    def test_hard_gate_holds_exactly_and_straight_through_reaches_the_gate_head(self):
        gd = self.api(); core = self.make()
        x = torch.tensor([[0, 1, 2, 2, 2, 1, 0, 2]])
        g = gd.GatedCore(core, mode='hard_st'); g.close_all()
        logits, gates, gate_logits = g(x)
        self.assertTrue(bool((gates == 0).all()))
        states = g.last_states
        for t in range(1, x.shape[1]):
            self.assertTrue(torch.equal(states[:, t], states[:, 0]))  # exact HOLD, bitwise
        loss = logits[:, -1].square().mean()
        loss.backward()
        head_grads = [p.grad for p in g.gate_parameters() if p.grad is not None]
        self.assertTrue(head_grads and any(float(p.abs().sum()) > 0 for p in head_grads))

    def test_gate_loss_and_rates_against_the_runaware_target(self):
        gd = self.api()
        x = torch.tensor([[0, 2, 2, 2, 2, 1]])
        target = gd.gate_target(x, k=2)  # 1 = WRITE expected, 0 = HOLD expected (3rd+ consecutive neutral)
        self.assertEqual(target.tolist(), [[1., 1., 1., 0., 0., 1.]])
        perfect = torch.tensor([[9., 9., 9., -9., -9., 9.]])
        self.assertLess(float(gd.gate_loss(perfect, target)), 1e-3)
        wrong = -perfect
        self.assertGreater(float(gd.gate_loss(wrong, target)), 1.)
        rates = gd.gate_rates(torch.sigmoid(torch.tensor([[9., -9., 9., 9., -9., 9.]])), target)
        self.assertAlmostEqual(rates['false_hold'], 1 / 4)   # one of four WRITE positions held
        self.assertAlmostEqual(rates['false_write'], 1 / 2)  # one of two HOLD positions written


if __name__ == '__main__':
    unittest.main()
