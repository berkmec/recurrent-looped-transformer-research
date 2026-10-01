"""R6 route, actual-update and storage accounting acceptance tests."""
import copy
import importlib.util
import tempfile
from pathlib import Path
import unittest

import torch
from core.model import Core, generate


class MemoryBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('review_cycle.memory_bench'),
                             'R6 runner is not implemented')
        from review_cycle import memory_bench as m
        return m

    def test_equivalent_routes_match_loss_all_gradients_and_adamw(self):
        m = self.api()
        torch.manual_seed(1701)
        initial = Core(width=8, heads=1).double()
        for base, arm in [('full', 'checkpoint'), ('detach', 'segment'),
                          ('detach', 'segment_encoder_checkpoint')]:
            a, b = copy.deepcopy(initial), copy.deepcopy(initial)
            oa, ob = [torch.optim.AdamW(z.parameters(), lr=.003) for z in (a, b)]
            for step in range(3):
                # Nonmultiple length exercises unequal final chunk scaling.
                x, y = generate('parity', 2, 19, 1701 + step)
                oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
                la = m.backward_loss(a, x, y, base, chunk=8)
                lb = m.backward_loss(b, x, y, arm, chunk=8)
                self.assertAlmostEqual(la, lb, places=12)
                for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
                    self.assertIsNotNone(pa.grad, name)
                    torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-9, atol=2e-11, msg=name)
                oa.step(); ob.step()
                for pa, pb in zip(a.parameters(), b.parameters()):
                    torch.testing.assert_close(pa, pb, rtol=2e-9, atol=2e-11)
                    for key in ('step', 'exp_avg', 'exp_avg_sq'):
                        torch.testing.assert_close(oa.state[pa][key], ob.state[pb][key], rtol=2e-9, atol=2e-11)

    def test_truncated_route_is_not_full_gradient(self):
        m = self.api()
        torch.manual_seed(91)
        a = Core(width=8, heads=1).double(); b = copy.deepcopy(a)
        x, y = generate('parity', 2, 19, 74)
        m.backward_loss(a, x, y, 'full', chunk=8)
        m.backward_loss(b, x, y, 'detach', chunk=8)
        self.assertGreater(float((a.merge.weight.grad-b.merge.weight.grad).abs().max()), 1e-7)

    def test_global_encoder_route_is_preserved_and_detects_removed_bridge(self):
        m = self.api()
        torch.manual_seed(91)
        a = Core(width=8, heads=1).double(); b = copy.deepcopy(a)
        x, y = generate('parity', 2, 19, 74)
        m.backward_loss(a, x, y, 'detach', chunk=8)
        m.backward_loss(b, x, y, 'segment', chunk=8, bridge_global=False)
        self.assertGreater(float(a.mk.weight.grad.norm()), 1e-7)
        self.assertIsNone(b.mk.weight.grad)
        self.assertIsNone(b.embedding.weight.grad)

    def test_saved_storage_deduplicates_aliases_and_tracks_lifetime(self):
        m = self.api()
        tracker = m.SavedStorageTracker()
        x = torch.ones(16)
        a = tracker.pack(x[:8]); b = tracker.pack(x[4:])
        self.assertEqual(tracker.snapshot()['live_unique_storage_bytes'], 64)
        self.assertEqual(tracker.snapshot()['save_requests'], 2)
        del a
        self.assertEqual(tracker.snapshot()['live_unique_storage_bytes'], 64)
        del b
        self.assertEqual(tracker.snapshot()['live_unique_storage_bytes'], 0)
        self.assertEqual(tracker.snapshot()['peak_unique_storage_bytes'], 64)

    def test_saved_hook_does_not_retain_autograd_graph(self):
        m = self.api()
        tracker = m.SavedStorageTracker()
        x = torch.ones(4, requires_grad=True)
        packed = tracker.pack(x.square())
        self.assertIsNone(packed.tensor.grad_fn)

    def test_real_cpu_worker_saves_initialized_optimizer_and_actual_updates(self):
        m = self.api()
        p = m.protocol('cpu'); p.update(width=8, heads=1, batch=2, lengths=[9], warmup=2, repeats=1)
        with tempfile.TemporaryDirectory() as d:
            result = m.worker(Path(d), p, 'segment', 9, diagnostic=False)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['optimizer_updates'], 3)
            self.assertNotEqual(result['initial_state_hash'], result['final_state_hash'])
            self.assertTrue(result['optimizer_initialized_before_measurement'])
            self.assertEqual(result['measurements'][0]['cuda'], None)
            state = torch.load(Path(d)/'final.pt', weights_only=True)
            self.assertTrue(all(float(v['step']) == 3 for v in state['optimizer']['state'].values()))
            self.assertTrue(result['sources']['core/model.py'])

    def test_freeze_detects_protocol_tamper(self):
        m = self.api()
        with tempfile.TemporaryDirectory() as d:
            m.freeze(Path(d), m.protocol('cpu'))
            path = Path(d)/'frozen.json'
            path.write_text(path.read_text().replace('"batch": 8', '"batch": 9'))
            with self.assertRaisesRegex(ValueError, 'protocol'):
                m.verify_frozen(Path(d))


if __name__ == '__main__':
    unittest.main()
