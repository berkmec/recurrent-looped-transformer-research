"""E-A10: the cleanup is closed at initialisation (exactly the sealed Core), opens through a learnable gate, snaps to a prototype when sharp, and works with the noise training and the saved-model path."""
import tempfile
import unittest
from pathlib import Path

import torch

from core.model import Core
from horizon import cleanup as cl
from horizon import depth_models as dm
from horizon import depth_run as dr
from horizon import state_noise as sno


def tokens(batch=3, length=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 6, (batch, length), generator=g)


class Cleanup(unittest.TestCase):
    def test_a_new_cleanup_core_computes_exactly_what_the_sealed_core_computes(self):
        plain, clean = dm.make_arm('core', 'perm3', 32, 5).eval(), dm.make_arm('core_clean', 'perm3', 32, 5).eval()
        for name, p in plain.named_parameters():
            q = dict(clean.named_parameters())[name.replace('pn.weight', 'pn.norm.weight')]
            self.assertTrue(torch.equal(p, q), name)                                   # the base weights are drawn first: same as the plain core of the seed
        x = tokens()
        with torch.no_grad():
            self.assertTrue(torch.equal(plain(x), clean(x)))                           # gate closed (open = 0): the function is the sealed Core's, bit for bit
        self.assertEqual(cl.gate_value(clean), 0.0)
        self.assertIsNone(cl.gate_value(plain))

    def test_the_gate_opens_and_changes_the_run_and_the_extra_parameters_are_counted(self):
        plain, clean = dm.make_arm('core', 'perm3', 32, 5).eval(), dm.make_arm('core_clean', 'perm3', 32, 5).eval()
        x = tokens()
        with torch.no_grad():
            before = clean(x)
            clean.pn.open.fill_(1.0)
            self.assertFalse(torch.allclose(clean(x)[:, 3:], before[:, 3:], atol=1e-4))
        self.assertAlmostEqual(cl.gate_value(clean), float(torch.tanh(torch.tensor(1.0))), places=6)
        extra = dm.parameter_count(clean) - dm.parameter_count(plain)
        self.assertEqual(extra, 2 * cl.PROTOTYPES * 32 + 2)                                # keys, values, log beta, open

    def test_a_sharp_open_cleanup_snaps_the_state_to_the_prototype_it_is_nearest_to(self):
        norm = Core(width=32, heads=1, classes=6).pn
        pn = cl.CleanupNorm(norm, 32, prototypes=8, beta=1e4)
        with torch.no_grad():
            pn.values.copy_(pn.keys)
            pn.open.fill_(20.0)                                                        # tanh = 1
            g = torch.Generator().manual_seed(2)
            target = pn.keys[3] / pn.keys[3].square().mean().sqrt()                     # the unit-rms version the RMS norm produces
            state = 5.0 * pn.keys[3] + 0.05 * torch.randn(32, generator=g)
            out = pn(state[None])[0]
        self.assertTrue(torch.allclose(out, pn.values[3], atol=1e-3))
        self.assertGreater(float(torch.cosine_similarity(out, target, dim=0)), 0.999)

    def test_the_gate_gets_a_gradient_before_it_opens_and_the_prototypes_only_after(self):
        clean = dm.make_arm('core_clean', 'perm3', 32, 5)
        x, y = tokens(), tokens(seed=1) % 6
        loss = torch.nn.functional.cross_entropy(clean(x).flatten(0, 1), y.flatten())
        loss.backward()
        self.assertGreater(abs(float(clean.pn.open.grad)), 0.0)                        # the gate can start to move
        self.assertEqual(float(clean.pn.keys.grad.abs().max()), 0.0)                   # g = 0: the prototypes are not used yet
        clean.zero_grad()
        with torch.no_grad():
            clean.pn.open.fill_(0.5)
        torch.nn.functional.cross_entropy(clean(x).flatten(0, 1), y.flatten()).backward()
        self.assertGreater(float(clean.pn.keys.grad.abs().max()), 0.0)

    def test_noise_training_and_the_saved_model_path_work(self):
        clean = dm.make_arm('core_clean', 'perm3', 32, 5)
        noisy = sno.noisy(clean, 0.1, seed=1)
        self.assertEqual(tuple(noisy(tokens()).shape), (3, 24, 6))
        sno.clean(noisy)
        self.assertIsInstance(noisy.pn, cl.CleanupNorm)                                # the cleanup survives the class swap and its undoing
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'core_clean', 32, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4], state_noise=0.1)
            model = dr.load_model(out.with_suffix('.pt')).eval()
        self.assertIsInstance(model.pn, cl.CleanupNorm)
        self.assertEqual(tuple(model(tokens()).shape), (3, 24, 6))

    def test_the_initial_gate_can_be_set_for_training_and_the_default_stays_closed(self):
        self.assertEqual(cl.gate_value(dm.make_arm('core_clean', 'perm3', 32, 5)), 0.0)
        opened = dm.make_arm('core_clean', 'perm3', 32, 5, open_init=0.3)
        self.assertAlmostEqual(cl.gate_value(opened), float(torch.tanh(torch.tensor(0.3))), places=6)
        plain = dm.make_arm('core', 'perm3', 32, 5)
        for name, p in plain.named_parameters():
            self.assertTrue(torch.equal(p, dict(opened.named_parameters())[name.replace('pn.weight', 'pn.norm.weight')]), name)            # the base weights do not depend on the gate

    def test_the_cleanup_parameters_can_be_exempted_from_weight_decay(self):
        # blind review F12: AdamW's weight decay also acts on open / log beta / keys / values unless exempted (default: decayed, as registered)
        kw = dict(steps=20, batch=4, curriculum=[4], weight_decay=5.0, lr=0.05, cleanup_open=0.3)
        decayed, trained = dr.train('perm3', 'core_clean', 32, 1, cleanup_no_decay=False, **kw)
        exempt, trained_exempt = dr.train('perm3', 'core_clean', 32, 1, cleanup_no_decay=True, **kw)
        self.assertEqual((trained['cleanup_no_decay'], trained_exempt['cleanup_no_decay']), (False, True))
        # with a huge decay the exempted log beta only moves through gradients (it stays near log 20 = 3.0); the decayed one is shrunk toward 0
        self.assertLess(float(decayed.pn.log_beta.detach().abs()), float(exempt.pn.log_beta.detach().abs()))
        self.assertGreater(float(exempt.pn.log_beta.detach()), 2.0)

    def test_a_cleanup_needs_the_feedback_path(self):
        with self.assertRaises(ValueError):
            cl.add_cleanup(dm.make_arm('core_nofb', 'perm3', 32, 1))


if __name__ == '__main__':
    unittest.main()
