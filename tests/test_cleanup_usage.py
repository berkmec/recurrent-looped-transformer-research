"""E-A10 descriptive: the cleanup usage diagnostic reports what the module does, on a model whose cleanup is known."""
import math
import unittest

import torch

from horizon import cleanup_usage as cu
from horizon import depth_models as dm


class Usage(unittest.TestCase):
    def test_a_closed_flat_module_and_a_plain_core(self):
        model = dm.make_arm('core_clean', 'perm3', 32, 1, open_init=0.0).eval()
        r = cu.usage(model, 'perm3', 1, n=60)
        self.assertEqual(r['gate'], 0.0)                                                                  # closed: tanh(0)
        self.assertAlmostEqual(r['relative_change'], 0.0, places=6)                                       # a closed gate leaves the feedback untouched, so r' = r
        self.assertAlmostEqual(r['purity_pre'], r['purity_post'], places=9)                               # ... and the two purities coincide
        self.assertAlmostEqual(r['beta'], 20.0, places=4)
        self.assertGreaterEqual(r['top_weight'], 1 / 128)
        self.assertLessEqual(r['slot_purity'], 1.0)
        plain = cu.usage(dm.make_arm('core', 'perm3', 32, 1).eval(), 'perm3', 1, n=60)
        self.assertEqual(set(plain), {'task', 'seed', 'n_states', 'classes', 'purity_pre'})

    def test_a_sharp_open_module_with_class_prototypes_snaps_and_is_pure(self):
        model = dm.make_arm('core_clean', 'perm3', 32, 1, open_init=0.0).eval()
        states, labels = cu.states_and_labels(model, 'perm3', 1, n=200)
        with torch.no_grad():
            pre = model.pn.norm(states)
            protos = torch.stack([pre[labels == c].mean(0) for c in range(6)])
            g = torch.Generator().manual_seed(3)
            book = torch.cat([protos, torch.randn(122, protos.shape[1], generator=g)])                    # six class prototypes, the other slots random (far away in cosine)
            model.pn.keys.copy_(book)
            model.pn.values.copy_(book)
            model.pn.log_beta.fill_(math.log(200.0))
            model.pn.open.fill_(math.atanh(0.999))
        r = cu.usage(model, 'perm3', 1, n=200)
        self.assertGreater(r['top_weight'], 0.2)                                                           # a sharp read-out concentrates mass on one prototype
        self.assertGreater(r['relative_change'], 0.0)
        self.assertTrue(0.0 <= r['purity_post'] <= 1.0 and 0.0 <= r['slot_purity'] <= 1.0)                  # (an untrained model's states do not cluster by class, so no ordering of the purities is asserted)
        self.assertLessEqual(r['active_slots'], 20)                                                        # the mass sits on the class prototypes, not on the 122 random slots


if __name__ == '__main__':
    unittest.main()
