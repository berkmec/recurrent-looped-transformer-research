import unittest

import torch

from horizon import depth_models as dm
from horizon import sk_pe
from horizon import sk_run as sr


class PositionalTransformerTests(unittest.TestCase):
    def test_tfm_pe_is_the_seed_s_tfm_plus_positions(self):
        plain = dm.make_arm('tfm', 'sk4', 64, 1)
        posed = sk_pe.make_arm('tfm_pe', 'sk4', 64, 1)
        sp, sq = plain.state_dict(), posed.state_dict()
        self.assertEqual(sorted(k.replace('embedding.tok.', 'embedding.') for k in sq), sorted(sp))
        for k, v in sq.items():
            self.assertTrue(torch.equal(v, sp[k.replace('embedding.tok.', 'embedding.')]), k)
        x = torch.full((1, 5), 3)
        e = posed.embedding(x)[0]
        self.assertGreater(float((e[1] - e[0]).abs().max()), 0.1)            # the same token at two positions differs

    def test_other_arms_pass_through_and_a_short_run_trains(self):
        self.assertIsInstance(sk_pe.make_arm('gru', 'sk4', 32, 1), dm.GRUArm)
        original = sr.make_arm
        try:
            sr.make_arm = sk_pe.make_arm
            model, trained = sr.train('sk4', 'tfm_pe', 32, 1, 3, batch=4, log_every=1000)
        finally:
            sr.make_arm = original
        self.assertTrue(hasattr(model.embedding, 'pe'))


if __name__ == '__main__':
    unittest.main()
