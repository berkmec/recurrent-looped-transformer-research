"""M-hattı M0 contracts: the model is the sealed Core at language width, and the needle set is what it claims to be."""
import unittest

import numpy as np
import torch


class MLineModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from mline import lm
        except ImportError as e:
            self.fail('mline.lm is not implemented: ' + str(e))
        return lm

    def test_arms_and_vocabulary(self):
        lm = self.api()
        self.assertEqual(tuple(lm.ARMS), ('plain', 'hard_st', 'soft'))
        self.assertEqual(lm.VOCAB, 256)
        with self.assertRaises(ValueError):
            lm.build('nonsense', width=64, heads=2)

    def test_every_arm_wraps_the_same_sealed_core(self):
        """The point of M1 is that all three arms share one architecture; only the gate differs."""
        lm = self.api()
        from core.model import Core
        models = {a: lm.build(a, width=64, heads=2, seed=117) for a in lm.ARMS}
        self.assertIsInstance(models['plain'], Core)
        for arm in ('hard_st', 'soft'):
            self.assertIsInstance(models[arm].core, Core)
            self.assertEqual(models[arm].mode, arm)
        counts = {a: lm.parameter_count(m) for a, m in models.items()}
        self.assertEqual(counts['plain']['core'], counts['hard_st']['core'])   # identical core size
        self.assertEqual(counts['plain']['gate'], 0)
        self.assertGreater(counts['hard_st']['gate'], 0)
        self.assertEqual(counts['hard_st']['vocabulary'], 256)

    def test_same_seed_gives_the_same_core_weights_across_arms(self):
        """Paired comparison rule: the arms must start from identical core weights, or the contrast is confounded."""
        lm = self.api()
        a = lm.build('plain', width=64, heads=2, seed=191)
        b = lm.build('hard_st', width=64, heads=2, seed=191)
        self.assertTrue(torch.equal(a.embedding.weight, b.core.embedding.weight))
        self.assertTrue(torch.equal(a.eq.weight, b.core.eq.weight))

    def test_forward_shapes_for_every_arm(self):
        lm = self.api()
        x = torch.randint(0, 256, (2, 16))
        for arm in lm.ARMS:
            logits = lm.forward_logits(lm.build(arm, width=64, heads=2), x)
            self.assertEqual(logits.shape, (2, 16, 256), arm)
            self.assertTrue(torch.isfinite(logits).all(), arm)

    def test_target_size_is_thirty_million_class(self):
        lm = self.api()
        n = lm.parameter_count(lm.build('hard_st'))['total']
        self.assertGreater(n, 25_000_000)
        self.assertLess(n, 40_000_000)


class NeedleTests(unittest.TestCase):
    def api(self):
        try:
            from mline import needle
        except ImportError as e:
            self.fail('mline.needle is not implemented: ' + str(e))
        return needle

    def test_answer_is_at_the_end_and_the_mask_selects_it(self):
        nd = self.api()
        items = nd.make(6, n_filler=64, kind='repeat', seed=3)
        x, y, mask = items['x'], items['y'], items['mask']
        self.assertEqual(x.shape, y.shape)
        for i, ans in enumerate(items['answers']):
            picked = bytes(y[i][mask[i]].tolist()).decode('ascii')
            self.assertEqual(picked, ans)                       # the mask selects exactly the answer bytes
            self.assertEqual(int(mask[i].sum()), len(ans))

    def test_the_key_appears_twice_and_the_filler_sits_between(self):
        nd = self.api()
        items = nd.make(1, n_filler=256, kind='repeat', seed=5)
        text = bytes(items['x'][0].tolist()).decode('ascii')
        key = text.split('The code for ')[1].split(' is ')[0]
        self.assertEqual(text.count(key), 2)
        first, second = text.index(key), text.rindex(key)
        self.assertGreaterEqual(second - first, 256)            # the filler really is between the two mentions

    def test_zero_filler_is_the_control_condition(self):
        nd = self.api()
        a = nd.make(4, n_filler=0, kind='repeat', seed=7)
        b = nd.make(4, n_filler=512, kind='repeat', seed=7)
        self.assertLess(a['width'], b['width'])
        self.assertEqual(a['answers'], b['answers'])            # same items, only the distance changes

    def test_text_filler_needs_a_corpus_and_uses_it(self):
        nd = self.api()
        with self.assertRaises(ValueError):
            nd.make(2, n_filler=32, kind='text', seed=1)
        corpus = np.frombuffer(b'x' * 4096, dtype=np.uint8)
        items = nd.make(2, n_filler=32, kind='text', seed=1, corpus=corpus)
        self.assertIn('x' * 16, bytes(items['x'][0].tolist()).decode('ascii'))

    def test_scoring_is_perfect_for_an_oracle_and_chance_for_noise(self):
        nd = self.api()
        items = nd.make(8, n_filler=32, kind='repeat', seed=11)
        oracle = lambda xb: torch.nn.functional.one_hot(
            torch.roll(xb, -1, dims=1), num_classes=256).float()
        s = nd.score(oracle, items, batch=4)
        self.assertEqual(s['exact_match'], 1.0)
        self.assertEqual(s['byte_accuracy'], 1.0)
        noise = lambda xb: torch.zeros(xb.shape[0], xb.shape[1], 256)
        self.assertLess(nd.score(noise, items, batch=4)['byte_accuracy'], 0.2)


if __name__ == '__main__':
    unittest.main()
