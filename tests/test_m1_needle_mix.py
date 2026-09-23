"""M1 must train on the needle format, or its three arms are compared on something none of them was taught.

The emergence probe found this the expensive way: a format absent from the training distribution produced five flat
checkpoints of below-chance byte accuracy. That fix was applied to the probe and NOT to the M1 runner, which is the
actual experiment. These tests close that gap and keep it closed.
"""
import unittest

import torch


class M1NeedleMixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        from mline import train as mt
        return mt

    def test_protocol_declares_a_needle_training_mix(self):
        p = self.api().protocol()
        self.assertGreater(p['needle_train_rate'], 0.)
        self.assertLess(p['needle_train_rate'], .5)                 # a mix, not a takeover
        self.assertEqual(tuple(p['needle_train_filler']), (0, 64))
        self.assertGreater(min(p['needle_filler'][1:]), max(p['needle_train_filler']),
                           'every evaluated filler beyond the control must exceed what training saw, or nothing generalises')

    def test_batch_is_the_measured_optimum_and_fits_vram(self):
        p = self.api().protocol()
        self.assertEqual(p['batch'], 32)                            # measured: 2.70x over batch 8; 48 spills to host

    def test_training_batch_mixes_needles_with_corpus_rows(self):
        mt = self.api()
        p = dict(mt.protocol(), batch=8, needle_train_rate=.25, length=512)
        x, y, meta = mt.training_batch(p, seed=117, step=3)
        self.assertEqual(x.shape, (8, 512))
        self.assertEqual(y.shape, (8, 512))
        self.assertEqual(meta['needles'], 2)
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        text = bytes(x[-1].tolist()).decode('ascii', errors='ignore')
        self.assertIn('The code for', text)                          # the tail rows are needles

    def test_needle_rows_use_training_keys_only(self):
        mt = self.api()
        from mline import needle as nd
        p = dict(mt.protocol(), batch=8, needle_train_rate=.5, length=512)
        x, _, _ = mt.training_batch(p, seed=117, step=9)
        allowed = set(nd.TRAIN_KEYS.decode('ascii'))
        for row in x[-4:]:
            text = bytes(row.tolist()).decode('ascii', errors='ignore')
            key = text.split('The code for ')[1].split(' is ')[0]
            self.assertTrue(set(key) <= allowed, f'{key} leaks an evaluation letter into training')

    def test_a_zero_rate_gives_a_pure_corpus_batch(self):
        mt = self.api()
        p = dict(mt.protocol(), batch=4, needle_train_rate=0., length=512)
        x, _, meta = mt.training_batch(p, seed=117, step=1)
        self.assertEqual(meta['needles'], 0)
        self.assertEqual(x.shape, (4, 512))


if __name__ == '__main__':
    unittest.main()
