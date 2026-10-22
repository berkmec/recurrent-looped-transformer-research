import tempfile
import unittest

import torch

from mline import checkpoint as ck
from mline import ews
from tests.test_longrun import lm_batch, tiny


def bats(values):
    return {k: dict(bits_per_byte=v) for k, v in zip(ews.POINTS, values)}


class WarmStartTests(unittest.TestCase):
    def test_init_from_gives_a_new_run_the_weights_and_a_fresh_schedule(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            tiny(src, steps=4, arm='a0')
            source = ck.load_latest(src)['model']
            tiny(dst, steps=2, arm='plain', init_from=str(src) + '/latest.pt', lr=0.0, warmup=0)
            got = ck.load_latest(dst)
            self.assertEqual(got['step'], 2)                                     # its own schedule, from update 1
            for k in source:
                self.assertTrue(torch.equal(source[k], got['model'][k]), k)      # lr 0: the weights are the source's

    def test_data_offset_shifts_the_batches_and_the_phase(self):
        asked = []

        def recording(step, n_rows, length=128):
            asked.append(step)
            return lm_batch(step, n_rows, length)
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=3, lm_batch=recording, data_offset=100)
        self.assertEqual(asked, [101, 102, 103])

    def test_reading_rule(self):
        self.assertEqual(ews.reading(bats([1.5] * 4), bats([1.49, 1.49, 1.48, 1.485]))['verdict'], 'WARM START HELPS')
        self.assertEqual(ews.reading(bats([1.5] * 4), bats([1.51, 1.5, 1.52, 1.515]))['verdict'], 'WARM START HURTS')
        self.assertEqual(ews.reading(bats([1.5] * 4), bats([1.45, 1.46, 1.5, 1.495]))['verdict'], 'NO DIFFERENCE')
        self.assertEqual(ews.reading(bats([1.5] * 4), bats([1.45, 1.46, 1.51, 1.48]))['verdict'], 'MIXED')
        self.assertEqual(ews.reading(bats([1.5] * 3), bats([1.5] * 4))['verdict'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
