import math
import unittest

import numpy as np

from mline import a0_where as aw


class WhereTests(unittest.TestCase):
    def test_byte_classes(self):
        got = aw.byte_class(np.array([32, 10, 65, 122, 48, 33, 0xC3, 0xA9, 7]))
        self.assertEqual([aw.CLASSES[i] for i in got],
                         ['space', 'newline', 'letter', 'letter', 'digit', 'punctuation', 'utf8-lead', 'utf8-continuation', 'other'])

    def test_word_offset_counts_the_letters_before_the_predicted_byte(self):
        x = np.frombuffer(b'ab cd', dtype=np.uint8)[None].astype(np.int64)
        self.assertEqual(aw.word_offset(x).tolist(), [[1, 2, 0, 1, 2]])

    def test_the_splits_add_up_to_the_total_gain(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 256, (4, 64))
        y = rng.integers(0, 256, (4, 64))
        a0 = rng.random((4, 64)) * 3
        l1 = a0 - rng.random((4, 64)) * 0.1
        r = aw.analyse(a0, l1, x, y)
        self.assertAlmostEqual(r['gain_bits_per_byte'], float(((a0 - l1) / math.log(2)).mean()))
        for key in ('by_position', 'by_target_class', 'by_a0_difficulty_quartile'):
            self.assertAlmostEqual(sum(v['share_of_gain'] for v in r[key].values()), 1.0, places=9, msg=key)
            self.assertEqual(sum(v['bytes'] for v in r[key].values()), 4 * 64, key)


if __name__ == '__main__':
    unittest.main()
