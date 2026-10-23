import unittest
from unittest import mock

import numpy as np

from mline import naw
from mline import needle as nd

TEXT = (b'Officials said on Tuesday that the river had risen overnight, and residents were told to leave the low streets near '
        b'the old bridge before the evening. The mayor spoke briefly and promised more help. ') * 400
CORPUS = np.frombuffer(TEXT, dtype=np.uint8)


def _row(r, i):
    return bytes(r['x'][i].tolist()) + bytes([int(r['y'][i, -1])])


class NAWTests(unittest.TestCase):
    def test_the_fact_sits_in_window_0_and_the_question_d_windows_later(self):
        w = 128
        r = naw.naw_rows(40, (1, 2, 3), seed=5, corpus=CORPUS, windows=4, window=w)
        self.assertEqual(tuple(r['x'].shape), (40, 4 * w))
        self.assertTrue(set(r['d'].tolist()) <= {1, 2, 3} and len(set(r['d'].tolist())) == 3)
        for i in range(40):
            row, d = _row(r, i), int(r['d'][i])
            q = row.rfind(b'\nThe code for ')
            self.assertEqual(q // w, d)                                            # the question in window d
            answer = row[q + len(b'\nThe code for ') + 4 + len(b' is '):][:4]
            self.assertEqual(row.count(b'The code for '), 2)
            fact = row.find(b'The code for ')
            self.assertLess(fact + len(naw.FACT % (b'ABCD', answer)), w + 1)       # the whole fact in window 0
            self.assertIn(naw.FACT % (row[q + 14:q + 18], answer), row[:w])
            self.assertEqual(row.count(answer), 2)                                  # the value nowhere else
            positions = r['four'][i].nonzero().flatten().tolist()
            self.assertEqual(bytes(r['y'][i, positions].tolist()), answer)        # the masks sit on the answer's targets
            self.assertEqual(r['first'][i].nonzero().flatten().tolist(), positions[:1])
            self.assertEqual(positions[-1] // w, d)

    def test_mqar_facts_and_the_decoy_force_the_key(self):
        r = naw.naw_rows(30, (2,), seed=9, corpus=CORPUS, windows=3, window=256, n_facts=4, decoy_share=1.0)
        for i in range(30):
            row = _row(r, i)
            window0 = row[:256]
            self.assertEqual(window0.count(b'The code for '), 5)                  # four facts and the decoy
            q = row.rfind(b'\nThe code for ')
            asked, answer = row[q + 14:q + 18], bytes(r['y'][i, r['four'][i]].tolist())
            self.assertIn(naw.FACT % (asked, answer), window0)
            keys = [window0[j + 13:j + 17] for j in range(len(window0)) if window0.startswith(b'The code for ', j)]
            self.assertEqual(len(set(keys)), 5)
            self.assertTrue(any(sum(a != b for a, b in zip(k, keys[0])) == 1 for k in keys[1:]) or
                            any(sum(a != b for a, b in zip(k, kk)) == 1 for k in keys for kk in keys if k != kk))

    def test_it_is_deterministic_and_refuses_what_cannot_be_clean(self):
        a = naw.naw_rows(5, (1,), seed=3, corpus=CORPUS, windows=2, window=128)
        b = naw.naw_rows(5, (1,), seed=3, corpus=CORPUS, windows=2, window=128)
        self.assertTrue(all(bool((a[k] == b[k]).all()) for k in a))
        with mock.patch.object(nd, '_value', lambda rng, length=4: b'1234'):
            leaky = np.frombuffer(b'year 1234 and ' * 2000, dtype=np.uint8)          # the value in every window of every row
            with self.assertRaises(RuntimeError):
                naw.naw_rows(1, (1,), seed=1, corpus=leaky, windows=2, window=128, max_tries=5)
            clean = naw.naw_rows(3, (1,), seed=1, corpus=CORPUS, windows=2, window=128)
            self.assertTrue(all(_row(clean, i).count(b'1234') == 2 for i in range(3)))
        with self.assertRaises(ValueError):
            naw.naw_rows(1, (2,), seed=1, corpus=CORPUS, windows=2, window=128)
        keys = [b'river', b'bridge', b'mayor']
        r = naw.naw_rows(4, (1,), seed=2, corpus=CORPUS, windows=2, window=128, key_source=keys)
        self.assertTrue(all(any(_row(r, i).count(naw.QUESTION % k) == 1 for k in keys) for i in range(4)))


if __name__ == '__main__':
    unittest.main()
