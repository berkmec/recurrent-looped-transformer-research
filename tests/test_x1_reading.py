import unittest

from mline import a0_reading as ar
from mline import x1_reading as xr


def _battery(bpb):
    """bpb: a number for every point, or a dict point -> number."""
    return {k: dict(bits_per_byte=bpb[k] if isinstance(bpb, dict) else bpb, train_k4_first=1.0) for k in ar.END_POINTS}


class X1ReadingTests(unittest.TestCase):
    def test_the_seat_of_the_transformer_decides_the_sign(self):
        l1 = _battery(1.40)
        self.assertEqual(xr.reading(_battery(1.45), l1)['verdict'], 'RLT BEATS TRANSFORMER')
        self.assertEqual(xr.reading(_battery(1.35), l1)['verdict'], 'TRANSFORMER BEATS RLT')
        self.assertEqual(xr.reading(_battery(1.41), l1)['verdict'], 'NO MEANINGFUL DIFFERENCE')
        self.assertAlmostEqual(xr.reading(_battery(1.45), l1)['delta_last'], 0.05)

    def test_the_last_point_and_the_majority_are_both_needed(self):
        l1 = _battery(1.40)
        mostly_worse = {k: 1.45 for k in ar.END_POINTS}
        mostly_worse[ar.END_POINTS[0]] = mostly_worse[ar.END_POINTS[1]] = 1.30        # 4 of 6 points positive
        self.assertEqual(xr.reading(_battery(mostly_worse), l1)['verdict'], 'MIXED')

    def test_missing_points_and_the_twin(self):
        t = _battery(1.38)
        del t[30518]
        self.assertEqual(xr.reading(t, _battery(1.40))['verdict'], 'INCOMPLETE')
        r = xr.reading(_battery(1.39), _battery(1.40), _battery(1.44))
        self.assertEqual((r['verdict'], r['transformer_vs_a0']['verdict']), ('NO MEANINGFUL DIFFERENCE', 'TRANSFORMER BEATS A0'))
        self.assertEqual(xr.reading(_battery(1.38), _battery(1.40))['verdict'], 'TRANSFORMER BEATS RLT')   # the threshold is inclusive
        r = xr.reading(_battery(1.50), _battery(1.40), _battery(1.44))
        self.assertEqual(r['transformer_vs_a0']['verdict'], 'A0 BEATS TRANSFORMER')


if __name__ == '__main__':
    unittest.main()
