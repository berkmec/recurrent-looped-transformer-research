"""The pooled first-error table of the permutation pilots."""
import unittest

from horizon import hazard_summary as hs


def window(start, stop, at_risk, first_errors):
    return dict(start=start, stop=stop, at_risk=at_risk, first_errors=first_errors,
                rate=(first_errors / (at_risk * (stop - start))) if at_risk else None)


RESULT = dict(task='perm4', arm='gru', width=8, train=dict(parameters=123), curve={
    16: dict(hazard=[window(0, 16, 100, 2)]),
    32: dict(hazard=[window(0, 16, 100, 1), window(16, 32, 98, 3)]),
    64: dict(hazard=[window(0, 16, 100, 0), window(16, 32, 100, 1), window(32, 64, 99, 0)]),
})


class Pooled(unittest.TestCase):
    def test_the_same_window_of_different_points_is_summed_with_its_exposure(self):
        pooled = hs.pooled_windows(RESULT)
        self.assertEqual(list(pooled), [(0, 16), (16, 32), (32, 64)])
        self.assertEqual(pooled[(0, 16)], [3, 3 * 100 * 16])                      # three draws reach the first window
        self.assertEqual(pooled[(16, 32)], [4, (98 + 100) * 16])                  # two reach the second
        self.assertEqual(pooled[(32, 64)], [0, 99 * 32])                          # one reaches the third
        self.assertAlmostEqual(hs.rate(pooled[(0, 16)]), 3 / 4800)
        self.assertEqual(hs.rate([0, 0]), None)

    def test_a_point_without_hazard_data_is_skipped_and_the_description_names_the_numbers(self):
        curve = dict(RESULT['curve'])
        curve[128] = dict(final=1.0)                                              # an old result without the hazard field
        self.assertEqual(hs.pooled_windows(dict(RESULT, curve=curve)), hs.pooled_windows(RESULT))
        text = hs.describe(RESULT, 'x.json')
        self.assertIn('[0,16) 3/4800', text)
        self.assertIn('all 7/', text)
        self.assertIn('123 parameters', text)


if __name__ == '__main__':
    unittest.main()
