import unittest

from mline import a0_reading as ar


def runs(deltas, k4=(0.99, 0.99)):
    """Two batteries whose bpb differ by `deltas` (A0 minus L1) at the eight points."""
    a0, l1 = {}, {}
    for k, d in zip(ar.POINTS, deltas):
        l1[k] = dict(bits_per_byte=1.7, train_k4_first=k4[1])
        a0[k] = dict(bits_per_byte=1.7 + d, train_k4_first=k4[0])
    return a0, l1


class A0ReadingTests(unittest.TestCase):
    def test_helps_needs_the_last_delta_and_a_majority(self):
        self.assertEqual(ar.reading(*runs([0.01] * 7 + [0.03]))['verdict'], 'FEEDBACK HELPS')
        self.assertEqual(ar.reading(*runs([-0.01] * 3 + [0.01] * 4 + [0.03]))['verdict'], 'MIXED')   # only 5 of 8 positive

    def test_hurts_is_symmetric(self):
        self.assertEqual(ar.reading(*runs([-0.01] * 7 + [-0.025]))['verdict'], 'FEEDBACK HURTS')

    def test_no_meaningful_difference_is_decided_by_the_last_point(self):
        self.assertEqual(ar.reading(*runs([0.05] * 7 + [0.019]))['verdict'], 'NO MEANINGFUL DIFFERENCE')

    def test_the_threshold_is_inclusive(self):
        self.assertEqual(ar.reading(*runs([0.01] * 7 + [0.020]))['verdict'], 'FEEDBACK HELPS')

    def test_a_missing_point_is_incomplete(self):
        a0, l1 = runs([0.01] * 8)
        del a0[5000]
        r = ar.reading(a0, l1)
        self.assertEqual(r['verdict'], 'INCOMPLETE')
        self.assertEqual(r['missing'], [5000])

    def test_the_end_of_schedule_rule_needs_five_of_six_points(self):
        def end(deltas):
            a0 = {k: dict(bits_per_byte=1.5 + d, train_k4_first=0.99) for k, d in zip(ar.END_POINTS, deltas)}
            l1 = {k: dict(bits_per_byte=1.5, train_k4_first=0.99) for k in ar.END_POINTS}
            return ar.reading(a0, l1, ar.END_POINTS, ar.END_MAJORITY)['verdict']
        self.assertEqual(end([-0.01, 0.01, 0.01, 0.01, 0.01, 0.03]), 'FEEDBACK HELPS')
        self.assertEqual(end([-0.01, -0.01, 0.01, 0.01, 0.01, 0.03]), 'MIXED')
        self.assertEqual(end([0.05] * 5 + [0.01]), 'NO MEANINGFUL DIFFERENCE')

    def test_retrieval_is_vacuous_when_a_run_did_not_learn_the_range(self):
        r = ar.reading(*runs([0.0] * 8, k4=(0.5, 0.99)))
        self.assertFalse(r['retrieval_valid'])
        self.assertTrue(str(r['retrieval']).startswith('VACUOUS'))


if __name__ == '__main__':
    unittest.main()
