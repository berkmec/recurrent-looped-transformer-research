"""C2/C7/F2 contracts: Holm correction arithmetic, full-log summaries that never read a prefix, zero-error risk bound, full job manifests."""
import math
import unittest


class StatsTests(unittest.TestCase):
    def api(self):
        try:
            from audit import stats
        except ImportError as e:
            self.fail('audit.stats is not implemented: ' + str(e))
        return stats

    def test_holm_step_down(self):
        st = self.api()
        # thresholds alpha/(m-i): 0.0125, 0.0167, 0.025, 0.05 — 0.03 > 0.025 stops the step-down
        r = st.holm({'a': 0.001, 'b': 0.015, 'c': 0.03, 'd': 0.5}, alpha=0.05)
        self.assertEqual(r, {'a': True, 'b': True, 'c': False, 'd': False})
        with self.assertRaises(ValueError):
            st.holm({'a': 1.5})

    def test_zero_error_bound_matches_closed_form(self):
        st = self.api()
        self.assertAlmostEqual(st.zero_error_upper_bound(299), 1 - 0.05 ** (1 / 299), places=12)
        self.assertLess(st.zero_error_upper_bound(299), 0.01)
        self.assertGreater(st.zero_error_upper_bound(298), 0.01)
        with self.assertRaises(ValueError):
            st.zero_error_upper_bound(0)

    def test_summary_consumes_all_records_not_a_prefix(self):
        st = self.api()
        ids = {f'p{i}' for i in range(40)}
        rows = [dict(id=f'p{i}', prediction=(0 if i < 16 else 3), accepted=True, correct=(i % 2 == 0)) for i in range(40)]
        s = st.summarize_records(rows, ids)
        self.assertEqual(s['status'], 'COMPLETE')
        self.assertEqual(s['prediction_counts'], {'0': 16, '3': 24})
        self.assertAlmostEqual(s['coverage'], 1.0); self.assertAlmostEqual(s['accepted_risk'], 0.5)
        partial = st.summarize_records(rows[:30], ids)
        self.assertEqual(partial['status'], 'INCOMPLETE'); self.assertEqual(len(partial['missing']), 10)
        with self.assertRaises(ValueError):
            st.summarize_records(rows + [rows[0]], ids)

    def test_manifest_is_full_cartesian_product(self):
        st = self.api()
        jobs = st.manifest_jobs(['p1', 'p2', 'p3'], ['a', 'b'], [117, 143, 191])
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({(j['problem'], j['arm'], j['seed']) for j in jobs}), 18)
        with self.assertRaises(ValueError):
            st.manifest_jobs(['p1', 'p1'], ['a'], [1])
        with self.assertRaises(ValueError):
            st.manifest_jobs([], ['a'], [1])


if __name__ == '__main__':
    unittest.main()
