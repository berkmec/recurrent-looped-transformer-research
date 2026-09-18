"""Behavioral contracts for the source-consistent, path-specific certificate."""
import copy
import math
import unittest
from fractions import Fraction
import numpy as np
from review_cycle import certificate as c


def root(name, value):
    return dict(kind='root', child=name, parent=None, value=value)


def edge(child, parent, value):
    return dict(kind='edge', child=child, parent=parent, value=value)


class PathContracts(unittest.TestCase):
    def test_correct_query_unrelated_branch_error(self):
        evidence = [root('a', 0), edge('q', 'a', 1), edge('b', 'a', 2)]
        result = c.verify_path(dict(a=0., q=1., b=100.), evidence, 'q')
        self.assertEqual(result['status'], 'VERIFIED')
        self.assertEqual(result['bound'], 0.)

    def test_missing_off_path_candidate_is_allowed(self):
        evidence = [root('a', 0), edge('q', 'a', 1), edge('b', 'a', 2)]
        self.assertEqual(c.verify_path(dict(a=0., q=1.), evidence, 'q')['status'], 'VERIFIED')

    def test_wrong_query_must_not_be_certified(self):
        result = c.verify_path(dict(a=0., q=1.3), [root('a', 0), edge('q', 'a', 1)], 'q')
        self.assertNotEqual(result['status'], 'VERIFIED')

    def test_contradiction_in_other_component_rejected(self):
        evidence = [root('a', 0), edge('q', 'a', 1), root('b', 0), root('z', 2), edge('z', 'b', 1)]
        result = c.verify_path(dict(a=0., q=1., b=0., z=2.), evidence, 'q')
        self.assertNotEqual(result['status'], 'VERIFIED')

    def test_multiple_anchors_select_tighter_complete_path(self):
        events = [root('a', 0), root('z', 3), edge('b', 'a', 1), edge('q', 'b', 1), edge('z', 'q', 1)]
        values = dict(a=100., b=-100., q=2.01, z=3.)
        result = c.verify_path(values, events, 'q')
        self.assertEqual(result['status'], 'VERIFIED')
        self.assertEqual(result['certificate']['path'], ['z', 'q'])
        self.assertEqual(result['candidate_answer'], values['q'])
        self.assertTrue(c.audit_certificate(result['certificate'], values, events, 'q'))

    def test_exact_boundary_no_float_sum_acceptance(self):
        events = [root('q', 0)]
        # IEEE 0.1 is strictly larger than the decimal threshold 1/10.
        above = c.verify_path(dict(q=.1), events, 'q', .1)
        below = c.verify_path(dict(q=math.nextafter(.1, 0.)), events, 'q', .1)
        self.assertEqual(above['status'], 'CONTINUE')
        self.assertEqual(below['status'], 'VERIFIED')
        self.assertEqual(c.verify_path(dict(q=0.), events, 'q', 0)['status'], 'VERIFIED')

    def test_display_overflow_does_not_change_exact_decision(self):
        events, candidate = [root('q', 0)], dict(q=10 ** 400)
        result = c.verify_path(candidate, events, 'q', 10 ** 400)
        self.assertEqual(result['status'], 'VERIFIED')
        self.assertIsNone(result['bound'])
        self.assertTrue(result['display_bound_overflow'])
        self.assertTrue(c.audit_certificate(result['certificate'], candidate, events, 'q', 10 ** 400))
        self.assertEqual(c.verify_path(candidate, events, 'q', .1)['status'], 'CONTINUE')

    def test_tiny_inconsistent_cycle_rejected_exactly(self):
        events = [root('a', 0), edge('b', 'a', 1), edge('a', 'b', -1.000000001)]
        self.assertEqual(c.verify_path(dict(a=0., b=1.), events, 'b')['reason'], 'INCONSISTENT_CYCLE')

    def test_latest_revision_is_authoritative(self):
        events = [root('a', 0), edge('q', 'a', 1), dict(kind='revision', child='q', parent='a', value=2)]
        self.assertEqual(c.verify_path(dict(a=0., q=1.), events, 'q')['status'], 'CONTINUE')
        self.assertEqual(c.verify_path(dict(a=0., q=2.), events, 'q')['status'], 'VERIFIED')

    def test_invalid_and_nonfinite_reject(self):
        good = [root('a', 0), edge('q', 'a', 1)]
        for bad in [float('nan'), float('inf'), True, '1']:
            with self.subTest(bad=str(bad)):
                self.assertNotEqual(c.verify_path(dict(a=0., q=bad), good, 'q')['status'], 'VERIFIED')
                self.assertNotEqual(c.verify_path(dict(a=0., q=1.), [root('a', bad), edge('q', 'a', 1)], 'q')['status'], 'VERIFIED')
        for events in [[], [edge('q', 'a', 1)], good + [dict(kind='invented')], [root('', 0)], [dict(kind='root', child='a', parent='q', value=0)]]:
            self.assertNotEqual(c.verify_path(dict(a=0., q=1.), events, 'q')['status'], 'VERIFIED')
        self.assertNotEqual(c.verify_path(dict(a=0., q=1., unknown=0.), good, 'q')['status'], 'VERIFIED')

    def test_stale_scope_query_and_corruption_rejected(self):
        events = [root('a', 0), edge('q', 'a', 1)]
        envelope = c.bind_candidate(dict(a=0., q=1.), events, 'q', 'task1')
        self.assertEqual(c.verify_envelope(envelope, events, 'q', 'task1')['status'], 'VERIFIED')
        revised = events + [dict(kind='revision', child='a', parent=None, value=2)]
        for ev, query, scope in [(revised, 'q', 'task1'), (events, 'q', 'task2'), (events, 'a', 'task1')]:
            self.assertNotEqual(c.verify_envelope(envelope, ev, query, scope)['status'], 'VERIFIED')
        broken = {**envelope, 'source_sha256': '0' * 64}
        self.assertNotEqual(c.verify_envelope(broken, events, 'q', 'task1')['status'], 'VERIFIED')

    def test_certificate_tampering_fails_independent_path_audit(self):
        events = [root('a', 0), edge('q', 'a', 1)]
        values = dict(a=0., q=1.02)
        cert = c.verify_path(values, events, 'q')['certificate']
        self.assertTrue(c.audit_certificate(cert, values, events, 'q'))
        for field, value in [('path', ['q']), ('bound', dict(numerator=0, denominator=1)), ('source_sha256', 'bad')]:
            broken = copy.deepcopy(cert)
            broken[field] = value
            self.assertFalse(c.audit_certificate(broken, values, events, 'q'))

    def test_randomized_soundness_against_linear_algebra(self):
        rng = np.random.default_rng(918330)
        for trial in range(240):
            n = int(rng.integers(2, 11))
            true = rng.integers(-32, 33, n) / 8
            events = [root('0', float(true[0]))]
            matrix = [np.eye(n)[0]]
            rhs = [true[0]]
            for child in range(1, n):
                parent = int(rng.integers(child))
                delta = float(true[child] - true[parent])
                events.append(edge(str(child), str(parent), delta))
                row = np.zeros(n)
                row[child], row[parent] = 1, -1
                matrix.append(row)
                rhs.append(delta)
            solved = np.linalg.solve(np.stack(matrix), np.array(rhs))
            np.testing.assert_allclose(solved, true, atol=1e-12, rtol=0)
            candidate = {str(j): float(true[j] + rng.normal(0, [.002, .03, .3][trial % 3])) for j in range(n)}
            query = str(int(rng.integers(n)))
            result = c.verify_path(candidate, events, query)
            exact_bound = Fraction(**result['bound_exact'])
            actual = abs(Fraction(candidate[query]) - Fraction(float(solved[int(query)])))
            self.assertLessEqual(actual, exact_bound)
            if result['status'] == 'VERIFIED':
                self.assertLessEqual(actual, Fraction(1, 10))
                self.assertTrue(c.audit_certificate(result['certificate'], candidate, events, query))


if __name__ == '__main__':
    unittest.main()
