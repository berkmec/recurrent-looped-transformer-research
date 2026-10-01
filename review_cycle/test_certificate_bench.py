import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fractions import Fraction
import numpy as np

from review_cycle import certificate_bench as b


class BenchContracts(unittest.TestCase):
    def test_fresh_source_reference_revision_and_depth(self):
        for split in ['iid', 'depth5', 'depth8', 'branched']:
            data = b.make_data(16, 91133, split)
            repeated = b.make_data(16, 91133, split)
            np.testing.assert_array_equal(data['events'], repeated['events'])
            for i, tape in enumerate(data['events']):
                ev = b.records(tape)
                first_query = next(j for j, e in enumerate(ev) if e['kind'] == 'query')
                before = b.reference_values(ev[:first_query + 1])
                after = b.reference_values(ev)
                query = ev[-1]['child']
                self.assertEqual(before[query] != after[query], bool(data['affected'][i]))
                self.assertEqual(len(after), 10)
                candidates = {k: float(v) for k, v in after.items()}
                self.assertEqual(b.verify_path(candidates, ev, query)['status'], 'VERIFIED')

    def test_chain_equality_control_and_branched_sensitivity(self):
        rng = np.random.default_rng(91138)
        for split in ['iid', 'depth5', 'depth8']:
            for tape in b.make_data(8, 91136, split)['events']:
                ev = b.records(tape)
                query = ev[-1]['child']
                values = {k: float(v) + float(rng.normal()) for k, v in b.reference_values(ev).items()}
                self.assertAlmostEqual(b.old_verify(values, ev, query)['bound'], b.verify_path(values, ev, query)['bound'], places=12)
        for tape in b.make_data(8, 91137, 'branched')['events']:
            ev = b.records(tape)
            query = ev[-1]['child']
            values = {k: float(v) for k, v in b.reference_values(ev).items()}
            check = b.verify_path(values, ev, query)
            self.assertEqual(len(check['certificate']['path']), 5)
            detected_gain = False
            for node in values.keys() - set(check['certificate']['path']):
                perturbed = {**values, node: values[node] + 10}
                old, new = b.old_verify(perturbed, ev, query), b.verify_path(perturbed, ev, query)
                if old['status'] != 'VERIFIED' and new['status'] == 'VERIFIED':
                    detected_gain = True
            self.assertTrue(detected_gain)

    def test_independent_solver_exact_inverse_and_contradiction(self):
        events = [dict(kind='root', child='b', value=2, parent=None), dict(kind='edge', child='b', parent='a', value=.5)]
        solved = b.reference_values(events)
        self.assertEqual(solved, dict(a=Fraction(3, 2), b=Fraction(2)))
        with self.assertRaisesRegex(ValueError, 'CONTRADICTORY'):
            b.reference_values(events + [dict(kind='root', child='a', parent=None, value=4)])

    def test_metric_denominators_and_zero_acceptance(self):
        empty = b.metrics([False, False], [True, False])
        self.assertIsNone(empty['selective_risk'])
        self.assertIsNone(empty['precision'])
        self.assertIsNone(empty['zero_error_upper_one_sided95'])
        result = b.metrics([True, False, True, False], [True, True, False, False])
        self.assertEqual(result['accepted_wrong'], 1)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['selective_risk'], .5)
        zero = b.metrics([True] * 83, [True] * 83)
        self.assertAlmostEqual(zero['zero_error_upper_one_sided95'], 1 - .05 ** (1 / 83))

    def test_source_pin_catches_imported_token_core_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'src'
            for rel in ['review_cycle/certificate.py', 'review_cycle/certificate_bench.py',
                        'learning/model.py', 'learning/token_core.py', 'learning/api.py', 'memory/api.py']:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((b.ROOT / rel).read_bytes())
            with patch.object(b, 'ROOT', root):
                out = Path(folder) / 'out'
                b.freeze(out)
                b.verify_frozen(out)
                (root / 'learning/token_core.py').write_text('changed')
                with self.assertRaisesRegex(RuntimeError, 'SOURCE_DRIFT'):
                    b.verify_frozen(out)

    def test_receipt_binds_protocol_source_condition_and_artifact_set(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'protocol.json').write_text('{}')
            (out / 'data-iid.npz').write_bytes(b'raw-source')
            files = ['raw-iid-17.npz', 'certificates-iid-17.json', 'lazy-iid-17.json']
            for rel in files:
                (out / rel).write_bytes(b'artifact')
            receipt = dict(split='iid', seed=17, protocol_sha256=b.sha(out / 'protocol.json'),
                           source_sha256=b.sha(out / 'data-iid.npz'), artifacts={rel: b.sha(out / rel) for rel in files})
            b.verify_receipt(out, {}, receipt, 'iid', 17)
            for field, value in [('protocol_sha256', 'old'), ('source_sha256', 'old'), ('seed', 43), ('split', 'depth8'), ('artifacts', {})]:
                with self.assertRaises(RuntimeError):
                    b.verify_receipt(out, {}, {**receipt, field: value}, 'iid', 17)

    def test_stale_corrupt_wrong_challenges_have_expected_outcomes(self):
        rows = b.challenge_checks()
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row['passed'] for row in rows))


if __name__ == '__main__':
    unittest.main()
