"""E12: completion receipts — COMPLETE requires matching run id, zero exit, all items, passing tests and matching artifact hashes."""
import json
import tempfile
import unittest
from pathlib import Path


class ReceiptTests(unittest.TestCase):
    def api(self):
        try:
            from audit import receipt
        except ImportError as e:
            self.fail('audit.receipt is not implemented: ' + str(e))
        return receipt

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name)
        self.a = root / 'final.json'; self.a.write_text('{"x": 1}', encoding='utf8')
        self.b = root / 'raw.jsonl'; self.b.write_bytes(b'{"id": 1}\n')
        self.artifacts = dict(final=self.a, raw=self.b)

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_receipt_round_trips_and_validates(self):
        rc = self.api()
        r = rc.make_receipt('run-1', self.artifacts, returncode=0, expected_items=600, completed_items=600, tests_passed=True)
        self.assertEqual(r['status'], 'COMPLETE')
        self.assertEqual(set(r['artifacts']), {'final', 'raw'})
        self.assertTrue(rc.validate_receipt(r, self.artifacts, expected_run_id='run-1', expected_items=600))
        path = Path(self.tmp.name) / 'receipt.json'
        rc.write_receipt(path, r)
        self.assertEqual(json.loads(path.read_text(encoding='utf8'))['status'], 'COMPLETE')

    def test_bad_exit_missing_items_failed_tests_are_incomplete(self):
        rc = self.api()
        self.assertEqual(rc.make_receipt('r', self.artifacts, returncode=1, expected_items=10, completed_items=10, tests_passed=True)['status'], 'INCOMPLETE')
        self.assertEqual(rc.make_receipt('r', self.artifacts, returncode=0, expected_items=10, completed_items=9, tests_passed=True)['status'], 'INCOMPLETE')
        self.assertEqual(rc.make_receipt('r', self.artifacts, returncode=0, expected_items=10, completed_items=10, tests_passed=False)['status'], 'INCOMPLETE')
        with self.assertRaises(ValueError):
            rc.make_receipt('r', self.artifacts, returncode=0, expected_items=10, completed_items=11, tests_passed=True)

    def test_tampered_artifact_or_wrong_run_fails_validation(self):
        rc = self.api()
        r = rc.make_receipt('run-1', self.artifacts, returncode=0, expected_items=5, completed_items=5, tests_passed=True)
        self.assertFalse(rc.validate_receipt(r, self.artifacts, expected_run_id='run-2', expected_items=5))
        self.a.write_text('{"x": 2}', encoding='utf8')
        self.assertFalse(rc.validate_receipt(r, self.artifacts, expected_run_id='run-1', expected_items=5))

    def test_string_booleans_and_empty_artifacts_rejected(self):
        rc = self.api()
        with self.assertRaises(ValueError):
            rc.make_receipt('r', self.artifacts, returncode='0', expected_items=1, completed_items=1, tests_passed=True)
        with self.assertRaises(ValueError):
            rc.make_receipt('r', self.artifacts, returncode=0, expected_items=1, completed_items=1, tests_passed='true')
        self.assertEqual(rc.make_receipt('r', {}, returncode=0, expected_items=1, completed_items=1, tests_passed=True)['status'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
