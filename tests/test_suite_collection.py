"""The suite runner must run every test in tests/, or the number it prints is not the number of tests that passed.

Measured on 2026-09-25: `tests/run_all.py` reported "244 tests, 244 PASS" while pytest collected 288 from the same
directory. The 42-test gap was not random - `unittest.TestLoader` collects `TestCase` subclasses and nothing else,
so six files written as plain module-level `test_*` functions contributed ZERO tests each:

    test_connected_evaluation.py  test_delivery_integrity.py  test_integration_audit.py
    test_integration_pipeline.py  test_integration_scale.py   test_reasoning_window.py

Every "full suite N/N PASS" sentence in this campaign's reports, ledger entries and audits was therefore computed
over a suite that silently excluded six whole files, and seven of the hidden tests were in fact failing. The defect
has the same shape as the one an independent reviewer named for audit checks: a number that is not computed over
what it claims to cover is not evidence. A green suite that does not run part of itself is a green light nobody
earned.

The runner now delegates to pytest. These tests pin that decision from the outside: they ask the runner what it
would execute and compare it against what is actually in the directory. Collection runs in a subprocess, because
starting pytest from inside a pytest session is not a supported configuration and a test that needs one is not a
test of anything.
"""
import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source_test_names():
    """Every test callable in tests/, read from the SOURCE rather than by importing.

    Reading the text means this check does not depend on the modules importing cleanly, on the runner, or on
    pytest - the three things it is meant to hold to account.
    """
    found = {}
    for path in sorted((ROOT / 'tests').glob('test_*.py')):
        tree = ast.parse(path.read_text(encoding='utf8'))
        names = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
                names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                names += [m.name for m in node.body
                          if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith('test_')]
        if names:
            found[path.stem] = names
    return found


class SuiteCollectionTests(unittest.TestCase):
    collected = None

    @classmethod
    def setUpClass(cls):
        # pytest also prints its own --collect-only listing to stdout, so our rows carry a marker of their own
        code = ('import sys; sys.path.insert(0, %r); import tests.run_all as r; '
                'ids = r.collect_ids(); print(chr(10).join("COLLECTED	" + i for i in ids))' % str(ROOT))
        done = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, cwd=str(ROOT))
        if done.returncode != 0:
            raise unittest.SkipTest('could not ask the runner what it collects: ' + done.stderr[-400:])
        cls.collected = [line.split('	', 1)[1].strip() for line in done.stdout.splitlines()
                         if line.startswith('COLLECTED	')]

    def test_the_runner_collects_something_at_all(self):
        self.assertGreater(len(self.collected), 200, 'collection returned implausibly few tests')

    def test_every_test_in_the_directory_is_collected(self):
        ids = set(self.collected)
        missing = []
        for module, names in _source_test_names().items():
            for name in names:
                if not any(f'{module}.py::' in i and i.rsplit('::', 1)[-1].split('[')[0] == name for i in ids):
                    missing.append(f'{module}.{name}')
        self.assertEqual(missing, [], 'tests/run_all.py would never run these')

    def test_the_six_function_style_files_are_covered(self):
        """Named explicitly: these are the files the previous runner dropped in full."""
        dropped = ('test_connected_evaluation', 'test_delivery_integrity', 'test_integration_audit',
                   'test_integration_pipeline', 'test_integration_scale', 'test_reasoning_window')
        for module in dropped:
            self.assertTrue(any(f'{module}.py::' in i for i in self.collected),
                            f'{module} is collected by pytest but not by the runner')

    def test_no_test_is_counted_twice(self):
        duplicates = sorted({i for i in self.collected if self.collected.count(i) > 1})
        self.assertEqual(duplicates, [], 'a test counted twice inflates the pass total')


if __name__ == '__main__':
    unittest.main()
