"""E11: single test entry for this worktree. Runs every tests/test_*.py, lists every test by name with its status
(PASS/FAIL/ERROR/SKIP — a skip whose reason starts with MISSING_ASSET is reported separately), writes a JSON report
and exits non-zero on any failure or error. Usage: python -B tests/run_all.py [--out .superpowers/sdd/phase2/tests-report.json]"""
import argparse
import json
import os
import platform
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Recorder(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.rows = []

    def addSuccess(self, test):
        super().addSuccess(test); self.rows.append(dict(test=test.id(), status='PASS'))

    def addFailure(self, test, err):
        super().addFailure(test, err); self.rows.append(dict(test=test.id(), status='FAIL', detail=self._exc_info_to_string(err, test)[-800:]))

    def addError(self, test, err):
        super().addError(test, err); self.rows.append(dict(test=test.id(), status='ERROR', detail=self._exc_info_to_string(err, test)[-800:]))

    def addSkip(self, test, reason):
        super().addSkip(test, reason); self.rows.append(dict(test=test.id(), status='MISSING_ASSET' if str(reason).startswith('MISSING_ASSET') else 'SKIP', reason=str(reason)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT / '.superpowers' / 'sdd' / 'phase2' / 'tests-report.json'))
    parser.add_argument('--pattern', default='test_*.py')
    args = parser.parse_args()
    os.chdir(ROOT)
    # tests/ is a namespace package (no __init__.py): load each module by name, exactly as `python -m unittest tests.<name>` does
    names = ['tests.' + p.stem for p in sorted((ROOT / 'tests').glob(args.pattern))]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    t0 = time.perf_counter()
    runner = unittest.TextTestRunner(verbosity=1, resultclass=Recorder, stream=sys.stderr)
    result = runner.run(suite)
    rows = sorted(result.rows, key=lambda r: r['test'])
    counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    try:
        import torch; torch_version = torch.__version__
    except Exception:
        torch_version = None
    report = dict(root=str(ROOT), python=platform.python_version(), torch=torch_version, platform=platform.platform(), seconds=time.perf_counter() - t0,
                  tests_run=result.testsRun, counts=counts, success=result.wasSuccessful(), tests=rows,
                  scope='unit/contract tests of this worktree; MISSING_ASSET marks tests that need a pinned model or checkpoint that is absent')
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf8')
    print(json.dumps(dict(tests_run=result.testsRun, counts=counts, success=result.wasSuccessful(), report=str(out)), ensure_ascii=False))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
