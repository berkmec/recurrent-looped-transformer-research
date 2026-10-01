"""E11: single test entry for this worktree. Runs every test in tests/, lists every test by name with its status
(PASS/FAIL/ERROR/SKIP — a skip whose reason starts with MISSING_ASSET is reported separately), writes a JSON report
and exits non-zero on any failure or error. Usage: python -B tests/run_all.py [--out .superpowers/sdd/phase2/tests-report.json]

2026-09-25 — WHY THIS DELEGATES TO PYTEST NOW. It used to drive `unittest`, which collects `TestCase` subclasses
and nothing else. Six files here are written as plain module-level `test_*` functions, half of them taking pytest
fixtures (`tmp_path`, `monkeypatch`, and fixtures defined in the modules themselves), so `unittest` contributed
ZERO tests from each of them. The runner printed "244 tests, 244 PASS" while the directory held 42 more, seven of
which were failing. Every "full suite N/N PASS" sentence in this campaign's reports, ledger and audits inherited
that number. A suite that does not run part of itself is not evidence that the part it skipped is green - the same
shape as the finding an independent reviewer named for audit checks whose evidence was a sentence. Wrapping the
functions in `FunctionTestCase` would not have fixed it either: without fixture support half of them cannot run
under `unittest` at all. So collection and execution go to pytest, which runs both styles, and this file keeps its
JSON report schema unchanged for the audits that read it.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Recorder:
    """One row per test, taken from pytest's report objects rather than parsed out of its console output."""

    def __init__(self):
        self.rows = {}
        self.collected = []

    def pytest_collection_modifyitems(self, items):
        self.collected = [item.nodeid for item in items]

    def pytest_runtest_logreport(self, report):
        row = self.rows.setdefault(report.nodeid, dict(test=report.nodeid, status='PASS'))
        if report.outcome == 'passed':
            return
        if report.outcome == 'skipped':
            reason = report.longrepr[2] if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3 else ''
            reason = str(reason).removeprefix('Skipped: ')
            row.update(status='MISSING_ASSET' if reason.startswith('MISSING_ASSET') else 'SKIP', reason=reason)
        elif report.outcome == 'failed':
            # a failure during setup or teardown is an error, the same distinction the unittest runner drew
            row.update(status='FAIL' if report.when == 'call' else 'ERROR', detail=str(report.longrepr)[-800:])

    def ordered(self):
        return sorted(self.rows.values(), key=lambda r: r['test'])


def _pytest():
    try:
        import pytest
    except ImportError as exc:              # the venv is read-only, so this is a setup fault and must not read as a pass
        raise SystemExit('pytest is required to run the whole suite (six files need its fixtures): %s' % exc)
    return pytest


def _argv(extra=()):
    # -p no:randomly keeps the order stable, so one report can be diffed against the previous one
    return ['tests', '-q', '-p', 'no:randomly', '--rootdir', str(ROOT), *extra]


def collect_ids():
    """Every test id this runner will execute. tests/test_suite_collection.py reads it to keep the gap closed."""
    recorder = Recorder()
    _pytest().main(_argv(['--collect-only']), plugins=[recorder])
    return list(recorder.collected)


def run():
    recorder = Recorder()
    code = _pytest().main(_argv(), plugins=[recorder])
    return recorder.ordered(), int(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT / '.superpowers' / 'sdd' / 'phase2' / 'tests-report.json'))
    args = parser.parse_args()
    os.chdir(ROOT)
    t0 = time.perf_counter()
    rows, code = run()
    counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    success = not (counts.get('FAIL') or counts.get('ERROR')) and code == 0
    try:
        import torch; torch_version = torch.__version__
    except Exception:
        torch_version = None
    report = dict(root=str(ROOT), python=platform.python_version(), torch=torch_version, platform=platform.platform(),
                  seconds=time.perf_counter() - t0, tests_run=len(rows), counts=counts, success=success, tests=rows,
                  runner='pytest; see this module docstring - unittest silently excluded 42 tests in six files until 2026-09-25',
                  scope='unit/contract tests of this worktree; MISSING_ASSET marks tests that need a pinned model or checkpoint that is absent')
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf8')
    print(json.dumps(dict(tests_run=len(rows), counts=counts, success=success, report=str(out)), ensure_ascii=False))
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
