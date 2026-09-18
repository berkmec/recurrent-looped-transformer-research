"""E12: completion receipts. COMPLETE only when the run id matches, the process exited 0, every expected item is present,
the required tests passed and every artifact hash matches at validation time. Everything else is INCOMPLETE.
CLI: python -B -m audit.receipt --out receipt.json --run-id ID --returncode N --expected-items N --completed-items N
     --tests-passed true|false --artifact name=path [--artifact ...]"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_receipt(run_id, artifacts, *, returncode, expected_items, completed_items, tests_passed):
    if not isinstance(run_id, str) or not run_id:
        raise ValueError('run_id must be a non-empty string')
    for name, value in (('returncode', returncode), ('expected_items', expected_items), ('completed_items', completed_items)):
        if type(value) is not int:
            raise ValueError(f'{name} must be an int, got {type(value).__name__}')
    if type(tests_passed) is not bool:
        raise ValueError('tests_passed must be a bool')
    if expected_items < 1 or completed_items < 0 or completed_items > expected_items:
        raise ValueError('invalid item counts')
    hashes = {name: dict(path=Path(path).as_posix(), sha256=sha(path), bytes=Path(path).stat().st_size) for name, path in artifacts.items()}
    complete = returncode == 0 and tests_passed and bool(hashes) and completed_items == expected_items
    return dict(run_id=run_id, returncode=returncode, tests_passed=tests_passed, expected_items=expected_items, completed_items=completed_items,
                artifacts=hashes, status='COMPLETE' if complete else 'INCOMPLETE', utc=time.time())


def validate_receipt(receipt, artifacts, *, expected_run_id, expected_items):
    try:
        if receipt['run_id'] != expected_run_id or receipt['expected_items'] != expected_items or receipt['status'] != 'COMPLETE':
            return False
        if set(receipt['artifacts']) != set(artifacts):
            return False
        for name, path in artifacts.items():
            if receipt['artifacts'][name]['sha256'] != sha(path):
                return False
        fresh = make_receipt(expected_run_id, artifacts, returncode=receipt['returncode'], expected_items=expected_items,
                             completed_items=receipt['completed_items'], tests_passed=receipt['tests_passed'])
        return fresh['status'] == 'COMPLETE'
    except (KeyError, TypeError, ValueError, OSError):
        return False


def write_receipt(path, receipt):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(receipt, f, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True); parser.add_argument('--run-id', required=True); parser.add_argument('--returncode', type=int, required=True)
    parser.add_argument('--expected-items', type=int, required=True); parser.add_argument('--completed-items', type=int, required=True)
    parser.add_argument('--tests-passed', choices=['true', 'false'], required=True); parser.add_argument('--artifact', action='append', default=[])
    args = parser.parse_args()
    artifacts, missing = {}, []
    for item in args.artifact:
        name, path = item.split('=', 1)
        (artifacts if Path(path).is_file() else missing).__setitem__(name, path) if Path(path).is_file() else missing.append(dict(name=name, path=path))
    receipt = make_receipt(args.run_id, artifacts, returncode=args.returncode, expected_items=args.expected_items, completed_items=args.completed_items,
                           tests_passed=args.tests_passed == 'true')
    if missing:
        receipt['missing_artifacts'] = missing; receipt['status'] = 'INCOMPLETE'
    write_receipt(args.out, receipt)
    print(json.dumps(dict(status=receipt['status'], run_id=receipt['run_id'], artifacts=sorted(receipt['artifacts']), missing=[m['name'] for m in missing])))


if __name__ == '__main__':
    main()
