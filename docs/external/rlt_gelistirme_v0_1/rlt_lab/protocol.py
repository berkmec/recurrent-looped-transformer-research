"""Small, executable reproducibility contracts (not the user's runner patch)."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
from .memory import canonical, sha256


def atomic_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = canonical(data)+b'\n'
    fd,temp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as file:
            file.write(blob)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp,path)
        # Persist rename where supported. Windows still uses its own durability rules.
        if os.name == 'posix':
            dirfd = os.open(path.parent,os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def make_receipt(run_id: str, artifacts: dict[str,str | Path], *,
                 returncode: int, expected_items: int, completed_items: int,
                 tests_passed: bool) -> dict[str,Any]:
    if (not isinstance(run_id,str) or not run_id or type(returncode) is not int
            or type(tests_passed) is not bool or type(expected_items) is not int
            or type(completed_items) is not int):
        raise ValueError('invalid receipt field types')
    if expected_items < 1 or completed_items < 0 or completed_items > expected_items:
        raise ValueError('invalid item counts')
    hashes = {name:sha256(Path(path).read_bytes()) for name,path in artifacts.items()}
    complete = (returncode == 0 and tests_passed and bool(artifacts)
                and completed_items == expected_items)
    return dict(run_id=run_id, returncode=returncode, tests_passed=tests_passed,
                expected_items=expected_items, completed_items=completed_items,
                artifacts=hashes, status='COMPLETE' if complete else 'INCOMPLETE')


def validate_receipt(receipt: dict[str,Any], artifacts: dict[str,str | Path],
                     *, expected_run_id: str, expected_items: int) -> bool:
    try:
        expected = make_receipt(expected_run_id,artifacts,
            returncode=receipt['returncode'],expected_items=expected_items,
            completed_items=receipt['completed_items'],tests_passed=receipt['tests_passed'])
        return receipt == expected and expected['status'] == 'COMPLETE'
    except (OSError,ValueError,KeyError,TypeError):
        return False


def zero_error_upper_bound(n: int, confidence: float = 0.95) -> float:
    """One-sided binomial bound for ZERO errors among n independent accepts.

    Not valid for selected-after-test thresholds, dependent repeated seed
    outputs, nonzero errors, or certification of unobserved distributions.
    """
    if n < 1 or not 0 < confidence < 1:
        raise ValueError('n>=1 and confidence in (0,1) required')
    return -math.expm1(math.log1p(-confidence)/n)


def holm(pvalues: dict[str,float], alpha: float = 0.05) -> dict[str,bool]:
    if not 0 < alpha < 1 or any(not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError('invalid probabilities')
    ordered=sorted(pvalues, key=pvalues.get)
    result={k:False for k in ordered}
    for rank,key in enumerate(ordered):
        if pvalues[key] <= alpha/(len(ordered)-rank):
            result[key]=True
        else:
            break
    return result


def summarize_records(records: Iterable[dict[str,Any]], expected_ids: set[str]) -> dict[str,Any]:
    """Consume ALL records, reject duplicates/unexpected IDs; report incompleteness."""
    seen=set()
    counts: dict[str,int]={}
    accepted=wrong=0
    for row in records:
        key=row['id']
        if key not in expected_ids or key in seen:
            raise ValueError('duplicate/unexpected record')
        seen.add(key)
        label=str(row['prediction'])
        counts[label]=counts.get(label,0)+1
        if row.get('accepted',False):
            accepted+=1
            wrong+=int(not row['correct'])
    return dict(expected=len(expected_ids),processed=len(seen),
                missing=sorted(expected_ids-seen),prediction_counts=counts,
                coverage=accepted/len(expected_ids) if expected_ids else None,
                accepted_risk=wrong/accepted if accepted else None,
                status='COMPLETE' if seen==expected_ids else 'INCOMPLETE')


def manifest_jobs(problem_ids: list[str], arms: list[str], seeds: list[int]) -> list[dict[str,Any]]:
    """Full Cartesian product: no arm/problem/seed is silently removed."""
    if not problem_ids or not arms or not seeds:
        raise ValueError('empty manifest dimension')
    if len(set(problem_ids))!=len(problem_ids) or len(set(arms))!=len(arms) or len(set(seeds))!=len(seeds):
        raise ValueError('duplicate manifest dimensions')
    return [dict(problem=p,arm=a,seed=s) for p in problem_ids for a in arms for s in seeds]
