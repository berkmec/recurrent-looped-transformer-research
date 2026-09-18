"""Resource-bounded launcher around the frozen training runner.

No scientific parameters are changed here. Workers have disjoint static queues;
all state/protocol/terminal validation belongs to training.run_one.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from . import training as t


def event(path, **value):
    value['utc'] = datetime.now(timezone.utc).isoformat()
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def worker(out, role, slot=0, condition_id=None):
    out = Path(out)
    manifest = t.verify_frozen(out)
    name = f'{role}-{slot}' if role == 'small' else role + ('-' + condition_id if condition_id else '')
    directory = out / 'workers' / name
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / 'events.jsonl'
    with t.lock(directory):
        event(log, state='STARTED', pid=os.getpid(), protocol_hash=manifest['protocol_hash'], role=role, slot=slot)
        started = time.perf_counter()
        try:
            if role == 'small':
                if slot not in (0, 1):
                    raise ValueError('exactly two static small-model slots')
                for c in t.conditions(manifest['protocol'])[slot::2]:
                    cid = t.ident(c)
                    event(log, state='CONDITION_STARTED', condition_id=cid)
                    receipt = t.run_one(out, cid)
                    event(log, state='CONDITION_COMPLETE', condition_id=cid, updates=receipt['updates'],
                          selected_step=receipt['selected_step'], wall_seconds=receipt['wall_seconds'])
            elif role == 'one':
                if not condition_id:
                    raise ValueError('one requires condition id')
                receipt = t.run_one(out, condition_id)
                event(log, state='CONDITION_COMPLETE', condition_id=condition_id, updates=receipt['updates'])
            elif role == 'selection':
                result = t.run_selection(out)
                event(log, state='LR_SELECTION_COMPLETE', selected_lr=result['selection']['selected_lr'])
            elif role == 'evaluate':
                result = t.evaluate(out)
                event(log, state='EVALUATION_COMPLETE', status=result['status'], variants=len(result['conditions']))
            else:
                raise ValueError('unknown role')
        except BaseException as exc:
            event(log, state='FAILED', error_type=type(exc).__name__, error=str(exc), wall_seconds=time.perf_counter() - started)
            raise
        event(log, state='COMPLETE', wall_seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--role', required=True, choices=['small', 'one', 'selection', 'evaluate'])
    parser.add_argument('--slot', type=int, default=0)
    parser.add_argument('--condition')
    args = parser.parse_args()
    worker(args.out, args.role, args.slot, args.condition)


if __name__ == '__main__':
    main()
