"""Analysis that tolerates INCOMPLETE conditions (unpinned; the frozen run.py analyze() refuses them by design).

Same contrasts, pairing checks and decision rule as run.analyze, but an arm/task with a missing
seed is reported on the finished seeds with the t coefficient for df = n-1 and labelled INCOMPLETE.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from drift import run as r

T_975 = {1: 12.706204736, 2: 4.302652729, 3: 3.182446305}


def seed_summary(diffs):
    n = len(diffs)
    if n < 2:
        return dict(seed_effects=diffs, mean=diffs[0] if diffs else None, t95=None, direction='UNDETERMINED', seeds=n)
    mean = statistics.mean(diffs); half = T_975[n - 1] * statistics.stdev(diffs) / math.sqrt(n)
    return dict(seed_effects=diffs, mean=mean, t95=[mean - half, mean + half], seeds=n, df=n - 1,
                direction='POSITIVE_INTERVAL' if mean - half > 0 else 'NEGATIVE_INTERVAL' if mean + half < 0 else 'INCLUDES_ZERO')


def status_of(out, c):
    d = Path(out) / r.ident(c)
    if (d / 'evaluation.json').exists() and (d / 'complete.json').exists():
        return 'COMPLETE'
    if any(d.glob('failure-step*.json')):
        return 'FAILED_NONFINITE'
    return 'MISSING' if not d.exists() else 'INCOMPLETE'


def analyze(out):
    out = Path(out); manifest = r.verify_frozen(out); p = manifest['protocol']
    cs = r.conditions(p)
    endpoints = [f'L{n}' for n in p['test_lengths']] + ['neutral-0-base'] + [f'neutral-{s}-{pl}' for s in p['neutral_slots'] if s for pl in ('prefix', 'middle', 'suffix')]
    status = {r.ident(c): status_of(out, c) for c in cs}
    absolute = {}
    for c in cs:
        if status[r.ident(c)] != 'COMPLETE':
            continue
        ev = json.loads((out / r.ident(c) / 'evaluation.json').read_text(encoding='utf8'))
        absolute[r.ident(c)] = {**{f"L{x['length']}": x['final_accuracy'] for x in ev['lengths']},
                                **{f"neutral-{x['slots']}-{x['placement']}": x['final_accuracy'] for x in ev['neutral']}}
    contrasts = []
    for task in p['tasks']:
        for arm in p['arms']:
            if arm == 'base':
                continue
            for endpoint in endpoints:
                diffs, per_seed, missing = [], [], []
                for seed in p['primary_seeds']:
                    ca = next(c for c in cs if c['task'] == task and c['seed'] == seed and c['arm'] == arm)
                    cb = next(c for c in cs if c['task'] == task and c['seed'] == seed and c['arm'] == 'base')
                    if status[r.ident(ca)] != 'COMPLETE' or status[r.ident(cb)] != 'COMPLETE':
                        missing.append(dict(seed=seed, arm_status=status[r.ident(ca)], base_status=status[r.ident(cb)])); continue
                    a, b = r.load_raw(out, ca, endpoint), r.load_raw(out, cb, endpoint)
                    for key in ('inputs', 'targets', 'input_ids'):
                        if not np.array_equal(a[key], b[key]):
                            raise ValueError('UNPAIRED_RAW_' + key)
                    ok_a = a['predictions'][:, -1] == a['targets'][:, -1]; ok_b = b['predictions'][:, -1] == b['targets'][:, -1]
                    d = float(ok_a.mean() - ok_b.mean()); diffs.append(d)
                    per_seed.append(dict(seed=seed, arm_correct=int(ok_a.sum()), base_correct=int(ok_b.sum()), gained=int((ok_a & ~ok_b).sum()), lost=int((~ok_a & ok_b).sum()), difference=d))
                contrasts.append(dict(task=task, arm=arm, endpoint=endpoint, per_seed=per_seed, missing=missing, **seed_summary(diffs),
                                      primary=(task == p['primary']['task'] and endpoint in p['primary']['endpoints'])))
    decisions = {}
    for task in p['tasks']:
        for arm in p['arms']:
            if arm == 'base':
                continue
            prim = [c for c in contrasts if c['task'] == task and c['arm'] == arm and c['endpoint'] in p['primary']['endpoints']]
            cost = next(c for c in contrasts if c['task'] == task and c['arm'] == arm and c['endpoint'] == p['primary']['cost_endpoint'])
            incomplete = any(c['missing'] for c in prim + [cost])
            if any(c['mean'] is None for c in prim + [cost]):
                verdict = 'UNDETERMINED'
            else:
                positive = all(c['mean'] * 100 >= p['primary']['gain_pp'] and c['direction'] == 'POSITIVE_INTERVAL' for c in prim) and cost['mean'] * 100 >= -p['primary']['max_cost_pp']
                negative = any(c['mean'] <= 0 for c in prim) or cost['mean'] * 100 < -p['primary']['max_cost_pp']
                verdict = 'MEASURED_POSITIVE' if positive else 'MEASURED_NEGATIVE' if negative else 'MIXED'
            decisions[f'{task}/{arm}'] = dict(status=verdict, incomplete=incomplete, seeds_used=min(c['seeds'] for c in prim + [cost]),
                                              primary=[dict(endpoint=c['endpoint'], mean_pp=None if c['mean'] is None else c['mean'] * 100,
                                                            t95_pp=None if c['t95'] is None else [v * 100 for v in c['t95']], direction=c['direction'], missing=c['missing']) for c in prim],
                                              cost=dict(endpoint=cost['endpoint'], mean_pp=None if cost['mean'] is None else cost['mean'] * 100,
                                                        t95_pp=None if cost['t95'] is None else [v * 100 for v in cost['t95']]),
                                              note='primary task per plan is five; parity rows are secondary' if task != p['primary']['task'] else None)
    result = dict(status='MEASURED_WITH_INCOMPLETE' if any(v != 'COMPLETE' for v in status.values()) else 'MEASURED',
                  protocol_hash=manifest['protocol_hash'], plan_sha256=r.sha(r.ROOT / 'docs/drift-mini-plan.md'),
                  condition_status=status, complete=sum(v == 'COMPLETE' for v in status.values()), declared=len(cs),
                  contrasts=contrasts, absolute=absolute, decisions=decisions,
                  caveat='t coefficient uses df = seeds-1 (12.706 for two seeds); exploratory, no multiplicity adjustment; INCOMPLETE arms never promoted to three-seed claims.')
    r.write_json(out / 'analysis-tolerant.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = analyze(args.out)
    print(json.dumps(dict(status=result['status'], complete=result['complete'], declared=result['declared'], decisions=result['decisions']), indent=1, allow_nan=False))


if __name__ == '__main__':
    main()
