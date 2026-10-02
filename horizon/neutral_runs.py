"""E-N: neutral runs (identity padding). Does a trained S_N arm's per-step first-error rate come from TIME (the state drifts while nothing happens) or from WORK
(every real update adds a little error)?

Every token is the identity permutation (token 0: `perms[0]` of the group table) with probability p_id and a uniform random group element otherwise; the label is
the running product, so an identity token must leave the state exactly unchanged. The first-error rate is counted per STEP and per real UPDATE (non-identity tokens).
If errors come from work, the rate per update stays put and the rate per step falls like (1 - p_id); if they come from time, the rate per step stays put.

  python -B -m horizon.neutral_runs results/Z1-pilot/perm5-core-w128-s1-cur.pt --out results/Z1-pilot/neutral-runs.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import depth_run as dr
from . import depth_tasks as dt

P_IDS = (0.0, 0.5, 0.9, 0.99)
EDGES = (0, 16, 64, 128, 256, 512)
RULE_P = (0.5, 0.9)           # the sigma levels the reading rule uses: at 0.99 only a handful of real updates are left
LATE_FROM = 128


def neutral_tokens(task, n, length, p_id, seed):
    """(n, length) tokens: identity (token 0) with probability p_id, else a uniform random group element."""
    rng = np.random.default_rng(dt.source_seed(dr.NAMESPACE, task, seed, 'dev', step=int(round(p_id * 1000)), length=length, depth=7))
    keep = rng.random((n, length)) < p_id
    random_tokens = rng.integers(0, dt.vocab(task), size=(n, length))
    return torch.from_numpy(np.where(keep, 0, random_tokens).astype(np.int64))


def window_stats(pred, target, x, edges=EDGES):
    """{(start, stop): (first errors, step exposure, update exposure)} over the sequences still error-free at the window start; update exposure counts the
    non-identity tokens inside the window."""
    wrong = pred != target
    length = wrong.shape[1]
    first = torch.where(wrong.any(1), wrong.int().argmax(1), torch.full((wrong.shape[0],), length))
    real = (x != 0).int()
    out = {}
    for start, stop in zip(edges[:-1], edges[1:]):
        if start >= length:
            break
        stop = min(stop, length)
        at_risk = first >= start
        events = int((at_risk & (first < stop)).sum())
        out[(start, stop)] = (events, int(at_risk.sum()) * (stop - start), int(real[at_risk, start:stop].sum()))
    return out


def measure(weights, p_ids=P_IDS, length=512, n=100, seed=None, batch=50):
    raw = torch.load(weights, weights_only=True)
    meta = raw['meta']
    if dt.is_count(meta['task']):
        raise ValueError('neutral runs are defined for the permutation tasks')
    model = dr.load_model(weights)
    seed = meta['seed'] if seed is None else seed
    rows = {}
    for p in p_ids:
        x = neutral_tokens(meta['task'], n, length, p, seed)
        y = dt.oracle(meta['task'], x)
        pred, target = dr._accuracy(model, x, y, batch)
        stats = window_stats(pred, target, x)
        late = [v for (a, _), v in stats.items() if a >= LATE_FROM]
        events, steps, updates = (sum(v[i] for v in late) for i in range(3))
        rows[p] = dict(windows={f'{a}-{b}': list(v) for (a, b), v in stats.items()}, final=float((pred == target)[:, -1].float().mean()),
                       late_events=events, late_step_exposure=steps, late_update_exposure=updates,
                       rate_per_step=(events / steps if steps else None), rate_per_update=(events / updates if updates else None))
    return dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], parameters=meta['parameters'], length=length, n=n, seed=seed,
                results=rows)


def classify(result, low=0.6, high=1 / 0.6):
    """The reading rule (program note, section 5.7): for p_id in (0.5, 0.9), compare the late (>= 128) rate per step and per update with the p_id = 0 rate.
    WORK: the rate per update stays within [0.6, 1.67] of the p_id = 0 rate and is nearer to it (in log) than the rate per step; TIME: the same for the rate
    per step; NEUTRAL FAILURE: the rate per step is more than 3 times the p_id = 0 rate; otherwise OTHER. A model with no late errors at p_id = 0 has nothing to attribute."""
    rows = {float(p): r for p, r in result['results'].items()}
    base = rows.get(0.0)
    if not base or not base['rate_per_step']:
        return dict(verdict='NO BASELINE ERRORS', per_p={})
    if base['late_events'] < 10:
        return dict(verdict='TOO FEW BASELINE EVENTS', per_p={}, base_events=base['late_events'])      # fewer than 10 first errors: reported, no interpretation (the rule)
    per_p, votes = {}, []
    for p in RULE_P:
        r = rows.get(p)
        if not r or r['rate_per_step'] is None or r['rate_per_update'] is None or r['late_events'] < 10:
            per_p[p] = dict(verdict='TOO FEW EVENTS', events=(r or {}).get('late_events'))
            continue
        step_ratio, update_ratio = r['rate_per_step'] / base['rate_per_step'], r['rate_per_update'] / base['rate_per_update']
        if step_ratio > 3:
            verdict = 'NEUTRAL FAILURE'
        else:
            work_near, time_near = low <= update_ratio <= high, low <= step_ratio <= high
            nearer_work = abs(math.log(max(update_ratio, 1e-9))) < abs(math.log(max(step_ratio, 1e-9)))
            verdict = 'WORK' if work_near and nearer_work else 'TIME' if time_near and not nearer_work else 'OTHER'
        per_p[p] = dict(verdict=verdict, step_ratio=step_ratio, update_ratio=update_ratio, events=r['late_events'])
        votes.append(verdict)
    # NEUTRAL FAILURE needs only one evaluable p_id (a rate per step more than 3 times the base is a strong effect). WORK / TIME / OTHER must hold for BOTH p_id in
    # (0.5, 0.9) (the rule): when one of them has fewer than 10 events the overall verdict is 'TOO FEW EVENTS (one p_id)' and the per-p verdicts stay as information.
    # (Added after a blinded review: the first version issued WORK from p_id = 0.5 alone when p_id = 0.9 had 4 events.)
    evaluable = [p for p in RULE_P if per_p.get(p, {}).get('verdict') != 'TOO FEW EVENTS']
    if 'NEUTRAL FAILURE' in votes:
        overall = 'NEUTRAL FAILURE'
    elif len(evaluable) < len(RULE_P):
        overall = 'TOO FEW EVENTS (one p_id)' if votes else 'TOO FEW EVENTS'
    else:
        overall = votes[0] if all(v == votes[0] for v in votes) else 'MIXED'
    return dict(verdict=overall, per_p=per_p)


def describe(r):
    lines = [f"{r['arm']} {r['task']} w{r['width']} ({r['parameters']} parameters), {r['n']} dev sequences of length {r['length']}"]
    for p, row in r['results'].items():
        f = lambda v: 'n/a' if v is None else f'{v:.2e}'
        lines.append(f"  p_id {p}: final {row['final']:.2f} | late (>= {LATE_FROM}) first errors {row['late_events']} | per step {f(row['rate_per_step'])} "
                     f"per update {f(row['rate_per_update'])}")
    lines.append('  reading: ' + json.dumps(classify(r), default=str))
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='+')
    p.add_argument('--length', type=int, default=512)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    results = []
    for path in a.weights:
        r = measure(path, length=a.length, n=a.n)
        results.append(r)
        print(describe(r), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
