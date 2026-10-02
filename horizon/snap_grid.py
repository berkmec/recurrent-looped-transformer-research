"""E-A9d: how sparse a cleanup does each architecture tolerate? The snap interval k for the Core, the GRU and the LSTM on ONE protocol (descriptive; program note 5.38). Evaluation only, CPU, dev split.

E-A9c (note 5.32.1) found that the self-snap, applied every step, makes the S5 GRU and LSTM exact like the Core, while every 16 steps only partly helps them and fully helps the Core (E-A9L: exact to
131,072 with k = 16). That is an efficiency difference (how long the carried state can be left alone), measured at k = 1 and 16 only and on different harnesses. Here the same measurement for the grid
k in 1, 2, 4, 8, 16, 32, 64, 128 on the SAME dev sequences (n = 200, 1,024 positions, the class selector: the class the model itself predicted at the previous step, centroids from the in-range
positions 8-15 of 2,000 dev sequences): the first-error hazard of [16, 1024) in events / person-steps per model and k, and `k_max` = the largest k in the grid up to which every smaller grid value also
gives 0 first errors (0 events in 201,600 person-steps bounds the hazard below ~1.5e-5). The Core runs with unrestricted attention paths, the recurrent arms through the hand-unrolled recurrence of `snap_recurrent`.

  python -B -m horizon.snap_grid results/Z1-pilot/perm5-core-w128-s1-cur.pt results/Z1-pilot/perm5-gru-w256-s1-cur.pt ... --out results/Z1-pilot/snap-grid.json

Descriptive: no verdict. Predictions (written before the first run, program note 5.38): k_max of at least three of the four S5 GRU / LSTM models is <= 4 (probability 0.65); k_max of both S5 Core
models is >= 32 (0.70); the Core's k_max is at least four times the largest recurrent k_max (0.60).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from . import depth_run as dr
from . import depth_tasks as dt
from .erosion_source import first_errors, window_counts, class_centroids
from .snap_long import run_length
from . import snap_recurrent as sr

GRID = (1, 2, 4, 8, 16, 32, 64, 128)
LENGTH, N = 1024, 200


def k_max(row):
    """The largest grid value k such that every grid value <= k has 0 first errors (0 when even k = 1 errs)."""
    best = 0
    for k in GRID:
        if row[str(k)]['events'] != 0:
            break
        best = k
    return best


def measure(weights, n=N, length=LENGTH, batch=50, log=print):
    meta = torch.load(weights, weights_only=True)['meta']
    if dt.is_count(meta['task']):
        raise ValueError('E-A9d is defined for the permutation tasks')
    model = dr.load_model(weights).eval()
    log(f'{weights}: {meta["arm"]} {meta["task"]} width {meta["width"]}')
    row = {}
    if meta['arm'] == 'core':
        prototypes = class_centroids(model, meta['task'], meta['seed'])
        out = run_length(model, prototypes, meta['task'], meta['seed'], length, n, None, batch, log, conditions=tuple(('class', k) for k in GRID))
        row = {str(k): dict(events=out[f'class-{k}']['events'], person_steps=out[f'class-{k}']['person_steps'], error_free=out[f'class-{k}']['error_free']) for k in GRID}
    elif meta['arm'] in ('gru', 'lstm'):
        x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
        centroids = sr.class_centroids(model, meta['task'], meta['seed'])
        for k in GRID:
            started = time.perf_counter()
            first = first_errors(sr.predictions(model, x, centroids, 'class', k, batch), y)
            events, person, _ = window_counts(first, 16, length)
            row[str(k)] = dict(events=events, person_steps=person, error_free=int((first >= length).sum()))
            log(f'  class-{k:<4} first errors {events:>3}/{person:>6}  error-free {row[str(k)]["error_free"]}/{n}  ({time.perf_counter() - started:.0f} s)')
    else:
        raise ValueError('E-A9d is defined for the core, gru and lstm arms')
    return dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, length=length, grid=row, k_max=k_max(row))


def describe(r):
    return (f'{r["arm"]} {r["task"]} w{r["width"]} ({Path(r["weights"]).stem}): k_max {r["k_max"]}  first errors by k: '
            + '  '.join(f'k={k} {r["grid"][str(k)]["events"]}' for k in GRID))


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='*')
    p.add_argument('--out', default=None)
    p.add_argument('--reading', default=None, help='a result file written by --out: print the table')
    a = p.parse_args()
    if a.reading:
        for r in json.loads(Path(a.reading).read_text(encoding='utf8')):
            print(describe(r))
        return
    results = []
    for path in a.weights:
        r = measure(path, log=lambda m: print(m, flush=True))
        results.append(r)
        print(describe(r), flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
