"""E-A9L: how long does the self-snapped S5 Core stay exact? Evaluation only (no training), CPU, dev split.

E-A9 (note 5.27) snaps the carried state toward the centroid of the class the model itself predicted (or the nearest centroid), every step or every 16 steps; the S5 Core that fails every
sequence within 1,024 positions without it (hazard ~2.6e-3 per step) makes no error in 201,600 sequence-steps with it. A zero count over 1,024 positions bounds the per-step hazard only
below ~1.5e-5, which still allows a failure within a long sequence (the E-X argument, note 5.13). This evaluates the snap on very long dev sequences (32,768 and 131,072 positions) and reports
the first-error count against the person-steps spent, with the same CLEAN / SPARSE / FLOOR labels as E-X (CLEAN: no error and >= 1e6 person-steps, so the hazard is below ~3e-6 per step).

Speed: the encoder self-attention over a 32,768-token prefix and the decoder's cross-attention over the whole memory dominate the cost. E-A5 (note 5.26.1) showed that restricting both to the last
64 positions leaves the constant-lr and annealed S5 Cores' predictions and hazard unchanged, so the long runs use that window by default (`--window 64`); the equivalence is checked again here on
shorter sequences (`equivalence`: the fraction of positions at which the windowed and the unwindowed snapped run predict the same class).

  python -B -m horizon.snap_long results/Z1-pilot/perm5-core-w128-s1-cur.pt --lengths 32768,131072 --n 40,8 --out results/Z1-pilot/snap-long.json

Reading rule (written before the first run, program note 5.29): per model and condition, first errors / person-steps over the positions after the first 16 of every sequence (the E-X
definition): CLEAN (0 errors, >= 1e6 person-steps), SPARSE (1-4 errors), FLOOR (>= 5 errors); the verdict of the experiment, read on the primary condition `class-16` and the two S5 Cores
(constant-lr and annealed): EXACT TO 131,072 if both are CLEAN or SPARSE at 131,072; EXACT TO 32,768 if both are CLEAN or SPARSE at 32,768 and not at 131,072; NOT EXACT otherwise (at least one
model has a FLOOR already at 32,768). Read monotonically: `EXACT TO L` needs BOTH models to be CLEAN or SPARSE at every evaluated length up to L (a model that is a FLOOR at 32,768 and SPARSE at 131,072 is NOT EXACT: blind
review F3). `SHORT` (no error but fewer than 1e6 person-steps, e.g. when a sequence fails before position 16 and leaves the risk set) counts as not OK, as registered. Descriptive: the other conditions, the first-error
window of any error, and the window / no-window equivalence.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from core.model import Core

from . import depth_run as dr
from . import depth_tasks as dt
from .erosion_source import ProbeCore, class_centroids, first_errors, predictions, window_counts
from .self_snap import SnapCore

CONDITIONS = (('class', 16), ('class', 1), ('nearest', 16))
PRIMARY = 'class-16'
ALL_CONDITIONS_UP_TO = 32768             # beyond this length only the primary condition runs (the sequential decoder makes every extra condition cost minutes)
BATCH_POSITIONS = 300_000                # sequences per batch = this / length, so the encoder and the memory of one batch stay near a gigabyte
CLEAN_EXPOSURE = 1_000_000
EDGES = (16, 256, 1024, 4096, 16384, 32768, 65536, 131072)


class LongCore(SnapCore, ProbeCore):
    """The sealed Core with the self-snap (class-chosen centroid every `every` steps) AND the optional attention windows of E-A5; plain Core when neither is configured."""


def configure(model, prototypes=None, selector='class', every=16, window=None):
    if not isinstance(model, Core):
        raise ValueError('the long snap evaluation is defined for the Core arm')
    if window is not None and window < 1:
        raise ValueError('the window must be >= 1')
    model.__class__ = LongCore
    model.prototypes, model.selector, model.every, model.strength = prototypes, selector, every, 1.0
    model.cross = model.enc = window
    model.at, model.state, model.reset_cache = None, None, False
    return model


def label(selector, every):
    return f'{selector}-{every}'


def reading_label(events, person_steps):
    if events == 0:
        return 'CLEAN' if person_steps >= CLEAN_EXPOSURE else 'SHORT'
    return 'SPARSE' if events < 5 else 'FLOOR'


def equivalence(model, prototypes, task, seed, length=4096, n=8, window=64, batch=8):
    """The fraction of positions at which the windowed and the unwindowed run of the primary condition predict the same class (and the fraction of sequences whose predictions agree everywhere)."""
    x, y, _ = dt.heldout(dr.NAMESPACE, task, seed, 'dev', n, length)
    configure(model, prototypes, 'class', 16, None)
    full = predictions(model, x, batch)
    configure(model, prototypes, 'class', 16, window)
    windowed = predictions(model, x, batch)
    configure(model)
    same = full == windowed
    return dict(length=length, n=n, window=window, positions=float(same.float().mean()), sequences=float(same.all(1).float().mean()),
                first_errors_full=int((first_errors(full, y) < length).sum()), first_errors_windowed=int((first_errors(windowed, y) < length).sum()))


def run_length(model, prototypes, task, seed, length, n, window, batch, log, conditions=CONDITIONS):
    x, y, _ = dt.heldout(dr.NAMESPACE, task, seed, 'dev', n, length)
    out = {}
    for selector, every in conditions:
        started = time.perf_counter()
        configure(model, prototypes, selector, every, window)
        pred = predictions(model, x, batch)
        configure(model)
        first = first_errors(pred, y)
        events, person, at_risk = window_counts(first, 16, length)
        edges = [e for e in EDGES if e < length] + [length]
        out[label(selector, every)] = dict(events=events, person_steps=person, label=reading_label(events, person), final=float((pred[:, -1] == y[:, -1]).float().mean()),
                                           error_free=int((first >= length).sum()), windows={f'{a}-{b}': dict(zip(('events', 'person_steps', 'at_risk'), window_counts(first, a, b))) for a, b in zip(edges[:-1], edges[1:])})
        log(f'  length {length:>6} n {n:>2} {label(selector, every):<11} first errors {events}/{person} person-steps  {out[label(selector, every)]["label"]}  '
            f'error-free sequences {out[label(selector, every)]["error_free"]}/{n}  ({time.perf_counter() - started:.0f} s)')
    return out


def measure(weights, lengths=(32768, 131072), ns=(40, 8), window=64, batch=20, log=print, equivalence_length=4096):
    meta = torch.load(weights, weights_only=True)['meta']
    if meta['arm'] not in ('core', 'core_clean') or dt.is_count(meta['task']):
        raise ValueError('E-A9L is defined for the Core on the permutation tasks')
    model = dr.load_model(weights).eval()
    log(f'{weights}: {meta["task"]} width {meta["width"]}')
    prototypes = class_centroids(model, meta['task'], meta['seed'])
    result = dict(weights=str(weights), task=meta['task'], width=meta['width'], seed=meta['seed'], window=window,
                  equivalence=equivalence(model, prototypes, meta['task'], meta['seed'], length=equivalence_length, window=window if window else 64), lengths={})
    log('  equivalence ' + json.dumps(result['equivalence']))
    for length, n in zip(lengths, ns):
        conditions = CONDITIONS if length <= ALL_CONDITIONS_UP_TO else CONDITIONS[:1]
        result['lengths'][str(length)] = dict(n=n, conditions=run_length(model, prototypes, meta['task'], meta['seed'], length, n, window, max(1, min(batch, BATCH_POSITIONS // length)), log, conditions))
    return result


def reading(results):
    """The registered verdict on the primary condition over the S5 Core models (module docstring)."""
    s5 = [r for r in results if r['task'] == 'perm5']
    if not s5:
        return dict(verdict='NO S5 MODEL')
    longest = sorted({int(k) for r in s5 for k in r['lengths']})
    per_length = {}
    for length in longest:
        labels = [r['lengths'][str(length)]['conditions'][PRIMARY]['label'] for r in s5 if str(length) in r['lengths']]
        per_length[length] = dict(labels=labels, ok=all(v in ('CLEAN', 'SPARSE') for v in labels) and len(labels) == len(s5))
    best = None
    for length in longest:                                           # monotone (blind review F3): EXACT TO L needs every evaluated length <= L to be CLEAN / SPARSE for both models
        if not per_length[length]['ok']:
            break
        best = length
    verdict = 'NOT EXACT' if best is None else f'EXACT TO {best:,}'
    return dict(verdict=verdict, per_length={str(k): v for k, v in per_length.items()})


def describe(r):
    lines = [f'{r["task"]} w{r["width"]} (window {r["window"]}); equivalence {json.dumps(r["equivalence"])}']
    for length, row in r['lengths'].items():
        lines.append(f'  {length}: ' + '  '.join(f'{k} {v["events"]}/{v["person_steps"]} {v["label"]}' for k, v in row['conditions'].items()))
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='*')
    p.add_argument('--lengths', default='32768,131072')
    p.add_argument('--n', default='40,8', help='sequences per length, comma list matching --lengths')
    p.add_argument('--window', type=int, default=64, help='encoder / cross-attention window (0 = none)')
    p.add_argument('--batch', type=int, default=20)
    p.add_argument('--out', default=None)
    p.add_argument('--reading', default=None, help='a result file written by --out: print the registered verdict')
    a = p.parse_args()
    if a.reading:
        results = json.loads(Path(a.reading).read_text(encoding='utf8'))
        for r in results:
            print(describe(r))
        print(json.dumps(reading(results), indent=1))
        return
    lengths = tuple(int(v) for v in a.lengths.split(','))
    ns = tuple(int(v) for v in a.n.split(','))
    results = []
    for path in a.weights:
        r = measure(path, lengths, ns, a.window or None, a.batch, log=lambda m: print(m, flush=True))
        results.append(r)
        print(describe(r), flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')
    if a.out:
        print(json.dumps(reading(results), indent=1))


if __name__ == '__main__':
    main()
