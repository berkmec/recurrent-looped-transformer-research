"""E-X: how far does exactness extend past the trained length (16)? Evaluation only, no training.

The Z1 permutation pilots were evaluated up to 1,024 positions (64x the trained length). A per-step first-error hazard of h means a sequence of length T survives with
probability about exp(-h T): 0 errors in 203,200 sequence-steps only bounds h below ~1.5e-5, which still allows ~30 % failures at T = 32,768. This evaluates saved S4 / S5
arms on fresh dev sequences of 2,048 ... 32,768 positions (up to 2,048x the trained length) and reports the first-error hazard per window beyond 1,024, pooled over
the evaluated lengths, plus the exactness horizon (the longest evaluated length at which EVERY sequence is error-free).

  python -B -m horizon.long_eval results/Z1-pilot/perm4-gru-w64-s1-cur.pt --lengths 2048,8192,32768 --n 100 --out results/Z1-pilot/long-eval.json

Reading rule (program note 5.13, written before the first run), applied to the windows beyond 1,024 positions, per model:
  CLEAN   no first error in any window beyond 1,024 and an exposure of at least 1e6 sequence-steps (so the hazard is below ~3e-6 per step, 95 %);
  BREAK   >= 5 first errors in the last window that has events, and its hazard is >= 10x the first window beyond 1,024 that has any exposure;
  FLOOR   >= 5 first errors beyond 1,024 and no BREAK (the rate does not rise 10x from the first to the last window): a constant per-step floor;
  SPARSE  1-4 first errors beyond 1,024 (a rate bound only);
  SHORT   no first error but less than 1e6 sequence-steps of exposure (not enough to call it clean).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from . import depth_run as dr
from . import depth_tasks as dt
from .hazard_summary import pooled_windows

EDGES = [0, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
BEYOND = 1024
CLEAN_EXPOSURE = 1_000_000


def measure(weights, lengths, n, batch=20):
    raw = torch.load(weights, weights_only=True)
    meta = raw['meta']
    if dt.is_count(meta['task']):
        raise ValueError('long evaluation is defined for the permutation tasks')
    model = dr.load_model(weights)
    curve = {}
    for length in lengths:
        started = time.perf_counter()
        x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
        pred, target = dr._accuracy(model, x, y, batch)
        ok = pred == target
        curve[str(length)] = dict(length=length, n=n, final=float(ok[:, -1].float().mean()), prefix=float(ok.float().mean()), error_free=int(ok.all(1).sum()),
                                  hazard=dr.first_error_hazard(pred, target, EDGES), seconds=time.perf_counter() - started)
    return dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, lengths=list(lengths), curve=curve,
                train=dict(parameters=dr.parameter_count(model)))


def exactness_horizon(result):
    """The longest evaluated length at which every sequence is error-free (None when even the shortest has an error)."""
    free = [int(k) for k, v in result['curve'].items() if v['error_free'] == v['n']]
    return max(free) if free else None


def reading(result):
    cells = {w: c for w, c in pooled_windows(result).items() if w[0] >= BEYOND and c[1] > 0}
    events, exposure = sum(c[0] for c in cells.values()), sum(c[1] for c in cells.values())
    if events == 0:
        verdict = 'CLEAN' if exposure >= CLEAN_EXPOSURE else 'SHORT'
    elif events < 5:
        verdict = 'SPARSE'
    else:
        with_events = [w for w, c in cells.items() if c[0] > 0]
        first, last = min(cells), max(with_events)
        rate = lambda w: cells[w][0] / cells[w][1]
        rising = last != first and cells[last][0] >= 5 and rate(last) >= 10 * rate(first)       # a zero first-window rate makes any later rate qualify
        verdict = 'BREAK' if rising else 'FLOOR'
    return dict(verdict=verdict, events=events, exposure=exposure, hazard=(events / exposure if exposure else None),
                windows={f'{a}-{b}': dict(events=c[0], exposure=c[1], hazard=c[0] / c[1]) for (a, b), c in cells.items()}, exactness_horizon=exactness_horizon(result))


def describe(result):
    r = reading(result)
    w = ' | '.join(f"[{k.replace('-', ',')}) {v['events']}/{v['exposure']}" for k, v in r['windows'].items())
    h = '' if r['hazard'] is None else f" = {r['hazard']:.2e}"
    return (f"{result['arm']} {result['task']} w{result['width']} ({result['train']['parameters']} parameters), n = {result['n']}, lengths {result['lengths']}: {r['verdict']}  "
            f"first errors beyond {BEYOND}: {r['events']}/{r['exposure']}{h}; exactness horizon {r['exactness_horizon']}; windows {w}")


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='+')
    p.add_argument('--lengths', default='2048,8192,32768')
    p.add_argument('--core-lengths', default=None, help='lengths for the Core arm (its decoder loop is sequential: use fewer / shorter)')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--core-n', type=int, default=40)
    p.add_argument('--batch', type=int, default=20)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    results = []
    for path in a.weights:
        arm = torch.load(path, weights_only=True)['meta']['arm']
        lengths = [int(v) for v in (a.core_lengths if (arm == 'core' and a.core_lengths) else a.lengths).split(',')]
        r = measure(path, lengths, a.core_n if arm == 'core' else a.n, a.batch)
        results.append(r)
        print(describe(r), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
