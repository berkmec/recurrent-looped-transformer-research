"""E-A9b: does the snap work without ANY label? The prototypes are found by k-means on the model's own in-range states. Evaluation only (no training), CPU, dev split.

E-A9 (note 5.27) calibrated the snap's class centroids with the true class labels of in-range dev sequences (the application of the snap needs no label, but the calibration did). A real model
has no such labels. Here the carried states of in-range sequences (positions 8..15 of 2,000 dev sequences; no label used) are clustered by k-means (k = the number of classes, k-means++
initialisation with a fixed seed, 25 Lloyd iterations) and the cluster centroids replace the labelled ones as prototypes of the `nearest` selector (the closest centroid in state space; no
readout, no label anywhere). The clusters are scored against the true classes only for reporting (purity: the share of states that fall in a cluster whose majority class is their own).

  python -B -m horizon.snap_labelfree results/Z1-pilot/perm5-core-w128-s1-cur.pt ... --out results/Z1-pilot/snap-labelfree.json

Reading rule (written before the first run, program note 5.30): the first-error hazard of [16, 1024) in events / person-steps on the same 200 dev sequences as E-A9, untouched run as reference,
the classification of E-A5 (`erosion_source.classify`: RESCUES R >= 5 and p < 0.01, PARTIAL 2 <= R < 5, HARMS, NEUTRAL). Primary condition `kmeans-16` (every 16 steps), the three S5 Cores
(constant-lr, annealed, annealed with state noise); the two S4 Cores are the control. Verdict: >= 2 of the 3 S5 models RESCUES -> LABEL-FREE SNAP WORKS; no S5 model RESCUES or PARTIAL and at least
two NEUTRAL -> LABEL-FREE SNAP DOES NOT WORK; otherwise MIXED (and INCONCLUSIVE when the S4 control is HARMS or the in-range accuracy under the primary snap falls below 0.99 on any model).
Descriptive: `kmeans-1` (every step), the cluster purity, the labelled-centroid `nearest-16` of E-A9 next to it.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from . import depth_run as dr
from . import depth_tasks as dt
from .erosion_source import classify, core_trajectory, first_errors, predictions, stored_control, window_counts, WINDOW_FROM
from .self_snap import configure

CONDITIONS = (('kmeans', 16), ('kmeans', 1))
PRIMARY = 'kmeans-16'
CONTROL_ACCURACY = 0.99
ITERATIONS = 25
SEED = 0


def kmeans(points, k, iterations=ITERATIONS, seed=SEED):
    """Lloyd's algorithm with k-means++ initialisation: (centroids (k, d), assignment (n,))."""
    g = torch.Generator().manual_seed(seed)
    n = len(points)
    centroids = points[torch.randint(n, (1,), generator=g)]
    for _ in range(1, k):
        dist = torch.cdist(points, centroids).min(1).values.square()
        centroids = torch.cat([centroids, points[torch.multinomial(dist / dist.sum(), 1, generator=g)]])
    for _ in range(iterations):
        assignment = torch.cdist(points, centroids).argmin(1)
        for c in range(k):
            members = points[assignment == c]
            if len(members):
                centroids[c] = members.mean(0)
    return centroids, torch.cdist(points, centroids).argmin(1)


def purity(assignment, labels, k):
    """The share of points whose cluster's majority label is their own label."""
    right = 0
    for c in range(k):
        members = labels[assignment == c]
        if len(members):
            right += int(torch.bincount(members).max())
    return right / len(labels)


def in_range_states(model, task, seed, n=2000, length=16, positions=(8, 16)):
    x, y, _ = dt.heldout(dr.NAMESPACE, task, seed, 'dev', n, length)
    with torch.no_grad():
        states, _ = core_trajectory(model.eval(), x)
    a, b = positions
    return states[:, a:b].reshape(-1, states.shape[-1]), y[:, a:b].reshape(-1)


def measure(weights, n=200, length=1024, batch=40, log=print):
    meta = torch.load(weights, weights_only=True)['meta']
    if meta['arm'] not in ('core', 'core_clean') or dt.is_count(meta['task']):
        raise ValueError('E-A9b is defined for the Core on the permutation tasks')
    model = dr.load_model(weights).eval()
    k = dt.classes(meta['task'])
    states, labels = in_range_states(model, meta['task'], meta['seed'])
    centroids, assignment = kmeans(states, k)
    x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
    log(f'{weights}: {meta["task"]} width {meta["width"]}, k = {k} clusters, purity {purity(assignment, labels, k):.3f}')
    out, base_first = {}, None
    for name, (selector, every) in [('base', (None, 1))] + [(f'{s}-{e}', (s, e)) for s, e in CONDITIONS]:
        started = time.perf_counter()
        configure(model, None if name == 'base' else centroids, 'nearest' if selector else 'class', every, 1.0)
        pred = predictions(model, x, batch)
        configure(model)
        first = first_errors(pred, y)
        base_first = first if name == 'base' else base_first
        events, person, at_risk = window_counts(first, WINDOW_FROM, length)
        out[name] = dict(events=events, person_steps=person, at_risk=at_risk, final=float((pred[:, -1] == y[:, -1]).float().mean()))
        log(f'  {name:<10} first errors {events:>3}/{person:>6} person-steps = {events / max(person, 1):.2e}  final {out[name]["final"]:.2f}  ({time.perf_counter() - started:.0f} s)')
    short_x, short_y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', 200, 16)
    accuracy = {}
    for name, (selector, every) in [(f'{s}-{e}', (s, e)) for s, e in CONDITIONS]:
        configure(model, centroids, 'nearest', every, 1.0)
        accuracy[name] = float((predictions(model, short_x, 50) == short_y).float().mean())
        configure(model)
    return dict(weights=str(weights), task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, length=length, parameters=dr.parameter_count(model), clusters=k,
                purity=purity(assignment, labels, k), conditions=out, in_range=accuracy, stored_control=stored_control(weights, base_first))


def labels_of(result):
    base = result['conditions']['base']
    return {name: classify((base['events'], base['person_steps']), (c['events'], c['person_steps']))[0] for name, c in result['conditions'].items() if name != 'base'}


def reading(results):
    s5 = [r for r in results if r['task'] == 'perm5']
    s4 = [r for r in results if r['task'] == 'perm4']
    if not s5:
        return dict(verdict='NO S5 MODEL')
    primary = [labels_of(r)[PRIMARY] for r in s5]
    control = [labels_of(r)[PRIMARY] for r in s4]
    weak = [r for r in s5 + s4 if r['in_range'][PRIMARY] < CONTROL_ACCURACY - 1e-9]
    rescues = primary.count('RESCUES')
    helped = rescues + primary.count('PARTIAL')
    if any(r.get('stored_control') is False for r in s5 + s4):
        verdict = 'CONTROL FAILED: the untouched run does not reproduce the stored first-error windows'
    elif rescues >= 2 and not weak:
        verdict = 'LABEL-FREE SNAP WORKS'
    elif weak or 'HARMS' in control:
        verdict = 'INCONCLUSIVE (the snap harms the in-range accuracy or the S4 control)'
    elif helped == 0 and primary.count('NEUTRAL') >= 2:
        verdict = 'LABEL-FREE SNAP DOES NOT WORK'
    else:
        verdict = 'MIXED'
    return dict(verdict=verdict, primary=primary, control=control, models={f'{r["task"]} w{r["width"]} {Path(r["weights"]).stem}': labels_of(r) for r in s5 + s4})


def describe(r):
    return (f'{r["task"]} w{r["width"]}: k = {r["clusters"]}, purity {r["purity"]:.3f}, stored-window control {r["stored_control"]}\n  [16,{r["length"]}): '
            + '  '.join(f'{k} {v["events"]}/{v["person_steps"]}' for k, v in r['conditions'].items())
            + '\n  in-range accuracy: ' + '  '.join(f'{k} {v:.3f}' for k, v in r['in_range'].items()) + '\n  labels: ' + '  '.join(f'{k} {v}' for k, v in labels_of(r).items()))


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='*')
    p.add_argument('--out', default=None)
    p.add_argument('--reading', default=None, help='a result file written by --out: print the registered verdict')
    a = p.parse_args()
    if a.reading:
        results = json.loads(Path(a.reading).read_text(encoding='utf8'))
        for r in results:
            print(describe(r))
        print(json.dumps(reading(results), indent=1))
        return
    results = []
    for path in a.weights:
        r = measure(path, log=lambda m: print(m, flush=True))
        results.append(r)
        print(describe(r), flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')
    if a.out:
        print(json.dumps(reading(results), indent=1))


if __name__ == '__main__':
    main()
