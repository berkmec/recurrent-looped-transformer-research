"""E-A9k: how rarely may the self-snap fire? A dose-response of the snap interval k against the renewal prediction of the E-A5 injection hazards. Evaluation only, CPU, dev split (program note 5.37).

E-A9 / E-A9L snap the carried state to the centroid of the class the model itself predicted every 16 steps (and every step): exact to 131,072 positions. E-A5x (note 5.26.2) found that the
carried state of the S5 Core diffuses (squared deviation linear in its age, D ~ 1e-3 M^2 per step, the same at every position) and that the first-error hazard after an oracle injection is zero for
the first 64 steps of age and returns in [64, 128). Until its first error the snap is the same intervention as a repeated oracle injection (the predicted class is the true class), so a snap every k
steps should have the first-error hazard of the injection experiment averaged over the ages 0 .. k-1 (a RENEWAL model: the age is reset every k steps and nothing else accumulates):

    h(k) = (1 / k) * sum over a < k of h_injection(a),      h_injection(a) = 0 for a < 64 (0 events in 38,400 person-steps) and h64 for 64 <= a < 128

with h64 the pooled E-A5 `state+cache` injection hazard of [64, 128) (events / person-steps over t0 = 128, 256, 512), read from the stored `erosion-source.json`; the prediction is made only for
k <= 128 (E-A5 followed 128 steps). Conditions `class-k` for k in 32, 64, 96, 128 (predicted) and 256 (descriptive); the attention paths windowed to the last 64 positions as in E-A9L (equivalence
measured there). Each S5 Core model: n = 40 dev sequences of 8,192 positions; the first-error hazard of [16, 8192) in events / person-steps.

  python -B -m horizon.snap_interval results/Z1-pilot/perm5-core-w128-s1-cur.pt results/Z1-pilot/perm5-core-w128-s1-cur-lrdecay0.3.pt --out results/Z1-pilot/snap-interval.json

Reading rule (written before the first run, program note 5.37): per model and per k in {96, 128} the measured hazard is CONSISTENT with the renewal prediction h(k) when the exact 95 % Poisson
interval of its event count (given its person-steps) intersects [0.4 h(k), 2.5 h(k)]; for k in {32, 64} it is CONSISTENT when the measured hazard is at most 5e-5 (no breakdown). RENEWAL HOLDS when
every one of those eight cells is consistent; SNAP BREAKS EARLIER THAN A RESET when some model has, at some k in {32, 64, 96, 128}, a measured hazard above both 2.5 h(k) and 5e-5; SNAP HOLDS LONGER THAN
A RESET when at k = 128 both models have a measured hazard below 0.4 h(128) (and none of the above); MIXED otherwise; INCOMPLETE without two S5 models. Descriptive: `class-256`, the number of error-free
sequences per condition, the windows of the hazard.
"""
import argparse
import json
from pathlib import Path

import torch
from scipy.stats import chi2

from . import depth_run as dr
from . import depth_tasks as dt
from .erosion_source import class_centroids
from .snap_long import run_length

INTERVALS = (32, 64, 96, 128, 256)
PREDICTED = (32, 64, 96, 128)
LENGTH, N = 8192, 40
BREAK_FLOOR = 5e-5
BAND = (0.4, 2.5)
T0S = ('128', '256', '512')
AGE_BINS = (('0-16', 0, 16), ('16-32', 16, 32), ('32-64', 32, 64), ('64-128', 64, 128))


def age_hazard(erosion_json, weights):
    """{bin: (events, person_steps)} of the E-A5 `state+cache` injection runs of this model, pooled over t0 = 128, 256, 512 (steps since the injection)."""
    results = json.loads(Path(erosion_json).read_text(encoding='utf8'))
    stem = Path(weights).stem
    for r in results:
        if Path(r['weights']).stem == stem:
            table = {name: [0, 0] for name, _, _ in AGE_BINS}
            for t0 in T0S:
                for name, _, _ in AGE_BINS:
                    cell = r['injection'][t0]['state+cache']['since'][name]
                    table[name][0] += cell['events']
                    table[name][1] += cell['person_steps']
            return {k: tuple(v) for k, v in table.items()}
    raise ValueError(f'{stem} is not in {erosion_json}')


def renewal_hazard(k, table):
    """The renewal prediction h(k) for a snap every k <= 128 steps (module docstring); the hazard of a bin is taken uniform inside it."""
    if k > 128:
        return None
    total = 0.0
    for name, lo, hi in AGE_BINS:
        events, person = table[name]
        covered = max(0, min(k, hi) - lo)
        total += covered * (events / person if person else 0.0)
    return total / k


def poisson_interval(events, alpha=0.05):
    """The exact (Garwood) two-sided interval of a Poisson mean given `events` observations."""
    lo = 0.0 if events == 0 else float(chi2.ppf(alpha / 2, 2 * events) / 2)
    hi = float(chi2.ppf(1 - alpha / 2, 2 * (events + 1)) / 2)
    return lo, hi


def consistent(events, person_steps, predicted, k):
    if person_steps == 0:
        return False
    if k in (32, 64):
        return events / person_steps <= BREAK_FLOOR
    lo, hi = poisson_interval(events)
    return lo / person_steps <= BAND[1] * predicted and hi / person_steps >= BAND[0] * predicted


def measure(weights, erosion_json='results/Z1-pilot/erosion-source.json', length=LENGTH, n=N, window=64, batch=20, log=print):
    meta = torch.load(weights, weights_only=True)['meta']
    if meta['arm'] != 'core' or dt.is_count(meta['task']):
        raise ValueError('E-A9k is defined for the plain Core on the permutation tasks')
    model = dr.load_model(weights).eval()
    table = age_hazard(erosion_json, weights)
    log(f'{weights}: {meta["task"]} width {meta["width"]}; injection hazard of [64,128): {table["64-128"][0]} / {table["64-128"][1]}')
    prototypes = class_centroids(model, meta['task'], meta['seed'])
    conditions = run_length(model, prototypes, meta['task'], meta['seed'], length, n, window, batch, log, conditions=tuple(('class', k) for k in INTERVALS))
    return dict(weights=str(weights), task=meta['task'], width=meta['width'], seed=meta['seed'], length=length, n=n, window=window,
                age_hazard={k: list(v) for k, v in table.items()}, conditions=conditions,
                predicted={str(k): renewal_hazard(k, table) for k in INTERVALS})


def cells(result):
    out = {}
    for k in PREDICTED:
        c = result['conditions'][f'class-{k}']
        predicted = result['predicted'][str(k)]
        out[k] = dict(events=c['events'], person_steps=c['person_steps'], hazard=c['events'] / max(c['person_steps'], 1), predicted=predicted,
                      consistent=consistent(c['events'], c['person_steps'], predicted, k), error_free=c['error_free'])
    return out


def reading(results):
    s5 = [r for r in results if r['task'] == 'perm5']
    if len(s5) < 2:
        return dict(verdict='INCOMPLETE', reason=f'{len(s5)} S5 Core model(s); two are registered')
    per_model = {Path(r['weights']).stem: cells(r) for r in s5}
    all_cells = [(m, k, c) for m, row in per_model.items() for k, c in row.items()]
    broken = [(m, k) for m, k, c in all_cells if c['hazard'] > BAND[1] * c['predicted'] and c['hazard'] > BREAK_FLOOR]
    longer = all(per_model[m][128]['hazard'] < BAND[0] * per_model[m][128]['predicted'] for m in per_model)
    if all(c['consistent'] for _, _, c in all_cells):
        verdict = 'RENEWAL HOLDS'
    elif broken:
        verdict = 'SNAP BREAKS EARLIER THAN A RESET'
    elif longer:
        verdict = 'SNAP HOLDS LONGER THAN A RESET'
    else:
        verdict = 'MIXED'
    return dict(verdict=verdict, broken=broken, models=per_model)


def describe(r):
    c = cells(r)
    lines = [f'{r["weights"]}: {r["task"]} w{r["width"]}, {r["n"]} sequences x {r["length"]} positions']
    for k in INTERVALS:
        d = r['conditions'][f'class-{k}']
        predicted = r['predicted'][str(k)]
        lines.append(f'  class-{k:<4} first errors {d["events"]:>3}/{d["person_steps"]:>7} = {d["events"] / max(d["person_steps"], 1):.2e}  predicted '
                     + ('n/a     ' if predicted is None else f'{predicted:.2e}') + f'  error-free {d["error_free"]}/{r["n"]}' + (f'  consistent {c[k]["consistent"]}' if k in c else ''))
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='*')
    p.add_argument('--erosion', default='results/Z1-pilot/erosion-source.json')
    p.add_argument('--out', default=None)
    p.add_argument('--reading', default=None, help='a result file written by --out: print the registered verdict')
    a = p.parse_args()
    if a.reading:
        results = json.loads(Path(a.reading).read_text(encoding='utf8'))
        for r in results:
            print(describe(r))
        print(json.dumps(reading(results), indent=1, default=str))
        return
    results = []
    for path in a.weights:
        r = measure(path, a.erosion, log=lambda m: print(m, flush=True))
        results.append(r)
        print(describe(r), flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')
    if a.out:
        print(json.dumps(reading(results), indent=1, default=str))


if __name__ == '__main__':
    main()
