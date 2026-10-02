"""E-A9: does a cleanup of the carried state toward the canonical state of the class the model itself predicts restore exactness? Evaluation only (no training), CPU, dev split.

E-A5 (note 5.26) located the S5 Core's long-position erosion in the history the carried state holds: replacing the state by the centroid of the TRUE class (an oracle) removes about 95 % of the
first errors in the next 128 steps, whatever the position, while restricting the attention paths does nothing. An oracle is not a method. Here the same replacement uses only what the model
itself produces: before decoder step t the fed-back state z (the previous step's final decoder output) is replaced by (1 - lambda) z + lambda mu[c], where c is

  class    the class the model PREDICTED at the previous step (argmax of its own readout of z: the output it already emitted),
  nearest  the class whose centroid is closest to z in state space (no readout at all),

and mu are the class centroids measured on in-range (positions 8..15) dev sequences (a calibration, as in E-A5; applying the snap needs no label). The snap happens every `every` steps. If the
state is still in the basin of the right class when it is snapped (the prediction is right with probability 1 - hazard), this is the oracle injection of E-A5 without the oracle: a cleanup
memory in the loop. If it works, an attractor-like correction (a prototype snap, trainable in the loop) is the architectural direction; if not, the drift cannot be removed by projecting
onto class centroids and the training side has to change (note 5.27).

  python -B -m horizon.self_snap results/Z1-pilot/perm5-core-w128-s1-cur.pt ... --out results/Z1-pilot/self-snap.json
  python -B -m horizon.self_snap --reading results/Z1-pilot/self-snap.json

Reading rule (written before the first run, program note 5.27). Same classification as E-A5 (`erosion_source.classify`; the first-error hazard of [16, 1024) in events / person-steps, the
untouched run as reference, R >= 5 and p < 0.01 RESCUES, 2 <= R < 5 PARTIAL, R <= 0.5 and p < 0.01 with >= 5 events HARMS, else NEUTRAL). Primary condition: `class-1` (the predicted-class
snap at every step, lambda = 1), the three S5 Core models; the two S4 Core models are the control. Verdict: >= 2 of the 3 S5 models RESCUES -> SELF-SNAP RESTORES EXACTNESS; no S5 model RESCUES
or PARTIAL and at least two NEUTRAL (and the S4 control not HARMS) -> SELF-SNAP DOES NOT HELP; no S5 model RESCUES or PARTIAL otherwise -> INCONCLUSIVE (the snap itself harms); anything else
-> MIXED. CONTROL FAILED when the untouched run does not reproduce the stored first-error windows of the model or when the primary snap leaves the in-range accuracy (length 16, n = 200)
of any model below 0.99. Descriptive (no verdict): the other conditions, and the first-error hazard beyond 1,024 positions of the primary condition against the untouched run on n = 40
sequences of 8,192 positions.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from core.model import Core

from . import depth_run as dr
from . import depth_tasks as dt
from .erosion_source import class_centroids, classify, first_errors, predictions, stored_control, window_counts, WINDOW_FROM

PRIMARY = 'class-1'
CONDITIONS = (('class', 1, 1.0), ('class', 4, 1.0), ('class', 16, 1.0), ('nearest', 1, 1.0), ('nearest', 16, 1.0), ('class', 1, 0.5))
CONTROL_ACCURACY = 0.99
LONG_LENGTH, LONG_N = 8192, 40
LONG_EDGES = (16, 64, 256, 1024, 2048, 4096, 8192)


class SnapCore(Core):
    """The sealed Core whose fed-back state is pulled toward the centroid of a class chosen from the state itself. Nothing changes unless it is configured."""

    prototypes = None       # (classes, width) class centroids
    selector = 'class'      # 'class': the model's own predicted class of the previous step; 'nearest': the closest centroid in state space
    every = 1
    strength = 1.0

    def decode(self, e, s, mk, mv, t):
        if s is not None and self.prototypes is not None and t % self.every == 0:
            z = s['s']
            chosen = self.read(self.outn(z)).argmax(-1) if self.selector == 'class' else torch.cdist(z, self.prototypes).argmin(-1)
            s = dict(s)
            s['s'] = (1.0 - self.strength) * z + self.strength * self.prototypes[chosen]
        return super().decode(e, s, mk, mv, t)


def configure(model, prototypes=None, selector='class', every=1, strength=1.0):
    """Set the snap in place (class swap); with no prototypes the trained behaviour at every step is restored."""
    if not isinstance(model, Core):
        raise ValueError('the self-snap is defined for the Core arm')
    if selector not in ('class', 'nearest'):
        raise ValueError('unknown selector: ' + str(selector))
    if every < 1 or not 0.0 <= strength <= 1.0:
        raise ValueError('every must be >= 1 and the strength in [0, 1]')
    model.__class__ = SnapCore
    model.prototypes, model.selector, model.every, model.strength = prototypes, selector, every, strength
    return model


def label(selector, every, strength):
    return f'{selector}-{every}' + ('' if strength == 1.0 else f'-soft{strength:g}')


def run_conditions(model, x, y, prototypes, batch, log=print):
    """({condition: counts}, first errors of the untouched run) on the same sequences."""
    out, base_first = {}, None
    for name, (selector, every, strength) in [('base', (None, 1, 1.0))] + [(label(*c), c) for c in CONDITIONS]:
        started = time.perf_counter()
        configure(model, None if name == 'base' else prototypes, selector or 'class', every, strength)
        pred = predictions(model, x, batch)
        configure(model)
        first = first_errors(pred, y)
        base_first = first if name == 'base' else base_first
        events, person, at_risk = window_counts(first, WINDOW_FROM, x.shape[1])
        out[name] = dict(selector=selector, every=every, strength=strength, events=events, person_steps=person, at_risk=at_risk, final=float((pred[:, -1] == y[:, -1]).float().mean()))
        log(f'  {name:<14} first errors {events:>3}/{person:>6} person-steps = {events / max(person, 1):.2e}  final {out[name]["final"]:.2f}  ({time.perf_counter() - started:.0f} s)')
    return out, base_first


def in_range(model, task, seed, prototypes, n=200, length=16, batch=50):
    x, y, _ = dt.heldout(dr.NAMESPACE, task, seed, 'dev', n, length)
    out = {}
    for name, (selector, every, strength) in [('base', (None, 1, 1.0))] + [(label(*c), c) for c in CONDITIONS]:
        configure(model, None if name == 'base' else prototypes, selector or 'class', every, strength)
        out[name] = float((predictions(model, x, batch) == y).float().mean())
        configure(model)
    return out


def long_part(model, task, seed, prototypes, batch=20, log=print):
    """Descriptive: the untouched run and the primary snap on LONG_N sequences of LONG_LENGTH positions, first-error windows beyond 16."""
    x, y, _ = dt.heldout(dr.NAMESPACE, task, seed, 'dev', LONG_N, LONG_LENGTH)
    out = {}
    for name, snap in (('base', False), (PRIMARY, True)):
        started = time.perf_counter()
        configure(model, prototypes if snap else None, 'class', 1, 1.0)
        pred = predictions(model, x, batch)
        configure(model)
        first = first_errors(pred, y)
        out[name] = dict(windows={f'{a}-{b}': dict(zip(('events', 'person_steps', 'at_risk'), window_counts(first, a, b))) for a, b in zip(LONG_EDGES[:-1], LONG_EDGES[1:])},
                         error_free=int((first >= LONG_LENGTH).sum()), final=float((pred[:, -1] == y[:, -1]).float().mean()))
        log(f'  long {name:<8} error-free sequences {out[name]["error_free"]}/{LONG_N}, final accuracy {out[name]["final"]:.2f}  ({time.perf_counter() - started:.0f} s)')
    return out


def measure(weights, n=200, length=1024, batch=40, long_eval=True, log=print):
    meta = torch.load(weights, weights_only=True)['meta']
    if meta['arm'] != 'core' or dt.is_count(meta['task']):
        raise ValueError('E-A9 is defined for the Core on the permutation tasks')
    model = dr.load_model(weights).eval()
    x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
    log(f'{weights}: {meta["task"]} width {meta["width"]}, {n} dev sequences of {length} positions')
    prototypes = class_centroids(model, meta['task'], meta['seed'])
    conditions, base_first = run_conditions(model, x, y, prototypes, batch, log)
    result = dict(weights=str(weights), task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, length=length, parameters=dr.parameter_count(model), conditions=conditions,
                  in_range=in_range(model, meta['task'], meta['seed'], prototypes))
    result['stored_control'] = stored_control(weights, base_first)
    if long_eval:
        result['long'] = long_part(model, meta['task'], meta['seed'], prototypes, log=log)
    return result


def labels(result, primary=PRIMARY):
    base = result['conditions']['base']
    return {name: classify((base['events'], base['person_steps']), (c['events'], c['person_steps']))[0] for name, c in result['conditions'].items() if name != 'base'}


def reading(results):
    s5 = [r for r in results if r['task'] == 'perm5']
    s4 = [r for r in results if r['task'] == 'perm4']
    if not s5:
        return dict(verdict='NO S5 MODEL')
    failed = []
    for r in s5 + s4:
        name = f'{r["task"]} w{r["width"]}'
        if r.get('stored_control') is False:
            failed.append(f'{name}: the untouched run does not reproduce the stored first-error windows')
        if r['in_range'][PRIMARY] < CONTROL_ACCURACY - 1e-9:
            failed.append(f'{name}: the primary snap leaves the in-range accuracy at {r["in_range"][PRIMARY]:.3f}')
    out = dict(controls=failed, models={f'{r["task"]} w{r["width"]} s{r["seed"]} {Path(r["weights"]).stem}': labels(r) for r in s5 + s4})
    if failed:
        out['verdict'] = 'CONTROL FAILED: ' + '; '.join(failed)
        return out
    primary = [labels(r)[PRIMARY] for r in s5]
    control = [labels(r)[PRIMARY] for r in s4]
    rescues = primary.count('RESCUES')
    helped = rescues + primary.count('PARTIAL')
    if rescues >= 2:
        out['verdict'] = 'SELF-SNAP RESTORES EXACTNESS'
    elif helped == 0:
        out['verdict'] = ('INCONCLUSIVE (the snap also harms the S4 control)' if 'HARMS' in control
                          else 'SELF-SNAP DOES NOT HELP' if primary.count('NEUTRAL') >= 2 else 'INCONCLUSIVE (the snap harms the S5 models)')
    else:
        out['verdict'] = 'MIXED'
    return out


def describe(r):
    lines = [f'{r["task"]} w{r["width"]} ({r["parameters"]} parameters), n = {r["n"]}, stored-window control {r["stored_control"]}']
    lines.append('  [16,%d): ' % r['length'] + '  '.join(f'{k} {v["events"]}/{v["person_steps"]}' for k, v in r['conditions'].items()))
    lines.append('  in-range accuracy (length 16): ' + '  '.join(f'{k} {v:.3f}' for k, v in r['in_range'].items()))
    lines.append('  labels: ' + '  '.join(f'{k} {v}' for k, v in labels(r).items()))
    for k, v in r.get('long', {}).items():
        lines.append(f'  long {k}: error-free {v["error_free"]}/{LONG_N}; ' + ' '.join(f'[{w}) {c["events"]}/{c["person_steps"]}' for w, c in v['windows'].items()))
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='*')
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--length', type=int, default=1024)
    p.add_argument('--batch', type=int, default=40)
    p.add_argument('--no-long', action='store_true')
    p.add_argument('--out', default=None)
    p.add_argument('--reading', default=None, help='a result file written by --out: print the registered verdict')
    a = p.parse_args()
    if a.reading:
        results = json.loads(Path(a.reading).read_text(encoding='utf8'))
        for r in results:
            print(describe(r))
        print(json.dumps(reading(results), indent=1, default=float))
        return
    results = []
    for path in a.weights:
        r = measure(path, a.n, a.length, a.batch, long_eval=not a.no_long, log=lambda m: print(m, flush=True))
        results.append(r)
        print(describe(r), flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(results, indent=1, default=float), encoding='utf8')
    if a.out:
        print(json.dumps(reading(results), indent=1, default=float))


if __name__ == '__main__':
    main()
