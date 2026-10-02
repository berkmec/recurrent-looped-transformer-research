"""E-A10p: does the gated Hopfield cleanup MODULE (horizon/cleanup.py) reproduce the benefit of the self-snap when its prototypes are set from the data? Evaluation only, CPU, dev split.

E-A10 (note 5.28) trains `core_clean` and asks whether training finds a useful cleanup. Before that matters, the module has to be CAPABLE of the cleanup: an attractor memory of 128 prototypes in the feedback
path (softmax over cosine similarities, read out as a mixture of values, gated by tanh(open)). Here a trained S5 Core gets the module added (`add_cleanup`, the sealed class untouched) with the
prototypes set from its own in-range carried states instead of learned: keys = values = the per-class mean of r = RMSNorm(state) over positions 8..15 of 2,000 in-range dev sequences (`labelled`: the
class is the true running product, calibration as in E-A9) or the k-means centroids of the same r vectors (`kmeans`: no label at all, `snap_labelfree.kmeans`, k = classes; the other 8 slots of the
128 are the same centroids repeated, i.e. unused), the softmax temperature beta = 50 and the gate tanh(open) = 0.999. The first-error hazard of [16, 1024) on 200 dev sequences of 1,024 positions is
compared with the untouched model.

  python -B -m horizon.cleanup_probe results/Z1-pilot/perm5-core-w128-s1-cur.pt ... --out results/Z1-pilot/cleanup-probe.json

Reading rule (written before the first run, program note 5.33): the classification of E-A5 (`erosion_source.classify`: RESCUES R >= 5 and p < 0.01, PARTIAL 2 <= R < 5, HARMS, NEUTRAL). Primary condition
`labelled`, the constant-lr and the annealed S5 Core; secondary `kmeans`. Verdict: both S5 models RESCUES on `labelled` and the in-range accuracy (length 16, n = 200) of every model >= 0.99 -> THE
MODULE CAN CLEAN (the module is a sound vehicle; E-A10's training question is then only whether training finds the prototypes); neither RESCUES or PARTIAL -> THE MODULE CANNOT CLEAN (a design flaw: the
softmax read-out or the normalisation destroys what the snap of E-A9 preserved); otherwise MIXED; CONTROL FAILED when the untouched run does not reproduce the stored first-error windows or the module at
gate 0 does not reproduce the untouched predictions. Descriptive: `kmeans`, the in-range accuracy.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from . import depth_run as dr
from . import depth_tasks as dt
from .cleanup import add_cleanup, PROTOTYPES
from .erosion_source import classify, first_errors, predictions, stored_control, window_counts, WINDOW_FROM
from .snap_labelfree import kmeans
from .state_growth import core_trajectory

BETA = 50.0
GATE = 0.999
PRIMARY = 'labelled'
CONDITIONS = ('labelled', 'kmeans')
CONTROL_ACCURACY = 0.99


def feedback_vectors(model, task, seed, n=2000, length=16, positions=(8, 16)):
    """(r, labels): r = RMSNorm(carried state) of the in-range states at the given positions and their true classes."""
    x, y, _ = dt.heldout(dr.NAMESPACE, task, seed, 'dev', n, length)
    with torch.no_grad():
        states, _ = core_trajectory(model.eval(), x)
        a, b = positions
        r = model.pn(states[:, a:b].reshape(-1, states.shape[-1]))
    return r, y[:, a:b].reshape(-1)


def retrofit(model, prototypes, beta=BETA, gate=GATE):
    """A deep copy of the Core with the cleanup module added and its prototypes (keys = values) set from `prototypes` (k, width), the rest of the 128 slots repeating them."""
    out = copy.deepcopy(model)
    add_cleanup(out)
    k = len(prototypes)
    slots = prototypes[torch.arange(PROTOTYPES) % k]
    with torch.no_grad():
        out.pn.keys.copy_(slots)
        out.pn.values.copy_(slots)
        out.pn.log_beta.fill_(math.log(beta))
        out.pn.open.fill_(math.atanh(gate))
    return out.eval()


def labelled_prototypes(r, labels, classes):
    protos = torch.zeros(classes, r.shape[-1])
    for c in range(classes):
        members = r[labels == c]
        if len(members) < 5:
            raise ValueError(f'class {c} has only {len(members)} states: draw more sequences')
        protos[c] = members.mean(0)
    return protos


def measure(weights, n=200, length=1024, batch=40, log=print):
    meta = torch.load(weights, weights_only=True)['meta']
    if meta['arm'] != 'core' or dt.is_count(meta['task']):
        raise ValueError('E-A10p is defined for the plain Core on the permutation tasks')
    model = dr.load_model(weights).eval()
    classes = dt.classes(meta['task'])
    r, labels = feedback_vectors(model, meta['task'], meta['seed'])
    x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
    short_x, short_y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', 200, 16)
    log(f'{weights}: {meta["task"]} width {meta["width"]}, {len(r)} in-range feedback vectors')
    base_pred = predictions(model, x, batch)
    base_first = first_errors(base_pred, y)
    closed = retrofit(model, labelled_prototypes(r, labels, classes), gate=0.0)
    identity = bool(torch.equal(predictions(closed, x[:20, :128], batch), base_pred[:20, :128]))
    out = {}
    events, person, at_risk = window_counts(base_first, WINDOW_FROM, length)
    out['base'] = dict(events=events, person_steps=person, at_risk=at_risk, final=float((base_pred[:, -1] == y[:, -1]).float().mean()), in_range=float((predictions(model, short_x, 50) == short_y).float().mean()))
    clusters = None
    for name in CONDITIONS:
        started = time.perf_counter()
        if name == 'labelled':
            protos = labelled_prototypes(r, labels, classes)
        else:
            protos, assignment = kmeans(r, classes)
            counts = torch.bincount(assignment, minlength=classes)
            clusters = int((counts > 0).sum())
        patched = retrofit(model, protos)
        pred = predictions(patched, x, batch)
        first = first_errors(pred, y)
        events, person, at_risk = window_counts(first, WINDOW_FROM, length)
        out[name] = dict(events=events, person_steps=person, at_risk=at_risk, final=float((pred[:, -1] == y[:, -1]).float().mean()),
                         in_range=float((predictions(patched, short_x, 50) == short_y).float().mean()))
        log(f'  {name:<9} first errors {events:>3}/{person:>6} person-steps = {events / max(person, 1):.2e}  final {out[name]["final"]:.2f}  in-range {out[name]["in_range"]:.3f}  ({time.perf_counter() - started:.0f} s)')
    log(f'  base      first errors {out["base"]["events"]:>3}/{out["base"]["person_steps"]:>6} person-steps = {out["base"]["events"] / max(out["base"]["person_steps"], 1):.2e}')
    return dict(weights=str(weights), task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, length=length, conditions=out, gate0_identity=identity, kmeans_clusters=clusters,
                stored_control=stored_control(weights, base_first))


def labels_of(result):
    base = result['conditions']['base']
    return {name: classify((base['events'], base['person_steps']), (c['events'], c['person_steps']))[0] for name, c in result['conditions'].items() if name != 'base'}


def reading(results):
    s5 = [r for r in results if r['task'] == 'perm5']
    if not s5:
        return dict(verdict='NO S5 MODEL')
    failed = []
    for r in results:
        name = f'{r["task"]} w{r["width"]}'
        if r.get('stored_control') is False:
            failed.append(f'{name}: the untouched run does not reproduce the stored windows')
        if not r.get('gate0_identity', True):
            failed.append(f'{name}: the module at gate 0 does not reproduce the untouched predictions')
    primary = [labels_of(r)[PRIMARY] for r in s5]
    low = [r for r in results if r['conditions'][PRIMARY]['in_range'] < CONTROL_ACCURACY - 1e-9]
    if failed:
        verdict = 'CONTROL FAILED: ' + '; '.join(failed)
    elif primary.count('RESCUES') == len(s5) and not low:
        verdict = 'THE MODULE CAN CLEAN'
    elif all(v in ('NEUTRAL', 'HARMS') for v in primary):
        verdict = 'THE MODULE CANNOT CLEAN'
    else:
        verdict = 'MIXED'
    return dict(verdict=verdict, primary=primary, models={f'{r["task"]} w{r["width"]} {Path(r["weights"]).stem}': labels_of(r) for r in results})


def describe(r):
    return (f'{r["task"]} w{r["width"]}, gate-0 identity {r["gate0_identity"]}, stored-window control {r["stored_control"]}, k-means clusters {r["kmeans_clusters"]}\n  [16,{r["length"]}): '
            + '  '.join(f'{k} {v["events"]}/{v["person_steps"]}' for k, v in r['conditions'].items())
            + '\n  in-range accuracy: ' + '  '.join(f'{k} {v["in_range"]:.3f}' for k, v in r['conditions'].items()) + '\n  labels: ' + '  '.join(f'{k} {v}' for k, v in labels_of(r).items()))


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
