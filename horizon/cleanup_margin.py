"""E-A11p: does a LABEL-FREE cleanup work when it only fires where it is confident? Evaluation only, CPU, dev split (program note 5.36).

E-A10p (5.33.1) showed that a Hopfield cleanup built from LABELLED class means reproduces the self-snap of E-A9, and that one built from label-free k-means centroids harms (in-range accuracy
0.38-0.76). The follow-up measurement (note 5.28.1) found the in-range states of the plain Core cluster by class with k-means purity 0.89-0.92: the harm is the ~9 % of states that are nearer
to the wrong centroid (12 of 120 classes own no cluster) compounding over steps (0.91^16 = 0.22), not an absence of class structure. Those states are the ambiguous ones: their margin
(cosine to the nearest prototype minus cosine to the second nearest) is small. Among the 50 % of calibration states with the largest margin the nearest-centroid class was right in 8,000 of 8,000.

The probe: the same retrofit as E-A10p (keys = values = prototypes repeated over the 128 slots, beta = 50, gate 0.999) with a HARD margin gate: the cleanup fires only where the margin over the
distinct prototypes is at least a threshold, otherwise the feedback passes through untouched (r' = r). The threshold is the q-quantile of the margins of the calibration states, so it needs no label.

  python -B -m horizon.cleanup_margin results/Z1-pilot/perm5-core-w128-s1-cur.pt ... --out results/Z1-pilot/cleanup-margin.json

Reading rule (written before the first run, program note 5.36): per model the first-error hazard of [16, 1024) on 200 dev sequences of 1,024 positions, the untouched run as reference, classified as in
E-A5 (`erosion_source.classify`: RESCUES R >= 5 and p < 0.01, PARTIAL 2 <= R < 5, HARMS R <= 0.5 and p(higher) < 0.01 with >= 5 events, NEUTRAL). Primary condition `kmeans-q50` (label-free:
k-means centroids, k = the class count, threshold = the median calibration margin), the two S5 Core models; the two S4 Cores are controls. CONTROL FAILED when the untouched run does not reproduce the stored
first-error windows or the retrofit at gate 0 does not reproduce the untouched predictions; INCONCLUSIVE when the primary condition leaves any model's in-range accuracy (length 16, n = 200) below 0.99 or
HARMS an S4 control; otherwise LABEL-FREE GATED CLEANUP WORKS when both S5 models RESCUES, LABEL-FREE GATED CLEANUP DOES NOT WORK when neither RESCUES nor PARTIAL, MIXED otherwise.
Descriptive: `kmeans-q30` (fires on 70 % of the calibration states), `kmeans-q70` (30 %), `labelled-q50` (class means, the label-built reference), the fraction of steps on which the gate fired.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from . import depth_run as dr
from . import depth_tasks as dt
from .cleanup import CleanupNorm
from .cleanup_probe import BETA, GATE, feedback_vectors, labelled_prototypes, retrofit
from .erosion_source import classify, first_errors, fresh_in_range, predictions, stored_control, window_counts, WINDOW_FROM
from .snap_labelfree import kmeans

PRIMARY = 'kmeans-q50'
CONDITIONS = (('kmeans-q30', 'kmeans', 0.3), ('kmeans-q50', 'kmeans', 0.5), ('kmeans-q70', 'kmeans', 0.7), ('labelled-q50', 'labelled', 0.5))
CONTROL_ACCURACY = 0.99


class MarginGatedNorm(CleanupNorm):
    """CleanupNorm whose gate is hard and state dependent: it fires only where the top-1 minus top-2 cosine margin over the distinct prototypes (`distinct`, normalised) is at least `threshold`."""

    threshold = None
    distinct = None
    counting = False
    fired = 0
    seen = 0

    def forward(self, state):
        r = self.norm(state)
        retrieved = torch.softmax(self.log_beta.exp() * F.normalize(r, dim=-1) @ F.normalize(self.keys, dim=-1).t(), dim=-1) @ self.values
        top = (F.normalize(r, dim=-1) @ self.distinct.t()).topk(2, dim=-1).values
        fire = ((top[..., 0] - top[..., 1]) >= self.threshold).to(r.dtype).unsqueeze(-1)
        if self.counting:
            self.fired += int(fire.sum())
            self.seen += int(fire.numel())
        return r + self.gate() * fire * (retrieved - r)


def margins(r, prototypes):
    """Cosine to the nearest prototype minus cosine to the second nearest, per row of r."""
    top = (F.normalize(r, dim=-1) @ F.normalize(prototypes, dim=-1).t()).topk(2, dim=-1).values
    return top[:, 0] - top[:, 1]


def retrofit_gated(model, prototypes, quantile, calibration, beta=BETA, gate=GATE):
    """The E-A10p retrofit plus the margin gate; the threshold is the `quantile` of the calibration margins (label-free)."""
    out = retrofit(model, prototypes, beta, gate)
    out.pn.__class__ = MarginGatedNorm
    out.pn.threshold = float(torch.quantile(margins(calibration, prototypes), quantile))
    out.pn.distinct = F.normalize(prototypes, dim=-1)
    out.pn.counting, out.pn.fired, out.pn.seen = False, 0, 0
    return out.eval()


def measure(weights, n=200, length=1024, batch=40, log=print):
    meta = torch.load(weights, weights_only=True)['meta']
    if meta['arm'] != 'core' or dt.is_count(meta['task']):
        raise ValueError('E-A11p is defined for the plain Core on the permutation tasks')
    model = dr.load_model(weights).eval()
    classes = dt.classes(meta['task'])
    r, labels = feedback_vectors(model, meta['task'], meta['seed'])
    x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
    short_x, short_y = fresh_in_range(meta['task'], meta['seed'])
    log(f'{weights}: {meta["task"]} width {meta["width"]}, {len(r)} in-range feedback vectors')
    base_pred = predictions(model, x, batch)
    base_first = first_errors(base_pred, y)
    centroids, assignment = kmeans(r, classes)
    books = dict(kmeans=centroids, labelled=labelled_prototypes(r, labels, classes))
    closed = retrofit_gated(model, books['kmeans'], 0.5, r, gate=0.0)
    identity = bool(torch.equal(predictions(closed, x[:20, :128], batch), base_pred[:20, :128]))
    events, person, at_risk = window_counts(base_first, WINDOW_FROM, length)
    out = dict(base=dict(events=events, person_steps=person, at_risk=at_risk, final=float((base_pred[:, -1] == y[:, -1]).float().mean()),
                         in_range=float((predictions(model, short_x, 50) == short_y).float().mean())))
    for name, book, quantile in CONDITIONS:
        started = time.perf_counter()
        patched = retrofit_gated(model, books[book], quantile, r)
        patched.pn.counting = True
        pred = predictions(patched, x, batch)
        first = first_errors(pred, y)
        events, person, at_risk = window_counts(first, WINDOW_FROM, length)
        fired = patched.pn.fired / max(patched.pn.seen, 1)
        patched.pn.counting = False
        out[name] = dict(events=events, person_steps=person, at_risk=at_risk, final=float((pred[:, -1] == y[:, -1]).float().mean()), fired=fired,
                         in_range=float((predictions(patched, short_x, 50) == short_y).float().mean()))
        log(f'  {name:<13} first errors {events:>3}/{person:>6} person-steps = {events / max(person, 1):.2e}  final {out[name]["final"]:.2f}  in-range {out[name]["in_range"]:.3f}  fired {fired:.2f}  ({time.perf_counter() - started:.0f} s)')
    log(f'  base          first errors {out["base"]["events"]:>3}/{out["base"]["person_steps"]:>6} person-steps = {out["base"]["events"] / max(out["base"]["person_steps"], 1):.2e}')
    return dict(weights=str(weights), task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, length=length, conditions=out, gate0_identity=identity,
                kmeans_clusters=int((torch.bincount(assignment, minlength=classes) > 0).sum()), stored_control=stored_control(weights, base_first))


def labels_of(result):
    base = result['conditions']['base']
    return {name: classify((base['events'], base['person_steps']), (c['events'], c['person_steps']))[0] for name, c in result['conditions'].items() if name != 'base'}


def reading(results):
    s5 = [r for r in results if r['task'] == 'perm5']
    s4 = [r for r in results if r['task'] == 'perm4']
    if not s5:
        return dict(verdict='NO S5 MODEL')
    failed = []
    for r in results:
        name = f'{r["task"]} w{r["width"]}'
        if r.get('stored_control') is False:
            failed.append(f'{name}: the untouched run does not reproduce the stored windows')
        if not r.get('gate0_identity', True):
            failed.append(f'{name}: the retrofit at gate 0 does not reproduce the untouched predictions')
    primary = [labels_of(r)[PRIMARY] for r in s5]
    low = [f'{r["task"]} w{r["width"]}' for r in results if r['conditions'][PRIMARY]['in_range'] < CONTROL_ACCURACY - 1e-9]
    harmed = [f'{r["task"]} w{r["width"]}' for r in s4 if labels_of(r)[PRIMARY] == 'HARMS']
    if failed:
        verdict = 'CONTROL FAILED: ' + '; '.join(failed)
    elif low or harmed:
        verdict = 'INCONCLUSIVE (' + ('in-range accuracy below 0.99: ' + ', '.join(low) if low else '') + ('; ' if low and harmed else '') + ('an S4 control is harmed: ' + ', '.join(harmed) if harmed else '') + ')'
    elif primary.count('RESCUES') == len(s5):
        verdict = 'LABEL-FREE GATED CLEANUP WORKS'
    elif all(v in ('NEUTRAL', 'HARMS') for v in primary):
        verdict = 'LABEL-FREE GATED CLEANUP DOES NOT WORK'
    else:
        verdict = 'MIXED'
    return dict(verdict=verdict, primary=primary, models={f'{r["task"]} w{r["width"]} {Path(r["weights"]).stem}': labels_of(r) for r in results})


def describe(r):
    return (f'{r["task"]} w{r["width"]}, gate-0 identity {r["gate0_identity"]}, stored-window control {r["stored_control"]}, k-means clusters {r["kmeans_clusters"]}\n  [16,{r["length"]}): '
            + '  '.join(f'{k} {v["events"]}/{v["person_steps"]}' for k, v in r['conditions'].items())
            + '\n  in-range accuracy: ' + '  '.join(f'{k} {v["in_range"]:.3f}' for k, v in r['conditions'].items())
            + '\n  gate fired on: ' + '  '.join(f'{k} {v["fired"]:.2f}' for k, v in r['conditions'].items() if 'fired' in v)
            + '\n  labels: ' + '  '.join(f'{k} {v}' for k, v in labels_of(r).items()))


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
