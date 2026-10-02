"""E-A9c: does the same self-snap help a plain GRU / LSTM? (the fairness control of E-A9) Evaluation only (no training), CPU, dev split.

E-A9 (note 5.27) made the S5 Core exact over 1,024 positions (and, E-A9L, much further) by pulling its carried state toward the centroid of the class it itself predicted. The annealed S5 GRU / LSTM
are 7-18x better than the Core without it (E-P2) but not exact (hazard ~1e-4). A cleanup that helps every recurrent tracker and merely closes the Core's gap is a different claim from one that
is specific to the Core, so the same snap is applied here to the S5 GRU w256 and LSTM w208 (constant-lr and annealed). The carried state is h (GRU) or the concatenation (h, c) (LSTM); the
class estimate is the argmax of the readout of the previous hidden state (`class`, the output the model already emitted) or the nearest centroid in state space (`nearest`); the centroids are the
per-class means of the state over positions 8..15 of 2,000 in-range dev sequences (calibration, as in E-A9). The recurrence is unrolled by hand (the same maths as `state_growth.recurrent_trajectory`)
and checked against the arm's own fused forward when no snap is configured.

  python -B -m horizon.snap_recurrent results/Z1-pilot/perm5-gru-w256-s1-cur-lrdecay0.3.pt ... --out results/Z1-pilot/snap-recurrent.json

Reading rule (written before the first run, program note 5.32): per model and condition the first-error hazard of [16, 1024) in events / person-steps on 200 dev sequences of 1,024 positions, the
untouched run as reference, classified as in E-A5 (`erosion_source.classify`: RESCUES R >= 5 and p < 0.01, PARTIAL 2 <= R < 5, HARMS, NEUTRAL; the zero-event floor 3 / person-steps). Primary
condition `class-16`, the four models (constant-lr and annealed GRU w256 and LSTM w208). Verdict: >= 2 models RESCUES -> THE SNAP HELPS THE RECURRENT BASELINES TOO; no model RESCUES or PARTIAL and at
least two NEUTRAL -> THE SNAP DOES NOT HELP THE RECURRENT BASELINES; otherwise MIXED; CONTROL FAILED when the unrolled recurrence does not reproduce the fused forward (max |logit difference| > 1e-4) or the
untouched run does not reproduce the stored first-error windows, or the primary snap leaves the in-range accuracy of a model below 0.99 (INCONCLUSIVE). Descriptive: the other conditions.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from . import depth_run as dr
from . import depth_tasks as dt
from .erosion_source import classify, first_errors, stored_control, window_counts, WINDOW_FROM
from .state_growth import recurrent_trajectory

CONDITIONS = (('class', 16), ('class', 1), ('nearest', 16))
PRIMARY = 'class-16'
CONTROL_ACCURACY = 0.99
IDENTITY_TOL = 1e-4


def unrolled_logits(arm, x, centroids=None, selector='class', every=1):
    """The arm's logits with its recurrence unrolled by hand; from step 1 on, every `every` steps the carried state (h, or h and c) is replaced by the centroid chosen by `selector`
    ('class': argmax of the readout of the previous hidden state; 'nearest': the closest centroid to the state). No centroids: the plain recurrence."""
    rnn, lstm = arm.rnn, isinstance(arm.rnn, nn.LSTM)
    emb = arm.embedding(x)
    b, steps, width = emb.shape
    h, c, out = emb.new_zeros(b, width), emb.new_zeros(b, width), []
    for t in range(steps):
        if centroids is not None and t >= 1 and t % every == 0:
            chosen = arm.read(h).argmax(-1) if selector == 'class' else torch.cdist(torch.cat([h, c], -1) if lstm else h, centroids).argmin(-1)
            target = centroids[chosen]
            h = target[:, :width]
            if lstm:
                c = target[:, width:]
        gi = F.linear(emb[:, t], rnn.weight_ih_l0, rnn.bias_ih_l0)
        gh = F.linear(h, rnn.weight_hh_l0, rnn.bias_hh_l0)
        if lstm:
            i, f, g, o = (gi + gh).chunk(4, -1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
        else:
            ir, iz, inn = gi.chunk(3, -1)
            hr, hz, hn = gh.chunk(3, -1)
            r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
            h = (1 - z) * torch.tanh(inn + r * hn) + z * h
        out.append(h)
    return arm.read(torch.stack(out, 1))


def predictions(arm, x, centroids=None, selector='class', every=1, batch=50):
    arm.eval()
    parts = []
    with torch.no_grad():
        for at in range(0, len(x), batch):
            parts.append(unrolled_logits(arm, x[at:at + batch], centroids, selector, every).argmax(-1))
    return torch.cat(parts)


def class_centroids(arm, task, seed, n=2000, length=16, positions=(8, 16)):
    """(classes, state width) per-class mean of the carried state (h, or the concatenation h, c) at the given positions of in-range dev sequences."""
    x, y, _ = dt.heldout(dr.NAMESPACE, task, seed, 'dev', n, length)
    with torch.no_grad():
        states, _ = recurrent_trajectory(arm.eval(), x)
    a, b = positions
    s, lab = states[:, a:b].reshape(-1, states.shape[-1]), y[:, a:b].reshape(-1)
    count = dt.classes(task)
    centroids = torch.zeros(count, states.shape[-1])
    for k in range(count):
        members = s[lab == k]
        if len(members) < 5:
            raise ValueError(f'class {k} has only {len(members)} states at positions {positions}: draw more sequences')
        centroids[k] = members.mean(0)
    return centroids


def measure(weights, n=200, length=1024, batch=50, log=print):
    meta = torch.load(weights, weights_only=True)['meta']
    if meta['arm'] not in ('gru', 'lstm') or dt.is_count(meta['task']):
        raise ValueError('E-A9c is defined for the GRU / LSTM arms on the permutation tasks')
    arm = dr.load_model(weights).eval()
    x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
    log(f'{weights}: {meta["arm"]} {meta["task"]} width {meta["width"]}, {n} dev sequences of {length} positions')
    with torch.no_grad():
        identity = float((unrolled_logits(arm, x[:8, :64]) - arm(x[:8, :64])).abs().max())
    centroids = class_centroids(arm, meta['task'], meta['seed'])
    out, base_first = {}, None
    for name, (selector, every) in [('base', (None, 1))] + [(f'{s}-{e}', (s, e)) for s, e in CONDITIONS]:
        started = time.perf_counter()
        pred = predictions(arm, x, None if name == 'base' else centroids, selector or 'class', every, batch)
        first = first_errors(pred, y)
        base_first = first if name == 'base' else base_first
        events, person, at_risk = window_counts(first, WINDOW_FROM, length)
        out[name] = dict(events=events, person_steps=person, at_risk=at_risk, final=float((pred[:, -1] == y[:, -1]).float().mean()))
        log(f'  {name:<11} first errors {events:>3}/{person:>6} person-steps = {events / max(person, 1):.2e}  final {out[name]["final"]:.2f}  ({time.perf_counter() - started:.0f} s)')
    short_x, short_y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', 200, 16)
    in_range = {f'{s}-{e}': float((predictions(arm, short_x, centroids, s, e, batch) == short_y).float().mean()) for s, e in CONDITIONS}
    return dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, length=length, parameters=dr.parameter_count(arm),
                identity=identity, conditions=out, in_range=in_range, stored_control=stored_control(weights, base_first))


def labels_of(result):
    base = result['conditions']['base']
    return {name: classify((base['events'], base['person_steps']), (c['events'], c['person_steps']))[0] for name, c in result['conditions'].items() if name != 'base'}


def reading(results):
    if not results:
        return dict(verdict='NO MODELS')
    failed = []
    for r in results:
        name = f'{r["arm"]} w{r["width"]} {Path(r["weights"]).stem}'
        if r['identity'] > IDENTITY_TOL:
            failed.append(f'{name}: the unrolled recurrence differs from the fused forward by {r["identity"]:.2e}')
        if r.get('stored_control') is False:
            failed.append(f'{name}: the untouched run does not reproduce the stored first-error windows')
    low = [f'{r["arm"]} w{r["width"]}' for r in results if r['in_range'][PRIMARY] < CONTROL_ACCURACY - 1e-9]
    labels = [labels_of(r)[PRIMARY] for r in results]
    rescues = labels.count('RESCUES')
    helped = rescues + labels.count('PARTIAL')
    if failed:
        verdict = 'CONTROL FAILED: ' + '; '.join(failed)
    elif low:
        verdict = 'INCONCLUSIVE (the primary snap lowers the in-range accuracy of ' + ', '.join(low) + ')'
    elif rescues >= 2:
        verdict = 'THE SNAP HELPS THE RECURRENT BASELINES TOO'
    elif helped == 0 and labels.count('NEUTRAL') >= 2:
        verdict = 'THE SNAP DOES NOT HELP THE RECURRENT BASELINES'
    else:
        verdict = 'MIXED'
    return dict(verdict=verdict, primary=labels, models={f'{r["arm"]} w{r["width"]} {Path(r["weights"]).stem}': labels_of(r) for r in results})


def describe(r):
    return (f'{r["arm"]} {r["task"]} w{r["width"]} ({r["parameters"]} parameters), n = {r["n"]}, identity {r["identity"]:.1e}, stored-window control {r["stored_control"]}\n  [16,{r["length"]}): '
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
