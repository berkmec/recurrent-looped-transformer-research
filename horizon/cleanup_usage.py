"""E-A10 descriptive: what did training make of the gated Hopfield cleanup? (no verdict; program note 5.28.2). Evaluation only, CPU, dev split.

E-A10(a) trains `core_clean` (note 5.28) and its registered verdict reads the first-error hazard. Whether the cleanup is *used* is a separate question the hazard cannot answer: the gate
g = tanh(open) may have stayed open while the Hopfield read-out is flat (a noise injection), or closed (the module unused), or open and sharp with prototypes that line up with the classes. Here, on
the in-range carried states (positions 8..15 of 2,000 dev sequences, the same states E-A9 / E-A10p calibrate on), for a `core_clean` model:

  gate, beta                      tanh(open) and exp(log beta) read from the weights
  top_weight                      mean over states of the largest softmax weight of the Hopfield read-out (1.0: a hard snap to one slot; 1 / 128: a flat average)
  active_slots                    slots that are the arg-max read-out of at least 1 % of the states
  slot_purity                     the share of states whose arg-max slot's majority class is their own class (1.0: every slot answers for one class)
  relative_change                 mean ||r' - r|| / ||r|| with r = RMSNorm(state) and r' the cleaned feedback
  purity_pre, purity_post         k-means (k = classes, the E-A9b / E-A10p procedure) purity of r and of r' against the true class: does the fed-back signal cluster by class?

For a plain Core only `purity_pre` exists (r = RMSNorm(state)), the reference the cleaned model is compared with.

  python -B -m horizon.cleanup_usage results/Z1-pilot/perm5-core_clean-...pt results/Z1-pilot/perm5-core-w128-s1-cur-lrdecay0.3-noise0.1.pt --out results/Z1-pilot/cleanup-usage.json
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from . import depth_run as dr
from . import depth_tasks as dt
from .cleanup import CleanupNorm
from .snap_labelfree import kmeans, purity
from .state_growth import core_trajectory

ACTIVE_SHARE = 0.01


def states_and_labels(model, task, seed, n=2000, length=16, positions=(8, 16)):
    x, y, _ = dt.heldout(dr.NAMESPACE, task, seed, 'dev', n, length)
    with torch.no_grad():
        states, _ = core_trajectory(model.eval(), x)
    a, b = positions
    return states[:, a:b].reshape(-1, states.shape[-1]), y[:, a:b].reshape(-1)


def usage(model, task, seed, n=2000, length=16, positions=(8, 16)):
    states, labels = states_and_labels(model, task, seed, n, length, positions)
    classes = dt.classes(task)
    module = model.pn
    cleaning = isinstance(module, CleanupNorm)
    with torch.no_grad():
        pre = module.norm(states) if cleaning else module(states)
        post = module(states) if cleaning else None
    out = dict(task=task, seed=seed, n_states=int(len(states)), classes=classes,
               purity_pre=purity(kmeans(pre, classes)[1], labels, classes))
    if not cleaning:
        return out
    with torch.no_grad():
        weights = torch.softmax(module.log_beta.exp() * F.normalize(pre, dim=-1) @ F.normalize(module.keys, dim=-1).t(), -1)
    slot = weights.argmax(1)
    count = torch.bincount(slot, minlength=len(module.keys))
    out.update(gate=float(module.gate().detach()), beta=float(module.log_beta.exp().detach()), top_weight=float(weights.max(1).values.mean()),
               active_slots=int((count >= ACTIVE_SHARE * len(states)).sum()), slot_purity=purity(slot, labels, len(module.keys)),
               relative_change=float(((post - pre).norm(dim=-1) / pre.norm(dim=-1)).mean()), purity_post=purity(kmeans(post, classes)[1], labels, classes))
    return out


def describe(weights, r):
    keys = ('gate', 'beta', 'top_weight', 'active_slots', 'slot_purity', 'relative_change', 'purity_pre', 'purity_post')
    return f'{weights}: {r["task"]} ({r["n_states"]} in-range states, {r["classes"]} classes)  ' + '  '.join(f'{k} {r[k]:.3f}' for k in keys if k in r)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='+')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    results = {}
    for path in a.weights:
        meta = torch.load(path, weights_only=True)['meta']
        if meta['arm'] not in ('core', 'core_clean') or dt.is_count(meta['task']):
            raise ValueError('the cleanup usage diagnostic is defined for the Core arms on the permutation tasks')
        r = usage(dr.load_model(path).eval(), meta['task'], meta['seed'])
        results[str(path)] = r
        print(describe(path, r), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
