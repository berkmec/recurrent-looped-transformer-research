"""E-A5x (descriptive, no verdict): how does the carried state's deviation from the canonical class state grow with its AGE since a clean injection? Evaluation only, CPU, dev split.

E-A5 (note 5.26) found for the S5 Core that after the carried state is replaced by the true class's centroid the next ~64 steps are error-free and the first-error hazard comes back only
in the window 64..128 steps later (3.9e-4, the in-range onset again): the model's reliability depends on the AGE of its state since the last clean reset, not on the absolute position.
This measures the cause directly. For `t0` in {8, 128, 256} the state is replaced by the true class's centroid before step t0 (state and cache, as E-A5's `state+cache`); the following
128 steps are followed and, for every age a = 0 .. 127 steps after the injection, the deviation of the carried state z from the centroid mu[c] of its TRUE class c is split into

  total       sqrt(mean |z - mu[c]|^2)                                          (relative to M, the mean nearest-other-centroid distance)
  systematic  the part that is the same for every sequence in the class: the unbiased estimate <v_even, v_odd> of |mean deviation of the class|^2 (v_even / v_odd = the mean deviation of
              the even / odd sequences of the class, so independent sampling noise does not inflate it); a class-specific constant offset that grows with age is a clock-like drift
  random      the rest: history-dependent scatter (random-walk-like when its square grows linearly with age)

and the untouched run (the same positions, no injection) is measured the same way for comparison. How the total grows with age (like sqrt(age): neutral accumulation of independent
per-step errors; like age: a coherent bias; saturating: a restoring force) says what an error-correcting fix has to do. Descriptive only.

  python -B -m horizon.drift_profile results/Z1-pilot/perm5-core-w128-s1-cur.pt ... --out results/Z1-pilot/drift-profile.json
"""
import argparse
import json
from pathlib import Path

import torch

from . import depth_run as dr
from . import depth_tasks as dt
from .erosion_source import class_centroids, configure
from .state_growth import core_trajectory

T0S = (8, 128, 256)
SPAN = 128
AGES = ((0, 1), (1, 4), (4, 16), (16, 32), (32, 64), (64, 128))


def decompose(dev, lab, parity):
    """(total, systematic) of the mean squared deviation: total = mean |dev|^2, systematic = count-weighted mean over classes of <mean dev of the even rows, mean dev of the odd rows>,
    the unbiased estimate of |class mean deviation|^2 (clipped at 0 at the caller); dev (N, W), lab (N,) class ids, parity (N,) bool."""
    total = float(dev.square().sum(-1).mean())
    weight, cross = 0.0, 0.0
    for c in torch.unique(lab).tolist():
        pick = lab == c
        even, odd = dev[pick & parity], dev[pick & ~parity]
        if len(even) and len(odd):
            n = int(pick.sum())
            cross += n * float((even.mean(0) * odd.mean(0)).sum())
            weight += n
    return total, (cross / weight if weight else 0.0)


def nearest_centroid_distance(centroids):
    dist = torch.cdist(centroids, centroids) + torch.eye(len(centroids)) * 1e30
    return float(dist.min(1).values.mean())


def profile(weights, t0s=T0S, span=SPAN, n=2000, batch=250, log=print):
    meta = torch.load(weights, weights_only=True)['meta']
    if meta['arm'] != 'core' or dt.is_count(meta['task']):
        raise ValueError('the drift profile is defined for the Core on the permutation tasks')
    model = dr.load_model(weights).eval()
    centroids = class_centroids(model, meta['task'], meta['seed'])
    scale = nearest_centroid_distance(centroids)
    length = max(t0s) + span
    x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
    parity = torch.arange(n) % 2 == 0
    out = {}
    for t0 in t0s:
        for variant in ('injected', 'untouched'):
            if variant == 'untouched' and t0 == t0s[0]:
                continue                                    # the in-range window is the same run as the injected one's first steps: skip the duplicate
            devs = []
            with torch.no_grad():
                for at in range(0, n, batch):
                    sl = slice(at, at + batch)
                    configure(model)
                    if variant == 'injected':
                        configure(model, at=t0, state=centroids[y[sl, t0 - 1]], reset_cache=True)
                    states, _ = core_trajectory(model, x[sl, :t0 + span])
                    configure(model)
                    devs.append(states[:, t0:t0 + span] - centroids[y[sl, t0:t0 + span]])
            dev = torch.cat(devs)                           # (n, span, W)
            rows = {}
            for a, b in AGES:
                if a >= span:
                    continue
                b = min(b, span)
                flat = dev[:, a:b].reshape(-1, dev.shape[-1])
                lab = y[:, t0 + a:t0 + b].reshape(-1)
                par = parity[:, None].expand(n, b - a).reshape(-1)
                total, systematic = decompose(flat, lab, par)
                systematic = max(0.0, systematic)
                rows[f'{a}-{b}'] = dict(total=total ** 0.5 / scale, systematic=systematic ** 0.5 / scale, random=max(0.0, total - systematic) ** 0.5 / scale)
            out[f'{variant}@{t0}'] = rows
            log(f'  t0 {t0:>3} {variant:<9} ' + '  '.join(f'age {k}: {v["total"]:.3f} (sys {v["systematic"]:.3f})' for k, v in rows.items()))
    return dict(weights=str(weights), task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, span=span, scale=scale, profile=out)


def describe(r):
    lines = [f'{r["task"]} w{r["width"]} (n = {r["n"]}, M = {r["scale"]:.2f}): deviation from the true class centroid / M, by age (total, systematic, random)']
    for key, rows in r['profile'].items():
        lines.append(f'  {key:<14} ' + '  '.join(f'{k}: {v["total"]:.3f} / {v["systematic"]:.3f} / {v["random"]:.3f}' for k, v in rows.items()))
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='+')
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--batch', type=int, default=250)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    results = []
    for path in a.weights:
        r = profile(path, n=a.n, batch=a.batch, log=lambda m: print(m, flush=True))
        results.append(r)
        print(describe(r), flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(results, indent=1, default=float), encoding='utf8')


if __name__ == '__main__':
    main()
