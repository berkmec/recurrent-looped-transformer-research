"""E-W: how much WEIGHT noise does a trained S_N arm tolerate? Evaluation only.

Program note 4.7 explains the constant-lr error floor of the GRU / LSTM as final-iterate weight noise (a frozen perturbation of the transition map): hazard = Phi(-m / s)
with s the logit noise the perturbation induces, and annealing shrinks s. That picture makes a direct, cheap prediction about the weights themselves: perturbing every weight
tensor w of a trained model by Gaussian noise of relative size eps (std eps x rms(w); biases, norm gains and embeddings included) should raise the first-error hazard along an
erfc-like curve, the models whose floor came from a noisy final iterate should tolerate LESS noise than their annealed twins, and an arm with a larger weight-space margin
(the Core?) should tolerate more. eps_tol = the smallest grid eps with a pooled hazard >= 1e-3 (infinity when none).

  python -B -m horizon.weight_noise results/Z1-pilot/perm4-gru-w64-s1-cur.pt results/Z1-pilot/perm4-core-w64-s1-cur.pt --out results/Z1-pilot/weight-noise.json
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import depth_run as dr
from .hazard_summary import pooled_windows

EPS_GRID = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
POINTS = (16, 64, 256, 1024)


def perturb(model, eps, seed):
    """A deep copy whose every parameter tensor w is replaced by w + eps x rms(w) x N(0, 1); the noise is drawn from its own generator (no global RNG use)."""
    out = copy.deepcopy(model)
    if eps <= 0:
        return out
    gen = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for p in out.parameters():
            rms = float(p.float().square().mean().sqrt())
            if rms > 0:
                p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * (eps * rms))
    return out


def noise_seed(draw, eps):
    """A seed that is distinct for every (draw, eps) pair (blind review R2-17: 1000 * (draw + 1) + round(eps * 1e4) gave eps 0.1 draw 1 and eps 0.2 draw 0 the SAME seed, so their noise
    directions were identical)."""
    return int(np.random.SeedSequence([int(draw), int(round(float(eps) * 1e9))]).generate_state(1)[0])


def measure(weights, eps_grid=EPS_GRID, draws=3, points=POINTS, n=100, batch=50):
    raw = torch.load(weights, weights_only=True)
    meta = raw['meta']
    if meta['arm'] not in ('core', 'gru', 'lstm'):
        raise ValueError('arm must be core, gru or lstm')
    base = dr.load_model(weights)
    rows = {}
    for eps in eps_grid:
        events = exposure = 0
        finals = []
        for d in range(draws if eps > 0 else 1):
            model = perturb(base, eps, seed=noise_seed(d, eps))
            curve = dr.evaluate(model, meta['task'], points, n, meta['seed'], batch=batch)
            cells = pooled_windows(dict(curve=curve))
            events += sum(c[0] for c in cells.values())
            exposure += sum(c[1] for c in cells.values())
            finals.append(curve[max(points)]['final'])
        rows[eps] = dict(events=events, exposure=exposure, hazard=(events / exposure if exposure else None), final_at_max_length=sum(finals) / len(finals), draws=len(finals))
    return dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], seed=meta['seed'], parameters=meta['parameters'], n=n, points=list(points),
                results=rows)


def eps_tol(rows, threshold=1e-3, excess=True):
    """Smallest eps > 0 whose pooled hazard is >= hazard(eps = 0) + `threshold` (the weight noise has ADDED `threshold` per step); infinity when none of the grid does. This is the E-W
    rule as amended after the second blind review (note 5.25, R2-2), before E-W ran: the first version used the absolute criterion hazard >= threshold (`excess=False`), which
    gives a model that already sits near the threshold unperturbed (a constant-lr arm) the smallest eps and a twin that starts near 0 the full distance, so the predicted order
    partly followed from where each arm starts. Grid keys may be strings (a JSON round trip) and are ordered numerically."""
    base = 0.0
    if excess:
        zero = next((rows[e] for e in rows if float(e) == 0.0), None)
        base = zero['hazard'] if zero and zero['hazard'] is not None else 0.0
    for eps in sorted((e for e in rows if float(e) > 0), key=float):
        h = rows[eps]['hazard']
        if h is not None and h >= base + threshold:
            return float(eps)
    return math.inf


def describe(r):
    cells = '  '.join(f"eps {float(e):g}: {v['hazard']:.2e} ({v['events']}/{v['exposure']})" for e, v in sorted(r['results'].items(), key=lambda kv: float(kv[0])))
    zero = next((v['hazard'] for e, v in r['results'].items() if float(e) == 0.0), None)
    return (f"{r['arm']} {r['task']} w{r['width']} ({r['parameters']} parameters), {r['n']} dev sequences, lengths {r['points']}: eps_tol {eps_tol(r['results'])} "
            f"(excess over the unperturbed hazard {zero if zero is None else format(zero, '.2e')}; absolute criterion: {eps_tol(r['results'], excess=False)})  |  {cells}")


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='+')
    p.add_argument('--eps', default=','.join(str(e) for e in EPS_GRID))
    p.add_argument('--draws', type=int, default=3)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    grid = tuple(float(v) for v in a.eps.split(','))
    results = []
    for path in a.weights:
        r = measure(path, grid, a.draws, n=a.n)
        results.append(r)
        print(describe(r), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
