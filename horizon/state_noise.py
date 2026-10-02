"""E-B: how much noise on the carried state does a trained recurrent arm tolerate? (and E-D's noisy-state training hook)

A per-step first-error hazard lambda is a margin statement: if each step perturbs the state by a relative sigma and the basin the state must stay in has
half-width r, the state leaves it with probability erfc(m / sqrt 2), m = r / sigma (Gaussian noise, spherical basin, independent steps: assumptions of
notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 4.2). Injecting noise of known size sigma into the carried state and measuring lambda(sigma) tests
that picture directly: lambda(sigma) should follow erfc(r / (sigma sqrt 2)) with ONE r per model (the implied radius r_i = sigma_i * m(lambda_i) must not
drift with sigma), and the ordering of r should follow the ordering of the noise-free hazard floors.

Carried state: the Core's previous final decoder output (what the feedback reads), a GRU's h, an LSTM's h and c; the noise is relative to the root-mean-square of
the vector it is added to, at every step from the second on (the first state is the initial one). Evaluation only here: noise changes the model's input at
inference, nothing is trained or saved. `noisy()` also returns a module that shares the arm's parameters, so it can drive training (E-D) as well.

  python -B -m horizon.state_noise results/Z1-pilot/perm4-core-w64-s1-cur.pt results/Z1-pilot/perm5-core-w128-s1-cur.pt --out results/Z1-pilot/state-noise-margin.json
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from core.model import Core

from . import depth_run as dr
from .hazard_summary import pooled_windows

SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]


def kick(h, sigma, gen):
    """h plus relative Gaussian noise: sigma times the rms of each row of h times a standard normal draw."""
    if sigma <= 0:
        return h
    return h + sigma * h.square().mean(-1, keepdim=True).sqrt() * torch.randn(h.shape, generator=gen, dtype=h.dtype)


class NoisyCore(Core):
    """The sealed Core whose previous final decoder output (the state the feedback reads) is perturbed before every decoder step but the first."""

    sigma = 0.0
    gen = None

    def decode(self, e, s, mk, mv, t):
        if s is not None and self.sigma > 0:
            s = dict(s)
            s['s'] = kick(s['s'], self.sigma, self.gen)
        return super().decode(e, s, mk, mv, t)


class NoisyRecurrent(nn.Module):
    """A GRU/LSTM arm unrolled cell by cell with the arm's OWN parameters (shared and differentiable) and relative noise on the carried state."""

    def __init__(self, arm, sigma, seed):
        super().__init__()
        self.arm, self.sigma = arm, float(sigma)
        self.gen = torch.Generator().manual_seed(seed)
        self.lstm = isinstance(arm.rnn, nn.LSTM)

    def forward(self, x):
        arm, rnn = self.arm, self.arm.rnn
        emb = arm.embedding(x)
        b, steps, width = emb.shape
        h, c, out = emb.new_zeros(b, width), emb.new_zeros(b, width), []
        for t in range(steps):
            if t > 0:
                h = kick(h, self.sigma, self.gen)
                if self.lstm:
                    c = kick(c, self.sigma, self.gen)
            gi = F.linear(emb[:, t], rnn.weight_ih_l0, rnn.bias_ih_l0)
            gh = F.linear(h, rnn.weight_hh_l0, rnn.bias_hh_l0)
            if self.lstm:
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


def noisy(model, sigma, seed):
    """The Core (changed in place) or a module around the GRU/LSTM arm that adds relative Gaussian noise to the carried state."""
    if isinstance(model, Core):
        model.__class__ = NoisyCore
        model.sigma = float(sigma)
        model.gen = torch.Generator().manual_seed(seed)
        return model
    if hasattr(model, 'rnn'):
        return NoisyRecurrent(model, sigma, seed)
    raise ValueError('state noise is defined for the Core, GRU and LSTM arms')


def clean(model):
    """Undo `noisy()` on a Core (back to the sealed class, no noise attributes left); other arms are never changed."""
    if isinstance(model, NoisyCore):
        model.__class__ = Core
        for name in ('sigma', 'gen'):
            model.__dict__.pop(name, None)
    return model


def margin_of(hazard):
    """m with erfc(m / sqrt 2) = hazard (bisection)."""
    lo, hi = 0.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > hazard:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def implied_radius(sigma, hazard):
    """r such that sigma * m = r: the basin half-width the Gaussian-basin model implies for one (sigma, hazard) pair; None outside (0, 0.5)."""
    if sigma <= 0 or not hazard or hazard <= 0 or hazard >= 0.5:
        return None
    return sigma * margin_of(hazard)


def pooled_hazards(curve):
    """(early, all): pooled first-error rate over the windows ending at or before position 64 and over all windows, plus the raw cells."""
    cells = pooled_windows(dict(curve=curve))
    def rate(selected):
        errors, exposure = sum(c[0] for k, c in cells.items() if selected(k)), sum(c[1] for k, c in cells.items() if selected(k))
        return (errors / exposure if exposure else None), errors, exposure
    return rate(lambda k: k[1] <= 64), rate(lambda _: True), cells


def stored_early(weights, points):
    """(first errors, exposure) of the windows ending at or before 64, summed over `points`, from the result file stored next to the weights
    (same draws as a sigma = 0 run); None when that file has no hazard data at those points."""
    path = Path(weights).with_suffix('.json')
    if not path.exists():
        return None
    curve = json.loads(path.read_text(encoding='utf8')).get('curve', {})
    events = exposure = 0
    for p in points:
        row = curve.get(str(p))
        if not row or 'hazard' not in row:
            return None
        for w in row['hazard']:
            if w['stop'] <= 64:
                events += w['first_errors']
                exposure += w['at_risk'] * (w['stop'] - w['start'])
    return events, exposure


def reproduces(stored, run, exact):
    """The sigma = 0 row against the stored numbers: Core exactly; GRU/LSTM run unrolled cell by cell (library layer agrees to 1e-5), so up to 2 first errors
    and 1 % of the exposure may differ."""
    if stored is None or run is None:
        return None
    if exact:
        return tuple(stored) == tuple(run)
    return abs(stored[0] - run[0]) <= 2 and abs(stored[1] - run[1]) <= 0.01 * max(stored[1], 1)


def margin_curve(weights, sigmas, points, n, seed, batch=50):
    raw = torch.load(weights, weights_only=True)
    meta = raw['meta']
    if meta['arm'] not in ('core', 'gru', 'lstm'):
        raise ValueError('arm must be core, gru or lstm, got ' + str(meta['arm']))
    rows = {}
    for sigma in sigmas:
        model = noisy(dr.load_model(weights), sigma, seed=1000 + seed)
        curve = dr.evaluate(model, meta['task'], points, n, seed, batch=batch)
        (early, e_err, e_exp), (total, t_err, t_exp), cells = pooled_hazards(curve)
        rows[sigma] = dict(final={p: curve[p]['final'] for p in points}, hazard_early=early, early_events=e_err, early_exposure=e_exp,
                           hazard_all=total, all_events=t_err, all_exposure=t_exp, implied_radius_early=implied_radius(sigma, early),
                           implied_radius_all=implied_radius(sigma, total),
                           windows={f'{a}-{b}': list(c) for (a, b), c in cells.items()})       # first errors and exposure per position window: does the noise sensitivity grow with position?
    check = None
    if 0.0 in rows and seed == meta['seed']:
        stored, run = stored_early(weights, points), (rows[0.0]['early_events'], rows[0.0]['early_exposure'])
        check = dict(stored=stored, run=run, reproduces=reproduces(stored, run, meta['arm'] == 'core'))
    return dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], parameters=meta['parameters'], seed=seed, n=n,
                points=list(points), sigma0_vs_stored=check, results=rows)


def sigma_tol(rows, threshold=1e-3):
    """Smallest sigma > 0 whose early (positions <= 64) hazard reaches `threshold`; infinity when none of the grid does."""
    for sigma in sorted(s for s in rows if s > 0):
        h = rows[sigma]['hazard_early']
        if h is not None and h >= threshold:
            return sigma
    return math.inf


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rank correlation with average ranks for ties; nan when one side is constant."""
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float('nan')


def radius_spread(rows):
    """Coefficient of variation of the early implied radius over the sigmas whose early hazard lies in (1e-4, 0.3); None with fewer than 3 such sigmas."""
    radii = [r['implied_radius_early'] for s, r in rows.items() if s > 0 and r['hazard_early'] is not None and 1e-4 < r['hazard_early'] < 0.3
             and r['implied_radius_early'] is not None]
    if len(radii) < 3:
        return None
    mean = sum(radii) / len(radii)
    return math.sqrt(sum((x - mean) ** 2 for x in radii) / len(radii)) / mean


def hazard_mle(cells, bracket=(1e-12, 1.0 - 1e-9)):
    """Maximum-likelihood CONSTANT per-step hazard from window cells [(first_errors, at_risk, width), ...] (blinded review finding F1).
    `first_error_hazard` divides the first errors of a window by at_risk x width, which counts every step of a sequence that has already failed inside the window as
    still at risk: the rate saturates at 1 / width (1/16 in [0, 16)) however large the true hazard is. Each at-risk sequence instead fails inside a window of width w
    with probability q = 1 - (1 - h)^w; the MLE of h solves the Bernoulli score equation (bisection). None without any sequence at risk, 0 without any first error."""
    cells = [(e, a, w) for e, a, w in cells if a > 0 and w > 0]
    if not cells:
        return None
    if sum(e for e, _, _ in cells) == 0:
        return 0.0

    def score(h):
        total = 0.0
        for e, a, w in cells:
            q = min(max(1.0 - (1.0 - h) ** w, 1e-15), 1.0 - 1e-15)
            total += (e / q - (a - e) / (1.0 - q)) * w * (1.0 - h) ** (w - 1)
        return total
    lo, hi = bracket
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if score(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def early_hazard_mle(row):
    """The MLE hazard over the windows ending at or before position 64 of one margin-curve row (cells stored as {'a-b': [first errors, exposure]})."""
    cells = []
    for key, (events, exposure) in row['windows'].items():
        a, b = (int(v) for v in key.split('-'))
        if b <= 64 and b > a:
            cells.append((events, int(round(exposure / (b - a))), b - a))
    return hazard_mle(cells)


def radius_spread_mle(rows):
    """radius_spread with the MLE hazard instead of events / (at_risk x width): the same selection (sigma > 0, hazard in (1e-4, 0.3), at least 3 sigmas)."""
    radii = []
    for sigma, row in rows.items():
        h = early_hazard_mle(row) if sigma > 0 and row.get('windows') else None
        if h is not None and 1e-4 < h < 0.3:
            radii.append(sigma * margin_of(h))
    if len(radii) < 3:
        return None
    mean = sum(radii) / len(radii)
    return math.sqrt(sum((x - mean) ** 2 for x in radii) / len(radii)) / mean


def verdict(results):
    """The reading rule written BEFORE the run (program note, section 5.1, E-B) applied to the list of per-model results (as saved in the JSON).
    The primary O3 keeps the pre-registered estimator; `radius_cv_mle` / `O3_mle` repeat it with the MLE hazard (review finding F1) so both are on record.
    A model whose sigma = 0 row does not reproduce its stored result is not read. O1: S4 Core early hazard at sigma 0.1 < 1e-3. O2: sigma_tol(S5 Core) <
    sigma_tol(S4 Core). O3: per model, the implied radius has a coefficient of variation <= 0.35 (else the Gaussian-basin model is rejected for it).
    O4: over the four S4 GRU/LSTM models, Spearman(sigma_tol, noise-free hazard) <= -0.6."""
    models = {}
    for r in results:
        rows = {float(s): row for s, row in r['results'].items()}
        check = (r.get('sigma0_vs_stored') or {}).get('reproduces')
        name = f"{r['task']}-{r['arm']}-w{r['width']}"
        models[name] = dict(rows=rows, valid=check is True, tol=sigma_tol(rows), spread=radius_spread(rows), spread_mle=radius_spread_mle(rows),
                            floor=rows.get(0.0, {}).get('hazard_all'))
    out = dict(models={n: dict(valid=m['valid'], sigma_tol=m['tol'], radius_cv=m['spread'], radius_cv_mle=m['spread_mle'], noise_free_hazard=m['floor'])
                       for n, m in models.items()})
    s4, s5 = models.get('perm4-core-w64'), models.get('perm5-core-w128')
    out['O1'] = (s4['rows'][0.1]['hazard_early'] < 1e-3) if s4 and s4['valid'] and 0.1 in s4['rows'] and s4['rows'][0.1]['hazard_early'] is not None else None
    out['O2'] = (s5['tol'] < s4['tol']) if s4 and s5 and s4['valid'] and s5['valid'] else None
    out['O3'] = {n: (m['spread'] <= 0.35 if m['spread'] is not None else None) for n, m in models.items() if m['valid']}
    out['O3_mle'] = {n: (m['spread_mle'] <= 0.35 if m['spread_mle'] is not None else None) for n, m in models.items() if m['valid']}
    four = [models.get(n) for n in ('perm4-gru-w64', 'perm4-lstm-w64', 'perm4-gru-w128', 'perm4-lstm-w112')]
    if all(m and m['valid'] and m['floor'] is not None for m in four):
        out['O4_spearman'] = spearman([m['tol'] if math.isfinite(m['tol']) else 1.0 for m in four], [m['floor'] for m in four])
        out['O4'] = out['O4_spearman'] <= -0.6
    else:
        out['O4_spearman'], out['O4'] = None, None
    return out


def describe(r):
    lines = [f"{r['arm']} {r['task']} w{r['width']} ({r['parameters']} parameters), dev seed {r['seed']}, n = {r['n']}, lengths {r['points']}"]
    c = r.get('sigma0_vs_stored')
    if c:
        lines.append(f"  sigma 0 vs the stored result (first errors, exposure in windows <= 64): stored {c['stored']} run {c['run']} -> "
                     + {True: 'REPRODUCES', False: 'DOES NOT REPRODUCE: do not read this model', None: 'no stored hazard data'}[c['reproduces']])
    for sigma, row in r['results'].items():
        def f(v, spec='.2e'):
            return 'n/a' if v is None else format(v, spec)
        lines.append(f"  sigma {sigma:<5} hazard early (<=64) {f(row['hazard_early'])} ({row['early_events']}/{row['early_exposure']})  all {f(row['hazard_all'])} "
                     f"({row['all_events']}/{row['all_exposure']})  implied radius early {f(row['implied_radius_early'], '.3f')} all {f(row['implied_radius_all'], '.3f')}  "
                     f"final " + ' '.join(f"{p}:{v:.2f}" for p, v in row['final'].items()))
        lines.append('        by window (x1e-4/step): ' + '  '.join(f"[{k.replace('-', ',')}) {(c[0] / c[1] * 1e4 if c[1] else float('nan')):.1f}"
                                                                for k, c in row['windows'].items()))
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='*')
    p.add_argument('--sigmas', default=','.join(str(s) for s in SIGMAS))
    p.add_argument('--points', default='16,64,256,1024')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=1, help='dev-draw seed')
    p.add_argument('--out', default=None)
    p.add_argument('--verdict', default=None, help='apply the pre-registered reading rule to this result file and exit (weights are ignored)')
    a = p.parse_args()
    if a.verdict:
        print(json.dumps(verdict(json.loads(Path(a.verdict).read_text(encoding='utf8'))), indent=1, default=str))
        return
    if not a.weights:
        p.error('give at least one weights file (or --verdict RESULT.json)')
    sigmas = [float(v) for v in a.sigmas.split(',') if v]
    points = [int(v) for v in a.points.split(',')]
    results = []
    for path in a.weights:
        r = margin_curve(path, sigmas, points, a.n, a.seed)
        results.append(r)
        print(describe(r), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
