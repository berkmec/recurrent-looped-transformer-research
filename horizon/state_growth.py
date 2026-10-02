"""E-B2: does a small kick to the carried state grow, shrink or die out? (the S5 Core break: an unstable mode, or something else?)

One relative kick of size eps (a random direction, eps times the rms of the state) is added to the carried state at position t0 of a clean run; the kicked
and the clean run are then followed side by side (float64, so a decaying kick stays visible). d_t = |state'_t - state_t| / |state_t| is tracked per position;
the mean over sequences of log d_t over a window of positions has a slope: the finite-time growth rate (> 0: expanding, < 0: contracting; 0: neutral).
Prediction flips (the argmax at position t differs between the two runs) show whether the kick matters for the OUTPUT. Carried state: the Core's previous
final decoder output (what the feedback reads; the three cached decoder keys/values are rebuilt from the kicked run, as in a real run), a GRU's h, an LSTM's
h and c (the diagnostic uses their concatenation).

  python -B -m horizon.state_growth results/Z1-pilot/perm4-core-w64-s1-cur.pt results/Z1-pilot/perm5-core-w128-s1-cur.pt --out results/Z1-pilot/state-growth.json
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.model import Core

from . import depth_run as dr
from . import depth_tasks as dt

FLOOR = 1e-14                 # |kick| below this relative size counts as "fully re-converged" in float64


def _rms(v):
    return v.square().mean(-1, keepdim=True).sqrt()


def core_trajectory(model, x, t0=None, delta=None):
    """(states, logits) of the Core's non-streaming forward; with t0 / delta the carried state is kicked before the decoder step at position t0."""
    e, _, _ = model.encode(x)
    n = model.mn(e)
    mk = model.rope(model.split(model.mk(n)))
    mv = model.split(model.mv(n))
    s, states, logits = None, [], []
    for t in range(x.shape[1]):
        if t0 is not None and t == t0:
            s = dict(s)
            s['s'] = s['s'] + delta
        y, s = model.decode(e[:, t], s, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
        states.append(s['s'])
        logits.append(y)
    return torch.stack(states, 1), torch.stack(logits, 1)


def recurrent_trajectory(arm, x, t0=None, delta=None):
    """(states, logits) of a GRU/LSTM arm unrolled by hand (same maths as `state_noise.NoisyRecurrent`); delta is (h kick, c kick)."""
    rnn, lstm = arm.rnn, isinstance(arm.rnn, nn.LSTM)
    emb = arm.embedding(x)
    b, steps, width = emb.shape
    h, c, states, outs = emb.new_zeros(b, width), emb.new_zeros(b, width), [], []
    for t in range(steps):
        if t0 is not None and t == t0:
            h = h + delta[0]
            if lstm:
                c = c + delta[1]
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
        states.append(torch.cat([h, c], -1) if lstm else h)
        outs.append(h)
    return torch.stack(states, 1), arm.read(torch.stack(outs, 1))


def state_rms_by_window(model, x, edges=(0, 16, 64, 128, 192, 256)):
    """Mean over sequences of the root-mean-square of the carried state (the Core's final decoder output, a GRU's h, an LSTM's h and c) inside position windows:
    a drift of the state's scale between the trained range and long positions (R2's `state distribution shift` reading of the S5 break) shows up here."""
    model = copy.deepcopy(model).double().eval()
    run = core_trajectory if isinstance(model, Core) else recurrent_trajectory
    with torch.no_grad():
        states, _ = run(model, x)
    rms = states.square().mean(-1).sqrt().mean(0)           # (steps,)
    return {f'{a}-{min(b, x.shape[1])}': float(rms[a:b].mean()) for a, b in zip(edges[:-1], edges[1:]) if a < x.shape[1]}


def class_margin_by_window(model, x, labels, edges=(0, 16, 64, 128, 192, 256), min_count=3):
    """Cluster geometry of the carried state by true class (R2's q(t) = R / M): inside each position window the states are grouped by the label (the running
    product), R is the mean root-mean-square distance of a state to its class centroid, M the mean distance from a class centroid to the nearest other centroid; a
    growing R / M with position is a loss of margin. Classes with fewer than `min_count` states in the window are skipped. Descriptive only (no threshold).
    Caveat (review finding F10): the within-class spread is estimated from fewer states in a short window, so the SAME geometry reads ~7-8 % lower in [0, 16) than in
    the later 48-64-position windows (120 classes, 50 sequences); compare windows of similar width (from [16, 64) on), not the first against the rest."""
    model = copy.deepcopy(model).double().eval()
    run = core_trajectory if isinstance(model, Core) else recurrent_trajectory
    with torch.no_grad():
        states, _ = run(model, x)
    out = {}
    for a, b in zip(edges[:-1], edges[1:]):
        if a >= x.shape[1]:
            break
        b = min(b, x.shape[1])
        s, lab = states[:, a:b].reshape(-1, states.shape[-1]), labels[:, a:b].reshape(-1)
        centroids, spreads = {}, []
        for c in torch.unique(lab).tolist():
            members = s[lab == c]
            if members.shape[0] >= min_count:
                centroids[c] = members.mean(0)
                spreads.append(float(((members - centroids[c]).square().sum(-1)).mean().sqrt()))
        if len(centroids) < 2:
            continue
        cent = torch.stack(list(centroids.values()))
        dist = torch.cdist(cent, cent) + torch.eye(len(cent), dtype=cent.dtype) * 1e30
        spread, nearest = sum(spreads) / len(spreads), float(dist.min(1).values.mean())
        out[f'{a}-{b}'] = dict(spread=spread, nearest_centroid=nearest, ratio=spread / nearest, classes=len(centroids))
    return out


def jacobian_norms(model, x, positions):
    """Largest singular value of the ONE-STEP Jacobian of the carried state, d state_t / d state_{t-1} (float64, one sequence at a time), at the given
    positions of x. The Core's three cached decoder keys/values are held at their clean values (only the feedback path is differentiated); the LSTM's
    state is h and c together. Returns {position: [norm per sequence]}. A norm below 1 is a local contraction, above 1 a local expansion."""
    model = copy.deepcopy(model).double().eval()
    out = {t: [] for t in positions}
    for i in range(x.shape[0]):
        xi = x[i:i + 1]
        if isinstance(model, Core):
            with torch.no_grad():
                e, _, _ = model.encode(xi)
                n = model.mn(e)
                mk = model.rope(model.split(model.mk(n)))
                mv = model.split(model.mv(n))
                s, states = None, []
                for t in range(xi.shape[1]):
                    _, s = model.decode(e[:, t], s, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
                    states.append(s)
            for t in positions:
                prev = states[t - 1]
                def step(v, t=t, prev=prev):
                    return model.decode(e[:, t], dict(prev, s=v), mk[:, :, :t + 1], mv[:, :, :t + 1], t)[1]['s'][0]
                jac = torch.autograd.functional.jacobian(step, prev['s'])[:, 0, :]
                out[t].append(float(torch.linalg.matrix_norm(jac, ord=2)))
        else:
            lstm = isinstance(model.rnn, nn.LSTM)
            with torch.no_grad():
                emb = model.embedding(xi)
                states = []
                h = emb.new_zeros(1, emb.shape[-1])
                c = emb.new_zeros(1, emb.shape[-1])
                for t in range(xi.shape[1]):
                    h, c = _cell(model.rnn, emb[:, t], h, c, lstm)
                    states.append((h, c))
            for t in positions:
                h0, c0 = states[t - 1]
                width = h0.shape[-1]
                def step(v, t=t, width=width):
                    h_, c_ = _cell(model.rnn, emb[:, t], v[:, :width], v[:, width:], lstm)
                    return torch.cat([h_, c_], -1)[0] if lstm else h_[0]
                start = torch.cat([h0, c0], -1) if lstm else h0
                jac = torch.autograd.functional.jacobian(step, start)[:, 0, :]
                out[t].append(float(torch.linalg.matrix_norm(jac, ord=2)))
    return out


def _cell(rnn, x_t, h, c, lstm):
    gi = F.linear(x_t, rnn.weight_ih_l0, rnn.bias_ih_l0)
    gh = F.linear(h, rnn.weight_hh_l0, rnn.bias_hh_l0)
    if lstm:
        i, f, g, o = (gi + gh).chunk(4, -1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c
    ir, iz, inn = gi.chunk(3, -1)
    hr, hz, hn = gh.chunk(3, -1)
    r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
    return (1 - z) * torch.tanh(inn + r * hn) + z * h, c


def kick_for(model, clean_states, t0, eps, gen):
    """A relative kick (random direction, eps times the rms of the clean state at t0 - 1) for the carried state the runner kicks at position t0."""
    prev = clean_states[:, t0 - 1]
    if isinstance(model, Core):
        return eps * _rms(prev) * torch.randn(prev.shape, generator=gen, dtype=prev.dtype)
    width = prev.shape[-1] // (2 if isinstance(model.rnn, nn.LSTM) else 1)
    h, c = prev[:, :width], prev[:, width:] if isinstance(model.rnn, nn.LSTM) else None
    kick_h = eps * _rms(h) * torch.randn(h.shape, generator=gen, dtype=h.dtype)
    return kick_h, (eps * _rms(c) * torch.randn(c.shape, generator=gen, dtype=c.dtype) if c is not None else None)


def growth_series(model, x, t0, eps, seed):
    """({'log_d': (n, steps) log relative difference, 'flip': (n, steps) prediction flips}, steps) for one batch of sequences x."""
    model = copy.deepcopy(model).double().eval()
    run = core_trajectory if isinstance(model, Core) else recurrent_trajectory
    with torch.no_grad():
        clean_states, clean_logits = run(model, x)
        delta = kick_for(model, clean_states, t0, eps, torch.Generator().manual_seed(seed))
        kicked_states, kicked_logits = run(model, x, t0, delta)
    d = (kicked_states - clean_states).norm(dim=-1) / clean_states.norm(dim=-1)
    flip = kicked_logits.argmax(-1) != clean_logits.argmax(-1)
    return dict(log_d=torch.log(d.clamp_min(FLOOR)).numpy(), flip=flip.numpy(), d=d.numpy())


def slope(values, start, stop):
    """Least-squares slope per position of the column means of `values` (n, steps) over positions start .. stop - 1 (the finite-time growth rate)."""
    y = values[:, start:stop].mean(0)
    t = np.arange(start, stop, dtype=np.float64)
    return float(np.polyfit(t, y, 1)[0]) if len(t) > 1 else float('nan')


def summarise(series, t0, steps, windows):
    out = dict(windows={}, converged=float((series['d'][:, -1] < 1e-10).mean()))
    for a, b in windows:
        a, b = max(a, t0 + 1), min(b, steps)
        if b - a < 2:
            continue
        out['windows'][f'{a}-{b}'] = dict(growth_per_step=slope(series['log_d'], a, b), flip_rate=float(series['flip'][:, a:b].mean()),
                                          mean_log_d_start=float(series['log_d'][:, a].mean()), mean_log_d_stop=float(series['log_d'][:, b - 1].mean()))
    return out


def measure(weights, steps, t0, eps_list, n, seed, windows=((0, 25), (25, 121), (121, 10 ** 9)), jacobian_positions=(4, 16, 32, 64, 128, 192, 255),
            jacobian_n=8):
    raw = torch.load(weights, weights_only=True)
    meta = raw['meta']
    if meta['arm'] not in ('core', 'gru', 'lstm'):
        raise ValueError('arm must be core, gru or lstm')
    model = dr.load_model(weights)
    x, _, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, steps)
    rel = [(a + t0, b + t0 if b < 10 ** 9 else b) for a, b in windows]      # windows are counted from the kick
    result = {}
    for eps in eps_list:
        series = growth_series(model, x, t0, eps, seed=seed)       # the kick directions; the dev draw itself uses the model's own seed (review finding F12: this was hard-coded 11)
        result[eps] = summarise(series, t0, steps, rel)
    # extra recorded detail, not part of the reading rule: the one-step Jacobian's largest singular value at a few positions (8 sequences)
    positions = [t for t in jacobian_positions if 1 <= t < steps]
    norms = jacobian_norms(model, x[:jacobian_n], positions)
    jac = {t: dict(median=float(np.median(v)), max=float(np.max(v)), above_one=float(np.mean(np.array(v) > 1.0))) for t, v in norms.items()}
    return dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], parameters=meta['parameters'], steps=steps, t0=t0, n=n,
                results=result, jacobian=jac, state_rms=state_rms_by_window(model, x),
                class_margin=class_margin_by_window(model, x, dt.oracle(meta['task'], x)))


TOLERANCE = 1e-9            # |growth per step| below this is numerical zero
GROWTH_G2 = 0.005           # note 5.4, G2: expansion per step that supports "unstable mode"
FLIP_FACTOR = 5.0           # note 5.4, G2: or a last-window flip rate this many times the S4 Core's


def _name(r):
    return f"{r['task']}-{r['arm']}-w{r['width']}"


def _width(window_key):
    a, b = (int(v) for v in str(window_key).split('-'))
    return b - a


def _vanished(window):
    """The perturbation has already reached the float64 floor (every position's log d is the clamp) when the window starts: the window carries no growth information
    (its 'slope' is rounding noise of averaging clamped values, e.g. +9e-18 for the S4 Core and +8.7e-8 / +4.4e-5 for the S4 LSTM), and the perturbation has contracted
    as far as it can. Added after the first reading, because the literal tolerance above was not enough for the S4 LSTM; both readings are recorded in the note."""
    return window['mean_log_d_start'] <= math.log(FLOOR) + 0.01


def _non_expanding(window):
    return window['growth_per_step'] <= TOLERANCE or _vanished(window)


def _contracting(window, converged):
    """A window contracts when the growth is negative, or the perturbation has already vanished (and every sequence re-converged)."""
    return window['growth_per_step'] < -TOLERANCE or (_vanished(window) and converged >= 0.99)


def verdict(records):
    """E-B2 reading (program note 5.4), applied to the records of `measure`; every eps is read separately and the rule must hold at both.
    G1: S4 Core: growth <= 0 in every window and flip rate < 1e-3 in the first two windows. G2: S5 Core: last-window growth > +0.005 per step OR last-window flip
    rate >= 5x the S4 Core's (undefined when the S4 Core's is 0: reported as such, not counted); refuted when every window's growth <= 0 and the flip rate is about the
    S4 Core's (within 1e-3). G3: every GRU / LSTM model contracts in every window. The tolerance for 'zero' is declared above (added after the first reading)."""
    by = {_name(r): r for r in records}
    out = dict(tolerance=TOLERANCE)
    core4, core5 = by.get('perm4-core-w64'), by.get('perm5-core-w128')
    if core4:
        g1 = {}
        for eps, row in core4['results'].items():
            wins = list(row['windows'].values())
            g1[eps] = dict(growth=[w['growth_per_step'] for w in wins], flips=[w['flip_rate'] for w in wins],
                           holds=all(_non_expanding(w) for w in wins) and all(w['flip_rate'] < 1e-3 for w in wins[:2]))
        out['G1'] = dict(per_eps=g1, holds=all(v['holds'] for v in g1.values()))
    if core4 and core5:
        base = {eps: list(row['windows'].values())[-1]['flip_rate'] for eps, row in core4['results'].items()}
        last_width = {eps: _width(list(row['windows'])[-1]) for eps, row in core4['results'].items()}
        g2 = {}
        for eps, row in core5['results'].items():
            wins = list(row['windows'].values())
            last = wins[-1]
            flip_defined = base[eps] > 0
            supported = last['growth_per_step'] > GROWTH_G2 or (flip_defined and last['flip_rate'] >= FLIP_FACTOR * base[eps])
            refuted = all(_non_expanding(w) for w in wins) and abs(last['flip_rate'] - base[eps]) <= 1e-3
            # second reading (review finding F3): the "5x the S4 Core's flip rate" clause is degenerate when that rate is 0 (any S5 rate then qualifies), so it is also
            # evaluated with the S4 rate floored at its detection limit 3 / (sequences x steps in the last window); both readings are recorded, the coded verdict is unchanged
            floor = 3.0 / (core4['n'] * last_width[eps]) if core4.get('n') and last_width[eps] else 0.0
            supported_floored = last['growth_per_step'] > GROWTH_G2 or last['flip_rate'] >= FLIP_FACTOR * max(base[eps], floor)
            g2[eps] = dict(growth=[w['growth_per_step'] for w in wins], last_flip=last['flip_rate'], s4_last_flip=base[eps], flip_clause_defined=flip_defined,
                           supported=supported, refuted=refuted, s4_flip_detection_limit=floor, supported_floored=supported_floored)
        out['G2'] = dict(per_eps=g2, verdict='SUPPORTED' if all(v['supported'] for v in g2.values()) else 'REFUTED' if all(v['refuted'] for v in g2.values())
                         else 'NEITHER (neutral: no contraction, no expansion above the threshold)',
                         verdict_with_floored_flip_clause=('SUPPORTED' if all(v['supported_floored'] for v in g2.values()) else
                                                           'REFUTED' if all(v['refuted'] for v in g2.values()) else 'NEITHER (neutral: no contraction, no expansion above the threshold)'))
    g3 = {}
    for name, r in by.items():
        if r['arm'] in ('gru', 'lstm'):
            g3[name] = {eps: dict(growth=[w['growth_per_step'] for w in row['windows'].values()], contracting=all(_contracting(w, row['converged']) for w in row['windows'].values()))
                        for eps, row in r['results'].items()}
    if g3:
        out['G3'] = dict(per_model={n: all(v['contracting'] for v in d.values()) for n, d in g3.items()}, detail=g3,
                         holds=all(all(v['contracting'] for v in d.values()) for d in g3.values()))
    return out


def describe(r):
    lines = [f"{r['arm']} {r['task']} w{r['width']} ({r['parameters']} parameters), {r['n']} dev sequences of length {r['steps']}, kick at position {r['t0']}"]
    for eps, row in r['results'].items():
        cells = '  '.join(f"[{k.replace('-', ',')}) growth {v['growth_per_step']:+.4f}/step flips {v['flip_rate']:.4f}" for k, v in row['windows'].items())
        lines.append(f"  eps {eps:g}: {cells}  | re-converged (d < 1e-10 at the end): {row['converged']:.2f}")
    if r.get('state_rms'):
        lines.append('  state rms by position window: ' + '  '.join(f'[{k.replace("-", ",")}) {v:.3f}' for k, v in r['state_rms'].items()))
    if r.get('class_margin'):
        lines.append('  class geometry R / M (within-class spread over nearest-centroid distance; classes present) by window: ' +
                     '  '.join(f"[{k.replace('-', ',')}) {v['ratio']:.3f} ({v['classes']})" for k, v in r['class_margin'].items()))
    if r.get('jacobian'):
        lines.append('  one-step Jacobian, largest singular value (median / max over sequences): ' +
                     '  '.join(f"t={t}: {v['median']:.3f}/{v['max']:.3f}" for t, v in r['jacobian'].items()))
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='*')
    p.add_argument('--verdict', default=None, help='apply the G1-G3 reading rule to a saved state-growth.json instead of measuring')
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--t0', type=int, default=8)
    p.add_argument('--eps', default='0.001,0.01')
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--seed', type=int, default=11, help='seed of the random kick directions (default 11, the value the stored results used)')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    if a.verdict:
        print(json.dumps(verdict(json.loads(Path(a.verdict).read_text(encoding='utf8'))), indent=1, default=str))
        return
    eps_list = [float(v) for v in a.eps.split(',')]
    results = []
    for path in a.weights:
        r = measure(path, a.steps, a.t0, eps_list, a.n, a.seed)
        results.append(r)
        print(describe(r), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
