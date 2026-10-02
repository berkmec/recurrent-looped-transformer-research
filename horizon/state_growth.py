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
    growing R / M with position is a loss of margin. Classes with fewer than `min_count` states in the window are skipped. Descriptive only (no threshold)."""
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
        series = growth_series(model, x, t0, eps, seed=11)
        result[eps] = summarise(series, t0, steps, rel)
    # extra recorded detail, not part of the reading rule: the one-step Jacobian's largest singular value at a few positions (8 sequences)
    positions = [t for t in jacobian_positions if 1 <= t < steps]
    norms = jacobian_norms(model, x[:jacobian_n], positions)
    jac = {t: dict(median=float(np.median(v)), max=float(np.max(v)), above_one=float(np.mean(np.array(v) > 1.0))) for t, v in norms.items()}
    return dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], parameters=meta['parameters'], steps=steps, t0=t0, n=n,
                results=result, jacobian=jac, state_rms=state_rms_by_window(model, x),
                class_margin=class_margin_by_window(model, x, dt.oracle(meta['task'], x)))


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
    p.add_argument('weights', nargs='+')
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--t0', type=int, default=8)
    p.add_argument('--eps', default='0.001,0.01')
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default=None)
    a = p.parse_args()
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
