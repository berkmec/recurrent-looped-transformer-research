"""E-Q: are the per-step first errors of a trained S_N arm DETERMINISTIC (the same few (state, token) transitions are wrong again and again) or NOISE-LIKE (spread
evenly over all transitions)?

E-B showed that the GRU/LSTM error floors hardly move with state noise up to sigma = 0.02-0.05 (the floor is already there at sigma = 0). A floor that does not depend on
noise is a misfit, not a margin: a handful of the |G| x |G| transition-table entries (previous true state, token) never got learned exactly, so every visit to one of them
fails. This counts, among the positions where the model was still error-free (so the previous state was right), the visits and the errors per (previous true state,
token) pair and asks two questions of the error counts c_p (E errors, P pairs):
  concentration  C = sum_p c_p^2 / (E + E (E - 1) / P): 1 for errors spread uniformly at random over the pairs (multinomial null), large when a few pairs hold them.
  replicability  the Pearson correlation of the per-pair error counts of two independent dev draws (seeds s and s + 1): deterministic misfit repeats, noise does not.
Reading rule (program note, section 5.9, written before any model was read): DETERMINISTIC when C >= 3 and the split-half correlation >= 0.3; NOISE-LIKE when C <= 1.5
and the correlation < 0.1; otherwise MIXED. A model with fewer than 30 errors in either draw is not read.

  python -B -m horizon.pair_errors results/Z1-pilot/perm4-gru-w64-s1-cur.pt --out results/Z1-pilot/pair-errors.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import depth_run as dr
from . import depth_tasks as dt

MIN_ERRORS = 30


def pair_counts(x, target, pred, vocab, start=1, stop=None):
    """(visits, errors) arrays of length vocab**2 over the at-risk positions in [start, stop): positions up to and including the sequence's first error. The pair is
    (the true state before the token, the token)."""
    correct = (pred == target)
    length = x.shape[1]
    wrong = ~correct
    first = np.where(wrong.any(1), wrong.int().argmax(1).numpy(), length)
    stop = length if stop is None else min(stop, length)
    t = np.arange(length)[None, :]
    at_risk = (t <= first[:, None]) & (t >= max(start, 1)) & (t < stop)
    prev = np.zeros_like(target.numpy())
    prev[:, 1:] = target.numpy()[:, :-1]
    pair = prev * vocab + x.numpy()
    event = at_risk & (t == first[:, None])
    visits = np.bincount(pair[at_risk], minlength=vocab * vocab)
    errors = np.bincount(pair[event], minlength=vocab * vocab)
    return visits, errors


def concentration(errors):
    e, p = float(errors.sum()), errors.size
    if e < 2:
        return None
    return float((errors.astype(np.float64) ** 2).sum() / (e + e * (e - 1) / p))


def correlation(a, b):
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def reading(conc, corr, errors_a, errors_b):
    if min(errors_a, errors_b) < MIN_ERRORS or conc is None or corr is None:
        return 'TOO FEW ERRORS'
    if conc >= 3 and corr >= 0.3:
        return 'DETERMINISTIC'
    if conc <= 1.5 and corr < 0.1:
        return 'NOISE-LIKE'
    return 'MIXED'


def marginals(errors, vocab, top=5):
    grid = errors.reshape(vocab, vocab)
    total = max(int(errors.sum()), 1)
    return dict(top_states_share=float(np.sort(grid.sum(1))[::-1][:top].sum() / total), top_tokens_share=float(np.sort(grid.sum(0))[::-1][:top].sum() / total),
                top_pairs_share=float(np.sort(errors)[::-1][:top].sum() / total))


def analyse(weights, n=3000, length=64, windows=((1, 64),), batch=50, seeds=None):
    raw = torch.load(weights, weights_only=True)
    meta = raw['meta']
    if dt.is_count(meta['task']):
        raise ValueError('pair errors are defined for the permutation tasks')
    vocab = dt.vocab(meta['task'])
    model = dr.load_model(weights)
    seeds = seeds or (meta['seed'], meta['seed'] + 1)
    draws = []
    for s in seeds:
        x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], s, 'dev', n, length)
        pred, target = dr._accuracy(model, x, y, batch)
        draws.append((x, target, pred))
    out = dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], n=n, length=length, seeds=list(seeds), windows={})
    for start, stop in windows:
        halves = [pair_counts(x, target, pred, vocab, start, stop) for x, target, pred in draws]
        errors = [h[1] for h in halves]
        pooled_v, pooled_e = sum(h[0] for h in halves), sum(h[1] for h in halves)
        conc, corr = concentration(pooled_e), correlation(*errors)
        out['windows'][f'{start}-{stop}'] = dict(
            visits=int(pooled_v.sum()), errors=int(pooled_e.sum()), hazard=(float(pooled_e.sum() / pooled_v.sum()) if pooled_v.sum() else None),
            errors_per_draw=[int(e.sum()) for e in errors], pairs_with_errors=int((pooled_e > 0).sum()), pairs=vocab * vocab, max_pair_errors=int(pooled_e.max()),
            concentration=conc, split_half_correlation=corr, reading=reading(conc, corr, int(errors[0].sum()), int(errors[1].sum())), **marginals(pooled_e, vocab))
    return out


def describe(r):
    lines = [f"{r['arm']} {r['task']} w{r['width']}: {r['n']} x {r['length']} dev positions per draw, seeds {r['seeds']}"]
    for w, v in r['windows'].items():
        f = lambda x: 'n/a' if x is None else f'{x:.3f}'
        lines.append(f"  positions [{w.replace('-', ',')}): hazard {f(v['hazard'])} ({v['errors']} errors over {v['visits']} at-risk visits; per draw {v['errors_per_draw']}) | "
                     f"pairs with errors {v['pairs_with_errors']}/{v['pairs']}, max per pair {v['max_pair_errors']} | concentration C {f(v['concentration'])}, "
                     f"split-half correlation {f(v['split_half_correlation'])} -> {v['reading']} | top-5 pairs/states/tokens hold "
                     f"{v['top_pairs_share']:.2f}/{v['top_states_share']:.2f}/{v['top_tokens_share']:.2f} of the errors")
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='+')
    p.add_argument('--n', type=int, default=3000)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--windows', default='1-64', help='comma list of start-stop position windows, e.g. 1-64,128-256 (length must reach the last stop)')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    windows = tuple(tuple(int(v) for v in w.split('-')) for w in a.windows.split(','))
    results = []
    for path in a.weights:
        r = analyse(path, n=a.n, length=a.length, windows=windows)
        results.append(r)
        print(describe(r), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
