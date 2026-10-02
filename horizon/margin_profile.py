"""Margin profile (descriptive, evaluation only): how large is the logit GAP of a correct prediction at each position, and how does its distribution move with position?

For a trained S_N arm the gap at position t is  logit[true class] - the largest other logit  (positive = right). A first error is a gap that crosses 0, so the first-error hazard is the
left tail of the gap distribution. The arm runs free (its own carried state from the first token), so the gaps mean something only until the first error of a sequence: every window below
uses SURVIVORS only (positions before the sequence's first error, the population of the first-error hazard). Per window of positions: the mean, the standard deviation and the 1 / 5 / 50 %
quantiles of the gap, and the fractions below 1, 2 and 4 nats (near misses: a smoother estimate of the tail than the error rate itself).

Two readings it separates (program note 5.20, written before the first run): the class cloud GROWS (mean gap falls and/or its spread rises with position, near misses climb smoothly) versus
a HEAVY TAIL (mean and spread stay put, rare outliers cross zero). No verdict, a description.

  python -B -m horizon.margin_profile results/Z1-pilot/perm5-core-w128-s1-cur.pt results/Z1-pilot/perm5-core-w128-s1-cur-lrdecay0.3.pt --out results/Z1-pilot/margin-profile-s5core.json
"""
import argparse
import json
from pathlib import Path

import torch

from . import depth_run as dr
from . import depth_tasks as dt

WINDOWS = ((0, 16), (16, 64), (64, 128), (128, 256), (256, 512), (512, 1024))
NEAR = (1.0, 2.0, 4.0)
QUANTILES = (0.01, 0.05, 0.5)


def gaps(logits, target):
    """(n, T) gap = logit of the true class minus the largest other logit."""
    true = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    other = logits.scatter(-1, target.unsqueeze(-1), float('-inf')).max(-1).values
    return true - other


def survivors(gap):
    """(n, T) bool: the positions strictly before the sequence's first error (a gap <= 0); a sequence without an error keeps every position."""
    wrong = gap <= 0
    first = torch.where(wrong.any(1), wrong.int().argmax(1), torch.full((gap.shape[0],), gap.shape[1]))
    return torch.arange(gap.shape[1]).unsqueeze(0) < first.unsqueeze(1)


def window_stats(gap, alive, windows=WINDOWS):
    """{'lo-hi': dict(n, mean, std, p1, p5, p50, below={'1.0': fraction, ...})} over the survivors of each window (windows past the evaluated length are dropped)."""
    out = {}
    for lo, hi in windows:
        if lo >= gap.shape[1]:
            continue
        hi = min(hi, gap.shape[1])
        g = gap[:, lo:hi][alive[:, lo:hi]]
        if g.numel() == 0:
            out[f'{lo}-{hi}'] = dict(n=0)
            continue
        q = torch.quantile(g.float(), torch.tensor(QUANTILES))
        out[f'{lo}-{hi}'] = dict(n=int(g.numel()), mean=float(g.mean()), std=float(g.std()) if g.numel() > 1 else None, p1=float(q[0]), p5=float(q[1]), p50=float(q[2]),
                                 below={str(t): float((g < t).float().mean()) for t in NEAR})
    return out


def profile(weights, length=1024, n=100, batch=50):
    """The margin profile of one saved arm on the dev sequences the hazard evaluation draws at `length` (same draw: seed, task, length, n)."""
    meta = torch.load(weights, weights_only=True)['meta']
    model = dr.load_model(weights).eval()
    x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
    parts = []
    with torch.no_grad():
        for at in range(0, n, batch):
            parts.append(gaps(model(x[at:at + batch]), y[at:at + batch]))
    gap = torch.cat(parts)
    alive = survivors(gap)
    return dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, length=length, windows=window_stats(gap, alive),
                sequences_without_error=int((alive.all(1)).sum()))


def describe(r):
    lines = [f"{r['arm']} {r['task']} w{r['width']} s{r['seed']} ({r['weights']}): n = {r['n']}, length {r['length']}, {r['sequences_without_error']} sequences without a first error"]
    for k, v in r['windows'].items():
        if not v.get('n'):
            lines.append(f'  [{k}) no survivors')
            continue
        lines.append(f"  [{k}) n {v['n']:>6}  gap mean {v['mean']:7.2f} std {v['std'] if v['std'] is None else round(v['std'], 2)}  p1 {v['p1']:7.2f} p5 {v['p5']:7.2f} p50 {v['p50']:7.2f}  "
                     + '  '.join(f'<{t}: {f:.2e}' for t, f in v['below'].items()))
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='+')
    p.add_argument('--length', type=int, default=1024)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    results = []
    for path in a.weights:
        r = profile(path, a.length, a.n)
        results.append(r)
        print(describe(r), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
