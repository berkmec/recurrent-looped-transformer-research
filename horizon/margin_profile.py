"""Margin profile (descriptive, evaluation only): how large is the logit GAP of a correct prediction at each position, and how does its distribution move with position?

For a trained S_N arm the gap at position t is  logit[true class] - the largest other logit  (positive = right). A first error is a gap that crosses 0, so the first-error hazard is the
left tail of the gap distribution. The arm runs free (its own carried state from the first token), so the gaps mean something only until the first error of a sequence: every window below
uses SURVIVORS only (positions before the sequence's first error, the population of the first-error hazard). Per window of positions: the mean, the standard deviation and the 1 / 5 / 50 %
quantiles of the gap, and the fractions below 1, 2 and 4 nats (near misses: a smoother estimate of the tail than the error rate itself).

Two readings it separates (program note 5.20, written before the first run): the class cloud GROWS (mean gap falls and/or its spread rises with position, near misses climb smoothly) versus
a HEAVY TAIL (mean and spread stay put, rare outliers cross zero). No verdict, a description.

Conventions: the window keys are HALF-OPEN position windows, '16-64' = positions 16 .. 63 (the E-L1K step bins and the E-S walk buckets name INCLUSIVE ranges instead: '32-63' = 32 .. 63).
The first error of a sequence is defined as in `depth_run.first_error_hazard` (argmax != target), not as gap <= 0: they differ only on exact ties and non-finite logits (blind review R2-14).
Only the permutation tasks are profiled (a counter arm would be evaluated on a different draw than `depth_run.evaluate` uses; R2-15).

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


def survivors(gap, wrong=None):
    """(n, T) bool: the positions strictly before the sequence's first error; a sequence without an error keeps every position. `wrong` = the boolean error mask (argmax != target, the
    definition of `depth_run`); without it the gap is used (an error = gap <= 0)."""
    if wrong is None:
        wrong = gap <= 0
    first = torch.where(wrong.any(1), wrong.int().argmax(1), torch.full((gap.shape[0],), gap.shape[1]))
    return torch.arange(gap.shape[1]).unsqueeze(0) < first.unsqueeze(1)


def window_stats(gap, alive, windows=WINDOWS):
    """{'lo-hi': dict(n, sequences, mean, std, p1, p5, p50, below={'1.0': fraction, ...})} over the survivors of each half-open window [lo, hi) (windows past the evaluated length are
    dropped); `sequences` = how many sequences contribute at least one position."""
    out = {}
    for lo, hi in windows:
        if lo >= gap.shape[1]:
            continue
        hi = min(hi, gap.shape[1])
        g = gap[:, lo:hi][alive[:, lo:hi]]
        if g.numel() == 0:
            out[f'{lo}-{hi}'] = dict(n=0, sequences=0)
            continue
        q = torch.quantile(g.float(), torch.tensor(QUANTILES))
        out[f'{lo}-{hi}'] = dict(n=int(g.numel()), sequences=int(alive[:, lo:hi].any(1).sum()), mean=float(g.mean()), std=float(g.std()) if g.numel() > 1 else None,
                                 p1=float(q[0]), p5=float(q[1]), p50=float(q[2]), below={str(t): float((g < t).float().mean()) for t in NEAR})
    return out


EROSION_BASE = '16-64'
EROSION_LATE = ('512-1024', '256-512')       # the latest window that still has enough surviving positions
MIN_LATE_N = 500
MIN_LATE_SEQUENCES = 20                      # ... contributed by at least this many sequences (blind review R2-16: 500 positions can come from a single survivor)
FLAT_BAND = 0.15
EROSION_RATIO = 0.7


def erosion(windows):
    """Descriptive summary used by E-K4 / E-K5 (program note 5.23): ratio = the mean gap of the latest window with >= MIN_LATE_N surviving positions from >= MIN_LATE_SEQUENCES sequences
    / the mean gap of [16, 64). EROSION when ratio <= 0.7, FLAT when within +-15 % of 1, GROWTH above 1.15, PARTIAL in between; NO DATA without both windows. Also the near-miss
    (< 2 nat) fractions. Windows stored before `sequences` existed count as having enough sequences."""
    base = windows.get(EROSION_BASE)
    late_key = next((k for k in EROSION_LATE if windows.get(k, {}).get('n', 0) >= MIN_LATE_N and windows[k].get('sequences', MIN_LATE_SEQUENCES) >= MIN_LATE_SEQUENCES), None)
    if not base or not base.get('n') or late_key is None:
        return dict(verdict='NO DATA')
    late = windows[late_key]
    ratio = late['mean'] / base['mean']
    verdict = 'EROSION' if ratio <= EROSION_RATIO else 'FLAT' if abs(ratio - 1) <= FLAT_BAND else 'GROWTH' if ratio > 1 + FLAT_BAND else 'PARTIAL'
    return dict(verdict=verdict, ratio=ratio, base_window=EROSION_BASE, late_window=late_key, base_mean=base['mean'], late_mean=late['mean'],
                near_miss_base=base['below']['2.0'], near_miss_late=late['below']['2.0'])


def profile(weights, length=1024, n=100, batch=50):
    """The margin profile of one saved arm on the dev sequences the hazard evaluation draws at `length` (same draw: seed, task, length, n)."""
    meta = torch.load(weights, weights_only=True)['meta']
    if dt.is_count(meta['task']):
        raise ValueError('the margin profile is defined for the permutation tasks')
    model = dr.load_model(weights).eval()
    x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
    parts, wrongs = [], []
    with torch.no_grad():
        for at in range(0, n, batch):
            logits = model(x[at:at + batch])
            parts.append(gaps(logits, y[at:at + batch]))
            wrongs.append(logits.argmax(-1) != y[at:at + batch])           # the error definition of depth_run.first_error_hazard
    gap = torch.cat(parts)
    alive = survivors(gap, torch.cat(wrongs))
    windows = window_stats(gap, alive)
    return dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, length=length, windows=windows,
                sequences_without_error=int((alive.all(1)).sum()), erosion=erosion(windows))


def describe(r):
    lines = [f"{r['arm']} {r['task']} w{r['width']} s{r['seed']} ({r['weights']}): n = {r['n']}, length {r['length']}, {r['sequences_without_error']} sequences without a first error"]
    for k, v in r['windows'].items():
        if not v.get('n'):
            lines.append(f'  [{k}) no survivors')
            continue
        lines.append(f"  [{k}) n {v['n']:>6}  gap mean {v['mean']:7.2f} std {v['std'] if v['std'] is None else round(v['std'], 2)}  p1 {v['p1']:7.2f} p5 {v['p5']:7.2f} p50 {v['p50']:7.2f}  "
                     + '  '.join(f'<{t}: {f:.2e}' for t, f in v['below'].items()))
    e = r.get('erosion')
    if e:
        lines.append('  erosion: ' + (e['verdict'] if 'ratio' not in e else f"{e['verdict']} (mean gap [{e['late_window']}) / [{e['base_window']}) = {e['ratio']:.2f}; near misses (< 2 nats) "
                                                                            f"{e['near_miss_base']:.1e} -> {e['near_miss_late']:.1e})"))
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
