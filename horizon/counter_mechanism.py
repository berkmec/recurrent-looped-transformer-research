"""Z1 counter pilots: where does a model get its first-descent positions right?

The first descent (`depth_tasks.reentry_mask`) is the stretch after an excursion deeper than the trained peak, from the first position back inside
the trained range up to the first return to depth 0. Only an exact count of the excursion answers it. A counter that saturated at the trained peak
is right at one place per sequence, the arrival at depth 0 (the floor clamps the true depth and the model's own count to 0 together); a counter
that really counted is right all along. This reports the accuracy of a saved model by the TRUE depth inside that window, on the same dev draw
`depth_run.evaluate` uses (same namespace, seed, length, n), so the totals equal the `reentry` numbers of the result file.

  python -B -m horizon.counter_mechanism results/Z1-pilot/count-gru-w64-s1.pt --depths 16 64
"""
import argparse
import json

import torch

from . import depth_run as dr
from . import depth_tasks as dt


def window_by_depth(pred, target, trajectory):
    """({true depth: [correct, positions]}, [correct, positions]) over the first-descent window of every sequence."""
    mask = dt.reentry_mask(trajectory)
    ok = pred == target
    by_depth = {}
    for depth in range(dt.TRAIN_DEPTH + 1):
        at = mask & (trajectory == depth)
        if bool(at.any()):
            by_depth[depth] = [int((ok & at).sum()), int(at.sum())]
    return by_depth, [int((ok & mask).sum()), int(mask.sum())]


def report(path, depths, n=100, split='dev', batch=8):
    raw = torch.load(path, weights_only=True)
    meta = raw['meta']
    model = dr.load_model(path)
    curve = {}
    for depth in depths:
        length = dt.count_eval_length(depth)
        x, y, _ = dt.heldout(dr.NAMESPACE, 'count', meta['seed'], split, n, length, depth=depth)
        pred, target = dr._accuracy(model, x, y, batch)
        by_depth, total = window_by_depth(pred, target, dt.depth_trajectory(x))
        curve[depth] = dict(length=length, window=total, by_true_depth=by_depth)
    return dict(path=str(path), arm=meta['arm'], task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, split=split, curve=curve)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='+')
    p.add_argument('--depths', default='16,64', help='comma list of peak depths')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--out', default=None, help='write the JSON here as well')
    a = p.parse_args()
    depths = [int(v) for v in a.depths.replace(' ', ',').split(',') if v]
    results = [report(path, depths, n=a.n) for path in a.weights]
    for r in results:
        print(f"{r['arm']} w{r['width']} seed {r['seed']} ({r['path']})")
        for depth, row in r['curve'].items():
            correct, positions = row['window']
            cells = ' '.join(f"d{d}:{c}/{m}" for d, (c, m) in sorted(row['by_true_depth'].items()))
            print(f"  peak {depth:>3} (length {row['length']}): first descent {correct}/{positions} = {correct / max(positions, 1):.3f} | {cells}")
    if a.out:
        with open(a.out, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
