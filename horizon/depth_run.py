"""Z1 runner: train one (task, arm, width, seed) on SHORT sequences and measure how far past the training depth it still works.

Training sees perm sequences of length TRAIN_LENGTH and counter sequences whose peak depth is at most TRAIN_DEPTH. Evaluation uses fresh
sequences from a separate source-seed namespace at lengths / peak depths up to 256x what was trained.

  python -B -m horizon.depth_run --task perm5 --arm core --width 128 --seed 1 --steps 6000 --out results/Z1-pilot/perm5-core-w128-s1.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import depth_tasks as dt
from .depth_models import make_arm, parameter_count
from .guards import GuardCounter, guarded_step

NAMESPACE = dict(train='z1-train', dev='z1-dev', test='z1-test')
THRESHOLD = 0.95


def train(task, arm, width, seed, steps, batch=64, lr=3e-3, clip=1.0, weight_decay=0.01, namespace=None, log_every=0, log=print):
    namespace = namespace or NAMESPACE
    model = make_arm(arm, task, width, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    guard, history, start = GuardCounter(), [], time.perf_counter()
    for step in range(1, steps + 1):
        b = dt.train_batch(namespace, task, seed, step, batch)
        model.train()
        loss = F.cross_entropy(model(b['x']).flatten(0, 1), b['y'].flatten())
        report = guard.record(guarded_step(opt, model.parameters(), loss, clip_norm=clip))
        if log_every and step % log_every == 0:
            history.append(dict(step=step, loss=report['loss'], grad_norm=report['grad_norm_before_clip']))
            log(f'{task} {arm} w{width} s{seed} step {step} loss {report["loss"]} ({time.perf_counter() - start:.0f}s)')
    return model, dict(steps=steps, batch=batch, lr=lr, clip=clip, guard=guard.summary(), history=history,
                       train_seconds=time.perf_counter() - start, parameters=parameter_count(model))


@torch.no_grad()
def _accuracy(model, x, y, batch):
    model.eval()
    correct_all, correct_final, ys, preds = 0, 0, [], []
    for at in range(0, len(x), batch):
        pred = model(x[at:at + batch]).argmax(-1)
        preds.append(pred)
        ys.append(y[at:at + batch])
    pred, target = torch.cat(preds), torch.cat(ys)
    return pred, target


def evaluate(model, task, points, n, seed, split='dev', namespace=None, batch=50):
    """points: sequence lengths for perm tasks, peak depths for 'count'. Returns {point: metrics}."""
    namespace = namespace or NAMESPACE
    out = {}
    for point in points:
        if task == 'count':
            length = max(4 * point, 64)
            x, y, _ = dt.heldout(namespace, task, seed, split, n, length, depth=point)
        else:
            length = point
            x, y, _ = dt.heldout(namespace, task, seed, split, n, length)
        pred, target = _accuracy(model, x, y, batch)
        ok = pred == target
        row = dict(length=length, final=float(ok[:, -1].float().mean()), prefix=float(ok.float().mean()))
        if task == 'count':
            unseen = target > dt.TRAIN_DEPTH
            row['unseen_depth'] = float(ok[unseen].float().mean()) if bool(unseen.any()) else None
            peak = target == point
            row['at_peak'] = float(ok[peak].float().mean()) if bool(peak.any()) else None
        else:
            beyond = torch.arange(x.shape[1]) >= dt.TRAIN_LENGTH
            row['beyond_horizon'] = float(ok[:, beyond].float().mean()) if bool(beyond.any()) else None
        out[point] = row
    return out


def effective_horizon(curve, key, threshold=THRESHOLD):
    """Largest point such that metric `key` is >= threshold at it and at every smaller evaluated point; 0 if none."""
    best = 0
    for point in sorted(curve):
        value = curve[point].get(key)
        if value is None or value < threshold:
            break
        best = point
    return best


def default_points(task):
    return [8, 16, 32, 64, 128, 256] if task == 'count' else [16, 32, 64, 128, 256, 512, 1024]


def run(task, arm, width, seed, steps, points=None, n_eval=100, out=None, **kw):
    started = time.perf_counter()
    model, trained = train(task, arm, width, seed, steps, **kw)
    points = points or default_points(task)
    curve = evaluate(model, task, points, n_eval, seed)
    key = 'unseen_depth' if task == 'count' else 'final'
    result = dict(task=task, arm=arm, width=width, seed=seed, train=trained, eval_points=points, curve=curve,
                  effective_horizon=effective_horizon(curve, key), horizon_metric=key,
                  trained_range=dt.TRAIN_DEPTH if task == 'count' else dt.TRAIN_LENGTH, wall_seconds=time.perf_counter() - started)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1), encoding='utf8')
    return result


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--points', default=None, help='comma list: lengths (perm) or peak depths (count)')
    p.add_argument('--n-eval', type=int, default=100)
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    points = [int(v) for v in a.points.split(',')] if a.points else None
    r = run(a.task, a.arm, a.width, a.seed, a.steps, points=points, n_eval=a.n_eval, out=a.out, batch=a.batch, lr=a.lr, log_every=a.log_every)
    print(json.dumps({k: r[k] for k in ('task', 'arm', 'width', 'seed', 'effective_horizon', 'horizon_metric', 'wall_seconds')}))
    for point, row in r['curve'].items():
        print(point, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == '__main__':
    main()
