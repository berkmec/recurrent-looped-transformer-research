"""Z1 runner: train one (task, arm, width, seed) on SHORT sequences and measure how far past the training depth it still works.

Training sees perm sequences of length TRAIN_LENGTH and counter sequences whose peak depth is at most TRAIN_DEPTH. Evaluation uses fresh
sequences from a separate source-seed namespace. The default points reach 64x the trained length (permutation tasks, 1,024 vs 16) and 32x the
trained peak depth (counter, 256 vs 8; its sequences are up to 1,536 long); other points can be given with --points.

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
TOLERANCE = 1e-6      # accuracies are float32 means over n = 100 sequences: 95/100 is stored as 0.94999999 and must still meet a 0.95 threshold


def meets(value, threshold=THRESHOLD):
    return value >= threshold - TOLERANCE


def _log(message):
    print(message, flush=True)    # redirected stdout is block-buffered: without the flush a long run shows nothing until it exits


def train(task, arm, width, seed, steps, batch=64, lr=3e-3, clip=1.0, weight_decay=0.01, namespace=None, log_every=0, log=_log,
          curriculum=None, advance_at=0.97, min_stage_steps=200, window=100, state_noise=0.0):
    """curriculum: increasing training lengths, e.g. [2, 4, 8, 16]. The stage advances when the mean per-position accuracy over the last
    `window` updates reaches `advance_at` (and the stage has run at least `min_stage_steps` updates). The last length is the trained range;
    nothing beyond it is ever trained on. Without a curriculum every update uses the task's trained length.
    state_noise: relative Gaussian noise (sigma times the rms) added to the carried state at every step but the first during TRAINING only
    (E-D, `horizon/state_noise.py`); the returned model is the plain arm and evaluation is clean. 0 = no noise (the previous behaviour, bit for bit)."""
    namespace = namespace or NAMESPACE
    lengths = list(curriculum) if curriculum else [dt.train_length(task)]
    model = make_arm(arm, task, width, seed)
    forward = model
    if state_noise > 0:
        from .state_noise import noisy         # imported here: state_noise imports this module
        forward = noisy(model, state_noise, seed=seed + 7)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    guard, history, start = GuardCounter(), [], time.perf_counter()
    stage, in_stage, recent, advances = 0, 0, [], []
    for step in range(1, steps + 1):
        b = dt.train_batch(namespace, task, seed, step, batch, length=lengths[stage])
        model.train()
        logits = forward(b['x'])
        loss = F.cross_entropy(logits.flatten(0, 1), b['y'].flatten())
        report = guard.record(guarded_step(opt, model.parameters(), loss, clip_norm=clip))
        recent.append(float((logits.detach().argmax(-1) == b['y']).float().mean()))
        recent = recent[-window:]
        in_stage += 1
        if stage < len(lengths) - 1 and in_stage >= min_stage_steps and len(recent) == window and sum(recent) / window >= advance_at:
            advances.append(dict(step=step, from_length=lengths[stage], to_length=lengths[stage + 1]))
            log(f'{task} {arm} w{width} s{seed} step {step} curriculum advance {lengths[stage]} -> {lengths[stage + 1]} ({time.perf_counter() - start:.0f}s)')
            stage, in_stage, recent = stage + 1, 0, []
        if log_every and step % log_every == 0:
            history.append(dict(step=step, loss=report['loss'], grad_norm=report['grad_norm_before_clip'], length=lengths[stage],
                                recent_accuracy=(sum(recent) / len(recent)) if recent else None))
            log(f'{task} {arm} w{width} s{seed} step {step} len {lengths[stage]} loss {report["loss"]} ({time.perf_counter() - start:.0f}s)')
    if state_noise > 0:
        from .state_noise import clean
        clean(model)
    return model, dict(steps=steps, batch=batch, lr=lr, clip=clip, guard=guard.summary(), history=history, curriculum=lengths, advances=advances,
                       final_stage_length=lengths[stage], train_seconds=time.perf_counter() - start, parameters=parameter_count(model),
                       state_noise=state_noise)


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


HAZARD_EDGES = [0, 16, 32, 64, 128, 256, 512, 1024]


def first_error_hazard(pred, target, edges):
    """Per-step probability of a FIRST error in each position window [start, stop), among sequences still error-free at `start`.
    A constant rate means the failures are a fixed per-step error (precision, budget); a rate that grows past the trained length
    means the mechanism itself stops working there. Windows past the end are dropped, the last is clipped to the sequence length."""
    wrong = pred != target
    length = wrong.shape[1]
    first = torch.where(wrong.any(1), wrong.int().argmax(1), torch.full((wrong.shape[0],), length))
    rows = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if start >= length:
            break
        stop = min(stop, length)
        at_risk = int((first >= start).sum())
        events = int(((first >= start) & (first < stop)).sum())
        rows.append(dict(start=start, stop=stop, at_risk=at_risk, first_errors=events,
                         rate=(events / (at_risk * (stop - start))) if at_risk else None))
    return rows


def evaluate(model, task, points, n, seed, split='dev', namespace=None, batch=50):
    """points: sequence lengths for perm tasks, peak depths for 'count'. Returns {point: metrics}."""
    namespace = namespace or NAMESPACE
    out = {}
    for point in points:
        if dt.is_count(task):
            length = dt.count_eval_length(point, task)
            x, y, _ = dt.heldout(namespace, task, seed, split, n, length, depth=point)
        else:
            length = point
            x, y, _ = dt.heldout(namespace, task, seed, split, n, length)
        pred, target = _accuracy(model, x, y, batch)
        ok = pred == target
        row = dict(length=length, final=float(ok[:, -1].float().mean()), prefix=float(ok.float().mean()))
        if dt.is_count(task):
            trajectory, trained_peak = dt.depth_trajectory(x), dt.peak(task)
            deep = trajectory > trained_peak
            reentry, resync = dt.reentry_mask(trajectory, trained_peak), dt.resync_mask(trajectory, trained_peak)
            # `deep` = accuracy where the true depth exceeds the trained peak. The label saturates there, so this measures whether the model keeps
            # emitting the saturated label (a model that always says LABEL_CAP scores 1.0, like a perfect counter); it does NOT measure counting.
            row['deep'] = float(ok[deep].float().mean()) if bool(deep.any()) else None
            row['reentry'] = float(ok[reentry].float().mean()) if bool(reentry.any()) else None   # first descent: needs the exact count
            row['reentry_positions'] = int(reentry.sum())
            row['resync'] = float(ok[resync].float().mean()) if bool(resync.any()) else None      # after the floor reset: re-synchronisation
            row['resync_positions'] = int(resync.sum())
        else:
            beyond = torch.arange(x.shape[1]) >= dt.TRAIN_LENGTH
            row['beyond_horizon'] = float(ok[:, beyond].float().mean()) if bool(beyond.any()) else None
            row['hazard'] = first_error_hazard(pred, target, HAZARD_EDGES)
        out[point] = row
    return out


def evaluate_length_control(model, lengths, n, seed, split='dev', namespace=None, batch=50, task='count'):
    """Counter only: the TRAINING distribution (depth never above the trained peak) on sequences far longer than the trained length.
    Separates 'cannot go deeper' from 'cannot go longer'."""
    namespace = namespace or NAMESPACE
    out = {}
    for length in lengths:
        x, y, _ = dt.heldout(namespace, task, seed, split, n, length)
        pred, target = _accuracy(model, x, y, batch)
        ok = pred == target
        out[length] = dict(final=float(ok[:, -1].float().mean()), prefix=float(ok.float().mean()))
    return out


def effective_horizon(curve, key, threshold=THRESHOLD):
    """Largest point such that metric `key` is >= threshold at it and at every larger-than-skipped evaluated point up to it; points where the
    metric does not apply (None) are skipped; 0 if no point passes."""
    best = 0
    for point in sorted(curve):
        value = curve[point].get(key)
        if value is None:
            continue
        if not meets(value, threshold):
            break
        best = point
    return best


def default_points(task):
    if dt.is_count(task):       # multiples of the trained peak; the sequences are 6 x the point long, so the largest point is capped at 512
        return [m * dt.peak(task) for m in (1, 2, 4, 8, 16, 32) if m * dt.peak(task) <= 512]
    return [16, 32, 64, 128, 256, 512, 1024]


def save_model(model, path, task, arm, width, seed, trained):
    """Weights as plain tensors plus primitive metadata only, so `torch.load(path, weights_only=True)` works."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(task=task, arm=arm, width=int(width), seed=int(seed), steps=int(trained['steps']), parameters=int(trained['parameters']),
                curriculum=[int(v) for v in trained['curriculum']])
    torch.save(dict(state_dict=model.state_dict(), meta=meta), path)


def load_model(path):
    raw = torch.load(path, weights_only=True)
    meta = raw['meta']
    model = make_arm(meta['arm'], meta['task'], meta['width'], meta['seed'])
    model.load_state_dict(raw['state_dict'])
    return model


def run(task, arm, width, seed, steps, points=None, n_eval=100, out=None, **kw):
    started = time.perf_counter()
    model, trained = train(task, arm, width, seed, steps, **kw)
    if out:   # before the slow evaluation: a crash there must not lose the trained model
        save_model(model, Path(out).with_suffix('.pt'), task, arm, width, seed, trained)
    # a horizon only means something if the task was learned inside the trained range, so that point is ALWAYS evaluated (never fail open)
    counting = dt.is_count(task)
    in_range_point = dt.peak(task) if counting else dt.TRAIN_LENGTH
    in_range_key = 'prefix' if counting else 'final'
    points = sorted(set(points or default_points(task)) | {in_range_point})
    eval_batch = 8 if arm == 'tfm' else 50          # attention is quadratic in the length: the transformer arm needs small batches at 1,000+ positions
    curve = evaluate(model, task, points, n_eval, seed, batch=eval_batch)
    key = 'reentry' if counting else 'final'
    in_range = curve[in_range_point][in_range_key]
    learned = meets(in_range)
    result = dict(task=task, arm=arm, width=width, seed=seed, train=trained, eval_points=points, curve=curve,
                  learned_in_range=learned, in_range_accuracy=in_range,
                  effective_horizon=effective_horizon(curve, key) if learned else 0, horizon_metric=key,
                  trained_range=in_range_point, wall_seconds=time.perf_counter() - started)
    if counting:
        trained_length = dt.train_length(task)       # 64 / 256 / 1,024 for the default counter (4 x and 16 x the trained length)
        result['length_control'] = evaluate_length_control(model, [trained_length, 4 * trained_length, 16 * trained_length], n_eval, seed,
                                                           batch=eval_batch, task=task)
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
    p.add_argument('--curriculum', default=None, help='comma list of increasing training lengths, e.g. 1,2,3,4,6,8,12,16')
    p.add_argument('--advance-at', type=float, default=0.97)
    p.add_argument('--min-stage-steps', type=int, default=200)
    p.add_argument('--state-noise', type=float, default=0.0, help='relative Gaussian noise on the carried state during training only (E-D)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    points = [int(v) for v in a.points.split(',')] if a.points else None
    curriculum = [int(v) for v in a.curriculum.split(',')] if a.curriculum else None
    r = run(a.task, a.arm, a.width, a.seed, a.steps, points=points, n_eval=a.n_eval, out=a.out, batch=a.batch, lr=a.lr, log_every=a.log_every,
            curriculum=curriculum, advance_at=a.advance_at, min_stage_steps=a.min_stage_steps, state_noise=a.state_noise)
    print(json.dumps({k: r[k] for k in ('task', 'arm', 'width', 'seed', 'learned_in_range', 'effective_horizon', 'horizon_metric', 'wall_seconds')}))
    for point, row in r['curve'].items():
        print(point, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    for length, row in r.get('length_control', {}).items():
        print('length control', length, {k: round(v, 3) for k, v in row.items()})


if __name__ == '__main__':
    main()
