"""Z1-G: the sealed U1 / U2c neutral-run protocol applied to a plain GRU.

The campaign's first positive architecture result (a learned HOLD/WRITE gate that keeps the Core's state through neutral runs) had no
standard-recurrence baseline. A GRU already has an update gate. This module trains a GRU under EXACTLY the U1 protocol (same numbers, read
from horizon/sweep.py, same source seeds so the training batches and test problems are the ones the Core arms saw) and evaluates it the way
U1 / U2c did: clean lengths 128..4096 and neutral insertions (128 / 512 slots at the prefix / middle / suffix of a length-128 problem).

  python -B -m horizon.gru_control --out results/Z1-G --lane 0 --lanes 2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import metrics as mt
from . import sweep as sw
from . import task_families as tf
from .depth_models import GRUArm
from .guards import GuardCounter, guarded_step

WIDTHS = (32, 64)    # 32: the width of the Core arms (about 6.7k GRU parameters); 64: about 25k parameters, matched to the Core arm's 24k


def protocol():
    return sw.protocol()


def conditions(p=None):
    p = p or protocol()
    return [dict(arm=f'gru{w}', width=w, task=t, seed=s) for w in WIDTHS for t in p['tasks'] for s in p['primary_seeds']]


def ident(c):
    return f"z1g-{c['arm']}-{c['task']}-s{c['seed']}"


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def make_model(c):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'])
        return GRUArm(tf.VOCAB[c['task']], tf.CLASSES[c['task']], c['width'])


@torch.no_grad()
def predict(model, x, y, batch=32):
    model.eval()
    logits = torch.cat([model(x[at:at + batch]) for at in range(0, len(x), batch)])
    pred = logits.argmax(-1)
    stats = dict(final_loss=float(F.cross_entropy(logits[:, -1], y[:, -1])), final_accuracy=float((pred[:, -1] == y[:, -1]).float().mean()),
                 prefix_accuracy=float((pred == y).float().mean()))
    return stats, pred


def train(c, p, steps=None, validation_every=None, validation_n=None):
    steps = p['steps'] if steps is None else steps
    validation_every = p['validation_every'] if validation_every is None else validation_every
    validation_n = p['validation_n'] if validation_n is None else validation_n
    model = make_model(c)
    opt = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    vx, vy, _ = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'validation', validation_n, p['train_length'])
    counter, history, best, selected, started = GuardCounter(), [], None, None, time.perf_counter()
    for step in range(1, steps + 1):
        batch = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], p['train_length'])
        model.train()
        loss = F.cross_entropy(model(batch['x']).flatten(0, 1), batch['y'].flatten())
        report = counter.record(guarded_step(opt, model.parameters(), loss, clip_norm=p['clip']))
        if step % validation_every == 0 or step == steps:
            stats, _ = predict(model, vx, vy)
            history.append(dict(step=step, loss=report['loss'], **stats))
            if best is None or stats['final_loss'] < best:        # strictly lower: the earliest step wins ties
                best = stats['final_loss']
                selected = dict(step=step, validation=stats, state={k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(selected['state'])
    return model, dict(selected_step=selected['step'], selected_validation=selected['validation'], history=history, guard=counter.summary(),
                       train_seconds=time.perf_counter() - started, steps=steps)


def evaluate(model, c, p, n=None, clean_lengths=None, neutral_slots=None, neutral_placements=None, train_length=None):
    n = p['test_n'] if n is None else n
    clean_lengths = p['test_lengths'] if clean_lengths is None else clean_lengths
    neutral_slots = p['neutral_slots'] if neutral_slots is None else neutral_slots
    neutral_placements = p['neutral_placements'] if neutral_placements is None else neutral_placements
    train_length = p['train_length'] if train_length is None else train_length
    row = dict(clean={}, neutral={})
    for length in clean_lengths:
        x, y, seed = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'test', n, length)
        stats, _ = predict(model, x, y)
        row['clean'][str(length)] = dict(final_accuracy=stats['final_accuracy'], prefix_accuracy=stats['prefix_accuracy'], n=n, source_seed=seed)
    x, y, seed = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'neutral', n, train_length)
    base_stats, base_pred = predict(model, x, y)
    row['neutral']['0-base'] = dict(final_accuracy=base_stats['final_accuracy'], n=n, source_seed=seed)
    for slots in [s for s in neutral_slots if s > 0]:
        for placement in neutral_placements:
            xx, yy = tf.insert_neutrals(x, c['task'], slots, placement)
            stats, pred = predict(model, xx, yy)
            row['neutral'][f'{slots}-{placement}'] = dict(
                final_accuracy=stats['final_accuracy'], loss_pp=100 * (base_stats['final_accuracy'] - stats['final_accuracy']),
                **mt.flip_counts(base_pred[:, -1].numpy(), pred[:, -1].numpy(), yy[:, -1].numpy()))
    return row


def run_one(out, c, steps=None, validation_every=None, validation_n=None, n=None, clean_lengths=None, neutral_slots=None, neutral_placements=None,
            train_length=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{ident(c)}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    p = protocol()
    started = time.perf_counter()
    model, trained = train(c, p, steps=steps, validation_every=validation_every, validation_n=validation_n)
    torch.save(dict(state_dict=model.state_dict(), meta=dict(task=c['task'], arm=c['arm'], width=int(c['width']), seed=int(c['seed']),
                                                            selected_step=int(trained['selected_step']), parameters=int(parameter_count(model)))),
               out / f'{ident(c)}.pt')                                    # before the slow evaluation
    row = evaluate(model, c, p, n=n, clean_lengths=clean_lengths, neutral_slots=neutral_slots, neutral_placements=neutral_placements, train_length=train_length)
    result = dict(condition=c, ident=ident(c), parameters=parameter_count(model), train=trained, wall_seconds=time.perf_counter() - started,
                  protocol=dict(steps=p['steps'], batch=p['batch'], train_length=p['train_length'], lr=p['lr'], source='horizon/sweep.py protocol() (U1)'), **row)
    path.write_text(json.dumps(result, indent=1), encoding='utf8')
    return result


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/Z1-G')
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--lanes', type=int, default=1)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    cs = conditions()
    if args.list:
        for c in cs:
            print(ident(c))
        return
    for c in cs[args.lane::args.lanes]:
        r = run_one(args.out, c)
        base = r['neutral']['0-base']['final_accuracy']
        print(ident(c), 'params', r['parameters'], 'selected', r['train']['selected_step'], 'base', round(base, 3),
              {k: round(v['final_accuracy'], 3) for k, v in r['neutral'].items() if k != '0-base'},
              {k: round(v['final_accuracy'], 3) for k, v in r['clean'].items()}, flush=True)


if __name__ == '__main__':
    main()
