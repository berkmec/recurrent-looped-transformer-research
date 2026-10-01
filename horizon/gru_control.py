"""Z1-G: the sealed U1 / U2c neutral-run protocol applied to a plain GRU.

The campaign's first positive architecture result (a learned HOLD/WRITE gate that keeps the Core's state through neutral runs) had no
standard-recurrence baseline. A GRU already has an update gate. This module trains a GRU under EXACTLY the U1 protocol (same numbers, read
from horizon/sweep.py, same source seeds so the training batches and test problems are the ones the Core arms saw) and evaluates it the way
U1 / U2c did: clean lengths 128..4096 and neutral insertions (128 / 512 slots at the prefix / middle / suffix of a length-128 problem).

  python -B -m horizon.gru_control --out results/Z1-G --lane 0 --lanes 2
"""
import argparse
import json
import math
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


# The U1 numbers, pinned literally. sweep.protocol() is still READ (nothing is copied) but is checked against these, so a change in the sealed
# file or an accidental edit cannot pass silently (independent review 001, finding B4).
SEALED = dict(steps=2048, batch=32, train_length=128, lr=.003, weight_decay=.01, clip=1., validation_every=128, validation_n=256, test_n=1000,
              primary_seeds=[117, 143, 191], source_seeds=dict(train=91010000, validation=92010000, test=93010000, neutral=93020000))


class NonFiniteError(ValueError):
    pass


def protocol():
    p = sw.protocol()
    for key, value in SEALED.items():
        if p[key] != value:
            raise ValueError(f'the sealed U1 protocol changed: {key} is {p[key]!r}, expected {value!r}')
    return p


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
    if steps < 1:
        raise ValueError('steps must be at least 1')
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
        if not report['applied']:        # U1 aborts on a non-finite loss or gradient and keeps the evidence; so does this control (review 001, B3)
            raise NonFiniteError(f'nonfinite loss/gradient at step {step}; failure preserved')
        if step % validation_every == 0 or step == steps:
            stats, _ = predict(model, vx, vy)
            history.append(dict(step=step, loss=report['loss'], **stats))
            # strictly lower: the earliest step wins ties; a NaN/inf candidate is never selected and never blocks a finite one (review 001, B9)
            if math.isfinite(stats['final_loss']) and (best is None or stats['final_loss'] < best):
                best = stats['final_loss']
                selected = dict(step=step, validation=stats, state={k: v.detach().clone() for k, v in model.state_dict().items()})
    if selected is None:
        raise ValueError('no finite validation loss: nothing to select')
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
    try:
        model, trained = train(c, p, steps=steps, validation_every=validation_every, validation_n=validation_n)
    except NonFiniteError as e:           # like U1: the failure is preserved, no result is written for this condition
        (out / f'{ident(c)}-failure.json').write_text(json.dumps(dict(status='FAIL_NONFINITE', condition=c, message=str(e)), indent=1), encoding='utf8')
        raise
    torch.save(dict(state_dict=model.state_dict(), meta=dict(task=c['task'], arm=c['arm'], width=int(c['width']), seed=int(c['seed']),
                                                            selected_step=int(trained['selected_step']), parameters=int(parameter_count(model)))),
               out / f'{ident(c)}.pt')                                    # before the slow evaluation
    row = evaluate(model, c, p, n=n, clean_lengths=clean_lengths, neutral_slots=neutral_slots, neutral_placements=neutral_placements, train_length=train_length)
    result = dict(condition=c, ident=ident(c), parameters=parameter_count(model), train=trained, wall_seconds=time.perf_counter() - started,
                  protocol=dict(steps=p['steps'], batch=p['batch'], train_length=p['train_length'], lr=p['lr'], source='horizon/sweep.py protocol() (U1)'), **row)
    path.write_text(json.dumps(result, indent=1), encoding='utf8')
    return result


CELLS = ('128-middle', '128-suffix', '512-middle', '512-suffix')    # the pre-registered cells (results/Z1-G/README.txt); prefix cells are reported only
HOLD_POINTS, DRIFT_POINTS = 2.0, 10.0


CELLS_ALL = ('128-prefix', '128-middle', '128-suffix', '512-prefix', '512-middle', '512-suffix')


def losses_from_gru(row):
    """The stored loss_pp, but only if it agrees with the accuracies it was computed from (review 001, B8: a read-back alone cannot catch a
    row whose two numbers contradict each other)."""
    base = row['neutral']['0-base']['final_accuracy']
    out = {}
    for cell in CELLS_ALL:
        stored = row['neutral'][cell]['loss_pp']
        derived = 100 * (base - row['neutral'][cell]['final_accuracy'])
        if abs(stored - derived) > 1e-6:
            raise ValueError(f'inconsistent row: {cell} stores loss_pp {stored} but its accuracies imply {derived}')
        out[cell] = stored
    return out


def losses_from_u2c(row):
    """U2c (and the U2c-v2 / lr variants) evaluation.json: neutral rows keyed '<slots>-<placement>' plus '0-base'."""
    base = row['neutral']['0-base']['final_accuracy']
    return {cell: 100 * (base - row['neutral'][cell]['final_accuracy']) for cell in CELLS_ALL}


def losses_from_u1(row):
    """U1 evaluation.json: neutral is a LIST of entries with a placement and consumed_tokens; slots = consumed_tokens minus the base length."""
    base = next(e for e in row['neutral'] if e['placement'] == 'base')
    out = {}
    for e in row['neutral']:
        if e['placement'] != 'base':
            out[f"{e['consumed_tokens'] - base['consumed_tokens']}-{e['placement']}"] = 100 * (base['final_accuracy'] - e['final_accuracy'])
    return {cell: out[cell] for cell in CELLS_ALL}


def mean_over_seeds(per_seed):
    return {cell: dict(mean=sum(s[cell] for s in per_seed) / len(per_seed), per_seed=[s[cell] for s in per_seed]) for cell in CELLS_ALL}


def comparison(results='results', task='five'):
    """Neutral-insertion loss (points of final accuracy, base minus inserted) per cell, mean over the three U1 seeds, computed from the raw
    per-condition files of every arm that was run under the U1 protocol: the Core without a gate (U1), the three gated Core arms (U2c-v2) and
    the plain GRU (Z1-G). Missing files are an error, not a silent skip."""
    seeds = protocol()['primary_seeds']
    root = Path(results)
    sources = {
        'Core, no gate (U1)': ('U1/u1-%s-w32-s{seed}/evaluation.json' % task, losses_from_u1),
        'Core + soft gate (U2c-v2)': ('U2c-v2/u2c-soft-%s-w32-s{seed}/evaluation.json' % task, losses_from_u2c),
        'Core + hard_st gate (U2c-v2)': ('U2c-v2/u2c-hard_st-%s-w32-s{seed}/evaluation.json' % task, losses_from_u2c),
        'Core + supervised gate (U2c-v2)': ('U2c-v2/u2c-supervised-%s-w32-s{seed}/evaluation.json' % task, losses_from_u2c),
        'GRU w32 (Z1-G)': ('Z1-G/z1g-gru32-%s-s{seed}.json' % task, losses_from_gru),
        'GRU w64 (Z1-G)': ('Z1-G/z1g-gru64-%s-s{seed}.json' % task, losses_from_gru),
    }
    table = {}
    for arm, (pattern, reader) in sources.items():
        per_seed = [reader(json.loads((root / pattern.format(seed=s)).read_text(encoding='utf8'))) for s in seeds]
        table[arm] = mean_over_seeds(per_seed)
    return table


def load_results(out):
    return [json.loads(path.read_text(encoding='utf8')) for path in sorted(Path(out).glob('z1g-*.json'))]


def verdict(rows):
    """The pre-registered reading, task 'five' only, from results rows:
    R1 (GRU HOLDS): loss <= 2 points in all four cells, in >= 2 of 3 seeds, at BOTH widths.
    R2 (GRU DRIFTS): loss >= 10 points in >= 2 of the four cells, in >= 2 of 3 seeds, at width 64.
    R3: anything else. INCOMPLETE until all six 'five' rows (2 widths x 3 seeds) exist."""
    five = [r for r in rows if r['condition']['task'] == 'five']
    detail = []
    for r in five:
        losses = {cell: r['neutral'][cell]['loss_pp'] for cell in CELLS}
        detail.append(dict(width=r['condition']['width'], seed=r['condition']['seed'], losses=losses,
                           holds=all(v <= HOLD_POINTS for v in losses.values()), drifts=sum(v >= DRIFT_POINTS for v in losses.values()) >= 2))
    out = dict(five_rows=len(five), detail=detail)
    if len(five) < len(WIDTHS) * len(protocol()['primary_seeds']):
        out['verdict'] = 'INCOMPLETE'
        return out
    by_width = {w: [d for d in detail if d['width'] == w] for w in WIDTHS}
    r1 = all(sum(d['holds'] for d in by_width[w]) >= 2 for w in WIDTHS)
    r2 = sum(d['drifts'] for d in by_width[64]) >= 2
    out['verdict'] = 'R1' if r1 else 'R2' if r2 else 'R3'
    return out


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/Z1-G')
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--lanes', type=int, default=1)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--verdict', action='store_true', help='print the pre-registered reading from the finished results and exit')
    parser.add_argument('--compare', action='store_true', help='print the neutral-loss comparison against the Core arms and write comparison-<task>.json')
    parser.add_argument('--task', default='five', choices=['five', 'stack'], help='task for --compare')
    args = parser.parse_args()
    cs = conditions()
    if args.compare:
        table = comparison('results', task=args.task)
        print('%-34s' % 'arm (mean loss, points; per seed)', *['%-22s' % c for c in CELLS_ALL])
        for arm, cells in table.items():
            print('%-34s' % arm, *['%-22s' % ('%5.1f (%s)' % (cells[c]['mean'], '/'.join('%.0f' % v for v in cells[c]['per_seed']))) for c in CELLS_ALL])
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f'comparison-{args.task}.json').write_text(json.dumps(table, indent=1), encoding='utf8')
        return
    if args.list:
        for c in cs:
            print(ident(c))
        return
    if args.verdict:
        rows = load_results(args.out)
        v = verdict(rows)
        for d in v['detail']:
            print('five', 'w%d' % d['width'], 's%d' % d['seed'], {k: round(x, 1) for k, x in d['losses'].items()}, 'holds' if d['holds'] else '', 'drifts' if d['drifts'] else '')
        print('verdict:', v['verdict'], '(five rows: %d of %d)' % (v['five_rows'], len(WIDTHS) * len(protocol()['primary_seeds'])))
        return
    for c in cs[args.lane::args.lanes]:
        r = run_one(args.out, c)
        base = r['neutral']['0-base']['final_accuracy']
        print(ident(c), 'params', r['parameters'], 'selected', r['train']['selected_step'], 'base', round(base, 3),
              {k: round(v['final_accuracy'], 3) for k, v in r['neutral'].items() if k != '0-base'},
              {k: round(v['final_accuracy'], 3) for k, v in r['clean'].items()}, flush=True)


if __name__ == '__main__':
    main()
