"""E-S runner: train one (task, arm, width, seed) on SHORT walks of the state-keyed retrieval task and measure how far past the trained walk length it still works.

Training: a walk of L group-element tokens (curriculum 0, 1, 2, 3, 4, 6, 8, 12, 16; a stage advances when the VALUE accuracy of the last `window` updates reaches 0.97),
up to four queries per sequence; two heads over one readout (state: first G logits, value: last V logits), loss = the sum of the two cross-entropies; the learning rate
is annealed (decayed to ~0 over the last 30 % of the updates: E-P) for EVERY arm. Evaluation: fresh dev sequences with a query after every 4th walk token for walks of
8, 16, 64 and 512 (the trained maximum is 16); value accuracy (at the queries) and state accuracy per bucket of cumulative walk tokens.

  python -B -m horizon.sk_run --task sk3 --arm core --width 64 --seed 1 --steps 15000 --out results/ES-pilot/sk3-core-w64-s1.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from . import depth_run as dr
from . import depth_tasks as dt
from . import sk_task as sk
from .depth_models import ARMS, make_arm, parameter_count
from .guards import GuardCounter, guarded_step

NAMESPACE = dict(train='sk-train', dev='sk-dev', test='sk-test')
CURRICULUM = (0, 1, 2, 3, 4, 6, 8, 12, 16)
EVAL_WALKS = (8, 16, 64, 512)
IN_RANGE_WALK = 16
EXT_BUCKETS = ('32-63', '64-127')       # cumulative walk tokens before the query: 2-8x the trained maximum of 16


def _log(message):
    print(message, flush=True)


def _batch(task, n, seed, step, walk, batch):
    s = dt.source_seed(NAMESPACE, task, seed, 'train', step=step, length=walk)
    return sk.generate(n, batch, walk, s, queries=max(1, min(4, walk)))


def heads(logits, g):
    """(state logits, value logits) of the shared readout: the first G and the last V classes."""
    return logits[..., :g], logits[..., g:]


def loss_and_accuracy(logits, b, g):
    state_logits, value_logits = heads(logits, g)
    loss = (F.cross_entropy(state_logits.flatten(0, 1), b['state'].flatten(), ignore_index=sk.IGNORE)
            + F.cross_entropy(value_logits.flatten(0, 1), b['value'].flatten(), ignore_index=sk.IGNORE))
    mask = b['value'] != sk.IGNORE
    value_acc = float((value_logits.detach().argmax(-1)[mask] == b['value'][mask]).float().mean())
    return loss, value_acc


def train(task, arm, width, seed, steps, batch=64, lr=3e-3, clip=1.0, weight_decay=0.01, curriculum=CURRICULUM, advance_at=0.97, min_stage_steps=200, window=100,
          lr_decay_fraction=0.3, lr_floor=0.0, log_every=0, log=_log):
    n = dt.TASKS[task]['n']
    g = sk.size(n)
    model = make_arm(arm, task, width, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    guard, history, start = GuardCounter(), [], time.perf_counter()
    stage, in_stage, recent, advances, lr_last = 0, 0, [], [], lr
    for step in range(1, steps + 1):
        if lr_decay_fraction > 0:
            lr_last = lr * (lr_floor + (1.0 - lr_floor) * min(1.0, (steps - step + 1) / (lr_decay_fraction * steps)))
            for group in opt.param_groups:
                group['lr'] = lr_last
        walk = curriculum[stage]
        b = _batch(task, n, seed, step, walk, batch)
        model.train()
        loss, value_acc = loss_and_accuracy(model(b['x']), b, g)
        report = guard.record(guarded_step(opt, model.parameters(), loss, clip_norm=clip))
        recent.append(value_acc)
        recent = recent[-window:]
        in_stage += 1
        if stage < len(curriculum) - 1 and in_stage >= min_stage_steps and len(recent) == window and sum(recent) / window >= advance_at:
            advances.append(dict(step=step, from_walk=curriculum[stage], to_walk=curriculum[stage + 1]))
            log(f'{task} {arm} w{width} s{seed} step {step} curriculum advance walk {curriculum[stage]} -> {curriculum[stage + 1]} ({time.perf_counter() - start:.0f}s)')
            stage, in_stage, recent = stage + 1, 0, []
        if log_every and step % log_every == 0:
            history.append(dict(step=step, loss=report['loss'], grad_norm=report['grad_norm_before_clip'], walk=curriculum[stage],
                                recent_value_accuracy=(sum(recent) / len(recent)) if recent else None))
            log(f'{task} {arm} w{width} s{seed} step {step} walk {curriculum[stage]} loss {report["loss"]} value acc {history[-1]["recent_value_accuracy"]} '
                f'({time.perf_counter() - start:.0f}s)')
    return model, dict(steps=steps, batch=batch, lr=lr, clip=clip, guard=guard.summary(), history=history, curriculum=list(curriculum), advances=advances,
                       final_stage_walk=curriculum[stage], train_seconds=time.perf_counter() - start, parameters=parameter_count(model), lr_decay_fraction=lr_decay_fraction,
                       lr_floor=lr_floor, lr_last=lr_last)


@torch.no_grad()
def evaluate(model, task, walks=EVAL_WALKS, n=100, seed=1, split='dev', every=4, batch=50):
    model.eval()
    gn = dt.TASKS[task]['n']
    g = sk.size(gn)
    out = {}
    for walk in walks:
        b = sk.generate(gn, n, walk, dt.source_seed(NAMESPACE, task, seed, split, length=walk), every=every)
        state_pred, value_pred = [], []
        for at in range(0, n, batch):
            s_logits, v_logits = heads(model(b['x'][at:at + batch]), g)
            state_pred.append(s_logits.argmax(-1))
            value_pred.append(v_logits.argmax(-1))
        values = sk.value_accuracy(torch.cat(value_pred), b['value'], b['walk'])
        states = sk.state_accuracy(torch.cat(state_pred), b['state'], b['walk'])
        correct, queries = values['all']
        out[walk] = dict(length=int(b['x'].shape[1]), queries=queries, accuracy=correct / queries, buckets=values, state_accuracy=states['all'][0] / states['all'][1],
                         state_buckets=states)
    return out


def _bucket_rate(row, key, buckets):
    cells = [row[key].get(b, [0, 0]) for b in buckets]
    total = sum(c[1] for c in cells)
    return (sum(c[0] for c in cells) / total if total else None), total


def summary(curve):
    """In-range value accuracy (walk 16: queries after 4, 8, 12 and 16 walk tokens), the extrapolation value accuracy (queries 32-127 walk tokens in, from the longest
    evaluated walk) and the same two for the state head."""
    inside = curve.get(IN_RANGE_WALK)
    longest = curve[max(curve)]
    ext, queries = _bucket_rate(longest, 'buckets', EXT_BUCKETS)
    state_ext, _ = _bucket_rate(longest, 'state_buckets', EXT_BUCKETS)
    return dict(in_range=(inside['accuracy'] if inside else None), state_in_range=(inside['state_accuracy'] if inside else None), extrapolation=ext,
                state_extrapolation=state_ext, extrapolation_queries=queries)


LEARNED = 0.95          # in-range value accuracy that counts as learned
ADVANTAGE_MARGIN = 0.15
MIN_EXTRAPOLATION = 0.6
NO_ADVANTAGE_MARGIN = 0.05


def _final_stage(r):
    """Where a run stopped in the walk curriculum (descriptive, NOT part of the verdict): the walk of the stage it ended in, the update at which it reached the last stage
    (None when it never did) and its budget. An arm that was 'not learned' while stuck at an early walk for the whole run fails differently from one that reached the last
    stage late, with the learning rate already decaying (the annealing starts at 70 % of the updates): the second may just have been short of budget."""
    t = r.get('train') or {}
    curriculum = t.get('curriculum') or []
    reached = bool(curriculum) and t.get('final_stage_walk') == curriculum[-1]
    advances = t.get('advances') or []
    return dict(stopped_at_walk=t.get('final_stage_walk'), reached_last_stage_at=((advances[-1]['step'] if advances else 0) if reached else None), steps=t.get('steps'))


BUDGET_LATE = 0.5           # a rival that reached the last walk stage after this fraction of its updates ...
BUDGET_PROGRESS_WALK = 4    # ... or never reached it although it had passed this walk, may simply have run out of budget (the lr decay starts at 70 % of the updates)


def _budget_limited(r):
    """True when a run's failure to learn the range can be a lack of budget (blind review R2-5): it reached the last stage late, or it was still climbing (past walk 4) at the end."""
    f = _final_stage(r)
    if f['reached_last_stage_at'] is not None:
        return f['steps'] is not None and f['reached_last_stage_at'] > BUDGET_LATE * f['steps']
    return f['stopped_at_walk'] is not None and f['stopped_at_walk'] >= BUDGET_PROGRESS_WALK


def reading(results, min_seeds=2):
    """The E-S reading rule (program note 5.16, written before the first run; refined after the second blind review, note 5.25, before E-S ran), applied to run results of ONE task:
    several arms, several seeds each. Per arm the mean in-range and extrapolation value accuracy over its seeds. An arm is LEARNED when its mean in-range accuracy is >= 0.95.
    Verdicts, for the Core arm 'core':
      NO CORE RUN / NO RIVAL RUNS   no 'core' result / no other arm at all (nothing to compare);
      CORE NOT LEARNED   mean in-range < 0.95: nothing can be said about an advantage (budget / trainability);
      INCOMPLETE         a rival learned the range but has no extrapolation value (a missing evaluation): the comparison cannot be made;
      ADVANTAGE          the Core is learned, its extrapolation is >= 0.6 AND >= the best OTHER learned arm's extrapolation + 0.15 (when no other arm learned the range the
                         advantage is reported as 'LEARNABILITY' instead: the others fail inside the trained range);
      NO ADVANTAGE       some other learned arm's extrapolation is >= the Core's - 0.05 (the boundary counts);
      MIXED              otherwise.
    A verdict is marked provisional when an arm has fewer than `min_seeds` seeds, and an ADVANTAGE verdict carries '(RIVAL BUDGET-LIMITED ...)' when some rival that did not learn the
    range may simply have run out of budget (`_budget_limited`): it is then not recorded as an advantage without a doubled-budget run. A seed that appears twice for one arm is an error."""
    arms = {}
    for r in results:
        arms.setdefault(r['arm'], []).append(r)
    table = {}
    for arm, rs in arms.items():
        seeds = [r.get('seed') for r in rs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'arm {arm}: the same seed appears twice {seeds}')
        in_range = sum(r['in_range'] for r in rs) / len(rs)
        ext = [r['extrapolation'] for r in rs if r.get('extrapolation') is not None]
        state_ext = [r['state_extrapolation'] for r in rs if r.get('state_extrapolation') is not None]
        learned = in_range >= LEARNED - 1e-6
        table[arm] = dict(seeds=len(rs), in_range=in_range, extrapolation=(sum(ext) / len(ext) if ext else None), learned=learned,
                          state_extrapolation=(sum(state_ext) / len(state_ext) if state_ext else None), final_stage=[_final_stage(r) for r in rs],
                          budget_limited=(not learned) and any(_budget_limited(r) for r in rs))
    core = table.get('core')
    if core is None:
        return dict(verdict='NO CORE RUN', arms=table)
    rivals = {a: t for a, t in table.items() if a != 'core'}
    if not rivals:
        return dict(verdict='NO RIVAL RUNS', arms=table)
    unfinished = sorted(a for a, t in rivals.items() if t['learned'] and t['extrapolation'] is None)
    others = {a: t for a, t in rivals.items() if t['learned'] and t['extrapolation'] is not None}
    best_other = max((t['extrapolation'] for t in others.values()), default=None)
    if not core['learned']:
        verdict = 'CORE NOT LEARNED'
    elif unfinished:
        verdict = 'INCOMPLETE (no extrapolation value for the learned rival(s) %s)' % ', '.join(unfinished)
    elif core['extrapolation'] is None:
        verdict = 'MIXED'
    elif best_other is None:
        verdict = 'ADVANTAGE (LEARNABILITY)' if core['extrapolation'] >= MIN_EXTRAPOLATION else 'MIXED'
    elif core['extrapolation'] >= MIN_EXTRAPOLATION and core['extrapolation'] >= best_other + ADVANTAGE_MARGIN - 1e-9:
        verdict = 'ADVANTAGE'
    elif best_other >= core['extrapolation'] - NO_ADVANTAGE_MARGIN - 1e-9:
        verdict = 'NO ADVANTAGE'
    else:
        verdict = 'MIXED'
    limited = sorted(a for a, t in rivals.items() if t['budget_limited'])
    if verdict.startswith('ADVANTAGE') and limited:
        verdict += ' (RIVAL BUDGET-LIMITED: %s reached the last stage late or was still climbing; not recorded as an advantage without a doubled-budget run)' % ', '.join(limited)
    provisional = any(t['seeds'] < min_seeds for t in table.values())
    return dict(verdict=verdict + (' (provisional: fewer than %d seeds for some arm)' % min_seeds if provisional else ''), arms=table, best_other_extrapolation=best_other)


def run(task, arm, width, seed, steps, walks=EVAL_WALKS, n_eval=100, out=None, **kw):
    started = time.perf_counter()
    model, trained = train(task, arm, width, seed, steps, **kw)
    if out:        # before the slow evaluation: a crash there must not lose the trained model
        dr.save_model(model, Path(out).with_suffix('.pt'), task, arm, width, seed, trained)
    curve = evaluate(model, task, walks, n_eval, seed)
    result = dict(task=task, arm=arm, width=width, seed=seed, train=trained, curve={str(k): v for k, v in curve.items()}, wall_seconds=time.perf_counter() - started,
                  **summary(curve))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=1), encoding='utf8')
    return result


def main():
    torch.set_num_threads(1)
    if len(sys.argv) > 2 and sys.argv[1] == '--reading':      # python -B -m horizon.sk_run --reading results/ES-pilot/sk3-*.json
        print(json.dumps(reading([json.loads(Path(f).read_text(encoding='utf8')) for f in sys.argv[2:]]), indent=1))
        return
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=('sk3', 'sk4'))
    p.add_argument('--arm', required=True, choices=ARMS)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--lr-decay-fraction', type=float, default=0.3)
    p.add_argument('--walks', default=','.join(str(w) for w in EVAL_WALKS))
    p.add_argument('--n-eval', type=int, default=100)
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r = run(a.task, a.arm, a.width, a.seed, a.steps, walks=tuple(int(v) for v in a.walks.split(',')), n_eval=a.n_eval, out=a.out, batch=a.batch, lr=a.lr,
            lr_decay_fraction=a.lr_decay_fraction, log_every=a.log_every)
    print(json.dumps({k: r[k] for k in ('task', 'arm', 'width', 'seed', 'in_range', 'state_in_range', 'extrapolation', 'state_extrapolation', 'extrapolation_queries',
                                       'wall_seconds')}))
    for walk, row in r['curve'].items():
        print(walk, 'length', row['length'], 'queries', row['queries'], 'value accuracy', round(row['accuracy'], 3), 'state accuracy', round(row['state_accuracy'], 3),
              {k: round(v[0] / v[1], 3) for k, v in row['buckets'].items() if k != 'all' and v[1]})


if __name__ == '__main__':
    main()
