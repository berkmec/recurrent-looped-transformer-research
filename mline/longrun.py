"""The long run: one from-scratch training that is not stopped for a plateau.

Phase transitions in this project arrive late and sharply - a small model sat at chance for 2000 updates and was at 0.96 by 2500; the 30M
fine-tunes moved between 500 and 1000 - so a run judged at checkpoints would be stopped exactly where it was about to change. This runner

  * never stops for a flat loss (it stops only for a non-finite loss that repeats: a marker file, not a restart loop),
  * records EVERY update the numbers a transition shows up in first: the loss on the answer digits of the synthetic rows, apart from
    the loss on the language rows, and the first-digit accuracy on the answers,
  * keeps a model snapshot every N updates, so a change can be dated and studied afterwards,
  * resumes as the same run after an interruption (batch, learning rate and weights are functions of the step alone),
  * follows warmup / stable / decay-over-the-last-fraction, so the schedule itself neither produces nor hides a change in the middle,
  * runs a battery every K updates (validation bits per byte, key matching on training letters / held-out letters / real words,
    single-fact distance, key-conditional attention) on the model as it is.

Usage: python -B -m mline.longrun --out results/L1-w928 --recipe m2-draft
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from horizon import guards as gu
from horizon import sweep as sw
from horizon import telemetry_std1 as ts
from mline import checkpoint as ck
from mline import decoy_train as dt
from mline import lm as ml

ROLE = 'the long run: one from-scratch training, not stopped for a plateau'
NATURAL_KEY_BYTES = 24                    # the longest natural key the generator draws, for the row-fit check


def lr_at(step, total, base, warmup=500, decay_fraction=.1, floor=.1):
    """Linear warmup, then flat, then a linear decay to floor * base over the last `decay_fraction` of the run."""
    if warmup > 0 and step <= warmup:
        return base * step / warmup
    if decay_fraction <= 0:
        return base
    start = total - int(total * decay_fraction)
    if step <= start:
        return base
    progress = (step - start) / max(1, total - start)
    return base * (1. - (1. - floor) * progress)


def allocate(n, shares):
    """n rows split in proportion to `shares` by largest remainder: deterministic, always sums to n."""
    total = float(sum(shares))
    exact = [n * s / total for s in shares]
    counts = [int(e) for e in exact]
    for i in sorted(range(len(shares)), key=lambda i: (-(exact[i] - counts[i]), i))[:n - sum(counts)]:
        counts[i] += 1
    return counts


def check_phases(phases):
    if not phases or phases[-1][0] is not None:
        raise ValueError('the last phase must be open (limit None)')
    closed = [limit for limit, _ in phases[:-1]]
    if any(limit is None for limit in closed) or any(a >= b for a, b in zip(closed, closed[1:])):
        raise ValueError('phase limits must be integers that increase')


def phase_at(phases, step):
    """(index, specs) of the phase in force at `step`: the first whose limit has not passed."""
    for i, (limit, specs) in enumerate(phases):
        if limit is None or step <= limit:
            return i, specs


def row_fits(spec, pad_to):
    """Whether the widest example this spec can draw fits one row (natural keys are counted at their longest)."""
    key = NATURAL_KEY_BYTES if spec.get('natural_share', 0) > 0 else max(spec['key_length'])
    widest = spec['facts'] * (23 + key) + max(spec['filler']) + spec['queries'] * (22 + key)
    return widest <= pad_to


def synthetic_batch(specs, n_rows, seed, step, pad_to, natural_keys=None):
    """(x, y, first_digit_mask, digit_mask) for n_rows synthetic rows drawn from `specs` in proportion to their shares.

    The masks sit on the TARGET positions of the answers: the first digit alone, and all four digits."""
    if n_rows == 0:
        empty = torch.zeros(0, pad_to, dtype=torch.long)
        return empty, empty.clone(), empty.bool(), empty.bool()
    xs, ys, firsts, digits = [], [], [], []
    for i, (spec, count) in enumerate(zip(specs, allocate(n_rows, [s['share'] for s in specs]))):
        if count == 0:
            continue
        natural = spec.get('natural_share', 0.0)
        if natural > 0 and not natural_keys:
            raise ValueError('spec %r asks for natural keys but none were given' % spec.get('name'))
        items = dt.decoy_items(count, tuple(spec['filler']), seed * 100_003 + step * 31 + i, spec.get('filler_kind', 'repeat'), None,
                               pad_to, True, spec['facts'], spec['queries'], tuple(spec['key_length']),
                               dt.ALPHABETS[spec['key_alphabet']], spec.get('with_replacement', True),
                               natural_keys if natural > 0 else None, natural if natural > 0 else 1.0)
        x, y = items['x'], items['y']
        first = torch.zeros_like(x, dtype=torch.bool)
        four = torch.zeros_like(first)
        for row, starts in enumerate(items['answer_starts']):
            for position, _question in starts:
                first[row, position - 1] = True
                four[row, position - 1:position + 3] = True
        xs.append(x), ys.append(y), firsts.append(first), digits.append(four)
    return torch.cat(xs), torch.cat(ys), torch.cat(firsts), torch.cat(digits)


# The draft recipe for the from-scratch run. It is a DRAFT: it is fixed when D6 has shown how real keys are best mixed in.
def _spec(name, share, facts, queries, filler, key_alphabet='broad', key_length=(4, 4), natural_share=0.0):
    return dict(name=name, share=share, facts=facts, queries=queries, filler=filler, key_length=key_length, key_alphabet=key_alphabet,
                natural_share=natural_share, with_replacement=True)


RECIPES = {
    'm2-draft': [
        # the position-locked single-fact start sped key matching up in the small Core (T1b vs T1a): early, a lot of it
        (2000, [_spec('single-fact', .5, 1, 1, (0, 64)), _spec('mqar4-random', .5, 4, 4, (0, 64))]),
        (8000, [_spec('single-fact-far', .2, 1, 1, (0, 384)), _spec('mqar4-random', .4, 4, 4, (0, 64), key_length=(1, 4)),
                _spec('mqar4-natural', .4, 4, 4, (0, 64), natural_share=1.0)]),
        (None, [_spec('single-fact-far', .15, 1, 1, (0, 448)), _spec('mqar2-far', .15, 2, 2, (0, 300)),
                _spec('mqar4-random', .3, 4, 4, (0, 128), key_length=(1, 4)),
                _spec('mqar4-natural', .4, 4, 4, (0, 128), natural_share=1.0)]),
    ],
}


def check_recipe(phases, pad_to):
    check_phases(phases)
    for _, specs in phases:
        for spec in specs:
            if not row_fits(spec, pad_to):
                raise ValueError('spec %r can draw an example wider than the row (%d bytes)' % (spec['name'], pad_to))


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def train(out, *, width, heads, steps, batch, length, lr, seed, needle_rate, phases, lm_batch, natural_keys=None, warmup=500,
          decay_fraction=.1, floor=.1, weight_decay=.01, clip=1., ckpt_every=250, snapshot_every=1000, eval_every=1000,
          eval_fn=None, device='cuda', arm='plain', window=4, alpha=.1, max_skips=50, verbose=True, telemetry='off', stop_at=None):
    """`stop_at` ends the run early ON THE FULL SCHEDULE (the learning rate still follows `steps`), leaving `stopped.json`
    instead of `complete.json`: a twin can retrace the first `stop_at` updates of a longer run exactly (A0 twin of L1)."""
    check_phases(phases)
    if telemetry not in ('off', 'std1'):
        raise SystemExit("telemetry 'off' veya 'std1' olmali (I1a:63/65; sessiz eksik-log yasak)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    done = out / 'complete.json'
    if done.exists() and json.loads(done.read_text(encoding='utf8')).get('steps', 0) >= steps:
        return out                                          # finished already: starting it again is a no-op
    model = ml.build(arm, width=width, heads=heads, window=window, alpha=alpha, seed=seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    parameters = sum(p.numel() for p in model.parameters())
    sw.write_json(out / 'probe.json', dict(
        role=ROLE, arm=arm, width=width, heads=heads, steps=steps, batch=batch, length=length, lr=lr, warmup=warmup,
        decay_fraction=decay_fraction, floor=floor, seed=seed, needle_rate=needle_rate, parameters=parameters,
        tokens_planned=steps * batch * length, phases=[[limit, specs] for limit, specs in phases], utc=time.time(),
        note='bytes are tokens (vocabulary 256); the schedule depends on `steps`, so extending a run changes the decay'))
    resume = ck.load_latest(out)
    start = 1
    if resume is not None and resume['step'] is not None:
        model.load_state_dict(resume['model'])
        if resume['optimizer'] is not None:
            opt.load_state_dict(resume['optimizer'])
        start = int(resume['step']) + 1
        log = out / 'probe.jsonl'                           # keep the log to the checkpoint: a restart leaves no duplicate steps
        if log.exists():
            kept = [line for line in log.read_text(encoding='utf8').splitlines()
                    if line.strip() and json.loads(line)['step'] <= resume['step']]
            log.write_text((chr(10).join(kept) + chr(10)) if kept else '', encoding='utf8')
        if verbose:
            print('resuming from step %d' % resume['step'], flush=True)
    counter = gu.GuardCounter()
    skipped_in_a_row = 0
    n_synthetic = round(batch * needle_rate)
    n_lm = batch - n_synthetic
    t0 = time.perf_counter()
    last = steps if stop_at is None else min(steps, int(stop_at))
    for step in range(start, last + 1):
        phase, specs = phase_at(phases, step)
        lm_x, lm_y = lm_batch(step, n_lm) if n_lm else (torch.zeros(0, length, dtype=torch.long),) * 2
        sx, sy, first, four = synthetic_batch(specs, n_synthetic, seed, step, length, natural_keys)
        x = torch.cat([lm_x, sx]).to(device)
        y = torch.cat([lm_y, sy]).to(device)
        current_lr = lr_at(step, steps, lr, warmup, decay_fraction, floor)
        for group in opt.param_groups:
            group['lr'] = current_lr
        model.train()
        logits = ml.forward_logits(model, x)
        ce = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none').view(y.shape)
        telemetry_on = telemetry == 'std1'
        prev_capture = None
        if telemetry_on:
            prev_capture = [(n, p.detach().clone()) for n, p in model.named_parameters()]
        report = counter.record(gu.guarded_step(opt, model.parameters(), ce.mean(), clip_norm=clip))
        with torch.no_grad():
            rows = ce.detach()
            first_d, four_d = first.to(device), four.to(device)
            answers = int(first_d.sum())
            hit = (logits.detach()[n_lm:].argmax(-1) == y[n_lm:])
            row = dict(step=step, phase=phase, loss=_finite(report['loss']), grad_norm=_finite(report['grad_norm_before_clip']),
                       applied=bool(report['applied']), lr=current_lr, bytes_seen=step * batch * length,
                       loss_lm=_finite(rows[:n_lm].mean()) if n_lm else None,
                       loss_synthetic=_finite(rows[n_lm:].mean()) if n_synthetic else None,
                       answer_loss=_finite(rows[n_lm:][four_d].mean()) if bool(four_d.any()) else None,
                       first_digit_accuracy=_finite(hit[first_d].float().mean()) if answers else None, answers=answers)
            if telemetry_on:
                row['clipped'] = bool(report['clipped'])
                row['wall_seconds'] = round(time.perf_counter() - t0, 6)
                row['update_norm'] = float(torch.sqrt(sum((p_detach - q.detach()).square().sum()
                                                          for p_detach, q in zip([q for _, q in prev_capture], model.parameters()))))
                prev_initial = dict(prev_capture)['initial']
                if step == 1:
                    row['init_update_norm'] = float(torch.linalg.vector_norm(model.initial.detach() - prev_initial))
                    row['init_grad_norm'] = None if model.initial.grad is None else float(torch.linalg.vector_norm(model.initial.grad))
        if not report['applied']:
            skipped_in_a_row += 1
            if skipped_in_a_row > max_skips:
                (out / 'DIVERGED.txt').write_text('%d updates in a row were skipped (non-finite loss or gradient), last at step %d: %s\n'
                                                  % (skipped_in_a_row, step, report.get('reason')), encoding='utf8')
                if telemetry_on:
                    train_seconds = time.perf_counter() - t0
                    block = ts.run_telemetry_block(status='STOPPED', guard_summary=counter.summary(),
                                                   cost={'train_seconds': train_seconds,
                                                         'sec_per_step': train_seconds / max(1, step - start + 1)},
                                                   reason='diverged: %d skipped in a row at step %d' % (skipped_in_a_row, step))
                    sw.append_line(out / 'telemetry_std1.jsonl', block)
                raise RuntimeError('the run diverged: %d skipped updates in a row at step %d' % (skipped_in_a_row, step))
        else:
            skipped_in_a_row = 0
        if eval_fn is not None and (step % eval_every == 0 or step == steps):
            try:
                with torch.no_grad():
                    row['eval'] = eval_fn(model, step)
            except Exception as error:      # the battery measures the run, it must not end it: a crash here resumed from the last checkpoint and failed at the same step again
                row['eval'] = dict(error='%s: %s' % (type(error).__name__, str(error)[:300]))
                print('BATTERY FAILED at step %d (the run goes on): %s' % (step, traceback.format_exc()), flush=True)
            model.train()
            if verbose:
                print('step %6d  %s' % (step, json.dumps(row['eval'], default=float)[:400]), flush=True)
        if snapshot_every and step % snapshot_every == 0:
            _atomic_save(out / 'snapshots' / ('step%06d.pt' % step), {k: v.detach().cpu() for k, v in model.state_dict().items()})
        if step % ckpt_every == 0 or step == last:
            ck.save(out / ck.LATEST, model, opt, step=step, meta=dict(seed=seed, width=width, tokens=step * batch * length))
        sw.append_line(out / 'probe.jsonl', row)
        if verbose and step % 100 == 0:
            print('step %6d  loss %.3f  answer_loss %s  first-digit %s  lr %.2e  %.1f min'
                  % (step, row['loss'] or float('nan'), row['answer_loss'], row['first_digit_accuracy'], current_lr,
                     (time.perf_counter() - t0) / 60), flush=True)
    if last < steps:
        if telemetry == 'std1':
            train_seconds = time.perf_counter() - t0
            sw.append_line(out / 'telemetry_std1.jsonl', ts.run_telemetry_block(
                status='STOPPED', guard_summary=counter.summary(),
                cost={'train_seconds': train_seconds, 'sec_per_step': train_seconds / max(1, last - start + 1)},
                reason='stop_at %d of %d planned updates' % (last, steps)))
        (out / 'stopped.json').write_text(json.dumps(dict(stopped_at=last, steps_planned=steps, tokens=last * batch * length,
                                                          wall_seconds=time.perf_counter() - t0, utc=time.time()), indent=1), encoding='utf8')
        return out
    if telemetry_on:
        train_seconds = time.perf_counter() - t0
        block = ts.run_telemetry_block(status='COMPLETE', guard_summary=counter.summary(),
                                       cost={'train_seconds': train_seconds,
                                             'sec_per_step': train_seconds / max(1, steps - start + 1)})
        sw.append_line(out / 'telemetry_std1.jsonl', block)
    done.write_text(json.dumps(dict(steps=steps, tokens=steps * batch * length, wall_seconds=time.perf_counter() - t0,
                                    utc=time.time()), indent=1), encoding='utf8')
    return out


def make_eval_fn(protocol, alphabet, natural_val_keys, device='cuda', items=32, length=512, filler_hi=128, validate_fn=None,
                 distances=(0, 64, 128, 256, 384, 448)):
    """The battery, run on the model as it is: validation bits per byte, key matching on the training letters / held-out T-Z /
    real validation words (the training-shaped score, questions drawn with replacement), single-fact distance, and the mean and the
    best key-conditionality over the attention heads."""
    from mline import attn_probe as ap
    from mline import mqar_eval as me
    from mline import train as mt
    validate = validate_fn or (lambda model: mt.validate(model, protocol, device, batches=8))

    def fn(model, step):
        out = dict(bits_per_byte=_finite(validate(model)['bits_per_byte']))
        out.update(me.extra_eval(alphabet, 4, 4, (0, filler_hi), length, n_items=items, key_lengths=(4,), with_replacement=True,
                                 device=device)(model))
        if natural_val_keys:
            r = me.mqar_accuracy(model, None, (4, 4), 4, 4, (0, filler_hi), length, items, device=device, with_replacement=True,
                                 key_source=natural_val_keys)
            out.update(natural_first=float(r['first_all']), natural_exact=float(r['exact_all']))
        for n in distances:
            r = me.mqar_accuracy(model, alphabet, (4, 4), 1, 1, (n, n), length, items, device=device)
            out['single_fact_exact@%d' % n] = float(r['exact_all'])
        probe = ap.probe(model, [(0, 0), (15, 15)], alphabet='TRAIN_KEYS', key_length=4, items=16, device=device)
        cond = [c for c in ap.conditionality(probe) if c['mean'] is not None]
        if cond:
            out['attention_conditionality_mean'] = float(sum(c['mean'] for c in cond) / len(cond))
            out['attention_conditionality_best'] = float(max(c['best'] for c in cond))
        return out

    return fn


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--recipe', default='m2-draft', choices=tuple(RECIPES))
    parser.add_argument('--width', type=int, default=928)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--tokens', type=float, default=500e6, help='total training tokens (bytes); steps = tokens / (batch * length)')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--decay-fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=401)
    parser.add_argument('--needle-rate', type=float, default=.25, help='share of each batch that is synthetic rows')
    parser.add_argument('--ckpt-every', type=int, default=250)
    parser.add_argument('--snapshot-every', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--eval-items', type=int, default=32, help='rows per cell of the battery')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--arm', default='plain', choices=ml.ARMS, help="'a0' = the same Core with feedback off (the A0 twin)")
    parser.add_argument('--stop-at', type=int, default=None, help='stop after this update, keeping the full schedule (see train)')
    parser.add_argument('--telemetry', default='off', choices=('off', 'std1'))
    args = parser.parse_args()
    from mline import data as md
    from mline import mqar_eval as me
    from mline import train as mt
    protocol = mt.protocol()
    steps = int(round(args.tokens / (args.batch * args.length)))
    phases = RECIPES[args.recipe]
    check_recipe(phases, args.length)
    corpus = md.load(protocol['data'], 'train')
    natural_train, natural_val = me.natural_key_sets(corpus, md.load(protocol['data'], 'validation'), n_train=2000, n_val=300, seed=7)

    def lm_batch(step, n_rows):
        x, y, _ = md.batch(corpus, args.seed, step, n_rows, args.length)
        return torch.from_numpy(x), torch.from_numpy(y)

    eval_fn = make_eval_fn(dict(protocol, batch=args.batch, length=args.length), dt.ALPHABETS['broad'], natural_val, args.device,
                           items=args.eval_items, length=args.length)
    train(args.out, width=args.width, heads=args.heads, steps=steps, batch=args.batch, length=args.length, lr=args.lr, seed=args.seed,
          needle_rate=args.needle_rate, phases=phases, lm_batch=lm_batch, natural_keys=natural_train, warmup=args.warmup,
          decay_fraction=args.decay_fraction, ckpt_every=args.ckpt_every, snapshot_every=args.snapshot_every,
          eval_every=args.eval_every, eval_fn=eval_fn, device=args.device, arm=args.arm, stop_at=args.stop_at,
          telemetry=args.telemetry)


if __name__ == '__main__':
    main()
