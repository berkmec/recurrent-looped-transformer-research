"""A small Core trained from scratch on multi-query associative recall (MQAR) rows only.

Why: every 30M fine-tune so far starts from a model that learned the single-fact needle task, whose cheapest solution is
positional (copy the nearest / first fact). The decoy fine-tunes (D1-D4) ask whether MQAR can be learned from there; they
cannot say whether the ARCHITECTURE can learn key-addressed retrieval at all, nor whether a single-fact start is what gets
in the way. This runner separates the two on CPU, in minutes-to-hours, beside the GPU queue:

    T1a  from scratch on MQAR                         -> can the Core learn key matching, and does it generalise to unseen letters?
    T1b  --init-from a single-fact (--facts 1) run    -> the same MQAR training from a position-locked start

The model is the sealed `core.model.Core` at a chosen width (nothing in core/ is edited), built exactly like `lm.build`'s
plain arm. `--feedback off` (the default) takes the Core's parallel path, which is what makes CPU training fast; the
retrieval path (encoder self-attention + decoder cross-attention) is the same either way, and the feedback was shown to
carry the template phase, not the fact.

Scored with `mline.decoy_probe.probe_model`, the probe the 30M runs were scored with: for each (alphabet, key length) the
share of answers that follow the ASKED key, when the first fact is asked and when the decoy is asked.

Usage: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1a --steps 3000 --width 128 --heads 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from core.model import Core
from horizon import guards as gu
from horizon import sweep as sw
from mline import checkpoint as ck
from mline import decoy_probe as dp
from mline import decoy_train as dt
from mline import emergence as em
from mline import induction as ind
from mline import lm as ml

ROLE = 'development diagnostic: can the Core learn key-addressed retrieval from scratch, and does a single-fact start get in the way?'


def build_small(width, heads, feedback, seed, window=4, alpha=.1):
    """The sealed Core at `width`, with a 256-byte embedding - `lm.build`'s plain arm, minus the 30M width."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        core = Core(width=width, heads=heads, window=window, classes=ml.VOCAB, feedback=feedback, alpha=alpha)
        core.embedding = nn.Embedding(ml.VOCAB, width)
    return core


def draw(n, seed, step, facts, queries, key_length, alphabet, filler, pad_to):
    """The training batch for (seed, step): a pure function of both, so a resumed run is the same run."""
    items = dt.decoy_items(n, filler_range=filler, seed=seed * 100_003 + step, kind='repeat', pad_to=pad_to, pack=True,
                           n_facts=facts, n_queries=queries, key_length=key_length, alphabet=alphabet)
    return items['x'], items['y']


def lr_at(step, lr, warmup):
    return lr if warmup <= 0 else lr * min(1., step / warmup)


def summarize(rows):
    """{'ALPHABET/key_length': {'first': share, 'decoy': share}}: the asked-fact share, averaged over the filler layouts.

    'decoy' is the number a position-locked model cannot reach: the answer follows the key when the NEARER fact is the wrong one."""
    cells = {}
    for r in rows:
        if r['asked'] is None:
            continue
        cells.setdefault('%s/%d' % (r['alphabet'], r['key_length']), {}).setdefault(r['ask'], []).append(r['asked'])
    return {cell: {ask: sum(v) / len(v) for ask, v in asks.items()} for cell, asks in cells.items()}


def train(out, width, heads, feedback, steps, batch, lr, warmup, seed, facts, queries, key_length, alphabet, filler, pad_to,
          eval_every, eval_items, eval_configs, eval_alphabets, eval_key_lengths, init_from=None, clip=1.,
          weight_decay=.01, verbose=True, batch_fn=None, extra_eval=None, source='mqar'):
    """`batch_fn(step) -> (x, y)` replaces the MQAR batch (induction rows use it); `extra_eval(model) -> dict` adds a number
    to every eval row. Both default to nothing, which is the MQAR run T1a/T1b were made with."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_small(width, heads, feedback, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    header = dict(role=ROLE, width=width, heads=heads, feedback=feedback, steps=steps, batch=batch, lr=lr, warmup=warmup,
                  seed=seed, source=source, facts=facts, queries=queries, key_length=list(key_length),
                  key_alphabet=alphabet.decode('ascii'),
                  filler=list(filler), pad_to=pad_to, eval_every=eval_every, eval_items=eval_items,
                  parameters=sum(p.numel() for p in model.parameters()), utc=time.time(),
                  init_from=None if init_from is None else str(init_from),
                  init_sha256=None if init_from is None else sw.sha(init_from))
    sw.write_json(out / 'probe.json', header)
    resume = ck.load_latest(out)
    start = 1
    if init_from is not None and (resume is None or resume['step'] is None):
        em.load_init(model, init_from)
    if resume is not None and resume['step'] is not None:
        model.load_state_dict(resume['model'])
        if resume['optimizer'] is not None:
            opt.load_state_dict(resume['optimizer'])
        start = int(resume['step']) + 1
        log = out / 'probe.jsonl'                    # keep the log to the checkpoint, so a restart leaves no duplicate steps
        if log.exists():
            kept = [line for line in log.read_text(encoding='utf8').splitlines()
                    if line.strip() and json.loads(line)['step'] <= resume['step']]
            log.write_text((chr(10).join(kept) + chr(10)) if kept else '', encoding='utf8')
        if verbose:
            print('resuming from step %d' % resume['step'], flush=True)
    counter = gu.GuardCounter()
    t0 = time.perf_counter()
    for step in range(start, steps + 1):
        x, y = batch_fn(step) if batch_fn is not None else draw(batch, seed, step, facts, queries, key_length, alphabet, filler,
                                                                pad_to)
        for group in opt.param_groups:
            group['lr'] = lr_at(step, lr, warmup)
        model.train()
        loss = F.cross_entropy(ml.forward_logits(model, x).flatten(0, 1), y.flatten())
        report = counter.record(gu.guarded_step(opt, model.parameters(), loss, clip_norm=clip))
        row = dict(step=step, loss=report['loss'], grad_norm=report['grad_norm_before_clip'])
        if step % eval_every == 0 or step == steps:
            rows = dp.probe_model(model, eval_configs, alphabets=eval_alphabets, items=eval_items,
                                  batch=min(16, eval_items), key_lengths=eval_key_lengths, verbose=False)
            row['eval'] = dict(summary=summarize(rows), rows=rows)
            if extra_eval is not None:
                row['eval']['extra'] = extra_eval(model)
            ck.save(out / ck.LATEST, model, opt, step=step, meta=dict(seed=seed, width=width, facts=facts, queries=queries))
            if verbose:
                cells = '  '.join('%s %s' % (cell, '/'.join('%.2f' % v[a] for a in ('first', 'decoy') if a in v))
                                  for cell, v in row['eval']['summary'].items())
                extra = '' if extra_eval is None else '  ' + ' '.join('%s %.3f' % kv for kv in row['eval']['extra'].items())
                print('step %5d  loss %.3f  (first/decoy asked-share)  %s%s  %.1f min'
                      % (step, report['loss'] or float('nan'), cells, extra, (time.perf_counter() - t0) / 60), flush=True)
        sw.append_line(out / 'probe.jsonl', row)
    return out


def parse_span(text):
    """'N' -> (N, N); 'LOW-HIGH' -> (LOW, HIGH): a length range with 1 <= LOW <= HIGH."""
    parts = [int(v) for v in text.split('-')]
    if len(parts) == 1:
        parts = parts * 2
    if len(parts) != 2 or not 1 <= parts[0] <= parts[1]:
        raise ValueError('a span is N or LOW-HIGH with 1 <= LOW <= HIGH')
    return parts[0], parts[1]


def parse_configs(text):
    return [tuple(int(v) for v in part.split(':')) for part in text.split(',')]


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--feedback', choices=('off', 'on'), default='off')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--seed', type=int, default=301)
    parser.add_argument('--facts', type=int, default=4)
    parser.add_argument('--queries', type=int, default=4)
    parser.add_argument('--key-length', default='1-4')
    parser.add_argument('--key-alphabet', default='wide', choices=tuple(dt.ALPHABETS))
    parser.add_argument('--train-filler', default='0,64')
    parser.add_argument('--pad-to', type=int, default=320)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--eval-items', type=int, default=64)
    parser.add_argument('--eval-configs', default='0:0,10:0,0:30,30:0,15:15')
    parser.add_argument('--eval-alphabets', default='TRAIN_KEYS,HELD_KEYS')
    parser.add_argument('--eval-key-lengths', default='1,4')
    parser.add_argument('--init-from', default=None)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--source', choices=('mqar', 'induction'), default='mqar',
                        help="'induction': rows of a random string repeated later (mline.induction) instead of MQAR facts")
    parser.add_argument('--seg-length', default='8-32', help='induction: length range of the repeated string')
    parser.add_argument('--gap', default='0,96', help='induction: filler between the two copies')
    parser.add_argument('--lead', default='0,16', help='induction: filler before the first copy')
    parser.add_argument('--skip', type=int, default=2, help='induction: bytes of a copy shown before it is scored')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)                  # a CPU worker shares the machine with a GPU trainer
    batch_fn = extra_eval = None
    if args.source == 'induction':
        shape = dict(seg=parse_span(args.seg_length), gap=em.parse_range(args.gap), lead=em.parse_range(args.lead), skip=args.skip)
        batch_fn = ind.batch_fn(args.batch, args.seed, args.pad_to, **shape)

        def extra_eval(model):
            return dict(copy_accuracy=ind.copy_accuracy(model, n_items=args.eval_items, batch=16, pad_to=args.pad_to, **shape))
    train(args.out, args.width, args.heads, args.feedback == 'on', args.steps, args.batch, args.lr, args.warmup, args.seed,
          args.facts, args.queries, dt.parse_key_length(args.key_length), dt.ALPHABETS[args.key_alphabet],
          em.parse_range(args.train_filler), args.pad_to, args.eval_every, args.eval_items, parse_configs(args.eval_configs),
          tuple(args.eval_alphabets.split(',')), tuple(int(v) for v in args.eval_key_lengths.split(',')),
          init_from=args.init_from, batch_fn=batch_fn, extra_eval=extra_eval, source=args.source)


if __name__ == '__main__':
    main()
