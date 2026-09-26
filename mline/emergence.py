"""M0 development probe: how much data does the needle capability need before it exists at all?

This is the measurement that decides how many rented GPU-hours to buy, and it is the one question whose answer does
NOT change if the step cost is later reduced: it is about bytes seen, not about seconds spent. So it is worth running
now, on the idle GPU, while the speed work happens in parallel.

Development only. It trains the `plain` arm, freezes no protocol, writes under results/M0-emergence, and touches no
M1 endpoint - the needle sets it scores are built from the VALIDATION corpus with their own seeds, and M1 will build
its own. Nothing here is evidence for or against the M1 hypothesis; it sizes the budget.

Usage: python -B -m mline.emergence --steps 6000 --batch 32
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from horizon import guards as gu
from horizon import sweep as sw
from mline import checkpoint as ck
from mline import data as md
from mline import lm as ml
from mline import needle as nd
from mline import train as mt

ROLE = 'development probe: sizes the training budget, never used as evidence about the gate'


def run(out, arm='plain', steps=6000, batch=32, every=250, device='cuda', seed=117, needle_rate=.125,
        train_filler=(0, 64), pack=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = mt.protocol()
    train = md.load(p['data'], 'train')
    corpus = md.load(p['data'], 'validation')
    model = ml.build(arm, width=p['width'], heads=p['heads'], window=p['window'], alpha=p['alpha'], seed=seed, gate_k=p['gate_k']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    counter = gu.GuardCounter()
    probes = {n: nd.make(48, n_filler=n, kind='repeat', seed=777 + n) for n in (0, 128, 512)}
    probes['text@512'] = nd.make(48, n_filler=512, kind='text', seed=999, corpus=corpus)
    # The leading indicator, added after the first probe returned exact_match 0.000 in all 76 of its measurements:
    # a set drawn from TRAIN_KEYS at a filler the training mix actually shows. Anything the model has learned must
    # appear HERE first. A zero here is a statement about the budget or the mix; a zero here with a non-zero on the
    # evaluation sets would be impossible, and a non-zero here with a zero there is the generalisation result the
    # experiment is about. Without it the two cannot be told apart, which cost the first probe its diagnosis.
    probes['train@32'] = nd.make(48, n_filler=32, kind='repeat', seed=555, alphabet=nd.TRAIN_KEYS)
    needles_per_batch = max(1, round(batch * needle_rate)) if needle_rate > 0 else 0
    header = dict(role=ROLE, arm=arm, steps=steps, batch=batch, length=p['length'], width=p['width'],
                  lr=p['lr'], seed=seed, bytes_per_step=batch * p['length'], utc=time.time(),
                  needle_rate=needle_rate, needles_per_batch=needles_per_batch, train_filler=list(train_filler),
                  pack=pack,
                  arm_note='pack=True fills each needle row with as many examples as fit; pack=False leaves the '
                           'single example padded to the training width, which is what the first probe ran and '
                           'what made the four answer bytes 0.098 % of the gradient. These are the two arms of '
                           'the packing comparison and they differ in this term alone.',
                  design='the needle FORMAT is taught at short filler; the evaluation uses fillers never seen, so what '
                         'has to generalise is the DISTANCE - the U1 shape (trained at 128, tested to 4096) in language. '
                         'Training and evaluation key alphabets are disjoint, so no evaluation key can be memorised.')
    sw.write_json(out / 'probe.json', header)
    print(json.dumps(header), flush=True)
    # Resumable for the same reason the M1 runner is: the batch at step N depends only on (seed, step), so a
    # restored model, optimizer and counter continue the same run rather than a similar one. A 9-hour probe is
    # long enough that losing it to a reboot is a real cost, and the first one was lost exactly that way.
    resume = ck.load_latest(out)
    start = 1
    if resume is not None and resume['step'] is not None:
        model.load_state_dict(resume['model'])
        if resume['optimizer'] is not None:
            opt.load_state_dict(resume['optimizer'])
        start = int(resume['step']) + 1
        # Truncate the log to the checkpoint, as mline/train.py does. Without this a resume APPENDS rows for
        # steps already written: measured on the padded control, where a restart left 7 duplicate rows. They were
        # byte-identical reruns - same input_hash, loss equal to 0.00e+00 - so nothing was wrong with the resume
        # itself, but a record carrying duplicate steps is no longer a clean record.
        log = out / 'probe.jsonl'
        if log.exists():
            kept = [line for line in log.read_text(encoding='utf8').splitlines()
                    if line.strip() and json.loads(line)['step'] <= resume['step']]
            log.write_text((chr(10).join(kept) + chr(10)) if kept else '', encoding='utf8')
        print('resuming from step %d' % resume['step'], flush=True)
    t0 = time.perf_counter()
    for step in range(start, steps + 1):
        x, y, input_hash = md.batch(train, seed, step, batch - needles_per_batch, p['length'])
        xb, yb = torch.from_numpy(x), torch.from_numpy(y)
        if needles_per_batch:
            mixed = nd.training_items(needles_per_batch, filler_range=train_filler, seed=seed * 100_003 + step,
                                      pad_to=p['length'], pack=pack)
            xb = torch.cat([xb, mixed['x']]); yb = torch.cat([yb, mixed['y']])
        xb = xb.to(device); yb = yb.to(device)
        model.train()
        logits = ml.forward_logits(model, xb)
        loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten())
        report = counter.record(gu.guarded_step(opt, model.parameters(), loss, clip_norm=p['clip']))
        row = dict(step=step, bytes_seen=step * batch * p['length'], loss=report['loss'],
                   bits_per_byte=(report['loss'] / np.log(2)) if report['loss'] else None,
                   grad_norm=report['grad_norm_before_clip'], input_hash=input_hash)
        if step % every == 0 or step == steps:
            val = mt.validate(model, dict(p, batch=batch), device, batches=8)
            forward = lambda xb_: ml.forward_logits(model, xb_)
            row['validation'] = val
            row['needle'] = {str(k): nd.score(forward, v, batch=8, device=device) for k, v in probes.items()}
            tr = row['needle']['train@32']
            print('step %5d  %6.1f MB  val bpb %.3f  TRAIN@32 %.3f (byte %.3f)  needle@0 %.3f  @128 %.3f  @512 %.3f  '
                  'text@512 %.3f  %.1f min'
                  % (step, row['bytes_seen'] / 1e6, val['bits_per_byte'], tr['exact_match'], tr['byte_accuracy'],
                     row['needle']['0']['exact_match'], row['needle']['128']['exact_match'],
                     row['needle']['512']['exact_match'], row['needle']['text@512']['exact_match'],
                     (time.perf_counter() - t0) / 60), flush=True)
            ck.save(out / ck.LATEST, model, opt, step=step,
                    meta=dict(bytes_seen=row['bytes_seen'], validation=val, arm=arm, seed=seed))
        sw.append_line(out / 'probe.jsonl', row)
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/M0-emergence')
    parser.add_argument('--arm', default='plain'); parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch', type=int, default=32); parser.add_argument('--every', type=int, default=250)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--seed', type=int, default=117)
    parser.add_argument('--needle-rate', type=float, default=.125)
    parser.add_argument('--no-pack', dest='pack', action='store_false', default=True,
                        help='run the CONTROL arm: one needle example per row, padded to the training width, as '
                             'the first probe did. The default packs the row, which is the treatment.')
    args = parser.parse_args()
    run(args.out, args.arm, args.steps, args.batch, args.every, args.device, args.seed, args.needle_rate,
        pack=args.pack)


if __name__ == '__main__':
    main()
