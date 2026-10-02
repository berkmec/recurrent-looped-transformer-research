"""E-L1X: does the L1 byte model keep working on documents far longer than the 512 bytes it was trained on, and if not, is it the attention memory or the carried state? Evaluation only, CPU,
on the snapshots the run saves every 1,000 updates, on validation articles (never trained on).

The L1 model is the sealed Core at language width (928, 8 heads), trained on 512-byte windows. Run continuously over a longer text, three things leave the range it was trained in: (a) the
encoder self-attention and the decoder cross-attention see more than 512 keys at relative offsets above 511; (b) the carried state is older than 512 steps; (c) positions are absolute RoPE
angles up to 4,095. The RLT promise is that the recurrence gives context beyond the training window at O(1) state: this measures whether it does, without changing a weight.

Conditions on the SAME bytes (the first 4,096 bytes of the first 32 validation articles with at least 4,097 bytes, in file order):

  full      the sealed Core run over the whole text (nothing restricted)
  win512    encoder and cross-attention restricted to the last 512 positions (relative offsets and key counts as in training), the carried state continues
  reset512  the carried state is replaced by the learned initial state every 512 steps (state age <= 512 as in training), the attention paths unrestricted
  chunk512  the text cut into eight independent 512-byte chunks (state and memory restarted): the in-distribution reference

Metric: per-article mean of the per-byte loss (bits) over the positions p of a bin whose in-chunk offset p mod 512 is at least 256 (every condition has at least 256 bytes of context there,
including the reference); the paired difference D_c(bin) = loss_c - loss_chunk512, mean over articles, standard error across articles. Bins: [0,512) control, [512,1024), [1024,2048), [2048,4096)
(and [4096,8192) in the descriptive 8,192-byte run, 16 articles; that run prints the same reading but the verdict is carried by the 4,096-byte run).

Reading rule (written before the first run, program note 5.35):
  CONTROL FAILED          the three conditions differ from chunk512 on [0,512) (where they must coincide) by more than 1e-3 bits at any position (max over articles and positions).
  CONTINUATION HURTS      some bin at or above [1024,2048) has D_full >= 0.15 bits and D_full >= 3 SE.
  CONTINUATION IS FINE    D_full <= 0.05 bits in all three bins above 512.
  MILD                    otherwise.
  Attribution, only for HURTS, at the bin b* of the largest D_full among the bins at or above 1,024: M = D_full - D_win512 (what the memory restriction removes), S = D_full - D_reset512 (what the
  state reset removes): MEMORY PATH if M >= 2/3 D_full and S <= 1/3 D_full; STATE PATH if S >= 2/3 D_full and M <= 1/3 D_full; BOTH if M >= 1/3 D_full and S >= 1/3 D_full; NEITHER if both are below 1/3 D_full;
  anything else (one at least a third and the other between) is reported as MIXED.
Descriptive (no verdict): every condition's absolute bits per byte per bin, D for every condition, the 8,192-byte run, later L1 snapshots.

  python -B -m mline.long_doc results/L1-scratch-w928-s401/snapshots/step006000.pt --out results/L1-long-doc
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from core.model import Core
from horizon.erosion_source import ProbeCore
from mline import data as md
from mline import train as mt
from mline.feedback_probe import load_model

LN2 = math.log(2)
TRAIN_LENGTH = 512
LENGTH = 4096
DOCS = 32
MIN_OFFSET = 256
BINS = ((0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192))      # the last one only exists in the descriptive 8,192-byte run
CONDITIONS = ('full', 'win512', 'reset512', 'chunk512')
CONTROL_TOL = 1e-3
HURT_BITS, HURT_SE, FINE_BITS = 0.15, 3.0, 0.05


class LongCore(ProbeCore):
    """The probe core of E-A5 plus a periodic reset of the carried state to the learned initial state (the decoder's own window-4 cache is left alone)."""

    reset_every = None

    def decode(self, e, s, mk, mv, t):
        if self.reset_every and s is not None and t % self.reset_every == 0:
            s = dict(s)
            s['s'] = self.initial.expand_as(s['s'])
        return super().decode(e, s, mk, mv, t)


def configure(model, condition):
    """Set the condition in place (class swap); `full` and `chunk512` leave the computation as trained."""
    if condition not in CONDITIONS:
        raise ValueError('unknown condition: ' + str(condition))
    model.__class__ = LongCore
    model.cross = model.enc = model.reset_every = model.at = model.state = None
    model.reset_cache = False
    if condition == 'win512':
        model.cross = model.enc = TRAIN_LENGTH
    elif condition == 'reset512':
        model.reset_every = TRAIN_LENGTH
    return model


def long_documents(n=DOCS, length=LENGTH, directory=None):
    """(n, length + 1) int64: the first length + 1 bytes of the first n validation articles that have that many, in file order (x = [:, :-1], y = [:, 1:])."""
    val = md.load(directory or mt.protocol()['data'], 'validation')
    raw = bytes(np.asarray(val[:len(val)]))
    docs = [np.frombuffer(d[:length + 1], dtype=np.uint8) for d in raw.split(md.SEPARATOR) if len(d) >= length + 1]
    if len(docs) < n:
        raise ValueError(f'only {len(docs)} validation articles have {length + 1} bytes; {n} were asked for')
    return np.stack(docs[:n]).astype(np.int64)


def bits(model, x, y, batch=8):
    """(n, T) per-byte loss in bits of the configured model."""
    model.eval()
    out = []
    with torch.no_grad():
        for at in range(0, len(x), batch):
            logits = model(torch.from_numpy(x[at:at + batch]))
            out.append((F.cross_entropy(logits.transpose(1, 2), torch.from_numpy(y[at:at + batch]), reduction='none') / LN2).numpy())
    return np.concatenate(out)


def chunk_bits(model, x, y, batch=64):
    """The reference: every article cut into independent TRAIN_LENGTH-byte chunks, the sealed forward on each, put back in place."""
    n, steps = x.shape
    if steps % TRAIN_LENGTH:
        raise ValueError('the length must be a multiple of the training length')
    plain = bits(model, x.reshape(-1, TRAIN_LENGTH), y.reshape(-1, TRAIN_LENGTH), batch)
    return plain.reshape(n, steps)


def bin_means(loss, steps):
    """{(lo, hi): (n,) per-article mean over the positions of the bin with offset p mod TRAIN_LENGTH >= MIN_OFFSET}."""
    positions = np.arange(steps)
    keep = positions % TRAIN_LENGTH >= MIN_OFFSET
    return {b: loss[:, (positions >= b[0]) & (positions < min(b[1], steps)) & keep].mean(1) for b in BINS if b[0] < steps}


def paired(values, reference):
    d = values - reference
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0


def run(snapshot, directory=None, docs=None, n=DOCS, length=LENGTH, batch=8, width=928, heads=8, log=print, model=None):
    data = long_documents(n, length, directory) if docs is None else docs
    x, y = data[:, :-1], data[:, 1:]
    steps = x.shape[1]
    model = load_model(snapshot, width=width, heads=heads) if model is None else model
    losses = {}
    for condition in CONDITIONS:
        started = time.perf_counter()
        configure(model, condition)
        losses[condition] = chunk_bits(model, x, y) if condition == 'chunk512' else bits(model, x, y, batch)
        log(f'  {condition:<9} mean {losses[condition].mean():.4f} bits/byte on {x.shape[0]} x {steps} bytes  ({time.perf_counter() - started:.0f} s)')
    configure(model, 'full')
    first = slice(0, min(TRAIN_LENGTH, steps))
    control = max(float(np.abs(losses[c][:, first] - losses['chunk512'][:, first]).max()) for c in CONDITIONS)
    per_doc = {c: {f'{b[0]}-{b[1]}': v.tolist() for b, v in bin_means(losses[c], steps).items()} for c in CONDITIONS}
    result = dict(snapshot=str(snapshot), n=int(x.shape[0]), length=int(steps), control=control, per_doc=per_doc)
    result['reading'] = reading(result)
    return result


def differences(result):
    """{condition: {bin: (mean D, SE)}} for the three conditions against chunk512, recomputed from the stored per-article values."""
    ref = result['per_doc']['chunk512']
    return {c: {b: paired(np.array(v), np.array(ref[b])) for b, v in result['per_doc'][c].items()} for c in CONDITIONS if c != 'chunk512'}


def reading(result):
    if result['control'] > CONTROL_TOL:
        return dict(verdict='CONTROL FAILED', control=result['control'])
    d = differences(result)
    names = [b for b in d['full'] if int(b.split('-')[0]) >= TRAIN_LENGTH]
    high = [b for b in names if int(b.split('-')[0]) >= 1024]
    if not high:
        return dict(verdict='INCOMPLETE', reason='no bin at or above 1,024 positions')
    hurts = [b for b in high if d['full'][b][0] >= HURT_BITS and d['full'][b][0] >= HURT_SE * d['full'][b][1]]
    if hurts:
        worst = max(high, key=lambda b: d['full'][b][0])
        full = d['full'][worst][0]
        memory, state = full - d['win512'][worst][0], full - d['reset512'][worst][0]
        if memory >= 2 / 3 * full and state <= 1 / 3 * full:
            attribution = 'MEMORY PATH'
        elif state >= 2 / 3 * full and memory <= 1 / 3 * full:
            attribution = 'STATE PATH'
        elif memory >= full / 3 and state >= full / 3:
            attribution = 'BOTH'
        elif memory < full / 3 and state < full / 3:
            attribution = 'NEITHER'
        else:
            attribution = 'MIXED'
        return dict(verdict='CONTINUATION HURTS', worst_bin=worst, d_full=full, memory_effect=memory, state_effect=state, attribution=attribution)
    if all(d['full'][b][0] <= FINE_BITS for b in names):
        return dict(verdict='CONTINUATION IS FINE')
    return dict(verdict='MILD')


def describe(result):
    d = differences(result)
    lines = [f'{result["snapshot"]}: {result["n"]} articles x {result["length"]} bytes, control {result["control"]:.1e}']
    for c in ('full', 'win512', 'reset512'):
        lines.append(f'  D({c:<8}) ' + '  '.join(f'[{b}] {m:+.3f}±{s:.3f}' for b, (m, s) in d[c].items()))
    lines.append('  reading: ' + json.dumps(result['reading'], default=float))
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('snapshots', nargs='+')
    p.add_argument('--n', type=int, default=DOCS)
    p.add_argument('--length', type=int, default=LENGTH)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--out', default=None, help='directory for <snapshot>-long-doc.json')
    a = p.parse_args()
    for path in a.snapshots:
        print(path, flush=True)
        r = run(path, n=a.n, length=a.length, batch=a.batch, log=lambda m: print(m, flush=True))
        print(describe(r), flush=True)
        if a.out:
            target = Path(a.out) / (Path(path).stem + f'-long-doc{"" if a.length == LENGTH else "-" + str(a.length)}.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(r, indent=1, default=float), encoding='utf8')


if __name__ == '__main__':
    main()
