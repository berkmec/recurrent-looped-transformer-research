"""Which channel carries the needle inside the trained range? Switch parts of a saved model off at inference and see.

The Core has FULL causal attention (encoder self-attention over the whole prefix, decoder cross-attention to every
encoder output) plus a weak recurrent feedback channel, and the gated arm adds a HOLD/WRITE gate that also freezes the
decoder's short window cache while it holds. Switches that separate them:

    nofeedback            alpha = 0: the recurrent feedback contributes nothing at any step
    gate_open             every HOLD/WRITE decision forced to WRITE: the gated weights run as the plain Core
                          (gated arm only)
    gate_open+nofeedback  both: attention is all that is left

Reading: if accuracy inside the trained range (filler <= 64) survives `nofeedback` (and `gate_open+nofeedback`), the fact
is carried by attention, and the cliff beyond the trained range is an attention / positional / training-distance
phenomenon rather than a failure of the recurrent state. If it collapses, the recurrent path or the gate was doing the
carrying, and the distance problem is about it.

Development diagnostic. Reads a checkpoint, trains nothing, freezes nothing. Every switch is pinned in
tests/test_channel_probe.py against an independently built equivalent model.

Usage: python -B -m mline.channel_probe --arm hard_st --checkpoint <selected.pt> --out <json>
"""
import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch

from horizon import sweep as sw
from mline import lm as ml
from mline import needle as nd
from mline import train as mt

ROLE = 'development diagnostic: which channel carries the needle inside the trained range?'
MODES = ('full', 'nofeedback', 'gate_open', 'gate_open+nofeedback')
DEFAULT_MODES = ','.join(MODES)
DEFAULT_FILLERS = '0,16,32,48,64,72,80,96'


@contextlib.contextmanager
def feedback_off(model):
    """alpha = 0 on the model's Core for the length of the block; the original scale is always restored."""
    core = getattr(model, 'core', model)
    saved = core.alpha
    core.alpha = 0.
    try:
        yield model
    finally:
        core.alpha = saved


@contextlib.contextmanager
def gate_open(model):
    """Force every HOLD/WRITE decision to WRITE (gated models only); the gate head is always restored."""
    if not (hasattr(model, 'head') and hasattr(model, 'open_all')):
        raise ValueError('only a gated model has a gate to open')
    saved = {k: v.clone() for k, v in model.head.state_dict().items()}
    model.open_all()
    try:
        yield model
    finally:
        model.head.load_state_dict(saved)


@contextlib.contextmanager
def apply(model, mode):
    if mode not in MODES:
        raise ValueError('mode must be one of ' + ', '.join(MODES))
    with contextlib.ExitStack() as stack:
        if 'gate_open' in mode:
            stack.enter_context(gate_open(model))
        if 'nofeedback' in mode:
            stack.enter_context(feedback_off(model))
        yield model


def run(arm, checkpoint, out, modes, fillers, alphabets=('TRAIN_KEYS', 'EVAL_KEYS'), kind='repeat', items=128,
        batch=32, device='cuda', seed=117):
    p = mt.protocol()
    model = ml.build(arm, width=p['width'], heads=p['heads'], window=p['window'], alpha=p['alpha'], seed=seed,
                     gate_k=p['gate_k']).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    forward = lambda xb: ml.forward_logits(model, xb)
    keys = dict(TRAIN_KEYS=nd.TRAIN_KEYS, EVAL_KEYS=nd.EVAL_KEYS)
    rows = []
    for mode in modes:
        if 'gate_open' in mode and arm == 'plain':
            print('skipping %s: the plain arm has no gate' % mode, flush=True)
            continue
        with apply(model, mode):
            for name in alphabets:
                for n in fillers:
                    sets = nd.make(items, n_filler=n, kind=kind, seed=2024 + n, alphabet=keys[name])
                    score = nd.score(forward, sets, batch=batch, device=device)
                    rows.append(dict(mode=mode, alphabet=name, n_filler=n, width=sets['width'],
                                     in_training_range=n <= max(p['needle_train_filler']),
                                     exact_match=score['exact_match'], byte_accuracy=score['byte_accuracy']))
                    print('%-21s %-10s filler %4d  exact %.4f  byte %.4f'
                          % (mode, name, n, score['exact_match'], score['byte_accuracy']), flush=True)
    report = dict(kind='M1_CHANNEL_PROBE', role=ROLE, utc=time.time(), arm=arm, checkpoint=str(checkpoint),
                  checkpoint_sha256=sw.sha(checkpoint), filler_kind=kind, items=items,
                  trained_filler_range=p['needle_train_filler'], rows=rows,
                  reading='Survival of in-range accuracy under nofeedback / gate_open+nofeedback means attention '
                          'carries the fact. A collapse means the recurrent path or the gate carries it.')
    sw.write_json(Path(out), report)
    print('written', out)
    return report


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=('plain', 'hard_st'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--modes', default=DEFAULT_MODES)
    parser.add_argument('--fillers', default=DEFAULT_FILLERS)
    parser.add_argument('--kind', default='repeat', choices=nd.KIND)
    parser.add_argument('--items', type=int, default=128)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run(args.arm, args.checkpoint, args.out, args.modes.split(','), [int(v) for v in args.fillers.split(',')],
        kind=args.kind, items=args.items, batch=args.batch, device=args.device)


if __name__ == '__main__':
    main()
