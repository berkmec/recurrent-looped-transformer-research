"""Re-scale the sealed Core's rotary positions at INFERENCE time, to ask whether the distance cliff is RoPE's.

The M1 checkpoints answer a needle almost perfectly up to the longest fact-to-question distance they were trained on
(64 bytes of filler) and collapse within 8-16 bytes beyond it. The Core has FULL causal attention with RoPE, so that is
the signature of rotary angles the model never saw. If it is, shrinking the positions (or stretching the base) should
move the cliff out by about the same factor, with no training at all. If it does not move, the cliff lives somewhere
else - in the attention pattern the model learned, or in the recurrent path - and that is worth knowing too.

Nothing here edits a sealed file: the patch is an instance attribute that shadows `Core.rope` for the length of a
`with` block and is deleted on exit. tests/test_rope_probe.py pins that factor 1 is bitwise the sealed behaviour.

Schemes (factor meaning):
    identity  the sealed rotary embedding, untouched
    pi        position interpolation: positions are multiplied by `factor` (< 1), so a distance D is read as factor*D.
              Uniform, so it also blurs local offsets.
    ntk       NTK-aware base scaling: base' = base * factor**(d/(d-2)) (factor > 1). Stretches the slow frequencies and
              barely moves the fast ones, so local offsets stay nearly exact.

Development diagnostic. Reads checkpoints, trains nothing, freezes nothing.

Usage: python -B -m mline.rope_probe --arm hard_st --checkpoint <selected.pt>
"""
import argparse
import contextlib
import sys
import time
import types
from pathlib import Path

import torch

from horizon import sweep as sw
from mline import lm as ml
from mline import needle as nd
from mline import train as mt

ROLE = 'development diagnostic: does rescaling RoPE positions move the distance cliff of the M1 checkpoints?'
BASE = 10000
SCHEMES = ('identity', 'pi', 'ntk')
DEFAULT_SCHEMES = 'identity:1,pi:0.5,pi:0.25,pi:0.125,ntk:2,ntk:4,ntk:8,ntk:16,ntk:32'
DEFAULT_FILLERS = '0,32,64,72,80,96,128,192,256,384,512,768,1024'


def rope_for(mode, factor):
    if mode not in SCHEMES:
        raise ValueError('mode must be one of ' + ', '.join(SCHEMES))

    def rope(self, x, start=0):
        d = x.shape[-1]
        position = torch.arange(start, start + x.shape[2], device=x.device, dtype=x.dtype)[:, None]
        base = BASE
        if mode == 'pi':
            position = position * factor
        elif mode == 'ntk':
            base = BASE * factor ** (d / (d - 2))
        angle = position * base ** (-torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
        c = angle.cos()[None, None]
        s = angle.sin()[None, None]
        return torch.stack((x[..., ::2] * c - x[..., 1::2] * s, x[..., ::2] * s + x[..., 1::2] * c), -1).flatten(-2)
    return rope


@contextlib.contextmanager
def rescaled(model, mode='identity', factor=1.):
    """Shadow `rope` on the model's Core for the length of the block; always restore the sealed method on exit."""
    core = getattr(model, 'core', model)
    patch = rope_for(mode, factor)                      # validates the scheme before anything is touched
    core.rope = types.MethodType(patch, core)
    try:
        yield model
    finally:
        del core.rope


def parse_schemes(text):
    out = []
    for part in text.split(','):
        mode, factor = part.split(':')
        out.append((mode, float(factor)))
    return out


def run(arm, checkpoint, out, schemes, fillers, alphabet='TRAIN_KEYS', kind='repeat', items=128, batch=32,
        device='cuda', seed=117):
    p = mt.protocol()
    model = ml.build(arm, width=p['width'], heads=p['heads'], window=p['window'], alpha=p['alpha'], seed=seed,
                     gate_k=p['gate_k']).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    forward = lambda xb: ml.forward_logits(model, xb)
    letters = dict(TRAIN_KEYS=nd.TRAIN_KEYS, EVAL_KEYS=nd.EVAL_KEYS)[alphabet]
    rows = []
    for mode, factor in schemes:
        with rescaled(model, mode, factor):
            for n in fillers:
                sets = nd.make(items, n_filler=n, kind=kind, seed=2024 + n, alphabet=letters)
                score = nd.score(forward, sets, batch=batch, device=device)
                rows.append(dict(mode=mode, factor=factor, n_filler=n, width=sets['width'],
                                 exact_match=score['exact_match'], byte_accuracy=score['byte_accuracy']))
                print('%-8s %-6g filler %5d  exact %.4f  byte %.4f' % (mode, factor, n, score['exact_match'],
                                                                      score['byte_accuracy']), flush=True)
    report = dict(kind='M1_ROPE_PROBE', role=ROLE, utc=time.time(), arm=arm, checkpoint=str(checkpoint),
                  checkpoint_sha256=sw.sha(checkpoint), alphabet=alphabet, filler_kind=kind, items=items,
                  trained_filler_range=p['needle_train_filler'], rows=rows,
                  reading='A cliff that moves out by ~1/factor under pi (or under ntk) is RoPE-relative-distance out '
                          'of distribution. A cliff that stays put is not; look at the learned attention pattern and '
                          'the recurrent path instead. Compare in-range accuracy (filler <= 64) to see what the '
                          'rescaling costs.')
    sw.write_json(Path(out), report)
    print('written', out)
    return report


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=('plain', 'hard_st'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--schemes', default=DEFAULT_SCHEMES)
    parser.add_argument('--fillers', default=DEFAULT_FILLERS)
    parser.add_argument('--alphabet', default='TRAIN_KEYS', choices=('TRAIN_KEYS', 'EVAL_KEYS'))
    parser.add_argument('--kind', default='repeat', choices=nd.KIND)
    parser.add_argument('--items', type=int, default=128)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run(args.arm, args.checkpoint, args.out, parse_schemes(args.schemes),
        [int(v) for v in args.fillers.split(',')], args.alphabet, args.kind, args.items, args.batch, args.device)


if __name__ == '__main__':
    main()
