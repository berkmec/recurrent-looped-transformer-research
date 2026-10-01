"""Where does the answer position look? The decoder's cross-attention at the position that predicts the first answer digit,
split by what it lands on: the asked fact's value digits, the other fact's, the keys, the filler, the question itself.

The decoy probe (mline/decoy_probe.py) says WHICH fact's value comes out. This says where the decoder is looking when it decides,
head by head. A model that answers from the right fact because it matched the key shows attention that follows the asked key;
a position-locked one looks at the same place whichever key is asked. 'Selectivity' is the share of the two value regions'
attention that lands on the ASKED fact's value: 0.5 = blind to the key, 1.0 = key-addressed.

The weights are read off the Core's own attention call (its output is returned unchanged, and the hook is removed afterwards), so
nothing in core/ is edited and nothing is re-implemented. Both of the Core's decode paths are supported: the sequential one
(feedback on - the 30M models) and the parallel one (feedback off - the small CPU models).

Development diagnostic: reads a checkpoint, trains nothing.

Usage: python -B -m mline.attn_probe --checkpoint <selected.pt> --out <json> --device cpu --configs 0:0,15:15
"""
import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import torch

from core.model import Core
from horizon import sweep as sw
from mline import decoy_probe as dp
from mline import emergence as em
from mline import lm as ml
from mline import mqar_scratch as ms
from mline import train as mt

ROLE = 'development diagnostic: does the answer position look where the asked KEY points, or always at the same place?'
PREFIX = len(b'The code for ')
IS = len(b' is ')
VALUE = dp.VALUE_LENGTH
END = len(b'.\n')


def regions(key_length, a, b):
    """name -> (start, end) byte spans of a decoy-probe row; they tile the row, the answer digits last."""
    spans, at = {}, 0

    def add(name, length):
        nonlocal at
        spans[name] = (at, at + length)
        at += length

    for which in ('first', 'decoy'):
        add('template_' + which, PREFIX)
        add('key_' + which, key_length)
        add('is_' + which, IS)
        add('value_' + which, VALUE)
        add('end_' + which, END)
        add('filler_a' if which == 'first' else 'filler_b', a if which == 'first' else b)
    add('template_question', 1 + PREFIX)                 # the newline before the question, then 'The code for '
    add('key_question', key_length)
    add('is_question', IS)
    add('answer', VALUE)
    return spans


def region_masses(weights, spans, position):
    """Attention mass per region, (batch, heads) each, over the regions visible at `position` (those starting after it are absent)."""
    out = {}
    for name, (start, end) in spans.items():
        if start > position:
            continue
        out[name] = weights[..., start:min(end, position + 1)].sum(-1)
    return out


def selectivity(masses, ask):
    """Share of the two value regions' attention that lands on the ASKED fact's value; nan where a head looks at neither."""
    asked, other = (masses['value_first'], masses['value_decoy']) if ask == 'first' \
        else (masses['value_decoy'], masses['value_first'])
    return asked / (asked + other)


@contextlib.contextmanager
def _record(core, position, sequential, store):
    """Wrap core.attend for one forward pass and keep the cross-attention weights at `position` in `store['weights']`."""
    calls = dict(n=0)
    original = core.attend
    # Call order in Core.forward: the encoder's attention comes first; the sequential decoder then calls a self-attention and a
    # cross-attention per position (so the cross-attention at position t is call 3 + 2t), while the parallel decoder makes a
    # single cross-attention call over the whole row (call 2) and does its windowed self-attention outside attend().
    target = 3 + 2 * position if sequential else 2

    def attend(q, k, v, causal=False):
        calls['n'] += 1
        out = original(q, k, v, causal)
        if calls['n'] == target:
            q_at = q[:, :, :1] if sequential else q[:, :, position:position + 1]
            scores = (q_at @ k[:, :, :position + 1].transpose(-1, -2)) / math.sqrt(q.shape[-1])
            store['weights'] = scores.softmax(-1)[:, :, 0].detach()
        return out

    core.attend = attend
    try:
        yield
    finally:
        if 'attend' in core.__dict__:
            del core.attend


def cross_attention_at(model, x, position, force_sequential=False):
    """The cross-attention weights, (batch, heads, position + 1), of the decoder step at `position`, for one forward pass over x."""
    if not isinstance(model, Core):
        raise TypeError('attention can be read from a plain Core, not from %s' % type(model).__name__)
    if not 0 <= position < x.shape[1]:
        raise ValueError('position %d is outside a row of %d bytes' % (position, x.shape[1]))
    sequential = bool(model.feedback or force_sequential)
    store = {}
    with _record(model, position, sequential, store), torch.no_grad():
        model(x, parallel_control=not force_sequential)          # no effect when the feedback is on: that path is sequential anyway
    return store['weights']


def probe(model, configs, asks=dp.ASK, alphabet='TRAIN_KEYS', key_length=4, items=32, batch=16, device='cpu', data_seed=777):
    """One row per (layout, ask): per-head attention mass on every region, and per-head selectivity, averaged over the items."""
    rows = []
    for a, b in configs:
        for ask in asks:
            sets = dp.make(items, a, b, ask, data_seed, dp.ALPHABETS[alphabet], key_length)
            position = sets['first']
            spans = regions(key_length, a, b)
            masses, picked = {}, []
            for start in range(0, items, batch):
                weights = cross_attention_at(model, sets['x'][start:start + batch].to(device), position).cpu()
                for name, mass in region_masses(weights, spans, position).items():
                    masses.setdefault(name, []).append(mass)
                picked.append(selectivity(region_masses(weights, spans, position), ask))
            per_head = {name: torch.cat(parts).mean(0).tolist() for name, parts in masses.items()}
            sel = torch.cat(picked)                                               # (items, heads), nan where undefined
            valid = ~torch.isnan(sel)
            per_head_sel = [float(sel[:, h][valid[:, h]].mean()) if valid[:, h].any() else None for h in range(sel.shape[1])]
            known = [v for v in per_head_sel if v is not None]
            rows.append(dict(alphabet=alphabet, key_length=key_length, a=a, b=b, ask=ask, items=items,
                             heads=sel.shape[1], masses=per_head, selectivity=per_head_sel,
                             selectivity_mean=sum(known) / len(known) if known else None))
    return dict(kind='M1_ATTENTION_PROBE', role=ROLE, utc=time.time(), rows=rows,
                reading='selectivity = share of the two value regions\' attention on the ASKED fact; 0.5 = blind to the key, '
                        '1.0 = key-addressed; compare the same head across asks and layouts')


def format_report(report):
    lines = []
    for r in report['rows']:
        top = sorted(((sum(v) / len(v), n) for n, v in r['masses'].items()), reverse=True)[:4]
        lines.append('%-10s key %d  %2d:%-2d ask %-5s  selectivity mean %s | per head %s | top regions %s'
                     % (r['alphabet'], r['key_length'], r['a'], r['b'], r['ask'], dp.fmt(r['selectivity_mean']),
                        ' '.join(dp.fmt(v) for v in r['selectivity']),
                        ', '.join('%s %.2f' % (n, m) for m, n in top)))
    return '\n'.join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--out')
    parser.add_argument('--show', help='print a saved report and exit')
    parser.add_argument('--configs', default='0:0,15:15,0:30,30:0')
    parser.add_argument('--asks', default='first,decoy')
    parser.add_argument('--alphabet', default='TRAIN_KEYS')
    parser.add_argument('--key-length', type=int, default=4)
    parser.add_argument('--items', type=int, default=32)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--width', type=int, default=None, help='a small Core of this width instead of the 30M protocol model')
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--feedback', choices=('on', 'off'), default='on')
    args = parser.parse_args()
    if args.show:
        print(format_report(json.loads(Path(args.show).read_text(encoding='utf8'))))
        return
    if not args.checkpoint or not args.out:
        parser.error('--checkpoint and --out are required unless --show is given')
    if args.device == 'cpu':
        torch.set_num_threads(1)
    if args.width is None:
        p = mt.protocol()
        model = ml.build('plain', width=p['width'], heads=p['heads'], window=p['window'], alpha=p['alpha'], seed=117,
                         gate_k=p['gate_k'])
    else:
        model = ms.build_small(args.width, args.heads, args.feedback == 'on', seed=0)
    model = model.to(args.device)
    em.load_init(model, args.checkpoint)
    model.eval()
    configs = [tuple(int(v) for v in part.split(':')) for part in args.configs.split(',')]
    report = probe(model, configs, tuple(args.asks.split(',')), args.alphabet, args.key_length, args.items, args.batch, args.device)
    report.update(checkpoint=str(args.checkpoint), checkpoint_sha256=sw.sha(args.checkpoint))
    sw.write_json(Path(args.out), report)
    print(format_report(report))
    print('written', args.out)


if __name__ == '__main__':
    main()
