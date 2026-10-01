"""Decoy facts: is the retrieval content-addressed (finds the fact by its KEY) or position-locked (copies the fact at the
typical distance, or simply the nearest one)?

Layout of a row:  fact1  filler(a)  fact2  filler(b)  question  answer
fact1 and fact2 have different keys and values whose first digits differ, so a predicted first digit can be attributed. The
question asks for fact1's key ('first') or fact2's key ('decoy').

    asks 'first':  answer follows the ASKED value  -> the key was used.   follows the DECOY value -> the nearest / most recent
                   fact was copied regardless of the key.
    asks 'decoy':  the mirror image. A content-addressed circuit answers both; a position-locked one fails one of them.

All distances stay inside the range the checkpoint was trained on, so a failure here is about HOW the fact is found, not about
how far away it is.

Development diagnostic. Reads a checkpoint, trains nothing, freezes nothing.

Usage: python -B -m mline.decoy_probe --checkpoint <selected.pt> --out <json>
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from horizon import sweep as sw
from mline import emergence as em
from mline import lm as ml
from mline import needle as nd
from mline import train as mt

ROLE = 'development diagnostic: is the retrieval content-addressed or position-locked?'
ASK = ('first', 'decoy')
# TRAIN_KEYS / EVAL_KEYS as in mline.needle; HELD_KEYS (T-Z) is unseen even by a model trained on the wide alphabet
# (mline.decoy_train.ALPHABETS['wide']), so it is the test of a general "are these letters equal" comparison.
ALPHABETS = dict(TRAIN_KEYS=nd.TRAIN_KEYS, EVAL_KEYS=nd.EVAL_KEYS, HELD_KEYS=b'TUVWXYZ')
VALUE_LENGTH = 4


def fact_bytes(key_length=4):
    """'The code for KKKK is VVVV.' plus its newline: 13 + key + 4 + 4 + 2 bytes."""
    return 23 + key_length
DEFAULT_CONFIGS = '0:0,0:10,10:0,0:30,30:0,15:15'


def make(n_items, a, b, ask='first', seed=0, alphabet=None, key_length=4):
    if ask not in ASK:
        raise ValueError('ask must be one of ' + ', '.join(ASK))
    alphabet = nd.EVAL_KEYS if alphabet is None else alphabet
    if len(alphabet) ** key_length < 2:
        raise ValueError('two distinct keys do not exist at length %d over %d letters' % (key_length, len(alphabet)))
    rng = np.random.default_rng(seed)
    rows, keys, decoy_keys, answers, decoy_answers = [], [], [], [], []
    for _ in range(n_items):
        while True:
            k1, k2 = nd._key(rng, key_length, alphabet), nd._key(rng, key_length, alphabet)
            if k1 != k2:
                break
        while True:
            v1, v2 = nd._value(rng), nd._value(rng)
            if v1[:1] != v2[:1]:                    # the first digits must differ so a prediction can be attributed
                break
        fact1 = b'The code for ' + k1 + b' is ' + v1 + b'.\n'
        fact2 = b'The code for ' + k2 + b' is ' + v2 + b'.\n'
        asked_key, asked_value = (k1, v1) if ask == 'first' else (k2, v2)
        query = b'\nThe code for ' + asked_key + b' is '
        rows.append(fact1 + nd.filler_bytes('repeat', a, rng) + fact2 + nd.filler_bytes('repeat', b, rng)
                    + query + asked_value)
        keys.append(k1.decode('ascii'))
        decoy_keys.append(k2.decode('ascii'))
        answers.append(v1.decode('ascii'))
        decoy_answers.append(v2.decode('ascii'))
    width = max(len(r) for r in rows)
    x = np.zeros((n_items, width), dtype=np.int64)
    y = np.zeros((n_items, width), dtype=np.int64)
    mask = np.zeros((n_items, width), dtype=bool)
    for i, row in enumerate(rows):
        arr = np.frombuffer((b' ' * (width - len(row))) + row, dtype=np.uint8).astype(np.int64)
        x[i] = arr
        y[i, :-1] = arr[1:]
        y[i, -1] = arr[-1]
        mask[i, width - VALUE_LENGTH - 1:width - 1] = True
    first = width - VALUE_LENGTH - 1
    asked = answers if ask == 'first' else decoy_answers
    other = decoy_answers if ask == 'first' else answers
    return dict(x=torch.from_numpy(x), y=torch.from_numpy(y), mask=torch.from_numpy(mask), width=width, first=first,
                keys=keys, decoy_keys=decoy_keys, answers=answers, decoy_answers=decoy_answers, ask=ask, a=a, b=b,
                key_length=key_length,
                asked_first=torch.tensor([ord(v[0]) for v in asked]), other_first=torch.tensor([ord(v[0]) for v in other]))


def attribute(got, asked, decoy):
    """Share of rows whose predicted first digit matches the asked fact, the other fact, or neither."""
    usable = asked != decoy
    n = int(usable.sum())
    if n == 0:
        return dict(usable=0, asked=None, decoy=None, neither=None)
    g, a, d = got[usable], asked[usable], decoy[usable]
    asked_rate = float((g == a).float().mean())
    decoy_rate = float((g == d).float().mean())
    return dict(usable=n, asked=asked_rate, decoy=decoy_rate, neither=1. - asked_rate - decoy_rate)


def fmt(value):
    return '-' if value is None else '%.2f' % value


def format_table(report):
    """One block per (alphabet, key length): for each filler layout a:b, what the answer followed when the FIRST fact was asked
    and when the DECOY was asked - the asked fact's value, the other fact's value, and the exact 4-digit match."""
    rows = report['rows']
    blocks = []
    for r in rows:
        block = (r['alphabet'], r.get('key_length', 4))              # reports from before key lengths existed were length 4
        if block not in blocks:
            blocks.append(block)
    lines = []
    for alphabet, key_length in blocks:
        mine = [r for r in rows if r['alphabet'] == alphabet and r.get('key_length', 4) == key_length]
        lines.append('=== %s, key length %d (%s items) ===' % (alphabet, key_length, report.get('items', '?')))
        lines.append('%-8s %5s | %-22s | %-22s' % ('a:b', 'gap', 'first fact asked', 'decoy asked'))
        lines.append('%-8s %5s | %-6s %-6s %-6s | %-6s %-6s %-6s' % ('', '', 'asked', 'other', 'exact', 'asked', 'other', 'exact'))
        layouts = []
        for r in mine:
            if (r['a'], r['b']) not in layouts:
                layouts.append((r['a'], r['b']))
        for a, b in layouts:
            cells = {r['ask']: r for r in mine if (r['a'], r['b']) == (a, b)}
            first, decoy = cells.get('first'), cells.get('decoy')
            gap = (first or decoy)['gap_to_question']
            lines.append('%-8s %5d | %-6s %-6s %-6s | %-6s %-6s %-6s' % (
                '%d:%d' % (a, b), gap,
                *(fmt(first[k]) if first else '-' for k in ('asked', 'decoy', 'exact_match')),
                *(fmt(decoy[k]) if decoy else '-' for k in ('asked', 'decoy', 'exact_match'))))
        lines.append('')
    return '\n'.join(lines)


def probe_model(model, configs, asks=ASK, alphabets=('TRAIN_KEYS',), items=64, batch=16, device='cpu', data_seed=777,
                key_lengths=(4,), verbose=True):
    """The probe itself, on a model that is already built: one row per (alphabet, key length, layout, ask).

    Takes any model that `mline.lm.forward_logits` can run - the 30M Core, a small one, either arm - and leaves it in the
    mode it was in, so a trainer can call it between steps."""
    was_training = model.training
    model.eval()
    rows = []
    try:
        for name, key_length in itertools.product(alphabets, key_lengths):
            for a, b in configs:
                for ask in asks:
                    sets = make(items, a, b, ask, data_seed, ALPHABETS[name], key_length)
                    got, exact = [], []
                    for start in range(0, items, batch):
                        xb = sets['x'][start:start + batch].to(device)
                        with torch.no_grad():
                            pred = ml.forward_logits(model, xb).argmax(-1).cpu()
                        got.append(pred[:, sets['first']])
                        yb, mb = sets['y'][start:start + batch], sets['mask'][start:start + batch]
                        exact.append(((pred == yb) & mb).sum(1) == mb.sum(1))
                    got, exact = torch.cat(got), torch.cat(exact)
                    row = dict(alphabet=name, key_length=key_length, a=a, b=b, ask=ask,
                               gap_to_question=a + fact_bytes(key_length) + b, decoy_to_question=b,
                               exact_match=float(exact.float().mean()))
                    row.update(attribute(got, sets['asked_first'], sets['other_first']))
                    rows.append(row)
                    if verbose:
                        print('%-10s key %d a %3d b %3d  ask %-6s  gap %3d  exact %.2f | asked %s  other-fact %s  neither %s'
                              % (name, key_length, a, b, ask, row['gap_to_question'], row['exact_match'], fmt(row['asked']),
                                 fmt(row['decoy']), fmt(row['neither'])), flush=True)
    finally:
        model.train(was_training)
    return rows


def run(checkpoint, out, configs, asks=ASK, alphabets=('TRAIN_KEYS',), items=64, batch=16, device='cpu', arm='plain',
        seed=117, data_seed=777, key_lengths=(4,)):
    p = mt.protocol()
    model = ml.build(arm, width=p['width'], heads=p['heads'], window=p['window'], alpha=p['alpha'], seed=seed,
                     gate_k=p['gate_k']).to(device)
    em.load_init(model, checkpoint)
    rows = probe_model(model, configs, asks, alphabets, items, batch, device, data_seed, key_lengths)
    report = dict(kind='M1_DECOY_PROBE', role=ROLE, utc=time.time(), checkpoint=str(checkpoint),
                  checkpoint_sha256=sw.sha(checkpoint), items=items, key_lengths=list(key_lengths), rows=rows,
                  reading='asked high for BOTH asks: content-addressed. "other-fact" high when asking for the first fact: the '
                          'nearest fact is copied regardless of the key (recency / position-locked). Chance for each is 0.1.')
    sw.write_json(Path(out), report)
    print('written', out)
    return report


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--out')
    parser.add_argument('--show', help='print a saved report as a table and exit (no model is loaded)')
    parser.add_argument('--configs', default=DEFAULT_CONFIGS, help='A:B[,A:B...] filler lengths before / after the decoy')
    parser.add_argument('--asks', default='first,decoy')
    parser.add_argument('--alphabets', default='TRAIN_KEYS')
    parser.add_argument('--key-lengths', default='4', help='comma-separated key lengths to probe (1-8)')
    parser.add_argument('--items', type=int, default=64)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--arm', default='plain', choices=('plain', 'hard_st'))
    args = parser.parse_args()
    if args.show:
        print(format_table(json.loads(Path(args.show).read_text(encoding='utf8'))))
        return
    if not args.checkpoint or not args.out:
        parser.error('--checkpoint and --out are required unless --show is given')
    if args.device == 'cpu':
        torch.set_num_threads(1)                 # a CPU worker shares the machine with a training run
    configs = [tuple(int(v) for v in part.split(':')) for part in args.configs.split(',')]
    run(args.checkpoint, args.out, configs, tuple(args.asks.split(',')), tuple(args.alphabets.split(',')), args.items,
        args.batch, args.device, args.arm, key_lengths=tuple(int(v) for v in args.key_lengths.split(',')))


if __name__ == '__main__':
    main()
