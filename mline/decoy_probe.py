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
VALUE_LENGTH = 4
FACT_BYTES = 27
DEFAULT_CONFIGS = '0:0,0:10,10:0,0:30,30:0,15:15'


def make(n_items, a, b, ask='first', seed=0, alphabet=None):
    if ask not in ASK:
        raise ValueError('ask must be one of ' + ', '.join(ASK))
    alphabet = nd.EVAL_KEYS if alphabet is None else alphabet
    rng = np.random.default_rng(seed)
    rows, keys, decoy_keys, answers, decoy_answers = [], [], [], [], []
    for _ in range(n_items):
        while True:
            k1, k2 = nd._key(rng, alphabet=alphabet), nd._key(rng, alphabet=alphabet)
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


def run(checkpoint, out, configs, asks=ASK, alphabets=('TRAIN_KEYS',), items=64, batch=16, device='cpu', arm='plain',
        seed=117, data_seed=777):
    p = mt.protocol()
    model = ml.build(arm, width=p['width'], heads=p['heads'], window=p['window'], alpha=p['alpha'], seed=seed,
                     gate_k=p['gate_k']).to(device)
    em.load_init(model, checkpoint)
    model.eval()
    keys = dict(TRAIN_KEYS=nd.TRAIN_KEYS, EVAL_KEYS=nd.EVAL_KEYS)
    rows = []
    for name in alphabets:
        for a, b in configs:
            for ask in asks:
                sets = make(items, a, b, ask, data_seed, keys[name])
                got, exact = [], []
                for start in range(0, items, batch):
                    xb = sets['x'][start:start + batch].to(device)
                    with torch.no_grad():
                        pred = ml.forward_logits(model, xb).argmax(-1).cpu()
                    got.append(pred[:, sets['first']])
                    yb, mb = sets['y'][start:start + batch], sets['mask'][start:start + batch]
                    exact.append(((pred == yb) & mb).sum(1) == mb.sum(1))
                got, exact = torch.cat(got), torch.cat(exact)
                row = dict(alphabet=name, a=a, b=b, ask=ask, gap_to_question=a + FACT_BYTES + b, decoy_to_question=b,
                           exact_match=float(exact.float().mean()))
                row.update(attribute(got, sets['asked_first'], sets['other_first']))
                rows.append(row)
                print('%-10s a %3d b %3d  ask %-6s  gap %3d  exact %.2f | asked %s  other-fact %s  neither %s'
                      % (name, a, b, ask, row['gap_to_question'], row['exact_match'], fmt(row['asked']),
                         fmt(row['decoy']), fmt(row['neither'])), flush=True)
    report = dict(kind='M1_DECOY_PROBE', role=ROLE, utc=time.time(), checkpoint=str(checkpoint),
                  checkpoint_sha256=sw.sha(checkpoint), items=items, rows=rows,
                  reading='asked high for BOTH asks: content-addressed. "other-fact" high when asking for the first fact: the '
                          'nearest fact is copied regardless of the key (recency / position-locked). Chance for each is 0.1.')
    sw.write_json(Path(out), report)
    print('written', out)
    return report


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--configs', default=DEFAULT_CONFIGS, help='A:B[,A:B...] filler lengths before / after the decoy')
    parser.add_argument('--asks', default='first,decoy')
    parser.add_argument('--alphabets', default='TRAIN_KEYS')
    parser.add_argument('--items', type=int, default=64)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--arm', default='plain', choices=('plain', 'hard_st'))
    args = parser.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(1)                 # a CPU worker shares the machine with a training run
    configs = [tuple(int(v) for v in part.split(':')) for part in args.configs.split(',')]
    run(args.checkpoint, args.out, configs, tuple(args.asks.split(',')), tuple(args.alphabets.split(',')), args.items,
        args.batch, args.device, args.arm)


if __name__ == '__main__':
    main()
