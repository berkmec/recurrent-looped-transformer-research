"""Decoy-aware needle training: several facts per example, the ASKED one chosen at random.

mline/decoy_probe.py showed that both M1-style checkpoints answer with the NEAREST (or, after the wide fine-tune, the first)
fact whatever key is asked: in the original mix one fact precedes its question and the question always asks it, so copying a
fact by position solves the task and matching on the key is never needed. Here an example is

    fact_1  filler  fact_2  filler  ...  fact_n  filler   question(fact_j)   answer

with distinct keys and j drawn uniformly, so the nearest fact is the right one only 1/n of the time and the key must be used.
The total filler of an example is drawn from the same range as before and split at random points between the facts.

With one fact this is the original generator byte for byte (tests/test_decoy_train.py pins it, so a run with --facts 1 is
the earlier recipe exactly); the packing and the next-byte targets are copied from mline.needle.training_items.

It is wired in by replacing `needle.training_items` for the length of one run, so mline/emergence.py - which the running
experiments import - is not edited.

Usage: python -B -m mline.decoy_train --out results/M1-finetune-distance/D1-decoy3 --init-from <selected.pt> --facts 3
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from mline import data as md
from mline import emergence as em
from mline import induction as ind
from mline import needle as nd
from mline import train as mt


# Training key alphabets. 'wide' adds letters the original mix kept for evaluation, but leaves T-Z (the probe's HELD_KEYS)
# unseen, so that a model which learned "are these letters equal" can be told from one that learned routing for particular
# letters. Values are fresh random digits per item, so no key can be memorised either way.
# 'broad' adds lower-case letters and digits: no letter-specific routing is left to learn, so what a model does with keys it
# has never seen (T-Z) is a statement about whether it compares keys, not about which letters it was shown.
ALPHABETS = dict(default=nd.TRAIN_KEYS, wide=b'ABCDEFGHJKLMNPQRS',
                 broad=b'ABCDEFGHJKLMNPQRS' + b'abcdefghijklmnopqrstuvwxyz' + b'0123456789')


# A filler that puts EVERY upper-case letter in the text without ever putting one in a key: a small model trained only on the facts
# never sees most of T-Z at all, so their embeddings stay at the random values they were created with and "does key matching
# generalise to T-Z" would be answered by construction. (The 30M models trained on language text; their T-Z embeddings are trained.)
PANGRAM = b'THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. '


def _filler(kind, n, rng, corpus):
    if kind == 'pangram':
        return (PANGRAM * (n // len(PANGRAM) + 1))[:n] if n > 0 else b''
    return nd.filler_bytes(kind, n, rng, corpus)


def parse_key_length(text):
    """'2' -> (2, 2); '1-4' -> (1, 4): the key length, or the range one length per example is drawn from."""
    try:
        parts = [int(v) for v in text.split('-')]
    except ValueError:
        raise ValueError('a key length is N or LOW-HIGH')
    if len(parts) == 1:
        parts = parts * 2
    if len(parts) != 2 or not 1 <= parts[0] <= parts[1] <= 8:
        raise ValueError('a key length is N or LOW-HIGH with 1 <= LOW <= HIGH <= 8')
    return parts[0], parts[1]


def example(rng, n_facts, lo, hi, kind='repeat', corpus=None, n_queries=1, key_length=(4, 4), alphabet=None,
            with_replacement=False):
    """One example. With n_facts == 1 the random draws are exactly the original generator's (key, value, filler length).

    With n_queries > 1 the example is MQAR-style: after the facts and the filler come n_queries questions, each about a
    DIFFERENT fact, in random order. That is far more key -> value signal for the same bytes, and "answer the first fact" is
    wrong for most of the questions.

    That design has a shortcut: with 4 facts and 4 questions the last question is answered by ELIMINATION (the one fact not asked
    yet) and the third by choosing between two - a small model scored 1.00 on the last question with no key use at all.
    `with_replacement` draws every asked fact independently, so each question needs the key (and a fact may be asked twice)."""
    if n_queries > n_facts and not with_replacement:
        raise ValueError('cannot ask %d distinct facts out of %d' % (n_queries, n_facts))
    klo, khi = key_length
    key_len = klo if klo == khi else int(rng.integers(klo, khi + 1))         # no draw for a fixed length: the original stream
    alphabet = nd.TRAIN_KEYS if alphabet is None else alphabet
    if len(alphabet) ** key_len < n_facts:
        raise ValueError('%d distinct keys do not exist at length %d over %d letters' % (n_facts, key_len, len(alphabet)))
    keys = []
    while len(keys) < n_facts:
        key = nd._key(rng, length=key_len, alphabet=alphabet)
        if key not in keys:
            keys.append(key)
    values = [nd._value(rng) for _ in keys]
    total = int(rng.integers(lo, hi + 1))
    if n_facts == 1:
        segments = [total]
    else:
        cuts = sorted(int(c) for c in rng.integers(0, total + 1, size=n_facts - 1))
        segments = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    if kind == 'mix':
        kind = 'repeat' if int(rng.integers(0, 2)) == 0 else 'text'       # one coin per example; no draw for the pure kinds
    body = b''
    for key, value, length in zip(keys, values, segments):
        body += b'The code for ' + key + b' is ' + value + b'.\n' + _filler(kind, length, rng, corpus)
    if n_queries == 1:
        asked_all = [int(rng.integers(0, n_facts)) if n_facts > 1 else 0]       # the draw the single-question mix always made
    elif with_replacement:
        asked_all = [int(i) for i in rng.integers(0, n_facts, size=n_queries)]
    else:
        asked_all = [int(i) for i in rng.permutation(n_facts)[:n_queries]]
    text = body
    value_starts = []
    for j in asked_all:
        text += b'\nThe code for ' + keys[j] + b' is '
        value_starts.append(len(text))
        text += values[j]
    return dict(text=text, total=total, asked=asked_all[0], asked_all=asked_all, keys=[k.decode('ascii') for k in keys],
                values=[v.decode('ascii') for v in values], key_length=key_len, value_starts=value_starts)


def decoy_items(n_items, filler_range=(0, 128), seed=0, kind='repeat', corpus=None, pad_to=None, pack=True, n_facts=1,
                n_queries=1, key_length=(4, 4), alphabet=None, with_replacement=False):
    """Same contract as mline.needle.training_items, with n_facts facts and n_queries questions per example."""
    rng = np.random.default_rng(seed + 31_337)
    lo, hi = filler_range
    rows, fillers, texts, starts = [], [], [], []
    for _ in range(n_items):
        blob, used, row_starts = b'', [], []
        while True:
            ex = example(rng, n_facts, lo, hi, kind, corpus, n_queries, key_length, alphabet, with_replacement)
            piece, n = ex['text'], ex['total']
            if blob and pack and pad_to is not None and len(blob) + 1 + len(piece) > pad_to:
                break
            offset = len(blob) + (1 if blob else 0)
            blob = blob + (b'\n' if blob else b'') + piece
            used.append(n)
            row_starts.extend((offset + start, q) for q, start in enumerate(ex['value_starts']))
            if not (pack and pad_to is not None):
                break
        rows.append(blob)
        fillers.append(used)
        texts.append(blob.decode('ascii', errors='ignore'))
        starts.append(row_starts)
    per_row, fillers = fillers, [n for row in fillers for n in row]
    width = max(len(r) for r in rows)
    if pad_to is not None:
        if width > pad_to:
            raise ValueError(f'needle example of {width} bytes does not fit the training width {pad_to}; '
                             'lower filler_range or the number of facts so the answer is never truncated away')
        width = pad_to
    x = np.zeros((n_items, width), dtype=np.int64)
    y = np.zeros((n_items, width), dtype=np.int64)
    answer_starts = []
    for i, row in enumerate(rows):
        pad = width - len(row)
        arr = np.frombuffer((b' ' * pad) + row, dtype=np.uint8).astype(np.int64)
        x[i] = arr
        y[i, :-1] = arr[1:]
        y[i, -1] = arr[-1]
        answer_starts.append([(pad + start, q) for start, q in starts[i]])      # (index of the first answer digit, query number)
    return dict(x=torch.from_numpy(x), y=torch.from_numpy(y), texts=texts, answer_starts=answer_starts, fillers=fillers,
                filler_range=filler_range,
                kind=kind, packed=bool(pack and pad_to is not None), fillers_per_row=per_row,
                examples_per_row=[len(f) for f in per_row], n_facts=n_facts, n_queries=n_queries, key_length=key_length)


@contextlib.contextmanager
def patched(n_facts, kind='repeat', corpus=None, n_queries=1, key_length=(4, 4), alphabet=None, induction_share=0.0,
            induction_shape=None, with_replacement=False):
    """Replace needle.training_items with the decoy generator for the length of the block; always restore it.

    `kind` is 'repeat', 'text' (a slice of `corpus` per filler) or 'mix' (a coin per example); the trainer calls the
    generator without either, so they are fixed here.
    """
    if kind in ('text', 'mix') and corpus is None:
        raise ValueError("filler kind %r needs a corpus" % kind)
    if not 0.0 <= induction_share <= 1.0:
        raise ValueError('induction_share is a fraction of the needle rows, between 0 and 1')
    if induction_share > 0 and induction_shape is None:
        raise ValueError('induction rows need induction_shape = dict(seg=, gap=, lead=, skip=)')
    original = nd.training_items

    def replacement(n_items, filler_range=(0, 128), seed=0, kind_=None, corpus_=None, pad_to=None, pack=True, **_ignored):
        n_induction = int(round(n_items * induction_share))          # the LAST rows of the batch are induction rows
        if n_induction and pad_to is None:
            raise ValueError('induction rows need a row width (pad_to)')
        mqar = decoy_items(n_items - n_induction, filler_range, seed, kind, corpus, pad_to, pack, n_facts=n_facts,
                           n_queries=n_queries, key_length=key_length, alphabet=alphabet,
                           with_replacement=with_replacement) if n_items - n_induction else None
        if not n_induction:
            return mqar
        x, y, _ = ind.items(n_induction, seed + 7919, pad_to, **induction_shape)
        texts = [bytes(row.tolist()).decode('ascii', errors='ignore') for row in x]
        if mqar is None:
            return dict(x=x, y=y, texts=texts, fillers=[], induction_rows=n_induction)
        return dict(mqar, x=torch.cat([mqar['x'], x]), y=torch.cat([mqar['y'], y]), texts=mqar['texts'] + texts,
                    induction_rows=n_induction)

    nd.training_items = replacement
    try:
        yield
    finally:
        nd.training_items = original


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--arm', default='plain')
    parser.add_argument('--init-from', default=None)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--every', type=int, default=250)
    parser.add_argument('--seed', type=int, default=217)
    parser.add_argument('--needle-rate', type=float, default=.25)
    parser.add_argument('--train-filler', default='0,128', help='LOW,HIGH: total filler per example, split between the facts')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--facts', type=int, default=3)
    parser.add_argument('--queries', type=int, default=1,
                        help='questions per example, each about a different fact in random order (MQAR-style when > 1)')
    parser.add_argument('--key-length', default='4',
                        help='N, or LOW-HIGH to draw one key length per example (short keys are easier to match on)')
    parser.add_argument('--key-alphabet', default='default', choices=tuple(ALPHABETS),
                        help="training key letters: 'default' A-M, 'wide' adds N P Q R S (T-Z stay unseen for the probe)")
    parser.add_argument('--with-replacement', action='store_true',
                        help='draw every asked fact independently: removes the elimination shortcut (the last question is otherwise '
                             'answered by the one fact not asked yet)')
    parser.add_argument('--induction-share', type=float, default=0.0,
                        help='fraction of the needle rows that are induction rows (a random string repeated later) instead of MQAR facts')
    parser.add_argument('--seg-length', default='8-32', help='induction rows: length range of the repeated string')
    parser.add_argument('--gap', default='0,96', help='induction rows: filler between the two copies')
    parser.add_argument('--lead', default='0,16', help='induction rows: filler before the first copy')
    parser.add_argument('--skip', type=int, default=2, help='induction rows: bytes of a copy shown before it is scored')
    parser.add_argument('--filler', default='repeat', choices=('repeat', 'text', 'mix', 'pangram'),
                        help='filler content: the repeated phrase, slices of the TRAINING corpus, a coin per example, or a pangram')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    corpus = None
    if args.filler in ('text', 'mix'):
        corpus = md.load(mt.protocol()['data'], 'train')
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, 'decoy_train.json').write_text(json.dumps(dict(facts=args.facts, queries=args.queries,
                                                                  key_length=args.key_length,
                                                                  key_alphabet=args.key_alphabet,
                                                                  induction_share=args.induction_share,
                                                                  with_replacement=args.with_replacement,
                                                                  train_filler=args.train_filler, filler=args.filler,
                                                                  note='needle.training_items replaced for this run by '
                                                                       'mline.decoy_train.decoy_items'), indent=1),
                                                  encoding='utf8')
    shape = dict(seg=ind.parse_span(args.seg_length), gap=em.parse_range(args.gap), lead=em.parse_range(args.lead),
                 skip=args.skip) if args.induction_share > 0 else None
    with patched(args.facts, args.filler, corpus, args.queries, parse_key_length(args.key_length),
                 ALPHABETS[args.key_alphabet], args.induction_share, shape, args.with_replacement):
        em.run(args.out, arm=args.arm, steps=args.steps, batch=args.batch, every=args.every, device=args.device,
               seed=args.seed, needle_rate=args.needle_rate, train_filler=em.parse_range(args.train_filler),
               pack=True, lr=args.lr, init_from=args.init_from)


if __name__ == '__main__':
    main()
