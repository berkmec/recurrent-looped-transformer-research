"""Scoring an MQAR model on the distribution it was trained on, teacher-forced, per question number.

mline.decoy_probe lays out two facts and one question - a diagnostic of WHICH fact comes out, not a measure of how well the task
was learned: a model that has learned the training task (four facts, four questions, the filler split between them) can still
read as chance there. This draws rows from the very generator the model was trained on (mline.decoy_train.decoy_items, a
held-out seed), and scores each answer: the first digit, and all four.
"""
import re

import numpy as np
import torch

from mline import decoy_probe as dp
from mline import decoy_train as dt
from mline import lm as ml


def mqar_accuracy(model, alphabet, key_length, facts, queries, filler, pad_to, n_items=64, batch=16, seed=777, device='cpu',
                  with_replacement=False, filler_kind='repeat', key_source=None):
    """{'first': [share of first digits right, per question number], 'exact': [...four digits right...], 'first_all', 'exact_all',
    'answers': how many answers were scored}. The digits fed in are the true ones (teacher forcing); chance for a first digit is 0.1."""
    items = dt.decoy_items(n_items, filler, seed, filler_kind, None, pad_to, True, facts, queries, key_length, alphabet,
                           with_replacement, key_source)
    x, starts = items['x'], items['answer_starts']
    first = [[0, 0] for _ in range(queries)]                  # [right, total] per question number
    exact = [[0, 0] for _ in range(queries)]
    was_training = model.training
    model.eval()
    try:
        for lo in range(0, n_items, batch):
            with torch.no_grad():
                pred = ml.forward_logits(model, x[lo:lo + batch].to(device)).argmax(-1).cpu()
            for r in range(pred.shape[0]):
                for position, q in starts[lo + r]:
                    right = [int(pred[r, position - 1 + d]) == int(x[lo + r, position + d]) for d in range(dp.VALUE_LENGTH)]
                    first[q][0] += right[0]
                    first[q][1] += 1
                    exact[q][0] += all(right)
                    exact[q][1] += 1
    finally:
        model.train(was_training)

    def share(cells):
        return [hit / total if total else float('nan') for hit, total in cells]

    total = sum(t for _, t in first)
    return dict(first=share(first), exact=share(exact), first_all=sum(h for h, _ in first) / total,
                exact_all=sum(h for h, _ in exact) / total, answers=total)


def natural_key_sets(train_corpus, val_corpus, n_train=2000, n_val=300, seed=7, sample_bytes=2_000_000):
    """(training keys, held-out keys): real words and names from the training text for a run to key on, and from the validation
    text for it to be scored on - minus every key the training list holds, so no scored key was ever a training key."""
    train = natural_keys(train_corpus, n_train, seed, sample_bytes)
    seen = set(train)
    pool = natural_keys(val_corpus, n_val * 3, seed + 1, sample_bytes)
    return train, [k for k in pool if k not in seen][:n_val]


def natural_keys(corpus, n=300, seed=0, sample_bytes=2_000_000):
    """Real keys pulled from a text corpus: half capitalised names (one or two words), half lower-case words, deterministic.

    Whole-word keys of 4-24 bytes, none of them a random string: the closest the generator gets to the headings a memory would be
    asked about. Draw them from a split the model never trained on."""
    raw = corpus[:sample_bytes]
    text = (raw if isinstance(raw, (bytes, bytearray)) else np.asarray(raw, dtype=np.uint8).tobytes()).decode('ascii', errors='ignore')
    names = list(dict.fromkeys(re.findall(r'[A-Z][a-z]{3,10}(?: [A-Z][a-z]{3,10})?', text)))
    words = list(dict.fromkeys(re.findall(r'\b[a-z]{5,11}\b', text)))
    rng = np.random.default_rng(seed)
    n_names = min(len(names), n // 2)
    picked = [names[int(i)] for i in rng.choice(len(names), size=n_names, replace=False)]
    n_words = min(len(words), n - n_names)
    picked += [words[int(i)] for i in rng.choice(len(words), size=n_words, replace=False)]
    order = rng.permutation(len(picked))
    return [picked[int(i)].encode('ascii') for i in order]


def extra_eval(alphabet, facts, queries, filler, pad_to, n_items=64, key_lengths=(1, 4), seed=777, with_replacement=False,
               filler_kind='repeat'):
    """model -> {'train_k1_first': ..., 'held_k4_exact': ...}: the training-shaped score on the training letters and on letters
    (T-Z) that no run ever trains on, at each key length. Averaged over the questions."""
    held = dp.ALPHABETS['HELD_KEYS']

    def fn(model):
        out = {}
        for tag, letters in (('train', alphabet), ('held', held)):
            for k in key_lengths:
                r = mqar_accuracy(model, letters, (k, k), facts, queries, filler, pad_to, n_items, seed=seed,
                                  with_replacement=with_replacement, filler_kind=filler_kind)
                out['%s_k%d_first' % (tag, k)] = float(r['first_all'])
                out['%s_k%d_exact' % (tag, k)] = float(r['exact_all'])
        return out

    return fn


def main():
    """Score a checkpoint on training-shaped MQAR rows: python -B -m mline.mqar_eval --checkpoint <pt> [--width 128 --heads 4]."""
    import argparse
    import json
    import sys
    from pathlib import Path

    from mline import emergence as em
    from mline import mqar_scratch as ms          # imported here: mqar_scratch imports this module
    from mline import train as mt
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default=None, help='also write the numbers here as JSON')
    parser.add_argument('--width', type=int, default=None, help='a small Core of this width instead of the 30M protocol model')
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--feedback', choices=('on', 'off'), default='on')
    parser.add_argument('--alphabet', choices=tuple(dt.ALPHABETS), default='default', help='the training letters to score on')
    parser.add_argument('--key-lengths', default='4')
    parser.add_argument('--facts', type=int, default=4)
    parser.add_argument('--queries', type=int, default=4)
    parser.add_argument('--with-replacement', action='store_true', help='score rows whose questions are drawn with replacement')
    parser.add_argument('--filler-kind', default='repeat', choices=('repeat', 'pangram'), help='the filler the rows are drawn with')
    parser.add_argument('--natural-keys', type=int, default=0,
                        help='also score on N real words and names from the VALIDATION text, minus every key of the 2000-key training '
                             'list a run can key on (use --pad-to 512: keys run to 24 bytes)')
    parser.add_argument('--filler', default='0,64')
    parser.add_argument('--pad-to', type=int, default=320)
    parser.add_argument('--items', type=int, default=32)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
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
    filler = em.parse_range(args.filler)
    report = {}
    for tag, letters in (('train', dt.ALPHABETS[args.alphabet]), ('held', dp.ALPHABETS['HELD_KEYS'])):
        for k in (int(v) for v in args.key_lengths.split(',')):
            r = mqar_accuracy(model, letters, (k, k), args.facts, args.queries, filler, args.pad_to, args.items,
                              args.batch, device=args.device, with_replacement=args.with_replacement,
                              filler_kind=args.filler_kind)
            report['%s_k%d' % (tag, k)] = r
            print('%-5s key %d  first digit per question %s | exact %s | all: first %.2f exact %.2f (%d answers)'
                  % (tag, k, ' '.join('%.2f' % v for v in r['first']), ' '.join('%.2f' % v for v in r['exact']),
                     r['first_all'], r['exact_all'], r['answers']), flush=True)
    if args.natural_keys:
        from mline import data as md
        _, keys = natural_key_sets(md.load(mt.protocol()['data'], 'train'), md.load(mt.protocol()['data'], 'validation'),
                                   n_train=2000, n_val=args.natural_keys, seed=7)
        r = mqar_accuracy(model, None, (4, 4), args.facts, args.queries, filler, args.pad_to, args.items, args.batch,
                          device=args.device, with_replacement=args.with_replacement, filler_kind=args.filler_kind, key_source=keys)
        r['sample_keys'] = [k.decode('ascii') for k in keys[:12]]
        report['natural'] = r
        print('natural (%d real words/names)  first digit per question %s | exact %s | all: first %.2f exact %.2f (%d answers)'
              % (len(keys), ' '.join('%.2f' % v for v in r['first']), ' '.join('%.2f' % v for v in r['exact']),
                 r['first_all'], r['exact_all'], r['answers']), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(dict(checkpoint=str(args.checkpoint), args=vars(args), report=report), indent=1),
                                  encoding='utf8')


if __name__ == '__main__':
    main()
