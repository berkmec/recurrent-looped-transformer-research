"""Scoring an MQAR model on the distribution it was trained on, teacher-forced, per question number.

mline.decoy_probe lays out two facts and one question - a diagnostic of WHICH fact comes out, not a measure of how well the task
was learned: a model that has learned the training task (four facts, four questions, the filler split between them) can still
read as chance there. This draws rows from the very generator the model was trained on (mline.decoy_train.decoy_items, a
held-out seed), and scores each answer: the first digit, and all four.
"""
import torch

from mline import decoy_probe as dp
from mline import decoy_train as dt
from mline import lm as ml


def mqar_accuracy(model, alphabet, key_length, facts, queries, filler, pad_to, n_items=64, batch=16, seed=777, device='cpu'):
    """{'first': [share of first digits right, per question number], 'exact': [...four digits right...], 'first_all', 'exact_all',
    'answers': how many answers were scored}. The digits fed in are the true ones (teacher forcing); chance for a first digit is 0.1."""
    items = dt.decoy_items(n_items, filler, seed, 'repeat', None, pad_to, True, facts, queries, key_length, alphabet)
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


def extra_eval(alphabet, facts, queries, filler, pad_to, n_items=64, key_lengths=(1, 4), seed=777):
    """model -> {'train_k1_first': ..., 'held_k4_exact': ...}: the training-shaped score on the training letters and on letters
    (T-Z) that no run ever trains on, at each key length. Averaged over the questions."""
    held = dp.ALPHABETS['HELD_KEYS']

    def fn(model):
        out = {}
        for tag, letters in (('train', alphabet), ('held', held)):
            for k in key_lengths:
                r = mqar_accuracy(model, letters, (k, k), facts, queries, filler, pad_to, n_items, seed=seed)
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
                              args.batch, device=args.device)
            report['%s_k%d' % (tag, k)] = r
            print('%-5s key %d  first digit per question %s | exact %s | all: first %.2f exact %.2f (%d answers)'
                  % (tag, k, ' '.join('%.2f' % v for v in r['first']), ' '.join('%.2f' % v for v in r['exact']),
                     r['first_all'], r['exact_all'], r['answers']), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(dict(checkpoint=str(args.checkpoint), args=vars(args), report=report), indent=1),
                                  encoding='utf8')


if __name__ == '__main__':
    main()
