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
