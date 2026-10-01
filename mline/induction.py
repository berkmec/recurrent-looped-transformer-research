"""Induction rows: a random string, some filler, then the SAME string again.

Predicting the second copy needs the circuit key-addressed retrieval needs - match what came just before against an earlier
occurrence, then copy what followed it - and nothing in these rows can be solved by position: the copy offset is random and the
string is fresh each time. The single-fact needle task is solved by position (T1s, the 30M checkpoints), so a model trained on it
never has to build that circuit; this is the data that does demand it, with no key/value template to shortcut on.

`items` packs examples into rows the way mline.needle.training_items does and returns a mask over the bytes the circuit is
responsible for: the second copy after its first `skip` bytes (a copy can only be recognised once some of it has been seen).
`copy_accuracy` is the score on exactly those bytes.
"""
import numpy as np
import torch

from mline import lm as ml
from mline import needle as nd

# 0-9, A-Z, a-z: wide enough that no letter set can be memorised, and it contains every needle key letter
ALPHABET = bytes(range(48, 58)) + bytes(range(65, 91)) + bytes(range(97, 123))


def example(rng, seg=(8, 32), gap=(0, 96), lead=(0, 16), alphabet=ALPHABET):
    """lead filler, a random string, gap filler, the same string. `copy_start` is where the second copy begins."""
    length = int(rng.integers(seg[0], seg[1] + 1))
    gap_length = int(rng.integers(gap[0], gap[1] + 1))
    lead_length = int(rng.integers(lead[0], lead[1] + 1))
    segment = bytes(alphabet[i] for i in rng.integers(0, len(alphabet), size=length))
    head = nd.filler_bytes('repeat', lead_length, rng) + segment + nd.filler_bytes('repeat', gap_length, rng)
    return dict(text=head + segment, copy_start=len(head), length=length, gap=gap_length, lead=lead_length, segment=segment)


def items(n_items, seed, pad_to, seg=(8, 32), gap=(0, 96), lead=(0, 16), skip=2, alphabet=ALPHABET):
    """(x, y, mask): rows of `pad_to` bytes, left-padded, packed with as many examples as fit; y is the next byte of x."""
    rng = np.random.default_rng(seed + 424_243)
    rows, marks = [], []
    for _ in range(n_items):
        blob, copies = b'', []
        while True:
            ex = example(rng, seg, gap, lead, alphabet)
            piece = ex['text']
            if len(piece) > pad_to:
                raise ValueError('an induction example of %d bytes does not fit the row width %d; lower seg/gap/lead'
                                 % (len(piece), pad_to))
            if blob and len(blob) + 1 + len(piece) > pad_to:
                break
            offset = len(blob) + (1 if blob else 0)
            blob = blob + (b'\n' if blob else b'') + piece
            copies.append((offset + ex['copy_start'], ex['length']))
        rows.append(blob)
        marks.append(copies)
    x = np.zeros((n_items, pad_to), dtype=np.int64)
    y = np.zeros((n_items, pad_to), dtype=np.int64)
    mask = np.zeros((n_items, pad_to), dtype=bool)
    for i, (row, copies) in enumerate(zip(rows, marks)):
        pad = pad_to - len(row)
        arr = np.frombuffer(b' ' * pad + row, dtype=np.uint8).astype(np.int64)
        x[i] = arr
        y[i, :-1] = arr[1:]
        y[i, -1] = arr[-1]
        for start, length in copies:
            for m in range(skip, length):                 # byte m of the copy is predicted at the position before it
                mask[i, pad + start + m - 1] = True
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def batch_fn(batch, seed, pad_to, seg=(8, 32), gap=(0, 96), lead=(0, 16), skip=2, alphabet=ALPHABET):
    """step -> (x, y): a pure function of (seed, step), so a resumed run is the same run."""
    def fn(step):
        x, y, _ = items(batch, seed * 100_003 + step, pad_to, seg, gap, lead, skip, alphabet)
        return x, y
    return fn


def copy_accuracy(model, n_items=64, batch=16, seed=777, pad_to=320, seg=(8, 32), gap=(0, 96), lead=(0, 16), skip=2,
                  alphabet=ALPHABET, device='cpu'):
    """Share of the masked second-copy bytes the model predicts exactly (chance is about 1/62)."""
    n = n_items
    x, y, mask = items(n, seed, pad_to, seg, gap, lead, skip, alphabet)
    was_training = model.training
    model.eval()
    hit = total = 0
    try:
        for start in range(0, n, batch):
            with torch.no_grad():
                pred = ml.forward_logits(model, x[start:start + batch].to(device)).argmax(-1).cpu()
            m = mask[start:start + batch]
            hit += int(((pred == y[start:start + batch]) & m).sum())
            total += int(m.sum())
    finally:
        model.train(was_training)
    return hit / total if total else float('nan')
