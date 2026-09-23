"""M-hattı M0: the language analogue of the campaign's neutral-run test.

Every toy experiment had the same shape - information first, a long stretch that does not change the answer, then the
question - and measured how much of the answer survived the stretch. In language that is a needle through filler:

    The code for GORM is 7431.
    <N bytes of filler>
    The code for GORM is |7431

Two filler kinds, because they are not the same question:
  `repeat` - a low-information phrase repeated, the direct analogue of the toy neutral token (an identity operation);
  `text`   - real corpus prose, which is distraction rather than emptiness.

Scoring is teacher-forced on the answer bytes only: exact match over all answer bytes, plus per-byte accuracy so a
near miss is visible. Deterministic given the seed; the keys and values never repeat within an evaluation set.
"""
import numpy as np
import torch

FILLER_PHRASE = b'and so it went on and on. '
KEY_ALPHABET = b'ABCDEFGHJKLMNPQRSTUVWXYZ'
VALUE_ALPHABET = b'0123456789'
KIND = ('repeat', 'text')


def _key(rng, length=4):
    return bytes(KEY_ALPHABET[i] for i in rng.integers(0, len(KEY_ALPHABET), size=length))


def _value(rng, length=4):
    return bytes(VALUE_ALPHABET[i] for i in rng.integers(0, len(VALUE_ALPHABET), size=length))


def filler_bytes(kind, n, rng, corpus=None):
    if n <= 0:
        return b''
    if kind == 'repeat':
        reps = n // len(FILLER_PHRASE) + 1
        return (FILLER_PHRASE * reps)[:n]
    if corpus is None:
        raise ValueError("filler kind 'text' needs a corpus")
    start = int(rng.integers(0, len(corpus) - n - 1))
    return bytes(np.asarray(corpus[start:start + n], dtype=np.uint8).tobytes())


def make(n_items, n_filler, kind='repeat', seed=0, corpus=None):
    """Build an evaluation set. Returns padded inputs, the answer span mask and the answers themselves."""
    if kind not in KIND:
        raise ValueError('kind must be one of ' + ', '.join(KIND))
    rng = np.random.default_rng(seed)
    rows, answers = [], []
    seen = set()
    for _ in range(n_items):
        while True:
            key = _key(rng)
            if key not in seen:
                seen.add(key); break
        value = _value(rng)
        prefix = b'The code for ' + key + b' is ' + value + b'.\n'
        middle = filler_bytes(kind, n_filler, rng, corpus)
        query = b'\nThe code for ' + key + b' is '
        rows.append(prefix + middle + query)
        answers.append(value)
    width = max(len(r) for r in rows) + len(answers[0])
    x = np.zeros((n_items, width), dtype=np.int64)
    mask = np.zeros((n_items, width), dtype=bool)
    y = np.zeros((n_items, width), dtype=np.int64)
    for i, (row, value) in enumerate(zip(rows, answers)):
        full = row + value
        pad = width - len(full)
        seq = (b' ' * pad) + full                      # left padding keeps the answer at the very end for every row
        arr = np.frombuffer(seq, dtype=np.uint8).astype(np.int64)
        x[i] = arr
        y[i, :-1] = arr[1:]; y[i, -1] = arr[-1]
        mask[i, width - len(value) - 1:width - 1] = True   # positions whose NEXT byte is an answer byte
    return dict(x=torch.from_numpy(x), y=torch.from_numpy(y), mask=torch.from_numpy(mask),
                answers=[a.decode('ascii') for a in answers], n_filler=n_filler, kind=kind, width=width)


@torch.no_grad()
def score(forward, items, batch=8, device='cpu'):
    """`forward` maps a byte tensor to logits. Teacher-forced: the answer bytes are scored, nothing else is."""
    x, y, mask = items['x'], items['y'], items['mask']
    hits, total, exact = 0, 0, 0
    for at in range(0, len(x), batch):
        xb = x[at:at + batch].to(device)
        logits = forward(xb)
        pred = logits.argmax(-1).cpu()
        mb, yb = mask[at:at + batch], y[at:at + batch]
        correct = (pred == yb) & mb
        hits += int(correct.sum()); total += int(mb.sum())
        exact += int(((correct.sum(1)) == mb.sum(1)).sum())
    return dict(exact_match=exact / len(x), byte_accuracy=hits / max(total, 1), items=len(x),
                n_filler=items['n_filler'], kind=items['kind'], width=items['width'])
