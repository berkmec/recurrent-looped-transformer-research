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
# Split so that an evaluation key CANNOT appear in training. A structural guarantee beats a probabilistic one:
# the zero-shot design this replaces failed because the format was never taught, and the fix - teaching the format -
# would leak the answer keys too unless the spaces are disjoint by construction.
TRAIN_KEYS = b'ABCDEFGHJKLM'
EVAL_KEYS = b'NPQRSTUVWXYZ'
KEY_ALPHABET = EVAL_KEYS
VALUE_ALPHABET = b'0123456789'
KIND = ('repeat', 'text')


def _key(rng, length=4, alphabet=None):
    alphabet = alphabet if alphabet is not None else EVAL_KEYS
    return bytes(alphabet[i] for i in rng.integers(0, len(alphabet), size=length))


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


PLACEMENT = ('far', 'near')


def make(n_items, n_filler, kind='repeat', seed=0, corpus=None, placement='far'):
    """Build an evaluation set. Returns padded inputs, the answer span mask and the answers themselves.

    `placement` selects which of two conditions of IDENTICAL sequence length is built:

      far  (the measurement) fact, filler, question  - the fact-to-question distance is `n_filler`
      near (the control)     filler, fact, question  - the distance is zero, the length is unchanged

    The control exists because M1's two decisive evaluation points are longer than the model's training length:
    565 bytes at filler 512 (1.10x of 512) and 2101 at filler 2048 (4.10x). Without it a zero at `far` confounds
    "cannot carry information across the distance" with "cannot process a sequence this long at any distance".
    With it the two separate: failing `far` while passing `near` is a memory result; failing both is a length
    result and says nothing about the gate. The same seed draws the same keys, values and filler for both, so the
    conditions are paired and differ in exactly one thing - where the fact sits.
    """
    if kind not in KIND:
        raise ValueError('kind must be one of ' + ', '.join(KIND))
    if placement not in PLACEMENT:
        raise ValueError('placement must be one of ' + ', '.join(PLACEMENT))
    rng = np.random.default_rng(seed)
    rows, answers, keys, distances = [], [], [], []
    seen = set()
    for _ in range(n_items):
        while True:
            key = _key(rng, alphabet=EVAL_KEYS)
            if key not in seen:
                seen.add(key); break
        value = _value(rng)
        prefix = b'The code for ' + key + b' is ' + value + b'.\n'
        middle = filler_bytes(kind, n_filler, rng, corpus)
        query = b'\nThe code for ' + key + b' is '
        rows.append(prefix + middle + query if placement == 'far' else middle + prefix + query)
        distances.append(len(middle) if placement == 'far' else 0)
        answers.append(value); keys.append(key.decode('ascii'))
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
                answers=[a.decode('ascii') for a in answers], answers_keys=keys,
                n_filler=n_filler, kind=kind, width=width, placement=placement, distances=distances)


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


def training_items(n_items, filler_range=(0, 128), seed=0, kind='repeat', corpus=None, pad_to=None):
    """Needle examples to MIX INTO TRAINING, so the format is learned and only the DISTANCE has to generalise.

    This is the U1 shape applied to language: U1 trained at horizon 128 and tested to 4096. Here training sees the
    format with short fillers and the evaluation uses fillers it never saw. Keys come from TRAIN_KEYS, which is
    disjoint from the EVAL_KEYS `make` draws on, so no evaluation key can be memorised.

    Returns full next-byte targets over the whole sequence, because the model has to learn the format as language,
    not just the answer span.
    """
    rng = np.random.default_rng(seed + 31_337)
    lo, hi = filler_range
    rows, fillers, texts = [], [], []
    for _ in range(n_items):
        key = _key(rng, alphabet=TRAIN_KEYS)
        value = _value(rng)
        n = int(rng.integers(lo, hi + 1))
        blob = (b'The code for ' + key + b' is ' + value + b'.\n'
                + filler_bytes(kind, n, rng, corpus)
                + b'\nThe code for ' + key + b' is ' + value)
        rows.append(blob); fillers.append(n); texts.append(blob.decode('ascii', errors='ignore'))
    width = max(len(r) for r in rows)
    if pad_to is not None:
        if width > pad_to:
            raise ValueError(f'needle example of {width} bytes does not fit the training width {pad_to}; '
                             'lower filler_range so the answer is never truncated away')
        width = pad_to
    x = np.zeros((n_items, width), dtype=np.int64)
    y = np.zeros((n_items, width), dtype=np.int64)
    for i, row in enumerate(rows):
        seq = (b' ' * (width - len(row))) + row
        arr = np.frombuffer(seq, dtype=np.uint8).astype(np.int64)
        x[i] = arr
        y[i, :-1] = arr[1:]; y[i, -1] = arr[-1]
    return dict(x=torch.from_numpy(x), y=torch.from_numpy(y), texts=texts, fillers=fillers,
                filler_range=filler_range, kind=kind)
