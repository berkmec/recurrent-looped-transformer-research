"""Cross-window fact rows (NAW) for RLT-M's stage 1 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md sections 5.40, 5.61).

L1's own fact and question format (mline.decoy_train.example: b'The code for KEY is VALUE.' and b'\\nThe code for KEY is ' + VALUE),
with every fact in window 0 and the question d windows later, each at a random place inside natural training text. A model whose
windows are independent can only guess the four digits; one that carries the fact across the window boundaries can answer. With
more than one fact (MQAR-style) or a decoy (a key one letter away with another value) the key has to be used: "the only fact in
memory" is not enough. The value never occurs in the row outside its fact and its answer (a row that would contain it is drawn
again), so the answer cannot be copied from the question's window (G0.5 of note 5.40).
"""
import numpy as np
import torch

from mline import needle as nd

FACT = b'The code for %s is %s.\n'
QUESTION = b'\nThe code for %s is '


def _keys(rng, n, alphabet, key_source):
    if key_source is not None:
        return [key_source[int(i)] for i in rng.choice(len(key_source), size=n, replace=False)]
    keys = []
    while len(keys) < n:
        k = nd._key(rng, alphabet=alphabet)
        if k not in keys:
            keys.append(k)
    return keys


def _similar(rng, key, alphabet, taken):
    """A key one letter away from `key` and different from every key in `taken` (the decoy)."""
    while True:
        i = int(rng.integers(0, len(key)))
        others = [c for c in alphabet if c != key[i]]
        k = key[:i] + bytes([others[int(rng.integers(0, len(others)))]]) + key[i + 1:]
        if k not in taken:
            return k


def naw_rows(n_rows, distances, seed, corpus, windows, window=512, n_facts=1, decoy_share=0.0, alphabet=nd.TRAIN_KEYS,
             key_source=None, max_tries=50):
    """dict(x, y: (n_rows, windows * window) int64 next-byte pairs; first, four: bool masks on the TARGET positions of the answer's
    first digit and of its four digits (mline.longrun.synthetic_batch's convention); d: the distance in windows, per row)."""
    rng = np.random.default_rng(seed)
    length = windows * window
    xs, ys, firsts, fours, ds = [], [], [], [], []
    for _ in range(n_rows):
        d = int(distances[int(rng.integers(0, len(distances)))])
        if not 1 <= d < windows:
            raise ValueError('distance %d does not fit in %d windows' % (d, windows))
        for _attempt in range(max_tries):
            keys = _keys(rng, n_facts, alphabet, key_source)
            values = [nd._value(rng) for _ in keys]
            facts = [FACT % (k, v) for k, v in zip(keys, values)]
            if key_source is None and decoy_share > 0 and rng.random() < decoy_share:
                facts.append(FACT % (_similar(rng, keys[0], alphabet, keys), nd._value(rng)))
            asked = int(rng.integers(0, n_facts))
            qa = QUESTION % keys[asked] + values[asked]
            start = int(rng.integers(0, len(corpus) - length - 1))
            row = bytearray(np.asarray(corpus[start:start + length + 1], dtype=np.uint8).tobytes())
            slot = max(len(f) for f in facts) + 1
            if window // slot < len(facts) or len(qa) > window:
                raise ValueError('%d facts of %d bytes do not fit one window of %d' % (len(facts), slot, window))
            for f, s in zip([facts[i] for i in rng.permutation(len(facts))],
                            sorted(int(v) for v in rng.choice(window // slot, size=len(facts), replace=False))):
                at = s * slot + int(rng.integers(0, slot - len(f) + 1))
                row[at:at + len(f)] = f
            q_at = d * window + int(rng.integers(0, window - len(qa) + 1))
            row[q_at:q_at + len(qa)] = qa
            if bytes(row).count(values[asked]) == 2:          # its fact and its answer, nowhere else
                break
        else:
            raise RuntimeError('no row without the value elsewhere in %d draws' % max_tries)
        a = q_at + len(qa) - len(values[asked])               # the row index of the answer's first digit
        flat = torch.tensor(list(row), dtype=torch.long)
        first, four = torch.zeros(length, dtype=torch.bool), torch.zeros(length, dtype=torch.bool)
        first[a - 1] = True
        four[a - 1:a + len(values[asked]) - 1] = True
        xs.append(flat[:-1]), ys.append(flat[1:]), firsts.append(first), fours.append(four), ds.append(d)
    return dict(x=torch.stack(xs), y=torch.stack(ys), first=torch.stack(firsts), four=torch.stack(fours), d=torch.tensor(ds))
