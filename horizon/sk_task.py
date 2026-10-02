"""SK: state-keyed retrieval, the task that needs BOTH halves of what an RLT-style model claims to have.

A recurrent state has to be tracked exactly through a long walk in a group, and the walk's current state is the KEY of a lookup into facts that are only in the
sequence prefix (an attention memory). A GRU / LSTM has no memory except its state (24 random 3-bit facts have to share it with the tracked state), a constant-depth
transformer has the memory but cannot track a long walk, the Core has both (recurrent state + cross-attention to the encoder memory): if its structure is worth
anything, it should show here.

Layout of one sequence (S_n, G = |S_n| = 6 or 24 group elements, V = 8 values):
  G facts [key_g, value_g], one per group element in a random order, values redrawn for every sequence;
  then a walk of L group-element tokens with query tokens '?' inserted after some of them (always after the last one; when L = 0 right after the facts).
Two heads share the readout (classes = G + V): the STATE head (the first G logits) is supervised at every walk and query position with the running product; the VALUE
head (the last V logits) is supervised only at a '?' with value_g for g = the CURRENT state. Everything else is ignored (-100). Dense state supervision keeps the
tracking learnable for every arm; what is left to find out is whether the arm can use its tracked state as a key into the prefix.

Token ids: walk g -> g, key g -> G + g, value v -> 2G + v, query -> 2G + V (vocabulary 2G + V + 1).
"""
import numpy as np
import torch

from . import depth_tasks as dt

N_VALUES = dt.SK_VALUES
IGNORE = -100
EDGES = (0, 4, 16, 32, 64, 128, 256, 512)       # cumulative walk tokens consumed: the buckets accuracy is reported in


def size(n):
    return len(dt.group(n)[0])


def vocab_size(n):
    return 2 * size(n) + N_VALUES + 1


def query_token(n):
    return 2 * size(n) + N_VALUES


def generate(n, count, walk, seed, queries=None, every=None):
    """dict(x, value, state, walk): `count` sequences with a walk of `walk` tokens. `queries` = number of queries at random walk positions (always including the last
    position; 1 when walk = 0); `every` = a query after every `every`-th walk token and after the last one (the evaluation layout). value[i, t] = the answer at a query
    position, state[i, t] = the running product at a walk or query position, walk[i, t] = walk tokens consumed up to position t (-1 inside the facts). All sequences of
    a call have the same length."""
    if walk < 0 or count < 1:
        raise ValueError('invalid budget')
    g = size(n)
    table = dt.group(n)[2]
    rng = np.random.default_rng(seed)
    if every:
        positions = [sorted(set(range(every, walk + 1, every)) | ({walk} if walk else {0})) for _ in range(count)]
    else:
        q = 1 if walk == 0 else max(1, min(queries or 1, walk))
        positions = []
        for _ in range(count):
            chosen = set(rng.choice(np.arange(1, walk), size=q - 1, replace=False).tolist()) if walk > 1 and q > 1 else set()
            positions.append(sorted(chosen | ({walk} if walk else {0})))
    lengths = {len(p) for p in positions}
    if len(lengths) != 1:
        raise ValueError('every sequence of a call must have the same number of queries')
    total = 2 * g + walk + lengths.pop()
    x = np.zeros((count, total), dtype=np.int64)
    value = np.full((count, total), IGNORE, dtype=np.int64)
    state_y = np.full((count, total), IGNORE, dtype=np.int64)
    walked = np.full((count, total), -1, dtype=np.int64)
    qtok = query_token(n)
    for i in range(count):
        values = rng.integers(0, N_VALUES, size=g)
        order = rng.permutation(g)
        row = []
        for key in order:
            row += [g + int(key), 2 * g + int(values[key])]
        steps = rng.integers(0, g, size=walk)
        state, ahead = 0, set(positions[i])
        if 0 in ahead:                                   # walk == 0: the query follows the facts directly
            row.append(qtok)
            value[i, len(row) - 1], state_y[i, len(row) - 1], walked[i, len(row) - 1] = values[state], state, 0
        for t in range(walk):
            row.append(int(steps[t]))
            state = int(table[state, steps[t]])
            state_y[i, len(row) - 1], walked[i, len(row) - 1] = state, t + 1
            if (t + 1) in ahead:
                row.append(qtok)
                value[i, len(row) - 1], state_y[i, len(row) - 1], walked[i, len(row) - 1] = values[state], state, t + 1
        x[i] = row
    return dict(x=torch.from_numpy(x), value=torch.from_numpy(value), state=torch.from_numpy(state_y), walk=torch.from_numpy(walked))


def bucket(value):
    """The EDGES bucket label of a walk length: '0-3', '4-15', ..."""
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        if lo <= value < hi:
            return f'{lo}-{hi - 1}'
    return f'{EDGES[-1]}+'


def _accuracy(pred, target, walk, mask):
    out = {'all': [int((pred[mask] == target[mask]).sum()), int(mask.sum())]}
    for w, ok in zip(walk[mask].tolist(), (pred[mask] == target[mask]).tolist()):
        cell = out.setdefault(bucket(w), [0, 0])
        cell[0] += int(ok)
        cell[1] += 1
    return out


def value_accuracy(pred, value, walk):
    """{bucket: [correct, queries]} at the query positions (value head), plus the overall [correct, queries] under 'all'."""
    return _accuracy(pred, value, walk, value != IGNORE)


def state_accuracy(pred, state, walk, query_mask=None):
    """{bucket: [correct, positions]} of the state head at every supervised position (walk and query positions), plus 'all'."""
    return _accuracy(pred, state, walk, state != IGNORE)
