"""Z1 depth-extrapolation task families.

Each task has a depth knob, so the amount of sequential computation can be pushed far past what was trained:

  perm3 / perm4 / perm5   composition in the symmetric group S_N. Every token names one permutation, the label at each position is the
                          index of the running product. S_5 has no constant-depth solution; a finite state (120) has to be carried
                          exactly through the whole sequence. Depth = sequence length.
  count                   a clipped counter: token 0 pushes (+1), token 1 pops (-1, floored at 0); the label is min(depth, CAP).
                          The depth that must be held is a free parameter (the peak D), unbounded in principle.

Training uses short sequences (perm: length <= TRAIN_LENGTH, count: peak <= TRAIN_DEPTH); the claim under test is how far beyond
that the learned recurrence still works.
"""
import hashlib
import itertools

import numpy as np
import torch

CAP = 255
TRAIN_DEPTH = 8
TRAIN_LENGTH = 16
TASKS = {'perm3': dict(kind='perm', n=3), 'perm4': dict(kind='perm', n=4), 'perm5': dict(kind='perm', n=5), 'count': dict(kind='count')}
_GROUPS = {}


def _build_group(n):
    perms = list(itertools.permutations(range(n)))               # perms[0] is the identity
    index = {p: i for i, p in enumerate(perms)}
    table = np.zeros((len(perms), len(perms)), dtype=np.int64)
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            table[i, j] = index[tuple(a[b[k]] for k in range(n))]   # table[a, b] = a o b
    return perms, index, table


def group(n):
    if n not in _GROUPS:
        _GROUPS[n] = _build_group(n)
    return _GROUPS[n]


def _spec(task):
    if task not in TASKS:
        raise ValueError('unknown task: ' + str(task))
    return TASKS[task]


def vocab(task):
    s = _spec(task)
    return len(group(s['n'])[0]) if s['kind'] == 'perm' else 2


def classes(task):
    s = _spec(task)
    return len(group(s['n'])[0]) if s['kind'] == 'perm' else CAP + 1


def oracle(task, x):
    """Per-position labels for the token matrix x (n, length)."""
    s = _spec(task)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError('invalid token matrix')
    arr = x.numpy()
    y = np.zeros_like(arr)
    if s['kind'] == 'perm':
        table = group(s['n'])[2]
        state = np.zeros(arr.shape[0], dtype=np.int64)
        for t in range(arr.shape[1]):
            state = table[state, arr[:, t]]
            y[:, t] = state
    else:
        depth = np.zeros(arr.shape[0], dtype=np.int64)
        for t in range(arr.shape[1]):
            depth = np.maximum(depth + np.where(arr[:, t] == 0, 1, -1), 0)
            y[:, t] = np.minimum(depth, CAP)
    return torch.from_numpy(y)


def _count_tokens(rng, n, length, depth):
    """Push/pop tokens whose true depth never exceeds the peak. With depth=None (training) the climb length is random up to
    TRAIN_DEPTH; with a given depth every sequence first climbs to that peak, then wanders below it (reflecting at the peak)."""
    peak = TRAIN_DEPTH if depth is None else int(depth)
    ramp = rng.integers(0, peak + 1, size=n) if depth is None else np.full(n, peak)
    current = np.zeros(n, dtype=np.int64)
    x = np.zeros((n, length), dtype=np.int64)
    for t in range(length):
        coin = rng.random(n) < 0.5
        token = np.where(t < ramp, 0, np.where(current >= peak, 1, np.where(coin, 0, 1)))
        x[:, t] = token
        current = np.maximum(current + np.where(token == 0, 1, -1), 0)
    return x


def generate(task, n, length, seed, depth=None):
    """(x, y): n sequences of the given length. depth only applies to 'count' (the peak that must be counted to)."""
    s = _spec(task)
    if n < 1 or length < 1:
        raise ValueError('invalid budget')
    rng = np.random.default_rng(seed)
    if s['kind'] == 'perm':
        x = torch.from_numpy(rng.integers(0, vocab(task), size=(n, length)).astype(np.int64))
    else:
        x = torch.from_numpy(_count_tokens(rng, n, length, depth))
    return x, oracle(task, x)


def source_seed(namespace, task, seed, split, step=0, length=0, depth=0):
    value = f'{namespace[split]}|{task}|{seed}|{step}|{length}|{depth}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def train_batch(namespace, task, seed, step, batch, length=None, depth=None):
    length = TRAIN_LENGTH if length is None else length
    s = source_seed(namespace, task, seed, 'train', step=step, length=length)
    x, y = generate(task, batch, length, s, depth=depth)
    return dict(x=x, y=y, source_seed=s)


def heldout(namespace, task, seed, split, n, length, depth=None):
    s = source_seed(namespace, task, seed, split, length=length, depth=0 if depth is None else depth)
    x, y = generate(task, n, length, s, depth=depth)
    return x, y, s
