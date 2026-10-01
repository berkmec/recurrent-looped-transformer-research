"""Z1 depth-extrapolation task families.

Each task has a depth knob, so the amount of sequential computation can be pushed far past what was trained:

  perm3 / perm4 / perm5   composition in the symmetric group S_N. Every token names one permutation, the label at each position is the
                          index of the running product. S_5 has no constant-depth solution; a finite state (120) has to be carried
                          exactly through the whole sequence. Depth = sequence length.
  count                   a stack-depth counter: token 0 pushes (+1), token 1 pops (-1, floored at 0); the label is min(depth, LABEL_CAP)
                          with LABEL_CAP = TRAIN_DEPTH, so every output class occurs in training. The depth that must be held is a free
                          parameter (the peak D), unbounded in principle: the test climbs to D, comes back down, and asks for the depth
                          once it is back inside the trained range ("re-entry"), which can only be answered from an exact count.

Training uses short sequences (perm: length <= TRAIN_LENGTH, count: peak <= TRAIN_DEPTH); the claim under test is how far beyond
that the learned recurrence still works.
"""
import hashlib
import itertools

import numpy as np
import torch

TRAIN_DEPTH = 8
LABEL_CAP = TRAIN_DEPTH     # a classifier cannot emit a class it never saw: the label saturates at the trained peak
TRAIN_LENGTH = 16           # permutation tasks: depth = length
COUNT_TRAIN_LENGTH = 64     # counter: long enough that the walk wanders for a long time inside depth <= TRAIN_DEPTH, so depth is the only thing extrapolated
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
    return len(group(s['n'])[0]) if s['kind'] == 'perm' else LABEL_CAP + 1


def _token_matrix(x):
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError('invalid token matrix')
    return x.numpy()


def depth_trajectory(x):
    """True (uncapped) stack depth after each token of the push/pop matrix x (n, length): 0 pushes, 1 pops, floored at zero."""
    arr = _token_matrix(x)
    out = np.zeros_like(arr)
    depth = np.zeros(arr.shape[0], dtype=np.int64)
    for t in range(arr.shape[1]):
        depth = np.maximum(depth + np.where(arr[:, t] == 0, 1, -1), 0)
        out[:, t] = depth
    return torch.from_numpy(out)


def _after_exceeding(trajectory):
    """(back_in_range, zeros_before): positions back inside the trained range after the sequence has been deeper than it, and for every
    position how many floor visits (depth 0 after having exceeded the trained depth) happened strictly before it."""
    exceeded = trajectory.cummax(dim=1).values > TRAIN_DEPTH
    floor = (trajectory == 0) & exceeded
    return exceeded & (trajectory <= TRAIN_DEPTH), floor.cumsum(dim=1) - floor.int()


def reentry_mask(trajectory):
    """The FIRST descent: positions back inside the trained range after the sequence has been deeper than it, up to and including the
    first return to depth 0. Only an exact count of the excursion answers these (a counter that saturated or lost count is wrong here)."""
    back, floor_visits = _after_exceeding(trajectory)
    return back & (floor_visits == 0)


def resync_mask(trajectory):
    """Positions back inside the trained range AFTER that first floor visit. The floor resets the count (pops at 0 stay 0), so these positions
    measure how fast a model re-synchronises, not whether it counted the excursion. Positions deeper than the trained peak after the reset are
    not here: they belong to `deep`. Every position after the first exceedance is in exactly one of deep (depth above the peak), reentry
    (first descent) and resync (back in range after the first floor visit)."""
    back, floor_visits = _after_exceeding(trajectory)
    return back & (floor_visits >= 1)


def oracle(task, x):
    """Per-position labels for the token matrix x (n, length)."""
    s = _spec(task)
    if s['kind'] == 'count':
        return torch.clamp(depth_trajectory(x), max=LABEL_CAP)
    arr = _token_matrix(x)
    y = np.zeros_like(arr)
    table = group(s['n'])[2]
    state = np.zeros(arr.shape[0], dtype=np.int64)
    for t in range(arr.shape[1]):
        state = table[state, arr[:, t]]
        y[:, t] = state
    return torch.from_numpy(y)


def count_eval_length(depth):
    """Sequence length for a test sequence with peak `depth`: room to climb (~2D steps), descend (~2D) and wander back inside the range."""
    return max(6 * int(depth), COUNT_TRAIN_LENGTH)


def _count_tokens(rng, n, length, depth):
    """Push (0) / pop (1) tokens.
    Training (depth=None): the true depth never exceeds TRAIN_DEPTH. Each sequence opens with a random run of pushes (0..TRAIN_DEPTH), then
    walks with its own push probability drawn from [0.25, 0.75]; at the peak a push becomes a pop (reflecting), so 'push at depth
    TRAIN_DEPTH' never occurs in training.
    Test (depth=D): every sequence first climbs to exactly D (push probability 0.75, forced after half the budget), then comes back down
    (push probability 0.25, reflecting at D) and wanders near the bottom. Both bias values lie inside the training range of statistics, so
    what changes at test is how deep the count has been, not which tokens occur."""
    training = depth is None
    peak = TRAIN_DEPTH if training else int(depth)
    ramp = rng.integers(0, peak + 1, size=n) if training else None
    p_push = rng.uniform(0.25, 0.75, size=n) if training else None
    current = np.zeros(n, dtype=np.int64)
    climbing = np.ones(n, dtype=bool)
    x = np.zeros((n, length), dtype=np.int64)
    for t in range(length):
        u = rng.random(n)
        if training:
            push = np.where(t < ramp, True, (u < p_push) & (current < peak))
        else:
            p = np.where(climbing, 0.75, 0.25)
            push = ((u < p) | (climbing & (t >= length // 2))) & (current < peak)
        x[:, t] = np.where(push, 0, 1)
        current = np.maximum(current + np.where(push, 1, -1), 0)
        if not training:
            climbing &= current < peak
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


def train_length(task):
    return COUNT_TRAIN_LENGTH if _spec(task)['kind'] == 'count' else TRAIN_LENGTH


def train_batch(namespace, task, seed, step, batch, length=None, depth=None):
    length = train_length(task) if length is None else length
    # the depth is part of the seed so that one source_seed names exactly one dataset; depth=None (the training distribution) maps to 0,
    # i.e. the derivation every recorded pilot used
    s = source_seed(namespace, task, seed, 'train', step=step, length=length, depth=0 if depth is None else depth)
    x, y = generate(task, batch, length, s, depth=depth)
    return dict(x=x, y=y, source_seed=s)


def heldout(namespace, task, seed, split, n, length, depth=None):
    """A fixed evaluation draw for the split 'dev' or 'test' (or 'validation' where the caller's namespace has one). The training split is drawn
    only by train_batch. n is NOT part of the seed: draws of different sizes share one seed (permutation draws nest), so use one n per point."""
    if split == 'train':
        raise ValueError("the 'train' split is drawn by train_batch only; a held-out draw from it could reproduce a training batch")
    s = source_seed(namespace, task, seed, split, length=length, depth=0 if depth is None else depth)
    x, y = generate(task, n, length, s, depth=depth)
    return x, y, s
