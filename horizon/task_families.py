"""Task families for U1.

five  — the existing five-state automaton (3 tokens, 5 classes): sufficient state is constant.
stack — push a / push b / neutral / pop with the top symbol as label (4 tokens, 3 classes: a, b, empty);
        the symbol stack grows with the problem, so a finite automaton cannot solve it.
Token 2 is the neutral (identity) operation in both families.
"""
import hashlib

import numpy as np
import torch

from core.model import generate as core_generate
from review_cycle.training_data import prefix_targets, tensor_hash

VOCAB = {'five': 3, 'stack': 4}
CLASSES = {'five': 5, 'stack': 3}
NEUTRAL = 2
STACK_PROBABILITIES = (0.2125, 0.2125, 0.15, 0.425)  # push a, push b, neutral, pop: pushes == pops in expectation


def _stack_trajectory(x):
    if x.ndim != 2 or torch.any((x < 0) | (x > 3)):
        raise ValueError('invalid stack input')
    n, length = x.shape
    store = torch.zeros(n, length + 1, dtype=torch.long)
    depth = torch.zeros(n, dtype=torch.long)
    rows = torch.arange(n)
    labels, depths = [], []
    for t in range(length):
        token = x[:, t]
        push = token < 2
        store[rows[push], depth[push]] = token[push]
        depth = depth + push.long()
        depth = torch.where(token == 3, (depth - 1).clamp(min=0), depth)
        top = store[rows, (depth - 1).clamp(min=0)]
        labels.append(torch.where(depth == 0, torch.full_like(top, 2), top))
        depths.append(depth.clone())
    return torch.stack(labels, 1), torch.stack(depths, 1)


def oracle_targets(x, task):
    if task == 'five':
        return prefix_targets(x, 'five')
    if task == 'stack':
        return _stack_trajectory(x)[0]
    raise ValueError('unknown task')


def max_depth(x):
    return _stack_trajectory(x)[1].max(1).values


def generate(task, n, length, seed):
    if n < 1 or length < 1:
        raise ValueError('invalid budget')
    if task == 'five':
        return core_generate('five', n, length, seed)
    if task == 'stack':
        rng = np.random.default_rng(seed)
        x = torch.from_numpy(rng.choice(4, (n, length), p=STACK_PROBABILITIES))
        return x, oracle_targets(x, 'stack')
    raise ValueError('unknown task')


def insert_neutrals(x, task, slots, placement):
    at = {'prefix': 0, 'middle': x.shape[1] // 2, 'suffix': x.shape[1], 'base': x.shape[1]}[placement]
    xx = torch.cat((x[:, :at], torch.full((len(x), slots), NEUTRAL, dtype=x.dtype), x[:, at:]), 1)
    yy = oracle_targets(xx, task)
    if not torch.equal(yy[:, -1], oracle_targets(x, task)[:, -1]):
        raise ValueError('neutral insertion changed the final answer')
    return xx, yy


def source_seed(namespace, task, seed, split, step=0, length=0):
    value = f'{namespace[split]}|{task}|{seed}|{step}|{length}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def heldout(namespace, task, seed, split, n, length):
    s = source_seed(namespace, task, seed, split, length=length)
    x, y = generate(task, n, length, s)
    return x, y, s


def train_batch(namespace, task, seed, step, batch, length):
    s = source_seed(namespace, task, seed, 'train', step=step, length=length)
    x, y = generate(task, batch, length, s)
    return dict(x=x, y=y, source_seed=s, input_hash=tensor_hash(x))
