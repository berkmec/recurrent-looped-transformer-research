"""Paired source generation: ordering and supervision never seed the inputs."""
import hashlib

import numpy as np
import torch

from core.model import generate


def tensor_hash(x):
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(str((str(x.dtype), tuple(x.shape))).encode() + x.numpy().tobytes()).hexdigest()


def source_seed(c, p, split, step=0, length=None):
    # Independent split namespaces; fixed before either validation or test opens.
    value = f"{p['source_seeds'][split]}|{c['task']}|{c['seed']}|{step}|{length or p['train_length']}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def prefix_targets(x, task):
    if task not in ('parity', 'five') or x.ndim != 2 or x.shape[1] == 0:
        raise ValueError('invalid task or input dimensions')
    if torch.any((x < 0) | (x > 2)):
        raise ValueError('invalid operation')
    # Explicit recurrence, independent of the training labels returned by generate.
    s = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    out = []
    for token in x.unbind(1):
        if task == 'parity':
            s = s ^ (token == 1).long()
        else:
            swapped = torch.where(s == 0, 1, torch.where(s == 1, 0, s))
            s = torch.where(token == 0, (s + 1) % 5, torch.where(token == 1, swapped, s))
        out.append(s.clone())
    return torch.stack(out, 1)


def block_neutrals(x, start_index=0):
    rows = []
    for i, row in enumerate(x):
        active, neutral = row[row != 2], row[row == 2]
        slot = (start_index + i) % 3
        at = (0, len(active) // 2, len(active))[slot]
        rows.append(torch.cat((active[:at], neutral, active[at:])))
    return torch.stack(rows)


def train_batch(c, p, step):
    if not 1 <= step <= p['steps']:
        raise ValueError('training step outside frozen budget')
    seed = source_seed(c, p, 'train', step)
    x, y = generate(c['task'], p['batch'], p['train_length'], seed)
    base_hash = tensor_hash(x)
    if c['arrangement'] == 'blocked':
        x = block_neutrals(x, (step - 1) * p['batch'])
        updated = prefix_targets(x, c['task'])
        if not torch.equal(updated[:, -1], y[:, -1]):
            raise ValueError('blocked rearrangement changed final target')
        y = updated
    return dict(x=x, y=y, source_seed=seed, input_hash=tensor_hash(x), base_input_hash=base_hash)


def sparse_positions(c, p, step):
    seed = source_seed(c, p, 'sparse', step)
    return torch.from_numpy(np.random.default_rng(seed).integers(0, p['train_length'], size=p['batch']))


def label_count(supervision, steps, batch, length):
    if supervision not in ('dense', 'sparse', 'final'):
        raise ValueError('unknown supervision')
    return steps * batch * (length if supervision == 'dense' else 1)


def heldout(c, p, split, n, length):
    seed = source_seed(c, p, split, length=length)
    x, y = generate(c['task'], n, length, seed)
    return x, y, seed


def insert_neutrals(x, task, slots, placement):
    at = {'prefix': 0, 'middle': x.shape[1] // 2, 'suffix': x.shape[1], 'base': x.shape[1]}[placement]
    xx = torch.cat((x[:, :at], torch.full((len(x), slots), 2, dtype=x.dtype), x[:, at:]), 1)
    yy = prefix_targets(xx, task)
    if not torch.equal(yy[:, -1], prefix_targets(x, task)[:, -1]):
        raise ValueError('neutral insertion changed final answer')
    return xx, yy
