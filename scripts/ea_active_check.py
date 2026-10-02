"""E-A companion check: the temperature rows were IDENTICAL to the baseline to the last digit. Is that a silent no-op or a real finding?

For the saved S5 Core: (1) how much do the logits move under each intervention, and does any argmax change? (2) how sharp are the encoder and memory-read attentions at
long positions (entropy against the uniform bound)? A temperature that moves logits by ~1e-3 but never an argmax, on attention that is near-uniform, is inert, not broken.

  PYTHONPATH=. python -B scripts/ea_active_check.py > results/Z1-pilot/perm5-core-intervention-active-check.txt
"""
import copy
import math
import sys

import torch

sys.path.insert(0, '.')
from horizon import attention_intervention as ai
from horizon import depth_run as dr
from horizon import depth_tasks as dt

torch.set_num_threads(1)
WEIGHTS = 'results/Z1-pilot/perm5-core-w128-s1-cur.pt'
LENGTH, SEQUENCES = 300, 4

base_model = dr.load_model(WEIGHTS).eval()
x, y, _ = dt.heldout(dr.NAMESPACE, 'perm5', 1, 'dev', SEQUENCES, LENGTH)
with torch.no_grad():
    base = base_model(x)
    for name in ('temp', 'temp_enc', 'temp_mem', 'w16', 'w16_enc', 'w16_mem'):
        out = ai.intervene(copy.deepcopy(base_model), name).eval()(x)
        d = (out - base).abs()
        changed = int((out.argmax(-1) != base.argmax(-1)).sum())
        print(f'{name:9s} max|dlogit| {float(d.max()):.3e}  mean {float(d.mean()):.3e}  argmax changed at {changed} of {out.shape[0] * out.shape[1]} positions')


def entropy_of(weights):
    p = weights.clamp_min(1e-12)
    return float(-(p * p.log()).sum(-1).mean())


enc, mem = [], []
original = base_model.__class__.attend


def spy(self, q, k, v, causal=False):
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        t = q.shape[2]
        scores = scores.masked_fill(~torch.tril(torch.ones(t, t, dtype=torch.bool)), float('-inf'))
        enc.append(entropy_of(torch.softmax(scores, -1)[..., 64:, :]))
    elif k.shape[2] > self.window:
        mem.append(entropy_of(torch.softmax(scores, -1)))
    return original(self, q, k, v, causal)


base_model.__class__.attend = spy
with torch.no_grad():
    base_model(x)
base_model.__class__.attend = original
uniform_enc = sum(math.log(i + 1) for i in range(64, LENGTH)) / (LENGTH - 64)
print(f'encoder self-attention entropy at positions >= 64: {enc[0]:.3f} nats (uniform over the visible keys, same positions: {uniform_enc:.3f})')
print(f'decoder memory-read entropy, mean over positions: {sum(mem) / len(mem):.3f} nats (uniform over the keys: mean of ln(t + 1) = '
      f'{sum(math.log(t + 1) for t in range(4, LENGTH)) / (LENGTH - 4):.3f})')
