"""Drift arms on top of the immutable Core.

run_sequence executes the Core's own encode/decode contract one token at a time (the same
computation as Core.forward without truncation) and returns per-step logits and hidden states,
so a neutral-invariance loss can see the states and the reread arm can re-anchor them.
The reread arm is a declared deviation from the RLT source: single-head causal attention from
the current state to every past (state, input-embedding) pair; W_o starts at zero so the first
forward pass equals the base arm.
"""
import torch
from torch import nn
from torch.nn import functional as F

from core.model import RMS

ARMS = ('base', 'win16', 'idinv', 'reread')


class RereadCore(nn.Module):
    def __init__(self, core, seed):
        super().__init__()
        self.core = core
        w = core.width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.reread_n = RMS(w)
            self.reread_q = nn.Linear(w, w, bias=False)
            self.reread_k = nn.Linear(2 * w, w, bias=False)
            self.reread_v = nn.Linear(2 * w, w, bias=False)
            self.reread_o = nn.Linear(w, w, bias=False)
        nn.init.zeros_(self.reread_o.weight)

    def reread(self, s, past_states, past_inputs):
        if past_states.shape[1] == 0:
            return s
        keys = torch.cat([past_states, past_inputs], -1)
        q = self.reread_q(self.reread_n(s))[:, None, None]
        k = self.reread_k(keys)[:, None]
        v = self.reread_v(keys)[:, None]
        a = F.scaled_dot_product_attention(q, k, v)[:, 0, 0]
        return s + self.reread_o(a)


def core_of(model):
    return model.core if isinstance(model, RereadCore) else model


def run_sequence(model, x, arm):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    if (arm == 'reread') != isinstance(model, RereadCore):
        raise ValueError('arm/model mismatch')
    core = core_of(model)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError('invalid input')
    e, _, _ = core.encode(x)
    n = core.mn(e)
    mk = core.rope(core.split(core.mk(n)))
    mv = core.split(core.mv(n))
    buffer = None if torch.is_grad_enabled() else e.new_zeros(x.shape[0], x.shape[1], core.width)
    state = None
    logits, states = [], []
    for t in range(x.shape[1]):
        z, state = core.decode(e[:, t], state, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
        if arm == 'reread':
            past = buffer[:, :t] if buffer is not None else (torch.stack(states, 1) if states else e[:, :0])
            s = model.reread(state['s'], past, e[:, :t])
            state = dict(state, s=s)
            z = core.read(core.outn(s))
        logits.append(z)
        states.append(state['s'])
        if buffer is not None:
            buffer[:, t] = state['s']
    return torch.stack(logits, 1), torch.stack(states, 1)


def neutral_invariance_loss(states, x, eps=1e-6):
    """Mean squared state change at positions whose input is the neutral token 2, scaled by mean squared state."""
    if states.shape[1] < 2:
        return states.sum() * 0.
    diff = (states[:, 1:] - states[:, :-1]).square().mean(-1)
    mask = (x[:, 1:] == 2).to(diff.dtype)
    scale = states.detach().square().mean() + eps
    return (diff * mask).sum() / mask.sum().clamp(min=1.) / scale
