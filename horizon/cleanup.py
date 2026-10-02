"""E-A10: a gated Hopfield cleanup in the Core's feedback path ("RLT-1 + cleanup"), a trainable, label-free version of the self-snap of E-A9 (note 5.27 / 5.28).

The Core feeds the previous step's final decoder output back through `r = RMSNorm(state)` and a gated linear merge. Nothing in that path ever pulls a drifted state back toward a class: a
perturbation neither grows nor decays (E-B2) and the deviation from the canonical class state accumulates with the state's age (E-A5). The cleanup inserts a modern-Hopfield retrieval
(an attractor memory: softmax over cosine similarities to M learned keys, read out as a mixture of M learned values; with a sharp softmax it snaps `r` to the nearest prototype) between the
normalisation and the merge:

    r' = r + g (Hop(r) - r),     Hop(r) = softmax(beta cos(r, keys)) values,     g = tanh(open),   open = 0 at initialisation

so a freshly built model computes EXACTLY what the sealed Core computes (K3: closable; the extra parameters are unused until `open` moves), the sequential cost per step stays O(1) (K4),
the feedback of the previous token's final decoder output through the gated merge is kept (K1) and the encoder memory and the windowed decoder are untouched (K2). It is the module that
replaces the Core's `pn` normalisation, so the sealed `core/model.py` is not edited, state-noise training (`state_noise.noisy`) and the E-A5 / E-A9 probes work unchanged.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

PROTOTYPES = 128
BETA = 20.0


class CleanupNorm(nn.Module):
    """RMSNorm followed by the gated Hopfield cleanup; replaces `Core.pn`."""

    def __init__(self, norm, width, prototypes=PROTOTYPES, beta=BETA, open_init=0.0):
        super().__init__()
        self.norm = norm
        self.keys = nn.Parameter(torch.randn(prototypes, width))
        self.values = nn.Parameter(torch.randn(prototypes, width))
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta)))
        self.open = nn.Parameter(torch.tensor(float(open_init)))

    def gate(self):
        return torch.tanh(self.open)

    def forward(self, state):
        r = self.norm(state)
        similarity = F.normalize(r, dim=-1) @ F.normalize(self.keys, dim=-1).t()
        retrieved = torch.softmax(self.log_beta.exp() * similarity, dim=-1) @ self.values
        return r + self.gate() * (retrieved - r)


def add_cleanup(core, prototypes=PROTOTYPES, beta=BETA, open_init=0.0):
    """Wrap the Core's feedback normalisation in place (the sealed class is untouched); returns the same model."""
    if not core.feedback:
        raise ValueError('a cleanup needs the feedback path')
    core.pn = CleanupNorm(core.pn, core.width, prototypes, beta, open_init)
    return core


def gate_value(core):
    """The current cleanup gate g = tanh(open) of a model built with `add_cleanup` (None for a plain Core)."""
    pn = getattr(core, 'pn', None)
    return float(pn.gate().detach()) if isinstance(pn, CleanupNorm) else None
