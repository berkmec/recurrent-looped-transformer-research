"""X1 baseline (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.59): a standard causal transformer language model with
L1's parameter budget, trained by the same runner (`mline.longrun`, recipe m2-draft) on the same bytes; only the architecture changes.

Pre-norm blocks with the Core's RMS norm, causal self-attention with the Core's rotary embedding, a GELU feed-forward of 4x width,
an RMS-normed byte readout, the usual GPT-2 initialisation (normal 0.02, residual projections scaled by 1/sqrt(2 * layers)) so the
baseline gets its best-known start. `tfm6` at width 528 with 8 heads has 20,365,744 parameters, L1 20,297,472 (+0.3 %).

Not an RLT variant: no feedback, no encoder/decoder split, no memory path. The A0 twin stays the single-factor control for feedback;
this arm answers the independent evaluation's question (X1): is the RLT language model as good as a plain transformer of its size?
`mline/lm.py` is part of the frozen M1 protocol (mline.train.SOURCES), so the arm lives here and `mline.longrun` picks it up.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from core.model import RMS

ARMS = ('tfm6',)
LAYERS = {'tfm6': 6}
VOCAB = 256


def rope(x, start=0):
    """core.model.Core.rope as a free function, on (batch, heads, positions, head width)."""
    angle = torch.arange(start, start + x.shape[2], device=x.device, dtype=x.dtype)[:, None] * 10000 ** (
        -torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype) / x.shape[-1])
    c, s = angle.cos()[None, None], angle.sin()[None, None]
    return torch.stack((x[..., ::2] * c - x[..., 1::2] * s, x[..., ::2] * s + x[..., 1::2] * c), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, width, heads, ffn=4):
        super().__init__()
        self.heads = heads
        self.an, self.fn = RMS(width), RMS(width)
        self.q, self.k, self.v, self.o = (nn.Linear(width, width, bias=False) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(width, ffn * width), nn.GELU(), nn.Linear(ffn * width, width))

    def split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

    def forward(self, x):
        n = self.an(x)
        q, k, v = rope(self.split(self.q(n))), rope(self.split(self.k(n))), self.split(self.v(n))
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape_as(x))
        return x + self.ff(self.fn(x))


class TransformerLM(nn.Module):
    def __init__(self, width=528, heads=8, layers=6, ffn=4):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError('width must split into heads of even width (rotary embedding)')
        self.width, self.heads, self.layers = width, heads, layers
        self.embedding = nn.Embedding(VOCAB, width)
        self.blocks = nn.ModuleList(Block(width, heads, ffn) for _ in range(layers))
        self.outn = RMS(width)
        self.read = nn.Linear(width, VOCAB)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * layers) if name.endswith(('.o.weight', '.ff.2.weight')) else 0.02)
            elif name.endswith('bias'):
                nn.init.zeros_(p)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        return self.read(self.outn(h))


def build(arm, width=528, heads=8, window=None, alpha=None, seed=117):
    """mline.lm.build's signature, so the runner does not branch; `window` and `alpha` are Core settings, unused here."""
    if arm not in ARMS:
        raise ValueError('arm must be one of ' + ', '.join(ARMS))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TransformerLM(width=width, heads=heads, layers=LAYERS[arm])
