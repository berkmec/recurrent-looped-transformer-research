"""Z1 arms. The sealed Core is never edited: its input vocabulary and class count are set the way U1 did it (replace the embedding).

  core       the sealed Core, state feedback on (alpha 0.1)
  core_nofb  the same weights layout with the feedback path off: causal attention plus a short local window, no recurrence
  gru        a plain GRU, the textbook recurrent baseline for state tracking and counting
  lstm       a plain LSTM: unlike the GRU it has an unbounded cell state, so it is the recurrent baseline that CAN count
  tfm        a causal pre-norm transformer with no positional encoding (NoPE), four layers: the constant-depth, non-recurrent baseline
  core_clean the Core with a gated Hopfield cleanup in the feedback path (`horizon/cleanup.py`, E-A10): identical to `core` while the gate is closed (its initial value)
"""
import torch
from torch import nn

from core.model import Core

from . import depth_tasks as dt
from .cleanup import add_cleanup

ARMS = ('core', 'core_nofb', 'gru', 'lstm', 'tfm', 'core_clean', 'hold', 'hold_nc')


class GRUArm(nn.Module):
    def __init__(self, vocab, classes, width):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.rnn = nn.GRU(width, width, batch_first=True)
        self.read = nn.Linear(width, classes)

    def forward(self, x):
        h, _ = self.rnn(self.embedding(x))
        return self.read(h)


class LSTMArm(nn.Module):
    def __init__(self, vocab, classes, width):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.rnn = nn.LSTM(width, width, batch_first=True)
        self.read = nn.Linear(width, classes)

    def forward(self, x):
        h, _ = self.rnn(self.embedding(x))
        return self.read(h)


class TfmArm(nn.Module):
    """Causal pre-norm transformer, no positional encoding: order is only visible through causal attention itself."""

    def __init__(self, vocab, classes, width, layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        layer = nn.TransformerEncoderLayer(width, max(1, width // 32), 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.read = nn.Linear(width, classes)

    def forward(self, x):
        n = x.shape[1]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        return self.read(self.norm(self.encoder(self.embedding(x), mask=mask, is_causal=True)))


def make_arm(arm, task, width, seed, window=4, alpha=0.1, open_init=0.0, hold_tmax=16):
    if arm not in ARMS:
        raise ValueError('unknown arm: ' + str(arm))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if arm == 'gru':
            return GRUArm(dt.vocab(task), dt.classes(task), width)
        if arm == 'lstm':
            return LSTMArm(dt.vocab(task), dt.classes(task), width)
        if arm == 'tfm':
            return TfmArm(dt.vocab(task), dt.classes(task), width)
        model = Core(width=width, heads=width // 32, window=window, classes=dt.classes(task), feedback=(arm in ('core', 'core_clean', 'hold', 'hold_nc')), alpha=alpha)
        if dt.vocab(task) != model.embedding.num_embeddings:
            model.embedding = nn.Embedding(dt.vocab(task), width)
        if arm == 'core_clean':
            add_cleanup(model, open_init=open_init)         # after the base weights are drawn: they equal the plain `core` of the same seed
        if arm in ('hold', 'hold_nc'):
            from .hold import HoldCore
            model = HoldCore(model, mode='hard_st', t_max=hold_tmax, chrono=(arm == 'hold'), seed=seed)
    return model


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
