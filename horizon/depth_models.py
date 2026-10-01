"""Z1 arms. The sealed Core is never edited: its input vocabulary and class count are set the way U1 did it (replace the embedding).

  core       the sealed Core, state feedback on (alpha 0.1)
  core_nofb  the same weights layout with the feedback path off: causal attention plus a short local window, no recurrence
  gru        a plain GRU, the textbook recurrent baseline for state tracking and counting
"""
import torch
from torch import nn

from core.model import Core

from . import depth_tasks as dt

ARMS = ('core', 'core_nofb', 'gru')


class GRUArm(nn.Module):
    def __init__(self, vocab, classes, width):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.rnn = nn.GRU(width, width, batch_first=True)
        self.read = nn.Linear(width, classes)

    def forward(self, x):
        h, _ = self.rnn(self.embedding(x))
        return self.read(h)


def make_arm(arm, task, width, seed, window=4, alpha=0.1):
    if arm not in ARMS:
        raise ValueError('unknown arm: ' + str(arm))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if arm == 'gru':
            return GRUArm(dt.vocab(task), dt.classes(task), width)
        model = Core(width=width, heads=width // 32, window=window, classes=dt.classes(task), feedback=(arm == 'core'), alpha=alpha)
        if dt.vocab(task) != model.embedding.num_embeddings:
            model.embedding = nn.Embedding(dt.vocab(task), width)
    return model


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
