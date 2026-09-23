"""M-hattı M0: the 30M-class language model is the SEALED Core at language width, not a new architecture.

`core.model.Core(width, heads, window, classes=256)` with a 256-entry byte embedding is already a language model, and
`horizon.gated.GatedCore` already wraps it with the one architectural result the campaign has replicated. So the M1
comparison tests, at 30M on real text, exactly the thing that was measured 3 seeds x 2 task families x 2 learning
rates on the toy automata - not an approximation of it.

Arms:
  plain    - the sealed Core, feedback on, no gate (the control)
  hard_st  - GatedCore with exact bitwise preservation (the replicated result)
  soft     - GatedCore with soft mixing (predicted to lose, because language state carries contents like `stack`)

Nothing here edits a sealed file.
"""
import torch
from torch import nn

from core.model import Core
from horizon import gated as gd

ARMS = ('plain', 'hard_st', 'soft')
VOCAB = 256


def build(arm, width=1152, heads=8, window=4, alpha=.1, seed=117, gate_k=2):
    if arm not in ARMS:
        raise ValueError('arm must be one of ' + ', '.join(ARMS))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        core = Core(width=width, heads=heads, window=window, classes=VOCAB, feedback=True, alpha=alpha)
        core.embedding = nn.Embedding(VOCAB, width)
        if arm == 'plain':
            return core
        torch.manual_seed(seed + 7)
        return gd.GatedCore(core, mode=arm, k=gate_k)


def parameter_count(model):
    core = model if isinstance(model, Core) else model.core
    total = sum(p.numel() for p in model.parameters())
    return dict(total=total, core=sum(p.numel() for p in core.parameters()),
                gate=total - sum(p.numel() for p in core.parameters()),
                embedding=core.embedding.weight.numel(), readout=core.read.weight.numel() + core.read.bias.numel(),
                width=core.width, heads=core.heads, vocabulary=VOCAB)


def forward_logits(model, x):
    """Logits for either arm, so the training loop does not branch on the arm."""
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def states_of(model):
    """Committed states of the last forward, or None for the plain Core, which does not record them."""
    return getattr(model, 'last_states', None)
