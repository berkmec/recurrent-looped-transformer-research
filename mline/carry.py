"""The RLT state as the only channel between windows (program note 5.74; bug hunt 5.71(b)): in L1's training every 512-byte row
started from the same learned initial state and the encoder memory covered the whole row, so nothing ever rewarded the state for
carrying information past the row, and E-L1X-F's "no state context" was the expected outcome. Here a row is n windows of w bytes:

  each window: the sealed Core's encoder over that window alone (its memory covers only the window, K2), then the Core's own
  sequential decoder (core.decode, K1: token-level gated feedback);
  carry: the decoder's token state s at a window's last position is the state the next window's first position starts from;
  the decoder's 4-position self-attention cache and its positions restart with each window.

So with carry the only path from window j to window j+1 is the RLT state, and the gradient of a later window's loss reaches the
earlier window through it (full backpropagation through the windows). read=False (carry off) starts every window from the
Core's `initial`, which is exactly the Core run on each window alone (tested): the manipulation check. A Core without feedback
(the A0 twin) ignores the state, so carry changes nothing for it (tested). core/model.py is used as it is, never edited. Cost
(K4): O(1) state per step; the windows run one after another, so a row of n windows costs n times one window's decoder loop.
"""
import torch
from torch import nn

from core.model import Core


def decode_carry(core, xw, carry=True):
    """xw (b, n, w) byte ids -> logits (b, n, w, classes) of the sealed Core run window by window, the token state handed on."""
    b, n, w = xw.shape
    out, state = [], None
    for j in range(n):
        e, _, _ = core.encode(xw[:, j])
        m = core.mn(e)
        mk, mv = core.rope(core.split(core.mk(m))), core.split(core.mv(m))
        s = None
        if carry and state is not None:
            empty = mk.new_zeros(b, core.heads, 0, core.width // core.heads)
            s = dict(s=state, dk=empty, dv=empty, t=0)
        ys = []
        for t in range(w):
            y, s = core.decode(e[:, t], s, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
            ys.append(y)
        state = s['s']
        out.append(torch.stack(ys, 1))
    return torch.stack(out, 1)


class CarryLM(nn.Module):
    """The sealed Core over rows of windows with the state carried between them; no new parameters (`memory` False for the
    trainer's parameter groups, `carry` True for its carry-off control)."""

    def __init__(self, core):
        super().__init__()
        if not isinstance(core, Core):
            raise TypeError('CarryLM wraps the sealed core.model.Core')
        self.core, self.memory, self.carry = core, False, True

    def gate_parameters(self):
        return []

    def forward(self, xw, read=True):
        """xw (b, n, w) -> logits (b, n, w, classes); read=False starts every window from the Core's initial state."""
        return decode_carry(self.core, xw, carry=read)
