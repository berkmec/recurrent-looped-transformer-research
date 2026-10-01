"""U4-inject: replace the Core's additive state feedback with attention injection (FLT, arXiv 2605.18797).

The sealed Core carries the recurrent state forward additively:

    z = e_t + alpha * sigmoid(gate([e_t, r])) * merge(r),   r = pn(s_{t-1})

`merge` is an unconstrained linear map, so the contribution's magnitude is bounded only by the learned weights and by
alpha - the shape the Fully Looped Transformer paper identifies as residual explosion. Attention injection replaces it:
the previous state becomes a Query over the decoder's sliding window, so its contribution arrives as a softmax-weighted
average of value vectors, which lies inside their convex hull whatever the weights do.

Arms:
  additive   - the sealed Core, untouched, as the control (bitwise identical; asserted by a test)
  inject     - bounded injection: z = e_t + io(attend(iq(r), K, V)); the recurrence flows only through the attention
  inject_mix - fully convex: z = w0*e_t + w1*a with softmax weights, so the state is an average, never an accumulation

In the injection arms the window K/V are derived from e_t alone rather than from the post-feedback z, so the state
reaches the next step only through the injected query - that is the point of the variant, and it makes the two paths
cleanly separable.

`core/model.py` is hashed inside every sealed manifest, so this variant lives beside it and never edits it.
"""
import torch
from torch import nn

MODES = ('additive', 'inject', 'inject_mix')


class InjectCore(nn.Module):
    def __init__(self, core, mode='inject'):
        super().__init__()
        if mode not in MODES:
            raise ValueError('mode must be one of ' + ', '.join(MODES))
        self.core = core; self.mode = mode
        width = core.width
        if mode != 'additive':
            self.iq = nn.Linear(width, width, bias=False)
            self.io = nn.Linear(width, width, bias=False)
            with torch.no_grad():
                self.io.weight.mul_(core.alpha)   # start near the sealed model's feedback scale, not at full strength
        if mode == 'inject_mix':
            self.mix = nn.Linear(2 * width, 2)
            with torch.no_grad():
                self.mix.weight.normal_(0., 0.02); self.mix.bias.copy_(torch.tensor([2., -2.]))  # start close to pass-through
        self.last_mix = None

    def injection_parameters(self):
        names = ('iq', 'io', 'mix')
        return [p for n in names if hasattr(self, n) for p in getattr(self, n).parameters()]

    def decode(self, e, s, mk, mv, t):
        core = self.core
        if self.mode == 'additive':
            return core.decode(e, s, mk, mv, t)
        prev = core.initial.expand(e.shape[0], -1) if s is None else s['s']
        n = core.dn(e)
        q = core.rope(core.split(core.dq(n)), t)
        k = core.rope(core.split(core.dk(n)), t)
        v = core.split(core.dv(n))
        if s is not None:
            k = torch.cat([s['dk'], k], 2); v = torch.cat([s['dv'], v], 2)
        r = core.pn(prev)
        a = self.io(core.attend(core.rope(core.split(self.iq(r)), t), k, v)[:, 0])
        if self.mode == 'inject':
            z = e + a
        else:
            w = torch.softmax(self.mix(torch.cat([e, a], -1)), -1)
            z = w[:, :1] * e + w[:, 1:] * a
            self.last_mix = dict(weights=w.detach(), e=e.detach(), injected=a.detach(), mixed=z.detach())
        z = z + core.do(core.attend(q, k, v)[:, 0])
        q2 = core.rope(core.split(core.cq(core.cn(z))), t)
        z = z + core.co(core.attend(q2, mk, mv)[:, 0])
        z = z + core.dff(core.dfn(z))
        keep = core.window - 1
        return core.read(core.outn(z)), dict(s=z, dk=k[:, :, -keep:] if keep else k[:, :, :0],
                                             dv=v[:, :, -keep:] if keep else v[:, :, :0], mk=mk, mv=mv, t=t + 1)

    def forward(self, x):
        """The sealed Core.forward loop (tbptt=0, no stream encoder), with the states returned for the diagnostics."""
        core = self.core
        e, _, _ = core.encode(x)
        n = core.mn(e)
        mk = core.rope(core.split(core.mk(n))); mv = core.split(core.mv(n))
        s, outs, states = None, [], []
        for t in range(x.shape[1]):
            y, s = self.decode(e[:, t], s, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
            outs.append(y); states.append(s['s'])
        return torch.stack(outs, 1), torch.stack(states, 1)
