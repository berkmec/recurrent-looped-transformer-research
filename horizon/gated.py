"""U2c: a learned HOLD/WRITE gate over the immutable Core (package P1, 'kanal C').

At every step the Core proposes a candidate state; the committed state is either the candidate (WRITE) or exactly the
previous committed state (HOLD). The gate head reads [e_t, s_{t-1}] — input and committed state only, no oracle.
  soft    — s_t = g·cand + (1-g)·s_{t-1}, differentiable, NOT exact (package: a soft gate does not preserve bitwise)
  hard_st — forward uses the hard decision (exact copy on HOLD), backward passes the sigmoid gradient (straight-through)
  oracle  — gate = run-aware neutral mask (HOLD from the (k+1)-th consecutive neutral): the U2b ceiling as a reference
On HOLD the hard modes also keep the previous window cache (dk/dv), as the freeze arm does. Readout is from the committed state.
"""
import torch
from torch import nn

from horizon import interventions as iv
from horizon.interventions_runaware import run_length

MODES = ('soft', 'hard_st', 'oracle')


class GatedCore(nn.Module):
    def __init__(self, core, mode='hard_st', k=2):
        super().__init__()
        if mode not in MODES:
            raise ValueError('mode must be one of ' + ', '.join(MODES))
        self.core = core; self.mode = mode; self.k = k
        width = core.initial.shape[-1]
        self.head = nn.Linear(2 * width, 1)
        with torch.no_grad():
            self.head.weight.normal_(0., 0.02); self.head.bias.fill_(2.)  # WRITE prior: starts close to the plain Core
        self.last_states = None

    def gate_parameters(self):
        return self.head.parameters()

    def open_all(self):
        with torch.no_grad():
            self.head.weight.zero_(); self.head.bias.fill_(20.)

    def close_all(self):
        with torch.no_grad():
            self.head.weight.zero_(); self.head.bias.fill_(-20.)

    def forward(self, x):
        core = self.core
        e, _, _ = core.encode(x)
        n = core.mn(e)
        mk = core.rope(core.split(core.mk(n))); mv = core.split(core.mv(n))
        B, T = x.shape
        committed = core.initial.expand(B, -1)
        lengths = run_length(x) if self.mode == 'oracle' else None
        state = None
        outs, gates, glogits, states = [], [], [], []
        for t in range(T):
            z, cand = core.decode(e[:, t], state, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
            if self.mode == 'oracle':
                g = (lengths[:, t] <= self.k).to(e.dtype); gl = (g * 2 - 1) * 12.
            else:
                gl = self.head(torch.cat([e[:, t], committed], -1)).squeeze(-1)
                gs = torch.sigmoid(gl)
                g = gs if self.mode == 'soft' else (gs > 0.5).to(gs.dtype) + gs - gs.detach()  # straight-through
            if self.mode == 'soft':
                gv = g.view(-1, 1)
                s = gv * cand['s'] + (1 - gv) * committed
                dk, dv = cand['dk'], cand['dv']
            else:
                m = (g.detach() > 0.5).view(-1, 1)
                s = torch.where(m, cand['s'], committed) + (g - g.detach()).view(-1, 1) * (cand['s'] - committed)  # exact value, soft gradient
                dk, dv = cand['dk'], cand['dv']
                if state is not None and state['dk'].shape == dk.shape:
                    m4 = m.view(-1, 1, 1, 1)
                    dk = torch.where(m4, dk, state['dk']); dv = torch.where(m4, dv, state['dv'])
            state = dict(cand, s=s, dk=dk, dv=dv); committed = s
            outs.append(core.read(core.outn(s))); gates.append(g); glogits.append(gl); states.append(s)
        self.last_states = torch.stack(states, 1).detach()
        return torch.stack(outs, 1), torch.stack(gates, 1), torch.stack(glogits, 1)


def gate_target(x, k=2):
    """1 = WRITE expected, 0 = HOLD expected: the run-aware neutral rule (HOLD from the (k+1)-th consecutive neutral)."""
    return (run_length(x) <= k).float()


def gate_loss(gate_logits, target):
    return nn.functional.binary_cross_entropy_with_logits(gate_logits, target)


def gate_rates(gates, target):
    """false_hold: WRITE-expected positions that were held; false_write: HOLD-expected positions that were written."""
    hard = (gates.detach() > 0.5).float(); t = target.float()
    write_pos = t == 1; hold_pos = t == 0
    return dict(false_hold=float((hard[write_pos] == 0).float().mean()) if bool(write_pos.any()) else None,
                false_write=float((hard[hold_pos] == 1).float().mean()) if bool(hold_pos.any()) else None,
                write_positions=int(write_pos.sum()), hold_positions=int(hold_pos.sum()), hold_rate=float((hard == 0).float().mean()))
