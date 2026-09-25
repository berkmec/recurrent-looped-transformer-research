"""U5: the same HOLD/WRITE gate, but with a switch for what the gate is allowed to read.

WHY. Measured on the M-line: 2068 bytes/s for a 30M model, because the decoder loop runs sequentially over 512
timesteps. Mamba and S4 meet the same shape of limit and defeat it with a parallel scan, which requires the
recurrence to be

    s_t = a_t * s_{t-1} + b_t      with a_t and b_t independent of s

Our update is `s_t = g_t*cand_t + (1-g_t)*s_{t-1}` and neither coefficient qualifies: the gate reads
`[e_t, s_{t-1}]`, and the candidate carries the Core's additive feedback `z = e + alpha*sigmoid(gate([e,r]))
*merge(r)` with `r = pn(s_{t-1})`. Two state dependences, and a scan needs both gone.

An input-only gate is therefore NECESSARY for a scan and **not sufficient** - the candidate's feedback would
have to go as well, which is a far larger change and would leave something close to a gated linear RNN over the
encoder outputs. This module tests the cheap condition first, and only that: does exact preservation - the one
result this campaign replicated across 3 seeds x 2 task families x 2 learning rates - survive the gate losing
its view of the state? A no ends the enquiry cheaply. A yes makes the second, expensive question worth asking.

`gate_input='state'` reproduces the sealed `horizon/gated.GatedCore` bit for bit, which tests/test_scanable.py
checks under gates that actually hold, not only at the WRITE-prior initialisation where every gate is open and
the differing branch never runs. `core/model.py` and `horizon/gated.py` are hashed inside sealed manifests, so
this variant lives beside them and never edits them.
"""
import torch
from torch import nn

from horizon.interventions_runaware import run_length

MODES = ('soft', 'hard_st', 'oracle')
GATE_INPUTS = ('state', 'input')


class ScanGatedCore(nn.Module):
    def __init__(self, core, mode='hard_st', k=2, gate_input='state'):
        super().__init__()
        if mode not in MODES:
            raise ValueError('mode must be one of ' + ', '.join(MODES))
        if gate_input not in GATE_INPUTS:
            raise ValueError('gate_input must be one of ' + ', '.join(GATE_INPUTS))
        self.core = core; self.mode = mode; self.k = k; self.gate_input = gate_input
        width = core.initial.shape[-1]
        # The ONLY difference between the arms: how much the gate head is allowed to see. Everything else -
        # initialisation, the WRITE prior, the straight-through rule, the window rollback - is held identical,
        # because a comparison whose arms differ in two things cannot attribute its own outcome.
        self.head = nn.Linear(width if gate_input == 'input' else 2 * width, 1)
        with torch.no_grad():
            self.head.weight.normal_(0., 0.02); self.head.bias.fill_(2.)   # WRITE prior, as in the sealed gate
        self.last_states = None

    def gate_parameters(self):
        return self.head.parameters()

    def open_all(self):
        with torch.no_grad():
            self.head.weight.zero_(); self.head.bias.fill_(20.)

    def close_all(self):
        with torch.no_grad():
            self.head.weight.zero_(); self.head.bias.fill_(-20.)

    def gate_logits(self, e_t, committed):
        if self.gate_input == 'input':
            return self.head(e_t).squeeze(-1)
        return self.head(torch.cat([e_t, committed], -1)).squeeze(-1)

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
                gl = self.gate_logits(e[:, t], committed)
                gs = torch.sigmoid(gl)
                g = gs if self.mode == 'soft' else (gs > 0.5).to(gs.dtype) + gs - gs.detach()  # straight-through
            if self.mode == 'soft':
                gv = g.view(-1, 1)
                s = gv * cand['s'] + (1 - gv) * committed
                dk, dv = cand['dk'], cand['dv']
            else:
                m = (g.detach() > 0.5).view(-1, 1)
                s = torch.where(m, cand['s'], committed) + (g - g.detach()).view(-1, 1) * (cand['s'] - committed)
                dk, dv = cand['dk'], cand['dv']
                if state is not None and state['dk'].shape == dk.shape:
                    m4 = m.view(-1, 1, 1, 1)
                    dk = torch.where(m4, dk, state['dk']); dv = torch.where(m4, dv, state['dv'])
            state = dict(cand, s=s, dk=dk, dv=dv); committed = s
            outs.append(core.read(core.outn(s))); gates.append(g); glogits.append(gl); states.append(s)
        self.last_states = torch.stack(states, 1).detach()
        return torch.stack(outs, 1), torch.stack(gates, 1), torch.stack(glogits, 1)


def scan_conditions(model):
    """What still stands between this model and a parallel scan, reported rather than assumed."""
    return dict(
        gate_is_state_free=model.gate_input == 'input',
        candidate_is_state_free=not getattr(model.core, 'feedback', True),
        scannable=model.gate_input == 'input' and not getattr(model.core, 'feedback', True),
        note='a scan needs BOTH coefficients free of s. This cycle moves only the gate; the candidate keeps the '
             'Core additive feedback, so `scannable` is expected to be False even in the arm under test.')
