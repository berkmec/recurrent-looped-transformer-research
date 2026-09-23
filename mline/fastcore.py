"""M0 speed work: the sealed Core's forward pass with STATIC shapes at every timestep.

Measured obstacle: the decoder attends over a growing memory (`mk[:, :, :t+1]`) and over a window that fills up over
the first few steps, so tensor shapes change at every t. torch.compile therefore recompiles per timestep - 1 709 CPU
seconds produced no completed step - and a CUDA graph cannot be captured at all.

The fix is to change the shapes, not the arithmetic: keep the memory at full length and mask positions after t, and
keep the window at a fixed `window` slots with a validity mask instead of a growing concatenation. Softmax over a
masked full range equals softmax over the slice, so this is the same function, expressed with shapes a compiler can
work with.

`core/model.py` is sealed, so this lives beside it and is asserted equal to `Core.forward` by tests/test_fastcore.py.
Equality is checked numerically rather than assumed: masking and slicing can take different kernel paths, and B2
already taught this campaign what an unnoticed representation change does to comparability.
"""
import torch
import torch.nn.functional as F


def _attend_masked(core, q, k, v, mask):
    """One query position against a masked key/value block, reshaped the way Core.attend does."""
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], core.width)[:, 0]


def forward_static(core, x, return_states=False):
    """Core.forward(x) with per-step shapes that never change. Same result, compilable shapes."""
    B, T = x.shape
    e, _, _ = core.encode(x)
    n = core.mn(e)
    mk = core.rope(core.split(core.mk(n)))
    mv = core.split(core.mv(n))
    heads, dh = mk.shape[1], mk.shape[3]
    window = core.window
    kbuf = torch.zeros(B, heads, window, dh, device=x.device, dtype=mk.dtype)
    vbuf = torch.zeros_like(kbuf)
    prev = core.initial.expand(B, -1)
    positions = torch.arange(T, device=x.device)
    slots = torch.arange(window, device=x.device)
    outs, states = [], []
    for t in range(T):
        z = e[:, t]
        if core.feedback:
            r = core.pn(prev)
            z = z + core.alpha * torch.sigmoid(core.gate(torch.cat([e[:, t], r], -1))) * core.merge(r)
        nz = core.dn(z)
        q = core.rope(core.split(core.dq(nz)), t)
        kbuf = torch.cat([kbuf[:, :, 1:], core.rope(core.split(core.dk(nz)), t)], 2)
        vbuf = torch.cat([vbuf[:, :, 1:], core.split(core.dv(nz))], 2)
        valid = window if t + 1 >= window else t + 1
        z = z + core.do(_attend_masked(core, q, kbuf, vbuf, (slots >= window - valid).view(1, 1, 1, window)))
        q2 = core.rope(core.split(core.cq(core.cn(z))), t)
        z = z + core.co(_attend_masked(core, q2, mk, mv, (positions <= t).view(1, 1, 1, T)))
        z = z + core.dff(core.dfn(z))
        prev = z
        outs.append(core.read(core.outn(z)))
        if return_states:
            states.append(z)
    logits = torch.stack(outs, 1)
    return (logits, torch.stack(states, 1)) if return_states else logits


def forward_static_gated(model, x):
    """The same treatment for GatedCore: soft and hard_st, states kept on the graph."""
    core = model.core
    B, T = x.shape
    e, _, _ = core.encode(x)
    n = core.mn(e)
    mk = core.rope(core.split(core.mk(n)))
    mv = core.split(core.mv(n))
    heads, dh = mk.shape[1], mk.shape[3]
    window = core.window
    kbuf = torch.zeros(B, heads, window, dh, device=x.device, dtype=mk.dtype)
    vbuf = torch.zeros_like(kbuf)
    committed = core.initial.expand(B, -1)
    state_k, state_v = None, None
    positions = torch.arange(T, device=x.device)
    slots = torch.arange(window, device=x.device)
    outs, gates, states = [], [], []
    for t in range(T):
        z = e[:, t]
        if core.feedback:
            r = core.pn(committed)
            z = z + core.alpha * torch.sigmoid(core.gate(torch.cat([e[:, t], r], -1))) * core.merge(r)
        nz = core.dn(z)
        q = core.rope(core.split(core.dq(nz)), t)
        k_new = core.rope(core.split(core.dk(nz)), t)
        v_new = core.split(core.dv(nz))
        kbuf = torch.cat([kbuf[:, :, 1:], k_new], 2)
        vbuf = torch.cat([vbuf[:, :, 1:], v_new], 2)
        valid = window if t + 1 >= window else t + 1
        wmask = (slots >= window - valid).view(1, 1, 1, window)
        cand = z + core.do(_attend_masked(core, q, kbuf, vbuf, wmask))
        q2 = core.rope(core.split(core.cq(core.cn(cand))), t)
        cand = cand + core.co(_attend_masked(core, q2, mk, mv, (positions <= t).view(1, 1, 1, T)))
        cand = cand + core.dff(core.dfn(cand))
        raw = torch.sigmoid(model.head(torch.cat([e[:, t], committed], -1)).squeeze(-1))
        g = raw if model.mode == 'soft' else (raw > 0.5).to(raw.dtype) + raw - raw.detach()
        if model.mode == 'soft':
            s = g.view(-1, 1) * cand + (1 - g.view(-1, 1)) * committed
        else:
            m = (g.detach() > 0.5).view(-1, 1)
            s = torch.where(m, cand, committed) + (g - g.detach()).view(-1, 1) * (cand - committed)
            if state_k is not None:
                m4 = m.view(-1, 1, 1, 1)
                kbuf = torch.where(m4, kbuf, state_k); vbuf = torch.where(m4, vbuf, state_v)
        state_k, state_v = kbuf, vbuf
        committed = s
        outs.append(core.read(core.outn(s))); gates.append(g); states.append(s)
    return torch.stack(outs, 1), torch.stack(gates, 1), torch.stack(states, 1)
