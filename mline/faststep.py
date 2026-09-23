"""M0 speed work, second attempt: compile ONE timestep, not the loop.

The first attempt handed the whole 512-iteration loop to torch.compile, which unrolled it; 25 minutes of compilation
produced no step. Static shapes were necessary but not sufficient, because two things still specialise the graph on
the timestep index:

  1. `Core.rope(x, start=t)` builds its angles from a Python int, so every t is a different graph;
  2. the attention masks are rebuilt from `t` each step.

Both are fixed the same way: precompute the rotary tables for all positions once and index them with a TENSOR, and
pass the masks in as tensors. What remains is a single function whose shapes and types never vary, so one compiled
graph serves all 512 calls and a CUDA graph can be captured.

The rotary application is reimplemented here and asserted identical to `Core.rope` in tests/test_faststep.py, because
this is exactly the kind of quiet reimplementation that has to be proved rather than trusted.
"""
import torch
import torch.nn.functional as F


def rope_tables(width_per_head, length, device, dtype):
    """cos/sin for every position, matching Core.rope's angle construction exactly."""
    inv = 10000 ** (-torch.arange(0, width_per_head, 2, device=device, dtype=dtype) / width_per_head)
    angle = torch.arange(length, device=device, dtype=dtype)[:, None] * inv
    return angle.cos(), angle.sin()


def apply_rope(x, cos, sin):
    """x: [B, heads, 1, dh]; cos/sin: [1, 1, 1, dh/2] for this position."""
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)


def make_step(core):
    """Return a function of tensors only: no Python int, no shape that depends on the timestep."""

    def step(e_t, prev, kbuf, vbuf, mk, mv, cos, sin, wmask, mmask):
        z = e_t
        if core.feedback:
            r = core.pn(prev)
            z = z + core.alpha * torch.sigmoid(core.gate(torch.cat([e_t, r], -1))) * core.merge(r)
        nz = core.dn(z)
        q = apply_rope(core.split(core.dq(nz)), cos, sin)
        kbuf = torch.cat([kbuf[:, :, 1:], apply_rope(core.split(core.dk(nz)), cos, sin)], 2)
        vbuf = torch.cat([vbuf[:, :, 1:], core.split(core.dv(nz))], 2)
        a = F.scaled_dot_product_attention(q, kbuf, vbuf, attn_mask=wmask)
        z = z + core.do(a.transpose(1, 2).reshape(z.shape[0], 1, core.width)[:, 0])
        q2 = apply_rope(core.split(core.cq(core.cn(z))), cos, sin)
        b = F.scaled_dot_product_attention(q2, mk, mv, attn_mask=mmask)
        z = z + core.co(b.transpose(1, 2).reshape(z.shape[0], 1, core.width)[:, 0])
        z = z + core.dff(core.dfn(z))
        return core.read(core.outn(z)), z, kbuf, vbuf

    return step


def forward(core, x, step=None, return_states=False):
    """Core.forward(x) driven by a single reusable step function."""
    B, T = x.shape
    device, dtype = x.device, core.initial.dtype
    e, _, _ = core.encode(x)
    n = core.mn(e)
    heads = core.heads
    dh = core.width // heads
    cos_t, sin_t = rope_tables(dh, T, device, dtype)
    mk = _rope_all(core.split(core.mk(n)), cos_t, sin_t)
    mv = core.split(core.mv(n))
    window = core.window
    kbuf = torch.zeros(B, heads, window, dh, device=device, dtype=dtype)
    vbuf = torch.zeros_like(kbuf)
    prev = core.initial.expand(B, -1)
    positions = torch.arange(T, device=device)
    slots = torch.arange(window, device=device)
    step = step or make_step(core)
    outs, states = [], []
    for t in range(T):
        cos = cos_t[t].view(1, 1, 1, -1); sin = sin_t[t].view(1, 1, 1, -1)
        valid = window if t + 1 >= window else t + 1
        wmask = (slots >= window - valid).view(1, 1, 1, window)
        mmask = (positions <= t).view(1, 1, 1, T)
        y, prev, kbuf, vbuf = step(e[:, t], prev, kbuf, vbuf, mk, mv, cos, sin, wmask, mmask)
        outs.append(y)
        if return_states:
            states.append(prev)
    logits = torch.stack(outs, 1)
    return (logits, torch.stack(states, 1)) if return_states else logits


def _rope_all(x, cos_t, sin_t):
    """Rotary over a whole sequence: [B, heads, T, dh], matching Core.rope with start=0."""
    cos = cos_t.view(1, 1, cos_t.shape[0], -1); sin = sin_t.view(1, 1, sin_t.shape[0], -1)
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)


class StepModule(torch.nn.Module):
    """The step as an nn.Module exposing ONLY the parameters the step actually uses.

    torch.cuda.make_graphed_callables differentiates with respect to every parameter the module reports, so handing it
    the whole Core fails: the encoder weights and the memory projections are used once before the loop, never inside
    the step, and autograd refuses. Registering just the fifteen submodules the step touches - shared, not copied, so
    gradients still land on the Core's own tensors - is what makes capture possible.

    CUDA graph capture needs no Triton, which is why it is the only route to removing launch overhead on this host.
    """

    USED = ('pn', 'gate', 'merge', 'dn', 'dq', 'dk', 'dv', 'do', 'cq', 'cn', 'co', 'dff', 'dfn', 'read', 'outn')

    def __init__(self, core):
        super().__init__()
        object.__setattr__(self, '_core_ref', (core,))     # kept out of the module tree on purpose
        for name in self.USED:
            setattr(self, name, getattr(core, name))

    @property
    def core(self):
        return self._core_ref[0]

    def forward(self, e_t, prev, kbuf, vbuf, mk, mv, cos, sin, wmask, mmask):
        core = self.core
        z = e_t
        if core.feedback:
            r = self.pn(prev)
            z = z + core.alpha * torch.sigmoid(self.gate(torch.cat([e_t, r], -1))) * self.merge(r)
        nz = self.dn(z)
        q = apply_rope(core.split(self.dq(nz)), cos, sin)
        kbuf = torch.cat([kbuf[:, :, 1:], apply_rope(core.split(self.dk(nz)), cos, sin)], 2)
        vbuf = torch.cat([vbuf[:, :, 1:], core.split(self.dv(nz))], 2)
        a = F.scaled_dot_product_attention(q, kbuf, vbuf, attn_mask=wmask)
        z = z + self.do(a.transpose(1, 2).reshape(z.shape[0], 1, core.width)[:, 0])
        q2 = apply_rope(core.split(self.cq(self.cn(z))), cos, sin)
        b = F.scaled_dot_product_attention(q2, mk, mv, attn_mask=mmask)
        z = z + self.co(b.transpose(1, 2).reshape(z.shape[0], 1, core.width)[:, 0])
        z = z + self.dff(self.dfn(z))
        return self.read(self.outn(z)), z, kbuf, vbuf
