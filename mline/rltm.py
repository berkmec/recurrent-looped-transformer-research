"""RLT-M (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.60): the sealed Core reads a cross-window associative memory
written with the shifted-key delta rule that learned the toy distance task (E-RM0R6). Only new modules: core/model.py is used as
it is (its encode / decode / rope / split and its parameters), never edited.

A row is n windows of w tokens. Every window runs the Core as L1 does, its token state and decoder caches starting from `initial`
(E-L1X: resetting L1 at 512 bytes changed it by 1.9e-8 bit), so the windows interact only through the memory M:

  write, after window j, from the encoder outputs e of that window, for i = 1 .. w-1:
      M <- M + beta_i (v_i - M k_i) k_i^T,   k_i = unit(Wk e[i-1]),  v_i = Wv e[i],  beta_i = sigmoid(wb . e[i])
      (the key comes from the previous token, the value from this one: an induction head's binding built into the write)
  read, in window j, from M_j (what windows 0 .. j-1 wrote; M_0 = 0):
      e'_t = e_t + gamma * Wo(M_j unit(Wq e_t)),   then the Core's cross-attention memory and decoder run on e'

The write depends on the encoder alone, so M_0 .. M_{n-1} exist before any decoder runs, and every window then decodes in
parallel: the decoder loop costs what L1's does, the write adds n (w - 1) small steps. M is d_v x d_k whatever the row length (K4);
gamma = 0 makes every window exactly the Core (K3, tested); feedback stays token-level inside the decoder (K1); the encoder memory
and the windowed decoder are the Core's own (K2).
"""
import torch
from torch import nn
from torch.nn import functional as F

from core.model import Core


def delta_write(M, k, v, beta):
    """M (b, d_v, d_k) <- M + beta (v - M k) k^T, k (b, d_k) unit, v (b, d_v), beta (b, 1); horizon.erm_r.delta_write."""
    return M + torch.einsum('bv,bk->bvk', beta * (v - torch.einsum('bvk,bk->bv', M, k)), k)


def decode_windows(core, e):
    """The Core's own sequential decoder path (Core.forward without stream_encoder or tbptt) on encoder outputs e (rows, w, width)."""
    n = core.mn(e)
    mk, mv = core.rope(core.split(core.mk(n))), core.split(core.mv(n))
    s, out = None, []
    for t in range(e.shape[1]):
        y, s = core.decode(e[:, t], s, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
        out.append(y)
    return torch.stack(out, 1)


class RLTM(nn.Module):
    def __init__(self, core, d_k=256, d_v=256, gamma_init=1.0, seed=0, memory=True):
        super().__init__()
        if not isinstance(core, Core):
            raise TypeError('RLT-M wraps the sealed core.model.Core')
        self.core, self.d_k, self.d_v, self.memory = core, d_k, d_v, memory
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.Wq, self.Wk = nn.Linear(core.width, d_k), nn.Linear(core.width, d_k)
            self.Wv, self.wb = nn.Linear(core.width, d_v), nn.Linear(core.width, 1)
            self.Wo = nn.Linear(d_v, core.width)
        self.gamma = nn.Parameter(torch.full((core.width,), float(gamma_init)))

    def gate_parameters(self):
        """The memory's parameters (decay-free in the trainers, as the toy arms)."""
        return [p for m in (self.Wq, self.Wk, self.Wv, self.wb, self.Wo) for p in m.parameters()] + [self.gamma]

    def memories(self, e):
        """e (b, n, w, width) -> the memory each window reads, M_0 .. M_{n-1}: (b, n, d_v, d_k), M_0 = 0."""
        b, n, w, _ = e.shape
        k = F.normalize(self.Wk(e[:, :, :-1]), dim=-1)
        v, beta = self.Wv(e[:, :, 1:]), torch.sigmoid(self.wb(e[:, :, 1:]))
        M, out = e.new_zeros(b, self.d_v, self.d_k), []
        for j in range(n):
            out.append(M)
            for i in range(w - 1):
                M = delta_write(M, k[:, j, i], v[:, j, i], beta[:, j, i])
        return torch.stack(out, 1)

    def forward(self, xw, read=True):
        """xw (b, n, w) tokens -> logits (b, n, w, classes). read=False skips the read (the memory-reset ablation)."""
        b, n, w = xw.shape
        e, _, _ = self.core.encode(xw.reshape(b * n, w))
        if self.memory and read:
            e = e.reshape(b, n, w, -1)
            r = torch.einsum('bnvk,bnwk->bnwv', self.memories(e), F.normalize(self.Wq(e), dim=-1))
            e = (e + self.gamma * self.Wo(r)).reshape(b * n, w, -1)
        return decode_windows(self.core, e).reshape(b, n, w, -1)


class FullRow(nn.Module):
    """The positive control: the same Core over the whole row as one sequence (no windows, no memory)."""

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, xw):
        b, n, w = xw.shape
        return self.core(xw.reshape(b, n * w)).reshape(b, n, w, -1)


def small_core(vocab, width=64, heads=2, window=4, seed=0):
    """A sealed Core at toy size with a `vocab`-entry embedding (as mline.lm.build does with 256 bytes), feedback on."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        core = Core(width=width, heads=heads, window=window, classes=vocab, feedback=True)
        core.embedding = nn.Embedding(vocab, width)
    return core
