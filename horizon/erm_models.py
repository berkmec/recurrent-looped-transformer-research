"""E-RM Aşama 0: oyuncak RLT-M + A/B merdiveni (R-5, kullanıcı onaylı adım 2).

§5.40 modelinin oyuncak ölçeği (genişlik 64, pencere 32, ≤4 pencere): kodlayıcı
pencere-içi nedensel + pencereler-arası bağımsız (paralel); bellek M=16 yuva;
yazma q_i = Wq(m_i + p_i), boş (null) anahtarlı 4-başlı toplama → ŝ_i;
a_i = σ(c + u·[ŝ_i; RMS(m_i)]), c=−2, u=0 ilkleme; m ← λm + aŝ,
λ_i = 1−1/τ_i, τ_i = 8·1024^{i/(M−1)} (M=16'da §5.40 ile aynı: 8…8.192);
okuma pencere çapraz-dikkatinden SONRA (oyuncakta kodlayıcı çıktısı üstüne),
konumsuz 8-başlı bellek dikkati; z ← z + γ⊙Wo(·), γ=0 ilkleme (ReZero).
Gerçek ölçekten sapmalar (Aşama 1'de yok): d_mem=64 (256 değil), kodlayıcı
2-katmanlı oyuncak transformer (L1 değil), okuma kodlayıcı çıktısına uygulanır
(ayrı kod çözücü yok). Kollar: A (belleksiz), B (RLT-M), B_comp (yuva-rekabetli
yazma: a=softmax), B_32 (32 yuva), B_rmt (RMT-sadık: kapısız yuva belirteci
taşıma), Apr (taşınan durum: pencere sıfırlamasız, betimleyici), B0 (M=0 yolu:
okuma sıfır döner; G0.1).
"""
import math

import torch
from torch import nn

from . import erm_tasks as et

ARMS = ('A', 'B', 'B_comp', 'B_32', 'B_rmt', 'Apr', 'B0')


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


class Encoder(nn.Module):
    def __init__(self, vocab, width, layers=2, heads=4, ff=256):
        super().__init__()
        self.width = width
        self.embedding = nn.Embedding(vocab, width)
        layer = nn.TransformerEncoderLayer(width, heads, ff, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        n = x.shape[1]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        return self.norm(self.tr(self.embedding(x), mask=mask, is_causal=True))

    def forward_windows(self, xw):
        b, nw, w = xw.shape
        e = self.forward(xw.reshape(b * nw, w))
        return e.reshape(b, nw, w, self.width)


class WriteModule(nn.Module):
    def __init__(self, d, d_mem, slots, heads=4, competitive=False):
        super().__init__()
        self.slots = slots
        self.heads = heads
        self.d_mem = d_mem
        self.competitive = competitive
        assert d_mem % heads == 0
        self.Wq = nn.Linear(d_mem, d_mem)
        self.pos = nn.Parameter(torch.randn(slots, d_mem) * 0.1)
        self.Wk = nn.Linear(d, d_mem)
        self.Wv = nn.Linear(d, d_mem)
        self.c = nn.Parameter(torch.tensor(-2.0))
        self.u = nn.Parameter(torch.zeros(d_mem + 1))

    def forward(self, e, m, lam):
        """e (B,W,d) pencere kodlaması, m (M,d_mem); → (m_yeni, a (B,M), s_hat)."""
        b, w, _ = e.shape
        h, dh = self.heads, self.d_mem // self.heads
        k = self.Wk(e).view(b, w, h, dh)
        v = self.Wv(e).view(b, w, h, dh)
        q = self.Wq(m + self.pos).view(b, self.slots, h, dh)
        scores = torch.einsum('bmhd,bwhd->bmhw', q, k) / math.sqrt(dh)   # (B,M,h,W)
        null = torch.zeros(b, self.slots, h, 1, device=e.device)
        alpha = torch.softmax(torch.cat([scores, null], -1), -1)[..., :w]
        s_hat = torch.einsum('bmhw,bwhd->bmhd', alpha, v).reshape(b, self.slots, self.d_mem)
        m_rms = m.square().mean(-1).sqrt()                              # (B,M)
        gate_in = torch.cat([s_hat, m_rms[:, :, None]], -1)
        s = self.c + (gate_in * self.u).sum(-1)                         # (B,M)
        a = torch.softmax(s, -1) if self.competitive else torch.sigmoid(s)
        m_new = lam[None, :, None] * m + a[:, :, None] * s_hat
        return m_new, a, s_hat


class ReadModule(nn.Module):
    def __init__(self, d, d_mem, heads=8):
        super().__init__()
        self.heads = heads
        assert d % heads == 0
        self.dh = d // heads
        self.Wqr = nn.Linear(d, d)
        self.Wkr = nn.Linear(d_mem, d)
        self.Wvr = nn.Linear(d_mem, d)
        self.Wo = nn.Linear(d, d)
        self.gamma = nn.Parameter(torch.zeros(d))

    def forward(self, z, m):
        """z (B,W,d); m (B,M,d_mem) ya da None (M=0 yolu: sıfır katkı)."""
        if m is None:
            return z
        b, w, _ = z.shape
        h, dh = self.heads, self.dh
        q = self.Wqr(rms(z)).view(b, w, h, dh)
        k = self.Wkr(rms(m)).view(b, -1, h, dh)
        v = self.Wvr(rms(m)).view(b, -1, h, dh)
        attn = torch.softmax(torch.einsum('bwhd,bmhd->bhmw', q, k) / math.sqrt(dh), -1)
        o = torch.einsum('bhmw,bmhd->bwhd', attn, v).reshape(b, w, -1)
        return z + self.gamma * self.Wo(o)


class RLTM(nn.Module):
    def __init__(self, vocab=et.VOCAB, width=64, slots=16, d_mem=64,
                 competitive=False, seed=0):
        super().__init__()
        self.slots = slots
        self.width = width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.enc = Encoder(vocab, width)
            if slots > 0:
                self.write = WriteModule(width, d_mem, slots, competitive=competitive)
                self.read = ReadModule(width, d_mem)
                tau = 8.0 * (1024.0 ** (torch.arange(slots).float() / max(1, slots - 1)))
                self.register_buffer('lam', 1.0 - 1.0 / tau)
            else:
                self.write, self.read, self.lam = None, None, None
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vocab))
        self.last_a = None
        self.last_m = None

    def gate_parameters(self):
        if self.slots == 0:
            return []
        mods = [self.write, self.read]
        return [p for m in mods for p in m.parameters()]

    def forward(self, xw):
        b, nw, w = xw.shape
        e = self.enc.forward_windows(xw)
        m = (torch.zeros(b, self.slots, self.write.d_mem, device=xw.device)
             if self.slots > 0 else None)
        outs, avals, mrms = [], [], []
        for t in range(nw):
            z = self.read(e[:, t], m) if self.slots > 0 else e[:, t]
            outs.append(self.head(z))
            if self.slots > 0:
                m, a, _ = self.write(e[:, t], m, self.lam)
                avals.append(a.detach())
                mrms.append(float(m.detach().square().mean().sqrt()))
        self.last_a = torch.stack(avals, 1) if avals else None   # (B,nW,M)
        self.last_m_rms = mrms
        return torch.stack(outs, 1)


class ArmA(nn.Module):
    """Belleksiz kol: aynı kodlayıcı, pencereler bağımsız."""

    def __init__(self, vocab=et.VOCAB, width=64, seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.enc = Encoder(vocab, width)
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vocab))

    def forward(self, xw):
        return self.head(self.enc.forward_windows(xw))


class ArmApr(nn.Module):
    """Taşınan durum (betimleyici): pencere sıfırlaması yok, bitişik çıkarım."""

    def __init__(self, vocab=et.VOCAB, width=64, seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.enc = Encoder(vocab, width)
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vocab))

    def forward(self, xw):
        b, nw, w = xw.shape
        return self.head(self.enc(xw.reshape(b, nw * w))).reshape(b, nw, w, -1)


class ArmRMT(nn.Module):
    """B_rmt: RMT-sadık yazma — S yuva belirteci pencere başına taşınır;
    yeni bellek = yazma belirteçlerinin kodlayıcı çıktısı; kapı yok."""

    def __init__(self, vocab=et.VOCAB, width=64, slots_mem=4, seed=0):
        super().__init__()
        self.slots_mem = slots_mem
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.enc = Encoder(vocab, width)
            self.mem0 = nn.Parameter(torch.randn(slots_mem, width) * 0.1)
            self.wr0 = nn.Parameter(torch.randn(slots_mem, width) * 0.1)
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vocab))

    def forward(self, xw):
        b, nw, w = xw.shape
        emb = self.enc.embedding(xw)
        mem = self.mem0.expand(b, -1, -1)
        outs = []
        for t in range(nw):
            inp = torch.cat([mem, emb[:, t],
                             self.wr0.expand(b, -1, -1)], 1)
            h = self.enc.norm(self.enc.tr(inp, mask=None, is_causal=True))
            outs.append(self.head(h[:, self.slots_mem:self.slots_mem + w]))
            mem = h[:, -self.slots_mem:]
        return torch.stack(outs, 1)


def make_arm(arm, seed=1, width=64, slots=16):
    if arm not in ARMS:
        raise ValueError('bilinmeyen kol: ' + str(arm))
    if arm == 'A':
        return ArmA(width=width, seed=seed)
    if arm == 'Apr':
        return ArmApr(width=width, seed=seed)
    if arm == 'B_rmt':
        return ArmRMT(width=width, seed=seed)
    if arm == 'B_comp':
        return RLTM(width=width, slots=slots, competitive=True, seed=seed)
    if arm == 'B_32':
        return RLTM(width=width, slots=32, seed=seed)
    if arm == 'B0':
        return RLTM(width=width, slots=0, seed=seed)
    return RLTM(width=width, slots=slots, seed=seed)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
