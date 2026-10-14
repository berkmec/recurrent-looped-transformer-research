"""E-RM Aşama 0: oyuncak NAW (Needle-Across-Windows) üretici (R-5, kullanıcı onaylı adım 2).

Protokol (§5.40 G0.3/G0.5, oyuncak ölçeği): 4 pencere × 32 belirteç; 16 olgu
(anahtar+değer çifti, 16 anahtar × 8 değer = 48 bit) OLGULAR YALNIZ 0-2.
pencerelerde; sorgu "[SORU, anahtar]" YALNIZ 3. pencerede rastgele konumda,
hedef = değer (sonraki konum). Dolgu kimlikleri değer kimlikleriyle ayrık
olduğundan sorgulanan değer yapısal olarak sorgu penceresinde YOKTUR (G0.5
özelliği); doldurucu kimlikler anahtar/değer kimlikleriyle de ayrık tutulur,
yani A kolu (pencereler bağımsız) sorguyu şanstan (1/8) yukarı çözemez.

  anahtarlar 0..15, değerler 16..23, SORU=24, dolgular 25..32 → V=33.
"""
import numpy as np
import torch

N_KEYS = 16
N_VALS = 8
KEYS = list(range(0, 16))
VALS = list(range(16, 24))
QMARK = 24
FILLERS = list(range(25, 33))
VOCAB = 33

N_WINDOWS = 4
WINDOW = 32
N_FACTS = 16
CHANCE = 1.0 / N_VALS


def _place(rng, taken, lo, hi, size):
    for _ in range(200):
        at = int(rng.integers(lo, hi - size + 1))
        cells = set(range(at, at + size))
        if not (cells & taken):
            return at, taken | cells
    raise RuntimeError('NAW toy: konum bulunamadı')


def generate(n, windows=N_WINDOWS, window=WINDOW, facts=N_FACTS, seed=0,
             query_window=None, min_fact_window0=2):
    """(x, answer_pos, target, meta): x (n, windows*window) int64; answer_pos
    (n,) sorgu yanıt konumu; target (n,) değer kimliği; meta dizisi
    (sorgu anahtarı, olgu penceresi, d). Olgular 0..windows-2 pencerelerinde,
    sorgu query_window (varsayılan son pencere) penceresinde."""
    qw = windows - 1 if query_window is None else query_window
    assert 0 < qw < windows
    rng = np.random.default_rng(seed)
    T = windows * window
    x = np.empty((n, T), dtype=np.int64)
    answer_pos = np.empty(n, dtype=np.int64)
    target = np.empty(n, dtype=np.int64)
    meta = []
    quota = [facts // qw + (1 if i < facts % qw else 0) for i in range(qw)]
    assert all(c * 2 <= window - 4 for c in quota), 'pencereye sigmaz'
    for s in range(n):
        seq = rng.choice(FILLERS, size=T).astype(np.int64)
        taken = [set() for _ in range(windows)]
        ws = list(rng.permutation([i for i, c in enumerate(quota) for _ in range(c)]))
        pairs = []
        for f in range(facts):
            w = ws[f]
            k = KEYS[int(rng.integers(0, N_KEYS))]
            v = VALS[int(rng.integers(0, N_VALS))]
            at, cells = _place(rng, taken[w], w * window, (w + 1) * window, 2)
            taken[w] = cells
            seq[at], seq[at + 1] = k, v
            pairs.append((w, k, v))
        qi, qk, qv = pairs[int(rng.integers(0, facts))]
        qat, qcells = _place(rng, taken[qw], qw * window, (qw + 1) * window, 2)
        taken[qw] = qcells
        seq[qat], seq[qat + 1] = QMARK, qk
        x[s] = seq
        answer_pos[s] = qat + 1
        target[s] = qv
        meta.append(dict(key=qk, fact_window=qi, query_window=qw, d=qw - qi))
    return (torch.from_numpy(x),
            torch.from_numpy(answer_pos),
            torch.from_numpy(target), meta)


def train_batch(seed, step, batch, windows=N_WINDOWS, window=WINDOW, facts=N_FACTS):
    x, apos, tgt, _ = generate(batch, windows, window, facts,
                               seed=(seed * 1_000_003 + step) % 2**31)
    return dict(x=x, answer_pos=apos, target=tgt)


def queried_value_absent(x, answer_pos, target, windows=N_WINDOWS, window=WINDOW,
                         query_window=None):
    """G0.5 özelliği: sorgulanan değer kimliği, yanıt konumu dışında sorgu
    penceresinde hiç geçmiyor mu (dizi başına bool tensör)."""
    qw = windows - 1 if query_window is None else query_window
    n = x.shape[0]
    ok = torch.ones(n, dtype=torch.bool)
    for s in range(n):
        lo, hi = qw * window, (qw + 1) * window
        win = x[s, lo:hi].clone()
        win[answer_pos[s] - lo] = -1
        ok[s] = bool(((win == int(target[s])).sum()) == 0)
    return ok
