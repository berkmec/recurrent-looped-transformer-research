"""E-RM0R (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.46): the toy E-RM task repaired so that it CAN be learned,
then the positive control first.

Stage 0 (erm_run / erm_tasks, sealed) could not have shown anything about memory: (1) every fact's key is drawn WITH replacement,
so the queried key is unique in only 38 % of sequences and a perfect memory reaches only 0.70 (the G0.3 threshold was 0.95);
(2) one supervised answer per 128-token sequence; (3) even the full-context arm `Apr` (one causal encoder over all four windows,
which can look straight from the query to the fact) stayed at chance in 6,000 updates. Here, with everything else as in stage 0
(arms from erm_models, AdamW lr 3e-3 with the memory modules decay-free, clip 1.0, batch 32):
  * keys are distinct inside a sequence (a permutation): the ceiling is 1.0 (tested);
  * the query window holds `queries` queries about distinct facts, every answer supervised;
  * an optional curriculum on the number of facts (e.g. 4,8,16), advancing when the last 200 updates reach 0.9.
erm_tasks.generate and erm_run are unchanged.

  python -B -m horizon.erm_r train --arm Apr --steps 12000 --seed 1 --out results/ERM-R/Apr-s1.json
  python -B -m horizon.erm_r --reading results/ERM-R/Apr-s1.json results/ERM-R/A-s1.json [results/ERM-R/B-s1.json ...]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import erm_models as em
from . import erm_tasks as et

QUERIES = 8
ADVANCE_AT = 0.9
ADVANCE_WINDOW = 200


def generate(n, facts=et.N_FACTS, queries=QUERIES, seed=0, windows=et.N_WINDOWS, window=et.WINDOW, with_facts=False):
    """x (n, windows*window); answer_pos and target (n, queries); d (n, queries) = query window - fact window.
    Facts (distinct keys, values with replacement) only in windows 0..windows-2; the queries, about distinct facts, in the last.
    with_facts: also (n, facts, 2) = (window, in-window position of the VALUE) in generation order, the cells the oracle-write
    arm B_orc writes (fact f into slot f mod 16, as erm_tasks.fact_cells_tensor gives stage 0)."""
    qw = windows - 1
    if queries > facts or 2 * queries > window:
        raise ValueError('too many queries')
    rng = np.random.default_rng(seed)
    quota = [facts // qw + (1 if i < facts % qw else 0) for i in range(qw)]
    if not all(2 * c <= window - 4 for c in quota):
        raise ValueError('facts do not fit the windows')
    x = np.empty((n, windows * window), dtype=np.int64)
    apos = np.empty((n, queries), dtype=np.int64)
    tgt = np.empty((n, queries), dtype=np.int64)
    dist = np.empty((n, queries), dtype=np.int64)
    cells = np.empty((n, facts, 2), dtype=np.int64)
    for s in range(n):
        seq = rng.choice(et.FILLERS, size=windows * window).astype(np.int64)
        taken = [set() for _ in range(windows)]
        where = list(rng.permutation([i for i, c in enumerate(quota) for _ in range(c)]))
        keys = rng.permutation(et.N_KEYS)[:facts]
        pairs = []
        for f in range(facts):
            w = where[f]
            v = et.VALS[int(rng.integers(0, et.N_VALS))]
            at, taken[w] = et._place(rng, taken[w], w * window, (w + 1) * window, 2)
            seq[at], seq[at + 1] = et.KEYS[int(keys[f])], v
            pairs.append((w, et.KEYS[int(keys[f])], v))
            cells[s, f] = (w, at + 1 - w * window)
        for j, f in enumerate(rng.permutation(facts)[:queries]):
            w, k, v = pairs[int(f)]
            at, taken[qw] = et._place(rng, taken[qw], qw * window, (qw + 1) * window, 2)
            seq[at], seq[at + 1] = et.QMARK, k
            apos[s, j], tgt[s, j], dist[s, j] = at + 1, v, qw - w
        x[s] = seq
    out = (torch.from_numpy(x), torch.from_numpy(apos), torch.from_numpy(tgt), torch.from_numpy(dist))
    return out + (torch.from_numpy(cells),) if with_facts else out


def ceiling(x, apos, tgt):
    """Accuracy of a perfect memory: the most frequent value among the facts whose key is the queried key."""
    xs, ap, tg = x.numpy(), apos.numpy(), tgt.numpy()
    best = []
    for s in range(xs.shape[0]):
        for j in range(ap.shape[1]):
            k = xs[s, ap[s, j]]
            fact_part = xs[s, :(et.N_WINDOWS - 1) * et.WINDOW]
            vals = [int(fact_part[i + 1]) for i in range(len(fact_part) - 1) if fact_part[i] == k]
            best.append(max(vals.count(v) for v in set(vals)) / len(vals))
    return float(np.mean(best))


class Positional(torch.nn.Module):
    """E-RM0R2: the token embedding plus fixed sinusoidal positions (no parameters). Windowed arms encode each 32-token window
    on its own, so positions count from 0 inside a window; the full-context arm `Apr` encodes all 128 tokens, so its positions
    are absolute. Stage 0's encoder had none: causal attention without positions cannot easily find 'the token before me',
    which associative recall (query key -> its earlier occurrence -> the next token, the value) needs."""

    def __init__(self, embedding, max_len=et.N_WINDOWS * et.WINDOW):
        super().__init__()
        self.tok = embedding
        d = embedding.embedding_dim
        pos = torch.arange(max_len, dtype=torch.float32)[:, None]
        freq = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2, dtype=torch.float32) / d)
        pe = torch.zeros(max_len, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        self.register_buffer('pe', pe, persistent=False)

    @property
    def embedding_dim(self):
        return self.tok.embedding_dim

    def forward(self, x):
        return self.tok(x) + self.pe[:x.shape[-1]]      # positions run along the last axis: (b, T) or B_rmt's (b, windows, w)


def delta_write(M, k, v, beta):
    """One delta-rule write of a batch: M (b, d_v, d_k) <- M + beta (v - M k) k^T, k (b, d_k) unit-norm, v (b, d_v), beta (b, 1).
    With beta = 1 and a unit k, reading M k afterwards returns v exactly (the old content along k is replaced, not added to)."""
    return M + torch.einsum('bv,bk->bvk', beta * (v - torch.einsum('bvk,bk->bv', M, k)), k)


class ArmDelta(torch.nn.Module):
    """E-RM0R5 (note 5.54): RLT-M with a delta-rule associative memory written token by token (fast weights) instead of stage 0's
    slot write, which pools a whole window into 16 slots and leaves each slot a mixture (E-RM0R4: with an oracle write the read
    works, 0.9994; with the learned pooled write it does not, 0.165). The toy window encoder (positions as in E-RM0R2) feeds a memory
    M (d_v x d_k), zero at the start and carried across windows. Every position of every window writes with the delta rule
    (k = unit(Wk e), v = Wv e, beta = sigmoid(wb . e): the model learns which tokens to write); each window first READS what the
    earlier windows wrote, o = M unit(Wq e), and adds z = e + gamma * Wo(o), gamma started open (E-RM0R4). Fixed-size memory:
    O(1) per step (K4); gamma = 0 closes the path exactly (K3)."""

    def __init__(self, vocab=et.VOCAB, width=64, d_k=64, d_v=64, seed=0, gamma_init=1.0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.enc = em.Encoder(vocab, width)
            self.Wq, self.Wk = torch.nn.Linear(width, d_k), torch.nn.Linear(width, d_k)
            self.Wv, self.wb = torch.nn.Linear(width, d_v), torch.nn.Linear(width, 1)
            self.Wo = torch.nn.Linear(d_v, width)
            self.gamma = torch.nn.Parameter(torch.full((width,), float(gamma_init)))
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, vocab))
        self.d_k, self.d_v = d_k, d_v

    def gate_parameters(self):
        """The memory's parameters: decay-free, as erm_run treats stage 0's write / read modules."""
        return [p for m in (self.Wq, self.Wk, self.Wv, self.wb, self.Wo) for p in m.parameters()] + [self.gamma]

    def forward(self, xw):
        b, nw, w = xw.shape
        e = self.enc.forward_windows(xw)
        M = torch.zeros(b, self.d_v, self.d_k, device=xw.device)
        outs = []
        for t in range(nw):
            et_ = e[:, t]
            o = torch.einsum('bvk,bwk->bwv', M, F.normalize(self.Wq(et_), dim=-1))
            outs.append(self.head(et_ + self.gamma * self.Wo(o)))
            k, v, beta = F.normalize(self.Wk(et_), dim=-1), self.Wv(et_), torch.sigmoid(self.wb(et_))
            for i in range(w):
                M = delta_write(M, k[:, i], v[:, i], beta[:, i])
        return torch.stack(outs, 1)


def make(arm, seed=1, positions=False, gamma_init=None):
    """An arm from erm_models, with sinusoidal positions in its encoder if `positions` (E-RM0R2), and its memory read gate gamma
    started at `gamma_init` instead of 0 (E-RM0R4). gamma = 0 still closes the read path exactly (K3), so closability is kept;
    only where training starts changes: at gamma = 0 every read parameter has zero gradient. `B_delta` is E-RM0R5's arm (its
    gate starts at gamma_init, 1 if not given)."""
    if arm == 'B_delta':
        model = ArmDelta(seed=seed, gamma_init=1.0 if gamma_init is None else gamma_init)
    else:
        model = em.make_arm(arm, seed=seed)
    if positions:
        if not hasattr(model, 'enc'):
            raise ValueError('arm %s has no encoder to give positions to' % arm)
        model.enc.embedding = Positional(model.enc.embedding)
    if gamma_init is not None and arm != 'B_delta':
        if not hasattr(model, 'read') or not hasattr(model.read, 'gamma'):
            raise ValueError('arm %s has no read gate gamma' % arm)
        with torch.no_grad():
            model.read.gamma.fill_(float(gamma_init))
    return model


def answers(logits, apos):
    """Logits at the answer positions: (batch, queries, vocab); every arm returns (batch, windows, window, vocab)."""
    b = logits.shape[0]
    flat = logits.reshape(b, -1, logits.shape[-1])
    return flat[torch.arange(b)[:, None], apos]


def train(arm, seed, steps, batch=32, lr=3e-3, wd=0.01, queries=QUERIES, curriculum=None, log_every=500, log=print, positions=False,
          gamma_init=None):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = make(arm, seed=seed, positions=positions, gamma_init=gamma_init)
    gate = list(model.gate_parameters()) if hasattr(model, 'gate_parameters') else []
    taken = {id(p) for p in gate}
    rest = [p for p in model.parameters() if id(p) not in taken]
    opt = torch.optim.AdamW(([dict(params=rest)] if rest else []) + ([dict(params=gate, weight_decay=0.0)] if gate else []),
                            lr=lr, weight_decay=wd)
    stages = list(curriculum) if curriculum else [et.N_FACTS]
    stage, recent, history, advanced = 0, [], [], []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        facts = stages[stage]
        x, apos, tgt, _, cells = generate(batch, facts=facts, queries=min(queries, facts), seed=(seed * 1_000_003 + step) % 2 ** 31,
                                          with_facts=True)
        model.train()
        xw = x.reshape(batch, et.N_WINDOWS, et.WINDOW)
        sel = answers(model(xw, cells) if arm == 'B_orc' else model(xw), apos)
        loss = F.cross_entropy(sel.reshape(-1, sel.shape[-1]), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        recent.append(float((sel.argmax(-1) == tgt).float().mean()))
        recent = recent[-ADVANCE_WINDOW:]
        if stage + 1 < len(stages) and len(recent) == ADVANCE_WINDOW and np.mean(recent) >= ADVANCE_AT:
            stage += 1
            advanced.append(dict(step=step, facts=stages[stage]))
            recent = []
        if step % log_every == 0 or step == steps:
            history.append(dict(step=step, loss=float(loss.detach()), train_acc=float(np.mean(recent)) if recent else None, facts=facts))
            log('%s s%d step %d facts %d loss %.4f acc %s (%.0fs)' % (arm, seed, step, facts, float(loss.detach()),
                                                                    history[-1]['train_acc'], time.perf_counter() - t0))
    return model, dict(arm=arm, seed=seed, steps=steps, batch=batch, lr=lr, weight_decay=wd, queries=queries,
                       curriculum=stages, advanced=advanced, final_facts=stages[stage], history=history,
                       train_seconds=time.perf_counter() - t0, positions='sinusoidal' if positions else None, gamma_init=gamma_init)


def evaluate(model, n=600, seed=777, queries=QUERIES, batch=50, arm=None):
    """Accuracy per fact distance d (over every query) and overall, at the full 16 facts."""
    torch.set_num_threads(1)
    model.eval()
    x, apos, tgt, dist, cells = generate(n, facts=et.N_FACTS, queries=queries, seed=seed, with_facts=True)
    hits = []
    with torch.no_grad():
        for at in range(0, n, batch):
            xw = x[at:at + batch].reshape(-1, et.N_WINDOWS, et.WINDOW)
            sel = answers(model(xw, cells[at:at + batch]) if arm == 'B_orc' else model(xw), apos[at:at + batch])
            hits.append(sel.argmax(-1) == tgt[at:at + batch])
    hit = torch.cat(hits)
    out = dict(all=dict(n=int(hit.numel()), acc=float(hit.float().mean())))
    for d in (1, 2, 3):
        m = dist == d
        out[str(d)] = dict(n=int(m.sum()), acc=float(hit[m].float().mean()) if bool(m.any()) else None)
    return out


def reading(results):
    """The registered 5.46 verdicts from finished runs ({arm: result}); arms that did not run are simply absent."""
    acc = {arm: r['eval']['3']['acc'] for arm, r in results.items()}
    out = dict(acc_d3=acc)
    if 'Apr' in acc:
        out['positive_control'] = 'LEARNABLE' if acc['Apr'] >= 0.90 else 'NOT LEARNABLE'
    if 'A' in acc:
        out['control'] = 'CONTROL OK' if acc['A'] <= 0.30 else 'CONTROL BROKEN'
    for arm in ('B', 'B_comp', 'B_orc', 'B_rmt', 'B_delta'):
        if arm in acc and out.get('positive_control') == 'LEARNABLE' and out.get('control') == 'CONTROL OK':
            out[arm] = 'MEMORY WORKS (toy)' if acc[arm] >= 0.90 else ('NOT LEARNED' if acc[arm] <= 0.30 else 'PARTIAL')
        elif arm in acc:
            out[arm] = 'VACUOUS (the positive control or the control failed)'
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', nargs='?', choices=('train',))
    p.add_argument('--arm', choices=em.ARMS + ('B_delta',))
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--queries', type=int, default=QUERIES)
    p.add_argument('--curriculum', default=None, help='facts per stage, e.g. 4,8,16')
    p.add_argument('--positions', action='store_true', help='sinusoidal positions in the encoder (E-RM0R2)')
    p.add_argument('--gamma-init', type=float, default=None, help='start the memory read gate here instead of 0 (E-RM0R4)')
    p.add_argument('--out')
    p.add_argument('--reading', nargs='+', default=None)
    a = p.parse_args()
    if a.reading:
        rs = {json.loads(Path(f).read_bytes().decode('utf8'))['arm']: json.loads(Path(f).read_bytes().decode('utf8')) for f in a.reading}
        print(json.dumps(reading(rs), indent=1))
        return
    cur = [int(v) for v in a.curriculum.split(',')] if a.curriculum else None
    model, info = train(a.arm, a.seed, a.steps, lr=a.lr, queries=a.queries, curriculum=cur, positions=a.positions, gamma_init=a.gamma_init)
    info['eval'] = evaluate(model, queries=a.queries, arm=a.arm)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), meta=dict(arm=a.arm, seed=a.seed)), out.with_suffix('.pt'))
    out.write_bytes((json.dumps(info, indent=1) + '\n').encode('utf8'))
    print(json.dumps(info['eval']))


if __name__ == '__main__':
    main()
