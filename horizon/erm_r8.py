"""E-RM0R8 (program note 5.72): does the shifted-key delta memory fail at the language scale because every token writes (interference)?

The toy memory task of E-RM0R6 (16 facts in windows 0-2, 8 queries in window 3; the toy window encoder with sinusoidal positions;
d_k = d_v = 64), with the windows lengthened from 32 to 128 tokens: the writes per window go from 31 to 127 and N/d from 0.48 to 1.98, the
ratio of stage 1 at the language scale (511 / 256 = 2.0), which did not learn (note 5.61.3). The facts stay 16; only the filler between them
grows. With the delta rule an association stored along a key k0 keeps, after N writes of unrelated unit keys, about exp(-beta N / d) of
itself, and the crosstalk of the others grows like N / d: at N / d = 2 and beta ~ 0.5, roughly a third of each fact survives a window and
the noise is twice its size. The fix under test starts the write gate closed, beta = sigmoid(wb . e + b) with b = -4 (beta ~ 0.018 at the
start everywhere), so the model has to open it where a fact is.

Arms (seed 1, 6,000 updates, batch 32, lr 3e-3: E-RM0R6's recipe; writes in the chunkwise-parallel form mline.rltm.chunked_write, tested
equal to the loop):
  ds32    window 32, default write gate        (E-RM0R6's B_delta_shift: the control that learns)
  ds128   window 128, default write gate
  ds128c  window 128, write gate started closed (bias -4)

Reading rule (written before the runs, note 5.72): per arm MEMORY WORKS (toy) iff accuracy at d = 3 >= 0.90, NOT LEARNED iff <= 0.30,
PARTIAL otherwise; then
  INTERFERENCE REPRODUCED   ds32 WORKS and ds128 NOT LEARNED
  CLOSED WRITE FIXES IT     INTERFERENCE REPRODUCED and ds128c WORKS
  NO INTERFERENCE AT 2.0    ds32 WORKS and ds128 WORKS (the hypothesis fails; the language-scale failure has another cause)
  INCONCLUSIVE              anything else
Descriptive: accuracy at d = 1, 2, 3; the mean write gate at the fact-value positions and at all other positions (selectivity).

  python -B -m horizon.erm_r8 train --arm ds128c --out results/ERM-R8/ds128c-s1.json
  python -B -m horizon.erm_r8 reading results/ERM-R8/ds32-s1.json results/ERM-R8/ds128-s1.json results/ERM-R8/ds128c-s1.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from horizon import erm_r as er
from horizon import erm_tasks as et
from mline.rltm import chunked_write

ARMS = dict(ds32=dict(window=32, beta_bias=None), ds128=dict(window=128, beta_bias=None), ds128c=dict(window=128, beta_bias=-4.0))
WORKS, NOT_LEARNED = 0.90, 0.30


class ArmDelta8(er.ArmDelta):
    """E-RM0R6's shifted-key delta arm with a configurable window, the write gate's bias optionally set, and the chunkwise-parallel write."""

    def __init__(self, window, seed=0, beta_bias=None, ut_chunk=64):
        super().__init__(seed=seed, shift_keys=True)
        self.window, self.ut_chunk = window, ut_chunk
        self.enc.embedding = er.Positional(self.enc.embedding, max_len=max(window, et.N_WINDOWS * et.WINDOW))   # positions restart in each window
        if beta_bias is not None:
            with torch.no_grad():
                self.wb.bias.fill_(float(beta_bias))

    def gates(self, e):
        """Per window: unit keys of the previous positions, values and write gates of positions 1 .. w-1."""
        return F.normalize(self.Wk(e[:, :-1]), dim=-1), self.Wv(e[:, 1:]), torch.sigmoid(self.wb(e[:, 1:]))

    def forward(self, xw):
        b, nw, w = xw.shape
        e = self.enc.forward_windows(xw)
        M = torch.zeros(b, self.d_v, self.d_k, device=xw.device)
        outs = []
        for t in range(nw):
            et_ = e[:, t]
            o = torch.einsum('bvk,bwk->bwv', M, F.normalize(self.Wq(et_), dim=-1))
            outs.append(self.head(et_ + self.gamma * self.Wo(o)))
            k, v, beta = self.gates(et_)
            if self.ut_chunk:
                M = chunked_write(M, k, v, beta, self.ut_chunk)
            else:
                for i in range(w - 1):
                    M = er.delta_write(M, k[:, i], v[:, i], beta[:, i])
        return torch.stack(outs, 1)


def make(arm, seed=1):
    cfg = ARMS[arm]
    return ArmDelta8(cfg['window'], seed=seed, beta_bias=cfg['beta_bias'])


def selectivity(model, window, n=200, seed=555):
    """Mean write gate at the positions that hold a fact's value (their key is the token before) and at every other writing position."""
    model.eval()
    x, _, _, _, cells = er.generate(n, facts=et.N_FACTS, queries=er.QUERIES, seed=seed, window=window, with_facts=True)
    with torch.no_grad():
        e = model.enc.forward_windows(x.reshape(n, et.N_WINDOWS, window))
        beta = torch.stack([model.gates(e[:, t])[2][..., 0] for t in range(et.N_WINDOWS)], 1)     # (n, windows, w - 1): position p -> index p - 1
    fact = torch.zeros_like(beta, dtype=torch.bool)
    for s in range(n):
        for w_, p in cells[s].tolist():
            if p >= 1:
                fact[s, w_, p - 1] = True
    return dict(fact=float(beta[fact].mean()), other=float(beta[~fact].mean()))


def train(arm, seed=1, steps=6000, batch=32, lr=3e-3, wd=0.01, log_every=500, log=print):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model, window = make(arm, seed), ARMS[arm]['window']
    gate = list(model.gate_parameters())
    taken = {id(p) for p in gate}
    opt = torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in taken]), dict(params=gate, weight_decay=0.0)],
                            lr=lr, weight_decay=wd)
    history, recent, t0 = [], [], time.perf_counter()
    for step in range(1, steps + 1):
        x, apos, tgt, _, _ = er.generate(batch, facts=et.N_FACTS, queries=er.QUERIES, seed=(seed * 1_000_003 + step) % 2 ** 31,
                                         window=window, with_facts=True)
        model.train()
        sel = er.answers(model(x.reshape(batch, et.N_WINDOWS, window)), apos)
        loss = F.cross_entropy(sel.reshape(-1, sel.shape[-1]), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        recent = (recent + [float((sel.argmax(-1) == tgt).float().mean())])[-200:]
        if step % log_every == 0 or step == steps:
            history.append(dict(step=step, loss=float(loss.detach()), train_acc=float(np.mean(recent)), selectivity=selectivity(model, window, n=50)))
            log('%s s%d step %d loss %.4f acc %.3f beta fact %.3f other %.3f (%.0fs)' % (arm, seed, step, float(loss.detach()), np.mean(recent),
                history[-1]['selectivity']['fact'], history[-1]['selectivity']['other'], time.perf_counter() - t0))
    return model, dict(arm=arm, seed=seed, steps=steps, batch=batch, lr=lr, weight_decay=wd, window=window, beta_bias=ARMS[arm]['beta_bias'],
                       history=history, train_seconds=time.perf_counter() - t0)


def evaluate(model, window, n=600, seed=777, batch=50):
    """Accuracy per fact distance d and overall at the full 16 facts (E-RM0R6's evaluation, at this window)."""
    model.eval()
    x, apos, tgt, dist, _ = er.generate(n, facts=et.N_FACTS, queries=er.QUERIES, seed=seed, window=window, with_facts=True)
    hits = []
    with torch.no_grad():
        for at in range(0, n, batch):
            hits.append(er.answers(model(x[at:at + batch].reshape(-1, et.N_WINDOWS, window)), apos[at:at + batch]).argmax(-1) == tgt[at:at + batch])
    hit = torch.cat(hits)
    out = dict(all=dict(n=int(hit.numel()), acc=float(hit.float().mean())))
    for d in (1, 2, 3):
        m = dist == d
        out[str(d)] = dict(n=int(m.sum()), acc=float(hit[m].float().mean()) if bool(m.any()) else None)
    return out


def arm_verdict(acc):
    return 'MEMORY WORKS (toy)' if acc >= WORKS else ('NOT LEARNED' if acc <= NOT_LEARNED else 'PARTIAL')


def reading(results):
    """{arm: result} -> the registered verdicts (note 5.72)."""
    per = {arm: arm_verdict(r['eval']['3']['acc']) for arm, r in results.items()}
    w = lambda a: per.get(a) == 'MEMORY WORKS (toy)'
    reproduced = w('ds32') and per.get('ds128') == 'NOT LEARNED'
    if reproduced and w('ds128c'):
        verdict = 'CLOSED WRITE FIXES IT'
    elif reproduced:
        verdict = 'INTERFERENCE REPRODUCED'
    elif w('ds32') and w('ds128'):
        verdict = 'NO INTERFERENCE AT 2.0'
    else:
        verdict = 'INCONCLUSIVE'
    return dict(verdict=verdict, arms=per, acc_d3={a: r['eval']['3']['acc'] for a, r in results.items()},
                selectivity={a: r.get('selectivity') for a, r in results.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--arm', choices=tuple(ARMS), required=True)
    t.add_argument('--seed', type=int, default=1)
    t.add_argument('--steps', type=int, default=6000)
    t.add_argument('--out', required=True)
    r = sub.add_parser('reading')
    r.add_argument('results', nargs='+')
    a = p.parse_args()
    if a.cmd == 'reading':
        res = {json.loads(Path(f).read_bytes().decode('utf8'))['train']['arm']: json.loads(Path(f).read_bytes().decode('utf8')) for f in a.results}
        print(json.dumps(reading(res), indent=1))
        return
    model, info = train(a.arm, a.seed, a.steps, log=lambda m: print(m, flush=True))
    window = ARMS[a.arm]['window']
    out = dict(train=info, eval=evaluate(model, window), selectivity=selectivity(model, window))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(json.dumps(out, indent=1).encode('utf8'))
    torch.save(model.state_dict(), str(Path(a.out).with_suffix('.pt')))
    print(json.dumps(dict(eval=out['eval'], selectivity=out['selectivity'])), flush=True)


if __name__ == '__main__':
    main()
