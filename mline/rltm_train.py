"""E-RM stage 1 at language scale (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.61): L1 with (B_ds) and without (A) the
RLT-M memory (mline.rltm), warm-started from an L1 checkpoint and trained on rows of `windows` x 512 bytes - language rows (consecutive
windows of one article), L1's own within-window fact rows (its recipe's last phase, one example per window) and cross-window fact rows
(mline.naw) - then scored on facts 1 ... 64 windows back, with the memory read switched off as the manipulation check.

    python -B -m mline.rltm_train --arm B_ds --init results/L1-scratch-w928-s401/latest.pt --seed 1 --out results/ERM-stage1/B_ds-s1
    python -B -m mline.rltm_train --reading results/ERM-stage1/*/final.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from horizon import guards as gu
from horizon import sweep as sw
from mline import checkpoint as ck
from mline import decoy_probe as dp
from mline import decoy_train as dt
from mline import lm as ml
from mline import longrun as lr
from mline import naw
from mline import rltm

ARMS = ('A', 'B_ds')
LN2 = math.log(2)
# (last update of the phase or None, shares, cross-window distances): note 5.40's mixture, the cross-window rows first
PHASES = ((300, dict(naw=1.0), (1, 2)), (1000, dict(naw=0.5, lm=0.5), (1, 2, 3)),
          (None, dict(lm=0.5, within=0.15, naw=0.35), (1, 2, 3)))
EVAL_D, FINAL_D = (1, 2, 3, 4, 8, 16), (1, 2, 3, 4, 8, 16, 32, 64)
KINDS = ('held', 'natural')                     # held-out letters T-Z and held-out real words; never trained on


def phase_at(step):
    for limit, shares, distances in PHASES:
        if limit is None or step <= limit:
            return shares, distances


def build(arm, init, seed, d=256, write_chunk=32, width=928, heads=8):
    if arm not in ARMS:
        raise ValueError('arm must be one of ' + ', '.join(ARMS))
    core = ml.build('plain', width=width, heads=heads, seed=401)
    if init:
        blob = torch.load(init, map_location='cpu', weights_only=False)
        core.load_state_dict(blob['model'] if isinstance(blob, dict) and 'model' in blob else blob)
    return rltm.RLTM(core, d_k=d, d_v=d, gamma_init=1.0, seed=seed, memory=(arm == 'B_ds'), write_chunk=write_chunk)


def batch(step, rows, windows, seed, corpus, natural_train, window=512, within_specs=None):
    """(x, y) of (rows, windows * window) bytes and, for the cross-window rows, the answer masks and the distance (0 elsewhere)."""
    shares, distances = phase_at(step)
    n_lm, n_within, n_naw = lr.allocate(rows, [shares.get(k, 0.0) for k in ('lm', 'within', 'naw')])
    length = windows * window
    xs, ys, firsts, fours, ds = [], [], [], [], []
    if n_lm:
        from mline import data as md
        x, y, _ = md.batch(corpus, seed, step, n_lm, length)
        xs.append(torch.from_numpy(x).long()), ys.append(torch.from_numpy(y).long())
        firsts.append(torch.zeros(n_lm, length, dtype=torch.bool)), fours.append(torch.zeros(n_lm, length, dtype=torch.bool))
        ds.append(torch.zeros(n_lm, dtype=torch.long))
    if n_within:
        specs = within_specs or lr.RECIPES['m2-draft'][-1][1]
        x, y, _, _ = lr.synthetic_batch(specs, n_within * windows, seed, step, window, natural_train)
        xs.append(x.reshape(n_within, length)), ys.append(y.reshape(n_within, length))
        firsts.append(torch.zeros(n_within, length, dtype=torch.bool)), fours.append(torch.zeros(n_within, length, dtype=torch.bool))
        ds.append(torch.zeros(n_within, dtype=torch.long))
    single = n_naw - n_naw // 2
    for k, (count, facts, keys) in enumerate(((single, 1, None), (n_naw // 2, 4, natural_train))):
        if count:
            r = naw.naw_rows(count, distances, (seed * 1_000_003 + step * 7 + k) % 2 ** 32, corpus, windows, window, n_facts=facts,
                             decoy_share=0.3, alphabet=dt.ALPHABETS['broad'], key_source=keys)
            xs.append(r['x']), ys.append(r['y']), firsts.append(r['first']), fours.append(r['four']), ds.append(r['d'])
    return torch.cat(xs), torch.cat(ys), torch.cat(firsts), torch.cat(fours), torch.cat(ds)


def _logits(model, x, windows, window, read=True):
    return model(x.reshape(x.shape[0], windows, window), read=read).reshape(x.shape[0], windows * window, -1)


@torch.no_grad()
def eval_naw(model, d, n, kind, corpus, natural_val, seed=4242, window=512, chunk=8, device='cpu', read=True):
    """First-digit and exact-four accuracy on n single-fact rows (decoy share 0.3) with the fact d windows back."""
    model.eval()
    keys = natural_val if kind == 'natural' else None
    r = naw.naw_rows(n, (d,), seed + d, corpus, d + 1, window, n_facts=1, decoy_share=0.3 if keys is None else 0.0,
                     alphabet=dp.ALPHABETS['HELD_KEYS'], key_source=keys)
    first, exact = [], []
    rows = max(1, chunk // max(1, (d + 1) // 4))
    for at in range(0, n, rows):
        x, y = r['x'][at:at + rows].to(device), r['y'][at:at + rows].to(device)
        hit = _logits(model, x, d + 1, window, read).argmax(-1) == y
        four = r['four'][at:at + rows].to(device)
        first.append(hit[r['first'][at:at + rows].to(device)].float())
        exact.append(hit.masked_fill(~four, True).all(-1).float())
    return dict(first=float(torch.cat(first).mean()), exact=float(torch.cat(exact).mean()), n=n)


@torch.no_grad()
def eval_lm(model, val, windows, rows=32, window=512, device='cpu', read=True):
    """Bits per byte by window index on consecutive windows of validation articles."""
    from mline import data as md
    model.eval()
    total = torch.zeros(windows, dtype=torch.float64)
    for step in range(1, rows // 8 + 1):
        x, y, _ = md.batch(val, 991, step, 8, windows * window)
        x, y = torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)
        ce = F.cross_entropy(_logits(model, x, windows, window, read).flatten(0, 1), y.flatten(), reduction='none')
        total += ce.view(x.shape[0], windows, window).mean((0, 2)).double().cpu()
    return [float(v) / (rows // 8) / LN2 for v in total]


def battery(model, corpus_val, natural_val, windows, distances, items, device, window=512):
    out = dict(naw={kind: {str(d): eval_naw(model, d, items, kind, corpus_val, natural_val, window=window, device=device)
                           for d in distances} for kind in KINDS},
               lm_bpb_by_window=eval_lm(model, corpus_val, windows, window=window, device=device))
    if model.memory:
        out['naw_read_off'] = {kind: {str(d): eval_naw(model, d, items, kind, corpus_val, natural_val, window=window, device=device,
                                                       read=False) for d in (1, 2)} for kind in KINDS}
        out['lm_bpb_by_window_read_off'] = eval_lm(model, corpus_val, windows, window=window, device=device, read=False)
        with torch.no_grad():
            out['gamma_abs_mean'] = float(model.gamma.abs().mean())
    return out


def train(out, *, arm, init, seed, steps, corpus, corpus_val, natural_train, natural_val, rows=8, windows=4, window=512, lr_core=3e-4,
          lr_new=9e-4, warmup=100, d=256, write_chunk=32, eval_every=500, ckpt_every=250, items=64, device='cuda', within_specs=None,
          width=928, heads=8, log=print, stop_at=None):
    """`stop_at` ends the process early (a checkpoint, no final battery); a later call with the same `steps` resumes the run."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'final.json').exists():
        return out
    model = build(arm, init, seed, d, write_chunk, width, heads).to(device)
    new = list(model.gate_parameters())
    taken = {id(p) for p in new}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in taken], lr=lr_core, base=lr_core)]
    if model.memory:
        groups.append(dict(params=new, lr=lr_new, base=lr_new, weight_decay=0.0))
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    start, resume = 1, ck.load_latest(out)
    if resume is not None and resume['step'] is not None:
        model.load_state_dict(resume['model'])
        if resume['optimizer'] is not None:
            opt.load_state_dict(resume['optimizer'])
        start = int(resume['step']) + 1
    sw.write_json(out / 'config.json', dict(arm=arm, init=str(init), seed=seed, steps=steps, rows=rows, windows=windows, window=window,
                                           lr_core=lr_core, lr_new=lr_new, warmup=warmup, d=d, write_chunk=write_chunk, items=items,
                                           parameters=sum(p.numel() for p in model.parameters()), phases=[list(p) for p in PHASES]))
    t0 = time.perf_counter()
    for step in range(start, steps + 1):
        for g in opt.param_groups:
            g['lr'] = g['base'] * min(1.0, step / warmup)
        model.train()
        x, y, first, four, dist = (t.to(device) for t in batch(step, rows, windows, seed, corpus, natural_train, window, within_specs))
        logits = _logits(model, x, windows, window)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        report = gu.guarded_step(opt, model.parameters(), loss, clip_norm=1.0)
        with torch.no_grad():
            hit = logits.argmax(-1) == y
            row = dict(step=step, loss=float(report['loss']), applied=bool(report['applied']), grad_norm=float(report['grad_norm_before_clip']),
                       seconds=round(time.perf_counter() - t0, 3))
            for dd in sorted(set(dist[dist > 0].tolist())):
                m = (dist == dd)[:, None] & first
                row['naw_first_d%d' % dd] = float(hit[m].float().mean())
        sw.append_line(out / 'train.jsonl', row)
        if step % ckpt_every == 0 or step == steps or step == stop_at:
            ck.save(out / 'latest.pt', model, opt, step)
        if step == stop_at and step < steps:
            return out
        if step % eval_every == 0 or step == steps:
            final = step == steps
            b = battery(model, corpus_val, natural_val, windows, FINAL_D if final else EVAL_D, items, device, window)
            sw.append_line(out / 'probe.jsonl', dict(step=step, eval=b))
            log('step %d loss %.4f naw held exact %s' % (step, row['loss'], {k: round(v['exact'], 3) for k, v in b['naw']['held'].items()}))
            if final:
                sw.write_json(out / 'final.json', dict(arm=arm, seed=seed, step=step, eval=b))
    return out


def reading(finals, primary='natural'):
    """Note 5.61's registered verdicts from the final batteries ({run: final.json}). Primary cell: held-out real words, exact four
    (L1's own within-window ceiling there is 0.964 at 16,000 updates; on held-out letters it is 0.659, under the 0.8 bar, so the
    letter cell is descriptive only)."""
    acc = lambda r, d, key='naw': r['eval'][key][primary][str(d)]['exact']
    a = [r for r in finals.values() if r['arm'] == 'A']
    b = {name: r for name, r in finals.items() if r['arm'] == 'B_ds'}
    r1 = {name: acc(r, 2) >= 0.8 and acc(r, 2) - acc(r, 2, 'naw_read_off') >= 0.5 for name, r in b.items()}
    a_chance = all(acc(r, 2) <= 0.2 for r in a) if a else None
    r2 = bool(b) and all(r1.values()) and all(acc(r, 16) >= 0.8 for r in b.values()) and a_chance is True
    ok = [r for name, r in b.items() if r1[name]]
    r3 = bool(ok) and float(np.median([acc(r, 64) for r in ok])) >= 0.6
    bpb = lambda r, k='lm_bpb_by_window': r['eval'][k]
    later_gain = [np.mean(bpb(ra)[1:]) - np.mean(bpb(rb)[1:]) for ra in a for rb in b.values()]
    regression = [bpb(rb)[0] - bpb(ra)[0] for ra in a for rb in b.values()]
    no_regression = bool(regression) and max(regression) <= 0.01
    r6 = bool(later_gain) and min(later_gain) >= 0.02
    if r2 and no_regression:
        label = 'WORKS' + (' +EXTRAPOLATES' if r3 else '') + (' +LM' if r6 else '')
    elif b and not any(r1.values()):
        label = 'NOT LEARNED'
    else:
        label = 'PARTIAL'
    return dict(label=label, R1=r1, R2=r2, R3=r3, R6=r6, no_regression=no_regression, a_at_chance=a_chance,
                primary_exact={name: {d: acc(r, d) for d in FINAL_D if str(d) in r['eval']['naw'][primary]} for name, r in finals.items()},
                window1_regression=regression, later_window_gain=later_gain)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--init', default='results/L1-scratch-w928-s401/latest.pt')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--out')
    p.add_argument('--device', default='cuda')
    p.add_argument('--items', type=int, default=64)
    p.add_argument('--reading', nargs='+', default=None)
    a = p.parse_args()
    if a.reading:
        finals = {str(Path(f).parent.name): json.loads(Path(f).read_bytes().decode('utf8')) for f in a.reading}
        print(json.dumps(reading(finals), indent=1, default=float))
        return
    from mline import data as md
    from mline import mqar_eval as me
    from mline import train as mt
    protocol = mt.protocol()
    corpus, corpus_val = md.load(protocol['data'], 'train'), md.load(protocol['data'], 'validation')
    natural_train, natural_val = me.natural_key_sets(corpus, corpus_val, n_train=2000, n_val=300, seed=7)
    train(a.out, arm=a.arm, init=a.init, seed=a.seed, steps=a.steps, corpus=corpus, corpus_val=corpus_val, natural_train=natural_train,
          natural_val=natural_val, items=a.items, device=a.device)


if __name__ == '__main__':
    main()
