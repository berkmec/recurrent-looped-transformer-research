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
from mline import carry
from mline import rltm

ARMS = ('A', 'B_ds', 'C')                       # C: no memory, the RLT state carried between the windows (mline.carry, note 5.74)
LN2 = math.log(2)
# (last update of the phase or None, shares, cross-window distances): note 5.40's mixture, the cross-window rows first
PHASES = ((300, dict(naw=1.0), (1, 2)), (1000, dict(naw=0.5, lm=0.5), (1, 2, 3)),
          (None, dict(lm=0.5, within=0.15, naw=0.35), (1, 2, 3)))
# the same mixture on rows of two windows, every cross-window fact one window back (E-SC1: the carried state is sequential, so a
# row costs its number of windows times one window's decoder loop)
PHASES_D1 = ((300, dict(naw=1.0), (1,)), (1000, dict(naw=0.5, lm=0.5), (1,)), (None, dict(lm=0.5, within=0.15, naw=0.35), (1,)))
PHASE_SETS = dict(stage1=PHASES, d1=PHASES_D1)
EVAL_D, FINAL_D = (1, 2, 3, 4, 8, 16), (1, 2, 3, 4, 8, 16, 32, 64)
KINDS = ('held', 'natural')                     # held-out letters T-Z and held-out real words; never trained on


def phase_at(step, phases=None):
    for limit, shares, distances in (phases or PHASES):
        if limit is None or step <= limit:
            return shares, distances


def build(arm, init, seed, d=256, write_chunk=32, width=928, heads=8, zero_out=True, ut_chunk=0):
    """L1 from `init` wrapped by RLT-M; `zero_out` starts the read's output at zero so the warm start is L1 itself (note 5.62)."""
    if arm not in ARMS:
        raise ValueError('arm must be one of ' + ', '.join(ARMS))
    core = ml.build('plain', width=width, heads=heads, seed=401)
    if init:
        blob = torch.load(init, map_location='cpu', weights_only=False)
        core.load_state_dict(blob['model'] if isinstance(blob, dict) and 'model' in blob else blob)
    if arm == 'C':
        return carry.CarryLM(core)
    return rltm.RLTM(core, d_k=d, d_v=d, gamma_init=1.0, seed=seed, memory=(arm == 'B_ds'), write_chunk=write_chunk,
                     zero_out=zero_out, ut_chunk=ut_chunk)


def batch(step, rows, windows, seed, corpus, natural_train, window=512, within_specs=None, phases=None):
    """(x, y) of (rows, windows * window) bytes and, for the cross-window rows, the answer masks and the distance (0 elsewhere)."""
    shares, distances = phase_at(step, phases)
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


def objective(logits, y, four, answer_weight=0.0):
    """(loss, the cross-window answers' mean cross-entropy or None). answer_weight 0: the mean over every byte, as note 5.61 trained
    stage 1. w > 0 with answers in the batch: (1 - w) x the mean over the other bytes + w x the mean over the answer bytes. In stage 1
    the answers were 12-32 of 16,384 bytes and 0.2 % of the loss, while the toy where the memory worked trained on them alone (bug
    hunt, note 5.71.2); E-RL0's SFT weights its answers 0.75 the same way."""
    if answer_weight <= 0:
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        with torch.no_grad():
            a = F.cross_entropy(logits[four], y[four]) if bool(four.any()) else None
        return loss, a
    ce = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none').view_as(y)
    if not bool(four.any()):
        return ce.mean(), None
    a = ce[four].mean()
    return (1 - answer_weight) * ce[~four].mean() + answer_weight * a, a


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
    if model.memory or getattr(model, 'carry', False):          # the memory read, or the carried state, switched off
        out['naw_read_off'] = {kind: {str(d): eval_naw(model, d, items, kind, corpus_val, natural_val, window=window, device=device,
                                                       read=False) for d in (1, 2)} for kind in KINDS}
        out['lm_bpb_by_window_read_off'] = eval_lm(model, corpus_val, windows, window=window, device=device, read=False)
    if model.memory:
        with torch.no_grad():
            out['gamma_abs_mean'] = float(model.gamma.abs().mean())
    return out


def train(out, *, arm, init, seed, steps, corpus, corpus_val, natural_train, natural_val, rows=8, windows=4, window=512, lr_core=3e-4,
          lr_new=9e-4, warmup=100, d=256, write_chunk=32, eval_every=500, ckpt_every=250, items=64, device='cuda', within_specs=None,
          width=928, heads=8, log=print, stop_at=None, ut_chunk=0, answer_weight=0.0, phases=None, eval_d=EVAL_D, final_d=FINAL_D):
    """`stop_at` ends the process early (a checkpoint, no final battery); a later call with the same `steps` resumes the run."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'final.json').exists():
        return out
    model = build(arm, init, seed, d, write_chunk, width, heads, ut_chunk=ut_chunk).to(device)
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
                                           lr_core=lr_core, lr_new=lr_new, warmup=warmup, d=d, write_chunk=write_chunk, ut_chunk=ut_chunk, items=items,
                                           answer_weight=answer_weight, ckpt_every=ckpt_every,
                                           parameters=sum(p.numel() for p in model.parameters()),
                                           phases=[list(p) for p in (phases or PHASES)], eval_d=list(eval_d), final_d=list(final_d)))
    t0 = time.perf_counter()
    for step in range(start, steps + 1):
        for g in opt.param_groups:
            g['lr'] = g['base'] * min(1.0, step / warmup)
        model.train()
        x, y, first, four, dist = (t.to(device) for t in batch(step, rows, windows, seed, corpus, natural_train, window, within_specs,
                                                                phases))
        logits = _logits(model, x, windows, window)
        loss, answer_ce = objective(logits, y, four, answer_weight)
        report = gu.guarded_step(opt, model.parameters(), loss, clip_norm=1.0)
        with torch.no_grad():
            hit = logits.argmax(-1) == y
            row = dict(step=step, loss=float(report['loss']), applied=bool(report['applied']), grad_norm=float(report['grad_norm_before_clip']),
                       seconds=round(time.perf_counter() - t0, 3))
            if answer_ce is not None:
                row['answer_bits'] = float(answer_ce) / LN2
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
            b = battery(model, corpus_val, natural_val, windows, final_d if final else eval_d, items, device, window)
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


def reading_objective(final, primary='natural'):
    """E-RM1W's registered verdict (note 5.73) from one B_ds final battery, primary cell held-out real words:
    OBJECTIVE FIX WORKS    exact four at d = 2 >= 0.8 and >= 0.5 above the read-off control (stage 1's R1)
    OBJECTIVE FIX PARTIAL  not WORKS, first digit at d = 1 >= 0.5
    OBJECTIVE FIX FAILS    first digit at d = 1 <= 0.2 (chance 0.1)
    INCONCLUSIVE           otherwise"""
    e = final['eval']
    naw_cells = e['naw'][primary]
    exact2, off2, first1 = naw_cells['2']['exact'], e['naw_read_off'][primary]['2']['exact'], naw_cells['1']['first']
    if exact2 >= 0.8 and exact2 - off2 >= 0.5:
        label = 'OBJECTIVE FIX WORKS'
    elif first1 >= 0.5:
        label = 'OBJECTIVE FIX PARTIAL'
    elif first1 <= 0.2:
        label = 'OBJECTIVE FIX FAILS'
    else:
        label = 'INCONCLUSIVE'
    return dict(label=label, exact_d2=exact2, exact_d2_read_off=off2, first_d1=first1,
                exact={d: c['exact'] for d, c in naw_cells.items()}, first={d: c['first'] for d, c in naw_cells.items()},
                held_exact={d: c['exact'] for d, c in e['naw']['held'].items()},
                lm_bpb_by_window=e['lm_bpb_by_window'], lm_bpb_by_window_read_off=e.get('lm_bpb_by_window_read_off'))


def reading_carry(final, primary='natural'):
    """E-SC1's registered verdict (note 5.74) from the carry arm's final battery, primary cell held-out real words at d = 1 (the
    trained distance; d = 2, two window boundaries, is descriptive):
    STATE CARRIES         exact four >= 0.8 and >= 0.5 above the carry-off control
    STATE CARRIES PARTLY  not the above, first digit >= 0.5 and >= 0.3 above the carry-off control's first digit
    STATE DOES NOT CARRY  first digit <= 0.2 (chance 0.1)
    INCONCLUSIVE          otherwise"""
    e = final['eval']
    on, off = e['naw'][primary]['1'], e['naw_read_off'][primary]['1']
    if on['exact'] >= 0.8 and on['exact'] - off['exact'] >= 0.5:
        label = 'STATE CARRIES'
    elif on['first'] >= 0.5 and on['first'] - off['first'] >= 0.3:
        label = 'STATE CARRIES PARTLY'
    elif on['first'] <= 0.2:
        label = 'STATE DOES NOT CARRY'
    else:
        label = 'INCONCLUSIVE'
    return dict(label=label, first_d1=on['first'], exact_d1=on['exact'], first_d1_off=off['first'], exact_d1_off=off['exact'],
                exact={d: c['exact'] for d, c in e['naw'][primary].items()}, first={d: c['first'] for d, c in e['naw'][primary].items()},
                lm_bpb_by_window=e['lm_bpb_by_window'], lm_bpb_by_window_carry_off=e.get('lm_bpb_by_window_read_off'))


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
    p.add_argument('--smoke', type=int, default=0, help='G0.4: this many updates, no battery; writes smoke.json (s / update, peak VRAM)')
    p.add_argument('--ut-chunk', type=int, default=0, help='the chunkwise-parallel write (chunks of this many writes); 0 = the loop')
    p.add_argument('--d', type=int, default=256, help='memory key / value size (note 5.61: 256; registered fallback 128)')
    p.add_argument('--answer-weight', type=float, default=0.0, help='weight of the cross-window answer bytes in the loss (objective; E-RM1W)')
    p.add_argument('--ckpt-every', type=int, default=250)
    p.add_argument('--reading-objective', default=None, help="one B_ds final.json -> E-RM1W's verdict (note 5.73)")
    p.add_argument('--reading-carry', default=None, help="the carry arm's final.json -> E-SC1's verdict (note 5.74)")
    p.add_argument('--windows', type=int, default=4, help='windows of 512 bytes per training row')
    p.add_argument('--phases', default='stage1', choices=tuple(PHASE_SETS), help='the mixture schedule (d1: every fact one window back)')
    p.add_argument('--eval-d', default=','.join(map(str, EVAL_D)), help='distances of the battery every eval_every updates')
    p.add_argument('--final-d', default=','.join(map(str, FINAL_D)), help='distances of the final battery')
    a = p.parse_args()
    run_kw = dict(windows=a.windows, phases=PHASE_SETS[a.phases], answer_weight=a.answer_weight, ckpt_every=a.ckpt_every,
                  eval_d=tuple(int(v) for v in a.eval_d.split(',')), final_d=tuple(int(v) for v in a.final_d.split(',')))
    if a.reading_objective:
        print(json.dumps(reading_objective(json.loads(Path(a.reading_objective).read_bytes().decode('utf8'))), indent=1, default=float))
        return
    if a.reading_carry:
        print(json.dumps(reading_carry(json.loads(Path(a.reading_carry).read_bytes().decode('utf8'))), indent=1, default=float))
        return
    if a.reading:
        finals = {str(Path(f).parent.name): json.loads(Path(f).read_bytes().decode('utf8')) for f in a.reading}
        print(json.dumps(reading(finals), indent=1, default=float))
        return
    from mline import data as md
    from mline import mqar_eval as me
    from mline import train as mt
    protocol = mt.protocol()
    from mline import corpus_rule
    corpus_rule.check_trainable(protocol['data'])          # user rule 2026-10-24: no training on the CNN corpus (note 5.70)
    corpus, corpus_val = md.load(protocol['data'], 'train'), md.load(protocol['data'], 'validation')
    natural_train, natural_val = me.natural_key_sets(corpus, corpus_val, n_train=2000, n_val=300, seed=7)
    if a.smoke:
        sw.write_json(Path(a.out) / 'smoke.json', smoke(a.out, a.smoke, arm=a.arm, init=a.init, seed=a.seed, steps=a.steps, corpus=corpus,
                                                       corpus_val=corpus_val, natural_train=natural_train, natural_val=natural_val,
                                                       items=a.items, device=a.device, ut_chunk=a.ut_chunk, d=a.d, **run_kw))
        return
    train(a.out, arm=a.arm, init=a.init, seed=a.seed, steps=a.steps, corpus=corpus, corpus_val=corpus_val, natural_train=natural_train,
          natural_val=natural_val, items=a.items, device=a.device, ut_chunk=a.ut_chunk, d=a.d, **run_kw)


def smoke(out, updates, **kw):
    """G0.4: `updates` training updates of the real run's shape (no battery; they stop on the full schedule), seconds per update
    after the first three (allocator and kernel warm-up) and the peak memory."""
    cuda = kw.get('device') == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    train(out, stop_at=updates, log=lambda *_: None, **kw)
    rows = [json.loads(l) for l in (Path(out) / 'train.jsonl').read_text(encoding='utf8').splitlines()]
    warm = min(3, len(rows) - 1)
    per = (rows[-1]['seconds'] - rows[warm - 1]['seconds']) / (len(rows) - warm) if warm else rows[-1]['seconds'] / len(rows)
    return dict(arm=kw['arm'], updates=len(rows), seconds_per_update=per,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30 if cuda else None)


if __name__ == '__main__':
    main()
