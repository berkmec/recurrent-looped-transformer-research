"""M-hattı M1: train the 30M-class byte model in three arms that differ ONLY in how the state is carried forward.

plain (the sealed Core) / hard_st (exact bitwise preservation) / soft (soft mixing). Same seed, so the three arms
start from identical core weights; same data, same budget, same evaluation. The gate adds 2 305 parameters to
31 131 904, so any difference cannot be attributed to capacity.

Evaluated on both things the campaign learned to measure:
  - bits per byte on a held-out split of DISJOINT documents (the language ability),
  - needle@N for N in {0, 128, 512, 2048} and two filler kinds (the drift robustness),
because C16 established that the first is blind to the second.

fp32 throughout: B2 measured that fp16 re-encoding is not bit-identical, and these arms must be comparable.

CLI: --freeze | --probe | --run-all   (--out results/M1)
"""
import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from horizon import gated_sweep as gs
from horizon import guards as gu
from horizon import sweep as sw
from mline import data as md
from mline import lm as ml
from mline import analyze as ma
from mline import needle as nd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('core/model.py', 'horizon/gated.py', 'horizon/guards.py', 'mline/lm.py', 'mline/data.py',
           'mline/needle.py', 'mline/train.py', 'mline/analyze.py')
DATA = os.environ.get('MLINE_DATA', 'E:/rlt-research/mline-data/cnn-bytes')


def protocol():
    return dict(version=1,
                purpose='M1: does the exact-preservation gate, the one result replicated across 3 seeds x 2 task families x 2 '
                        'learning rates on the toy automata, still help at 30M on real text?',
                arms=list(ml.ARMS), seeds=[117], width=1152, heads=8, window=4, alpha=.1, gate_k=2,
                steps=4000, batch=32, length=512, lr=.0006, weight_decay=.01, clip=1.,
                validation_every=250, validation_batches=24, data=DATA, vocabulary=ml.VOCAB,
                needle_train_rate=.125, needle_train_filler=(0, 64),
                needle_filler=[0, 128, 512, 2048], needle_kinds=['repeat', 'text'],
                needle_items=ma.THRESHOLDS['items'], needle_batch=32,
                needle_placements=['far', 'near'], long_filler=[512, 2048],
                device='cuda', dtype='float32',
                selection='minimum validation bits per byte; earliest step on ties',
                targets=dict(bpb_regression=.02, needle_gain=.05),
                # The pre-registration lives HERE, inside `protocol`, so `protocol_hash` covers it and
                # `verify_frozen` refuses any post-hoc edit. For U4-sep2 the thresholds were fixed in advance but
                # only in a gitignored ledger that I could rewrite at any time and that carries no independent
                # date; that is a promise, not a record. `mline/analyze.py` is in SOURCES for the same reason, so
                # the sealed evidence pins the SCORING RULE and not merely the numbers it happens to read.
                preregistration=dict(
                    thresholds=dict(ma.THRESHOLDS),
                    statuses=list(ma.STATUSES),
                    predictions=dict(
                        a='hard_st costs no language ability: its bits per byte is within 0.02 of plain',
                        b='hard_st beats plain on needle exact match by more than 0.05 at BOTH long fillers, '
                          '512 and 2048, in the far placement',
                        c='soft stays below hard_st at both long fillers, because language state carries contents '
                          'the way the toy `stack` task did and U2c measured soft mixing leaking on that kind of '
                          'state; recorded as a prediction about WHY, and it does not change the status'),
                    falsification='if (b) fails while the length-matched near control shows the arms CAN process '
                                  'sequences of that length, the gate is a property of the toy setting and the '
                                  'M-line stops at rung one',
                    resolution=dict(
                        items=ma.THRESHOLDS['items'], batch=32,
                        # The share of the next-byte objective that must reach the four answer bytes. The first M0
                        # probe ran at 0.00098 - one gradient in a thousand, because needle rows were 83.7 % blank
                        # padding and every padded position is still a target - and learned no format at all in
                        # 81.9 MB. Packing brings it to 0.0054. The floor sits between the two so the failure is
                        # excluded and the current mix is still falsifiable; tests/test_needle_packing.py measures
                        # it rather than trusting this sentence.
                        objective_floor=.003,
                        basis='the registered effect size must be resolvable by the measurement it is read from. '
                              'The standard error of a difference of two proportions is sqrt(2p(1-p)/n). At the '
                              'original 64 items that is 0.053 near p=0.10 and 0.081 near p=0.30, so a 0.05 '
                              'threshold sat inside its own noise for any score above about 7 per cent - a coin '
                              'flip reportable as GATE_TRANSFERS. 256 items brings it to 0.041 near p=0.30, and '
                              'the rule additionally requires the gain to clear two standard errors computed from '
                              'the scores observed, which can only make a positive claim harder.',
                        cost='cheaper than the setting it replaces, not more expensive: 8 launches of 32 against '
                             '16 of 4, on a workload measured to be launch-bound rather than arithmetic-bound',
                        inherited_from='the thresholds were chosen while the needle task was still the zero-shot '
                                       'version whose format no arm was ever taught; nothing rechecked them after '
                                       'the task was redesigned'),
                    registered_before='every M1 arm: at freeze time no condition directory exists, which the '
                                      'manifest frozen_utc and the empty output directory both witness. The only '
                                      '30M training in flight is the M0 emergence probe, whose recorded role is '
                                      'budget sizing and which is excluded from evidence about the gate; its '
                                      'thresholds are not derived from any probe measurement.'),
                note='fp32 on purpose (B2: fp16 re-encoding is not bit-identical and the arms must be comparable); '
                     'the three arms share identical core weights at initialisation, verified by tests/test_mline.py. '
                     'batch 32 is the measured optimum on this card (2.70x the throughput of batch 8; batch 48 spills '
                     'to host memory and loses a factor of ten). 4000 steps is 65.5 MB per arm, about 8.5 hours each '
                     'at the measured 2148 bytes/s, which fits three arms inside the available window. '
                     'The needle FORMAT is mixed into training at short filler (0-64) and evaluated at 128, 512 and 2048 - '
                     'two, eight and thirty-two times the longest distance training ever saw, the same ladder U1 used '
                     '(trained at 128, tested to 4096). Every non-zero evaluated distance is therefore out of '
                     'distribution, so what has to generalise is the DISTANCE and not the format. Without '
                     'this the arms would be compared on a zero-shot format none of them was taught, which is exactly '
                     'what the emergence probe demonstrated before it was corrected.')


def conditions(p):
    return [dict(arm=arm, seed=seed, width=p['width'], heads=p['heads'], stage='m1') for arm in p['arms'] for seed in p['seeds']]


def ident(c):
    return f"m1-{c['arm']}-w{c['width']}-s{c['seed']}"


def known(manifest, cid):
    for c in conditions(manifest['protocol']):
        if ident(c) == cid:
            return c
    raise ValueError('unknown condition ' + cid)


def make_model(c, p):
    return ml.build(c['arm'], width=p['width'], heads=p['heads'], window=p['window'], alpha=p['alpha'], seed=c['seed'], gate_k=p['gate_k'])


def source_hashes():
    return {name: sw.sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol())
    corpus = json.loads((Path(p['data']) / 'corpus.json').read_text(encoding='utf8'))
    manifest = dict(protocol=p, protocol_hash=sw.digest(p), sources=source_hashes(),
                    corpus={k: corpus[k] for k in ('dataset', 'config', 'split', 'train_bytes', 'validation_bytes',
                                                   'train_documents', 'validation_documents', 'train_sha256', 'validation_sha256')},
                    environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                     gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                    frozen_utc=time.time())
    if (out / 'frozen.json').exists():
        old = verify_frozen(out)
        if old['protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('refusing frozen protocol change')
        return old
    if any(out.iterdir()):
        raise ValueError('new freeze requires empty output directory')
    sw.write_json(out / 'frozen.json', manifest)
    (out / 'frozen.sha256').write_text(sw.sha(out / 'frozen.json'), encoding='ascii')
    return manifest


def verify_frozen(out):
    out = Path(out)
    if sw.sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text(encoding='ascii').strip():
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((out / 'frozen.json').read_text(encoding='utf8'))
    if manifest['protocol_hash'] != sw.digest(manifest['protocol']) or manifest['sources'] != source_hashes():
        raise ValueError('protocol/source drift')
    torch.set_num_threads(1)
    return manifest


def bits_per_byte(loss):
    return loss / np.log(2)


def training_batch(p, seed, step, train=None):
    """Corpus rows plus a fixed share of needle examples, so the format is in the training distribution.

    The needles use TRAIN_KEYS, disjoint from the evaluation alphabet, and fillers of 0-128 bytes, below every
    evaluated distance except the zero-filler control. So memorising an evaluation key is impossible and the only
    thing that can carry the answer across 512 or 2048 bytes is the state.
    """
    needles = round(p['batch'] * p['needle_train_rate']) if p['needle_train_rate'] > 0 else 0
    corpus_rows = p['batch'] - needles
    x, y, input_hash = md.batch(train if train is not None else md.load(p['data'], 'train'),
                                seed, step, corpus_rows, p['length'])
    xb, yb = torch.from_numpy(x), torch.from_numpy(y)
    if needles:
        mixed = nd.training_items(needles, filler_range=tuple(p['needle_train_filler']),
                                  seed=seed * 100_003 + step, pad_to=p['length'])
        xb = torch.cat([xb, mixed['x']]); yb = torch.cat([yb, mixed['y']])
    return xb, yb, dict(needles=needles, corpus_rows=corpus_rows, input_hash=input_hash)


@torch.no_grad()
def validate(model, p, device, batches=None):
    model.eval()
    val = md.load(p['data'], 'validation')
    total, n, holds = 0., 0, []
    for step in range(1, (batches or p['validation_batches']) + 1):
        x, y, _ = md.batch(val, 991, step, p['batch'], p['length'])
        xb = torch.from_numpy(x).to(device); yb = torch.from_numpy(y).to(device)
        logits, gates = ml.forward_with_gates(model, xb)
        total += float(F.cross_entropy(logits.flatten(0, 1), yb.flatten())); n += 1
        h = ml.hold_rate(gates)
        if h is not None:
            holds.append(h)
    return dict(loss=total / n, bits_per_byte=bits_per_byte(total / n), batches=n,
                hold_rate=(sum(holds) / len(holds)) if holds else None)


def needle_suite(model, p, device):
    corpus = md.load(p['data'], 'validation')
    model.eval()
    seen = {}

    def forward(xb):
        logits, gates = ml.forward_with_gates(model, xb)
        h = ml.hold_rate(gates)
        if h is not None:
            seen.setdefault('holds', []).append(h)
        return logits

    out = {}
    for kind in p['needle_kinds']:
        for n in p['needle_filler']:
            for placement in p.get('needle_placements', ['far']):
                # `near` is the length-matched control: same sequence width, fact next to the question. At filler 0
                # there is no filler to move, so the two conditions are the same sequence and one of them is skipped.
                if placement == 'near' and n == 0:
                    continue
                seen['holds'] = []
                items = nd.make(p['needle_items'], n_filler=n, kind=kind,
                                seed=4242 + n + (0 if kind == 'repeat' else 7),
                                corpus=corpus if kind == 'text' else None, placement=placement)
                score = nd.score(forward, items, batch=p['needle_batch'], device=device)
                holds = seen.get('holds') or []
                score['hold_rate'] = (sum(holds) / len(holds)) if holds else None
                score['placement'] = placement
                out[f'{kind}@{n}' + ('' if placement == 'far' else '@' + placement)] = score
    return out


def run_one(out, c, device='cuda'):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']; known(manifest, ident(c))
    directory = out / ident(c); directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        return json.loads((directory / 'train.json').read_text(encoding='utf8'))
    if any(directory.iterdir()):
        raise ValueError('incomplete condition directory requires audit; no automatic reuse: ' + str(directory))
    (directory / 'active.lock').write_text(json.dumps(dict(pid=os.getpid(), time=time.time())), encoding='utf8')
    try:
        wall = time.perf_counter()
        model = make_model(c, p).to(device)
        counts = ml.parameter_count(model)
        opt = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
        train = md.load(p['data'], 'train')
        counter = gu.GuardCounter(); best, selected, train_seconds = None, None, 0.
        for step in range(1, p['steps'] + 1):
            start = time.perf_counter()
            xb, yb, meta = training_batch(p, c['seed'], step, train)
            xb = xb.to(device); yb = yb.to(device)
            input_hash = meta['input_hash']
            model.train()
            logits, gates = ml.forward_with_gates(model, xb)
            loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten())
            report = counter.record(gu.guarded_step(opt, model.parameters(), loss, clip_norm=p['clip']))
            train_seconds += time.perf_counter() - start
            row = dict(step=step, applied=report['applied'], loss=report['loss'], grad_norm=report['grad_norm_before_clip'],
                       bits_per_byte=bits_per_byte(report['loss']) if report['loss'] else None, input_hash=input_hash,
                       hold_rate=ml.hold_rate(gates))
            if step % p['validation_every'] == 0 or step == p['steps']:
                stats = validate(model, p, device); row['validation'] = stats
                if best is None or stats['loss'] < best:
                    best = stats['loss']; selected = dict(step=step, model={k: v.cpu().clone() for k, v in model.state_dict().items()}, validation=stats)
                print('  step %5d  train bpb %.3f  val bpb %.3f  %.2fs/step' % (step, row['bits_per_byte'] or float('nan'), stats['bits_per_byte'], train_seconds / step), flush=True)
            sw.append_line(directory / 'updates.jsonl', gs.json_safe(row))
        torch.save(selected['model'], directory / 'selected.pt')
        model.load_state_dict(selected['model']); model.to(device)
        needles = needle_suite(model, p, device)
        result = dict(condition=c, ident=ident(c), steps=p['steps'], selected_step=selected['step'], selected_validation=selected['validation'],
                      parameters=counts, guard=counter.summary(), needle=needles, train_seconds=train_seconds,
                      wall_seconds=time.perf_counter() - wall, protocol_hash=manifest['protocol_hash'],
                      selected_sha256=sw.sha(directory / 'selected.pt'), updates_sha256=sw.sha(directory / 'updates.jsonl'))
        sw.write_json(directory / 'train.json', gs.json_safe(result))
        sw.write_json(directory / 'complete.json', dict(ident=ident(c), utc=time.time(), train_sha256=sw.sha(directory / 'train.json')))
        return result
    finally:
        (directory / 'active.lock').unlink(missing_ok=True)


def probe(device='cuda', steps=8):
    """Measure before launching: step time and peak VRAM for every arm, at the frozen shape."""
    p = protocol()
    train = md.load(p['data'], 'train')
    print('%-9s %9s %12s %12s' % ('arm', 's/step', 'peak VRAM', 'params'))
    for arm in p['arms']:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = ml.build(arm, width=p['width'], heads=p['heads'], window=p['window'], alpha=p['alpha'], seed=117, gate_k=p['gate_k']).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
        t0 = None
        for step in range(1, steps + 1):
            x, y, _ = md.batch(train, 117, step, p['batch'], p['length'])
            xb = torch.from_numpy(x).to(device); yb = torch.from_numpy(y).to(device)
            model.train()
            logits = ml.forward_logits(model, xb)
            loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten())
            gu.guarded_step(opt, model.parameters(), loss, clip_norm=p['clip'])
            torch.cuda.synchronize()
            if step == 2:
                t0 = time.perf_counter()                       # skip warm-up steps
        per = (time.perf_counter() - t0) / (steps - 2)
        print('%-9s %9.3f %10.2f GB %12d' % (arm, per, torch.cuda.max_memory_allocated() / 1e9, ml.parameter_count(model)['total']))
        print('           -> %d steps = %.1f h' % (p['steps'], per * p['steps'] / 3600))
        del model, opt
        torch.cuda.empty_cache()


def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--run-all', action='store_true'); group.add_argument('--probe', action='store_true')
    parser.add_argument('--out', default='results/M1'); parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.probe:
        probe(args.device); return
    if args.freeze:
        m = freeze(args.out)
        print(json.dumps(dict(out=args.out, protocol_hash=m['protocol_hash'], arms=m['protocol']['arms'], steps=m['protocol']['steps'],
                              corpus_sha256=m['corpus']['train_sha256'][:16]))); return
    m = verify_frozen(args.out)
    for c in conditions(m['protocol']):
        r = run_one(args.out, c, args.device)
        n = r['needle']
        print(ident(c), 'val bpb %.3f' % r['selected_validation']['bits_per_byte'],
              '| needle repeat', json.dumps({k.split('@', 1)[1]: round(v['exact_match'], 3) for k, v in n.items() if k.startswith('repeat')}),
              '| text', json.dumps({k.split('@', 1)[1]: round(v['exact_match'], 3) for k, v in n.items() if k.startswith('text')}),
              '| %.0fs' % r['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
