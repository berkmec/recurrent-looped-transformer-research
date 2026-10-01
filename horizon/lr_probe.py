"""U2c-lr: a development probe asking whether the hard_st instability is an artifact of the learning rate.

Context: the U2c hard_st arm destabilised at the lr inherited from U1 (0.003) — gate decisions flipped en masse
(hold_rate 0.31 → 0.56 in one step), the loss jumped 0.026 → 3.27 and the straight-through gradient reached 9.15e16.
R5 had already measured the plain Core to be lr-sensitive, so "untrainable" cannot be claimed without this sweep.

This is a DEVELOPMENT probe, deliberately narrow:
- it trains only, and looks only at training stability and the validation namespace — never test, never the neutral suites;
- it writes its own frozen protocol in its own directory and touches nothing inside the frozen U2c source set;
- it selects nothing by itself: it reports, and any protocol change remains an explicit, separate decision.

CLI: --freeze | --run-all | --analyze   (--out results/U2c-lr)
"""
import argparse
import copy
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from horizon import gated as gd
from horizon import gated_sweep as gs
from horizon import guards as gu
from horizon import sweep as sw
from horizon import task_families as tf

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('core/model.py', 'horizon/gated.py', 'horizon/guards.py', 'horizon/gated_sweep.py', 'horizon/lr_probe.py')


def protocol():
    b = sw.protocol()
    return dict(version=1, role='development probe: training stability and validation only; never test or neutral',
                purpose='is the U2c hard_st instability an artifact of the inherited learning rate 0.003?',
                steps=b['steps'], batch=b['batch'], train_length=b['train_length'], weight_decay=b['weight_decay'], clip=b['clip'],
                width=b['width'], heads=b['heads'], window=b['window'], alpha=b['alpha'], gate_k=2, gate_loss_weight=1.,
                validation_every=b['validation_every'], validation_n=b['validation_n'], eval_batch=32,
                arms=['hard_st', 'soft'], lrs=[0.003, 0.001, 0.0003, 0.0001], seeds=[117], tasks=['five'],
                source_seeds=dict(b['source_seeds']), device='cpu', threads=1,
                instability=dict(loss_jump_ratio=10., loss_jump_absolute=.5, grad_norm=1e4),
                note='soft is the stable control; the probe reports, it does not select. Run concurrently with U2c-v2, so wall times are contended and must not be used as timing evidence.')


def conditions(p):
    return [dict(arm=arm, task=task, seed=seed, lr=lr, width=p['width'], heads=p['heads'], window=p['window'], stage='u2c-lr')
            for arm in p['arms'] for task in p['tasks'] for seed in p['seeds'] for lr in p['lrs']]


def ident(c):
    return f"lr-{c['arm']}-{c['task']}-s{c['seed']}-lr{c['lr']:g}"


def source_hashes():
    return {name: sw.sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol()); sw.configure(p)
    manifest = dict(protocol=p, protocol_hash=sw.digest(p), sources=source_hashes(),
                    environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__), frozen_utc=time.time())
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
    sw.configure(manifest['protocol'])
    return manifest


def run_one(out, c):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    directory = out / ident(c); directory.mkdir(exist_ok=True)
    if (directory / 'probe.json').exists():
        return json.loads((directory / 'probe.json').read_text(encoding='utf8'))
    model = gs.make_model(dict(c, arm=c['arm']), p)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=p['weight_decay'])
    vx, vy, _ = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'validation', p['validation_n'], p['train_length'])
    counter = gu.GuardCounter(); w = gs.gate_weight(c['arm'], p)
    prev_loss = None; jumps = []; best = None; last_hold = None; t0 = time.perf_counter()
    for step in range(1, p['steps'] + 1):
        batch = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], p['train_length'])
        model.train()
        logits, gates, glogits = model(batch['x'])
        task_loss = F.cross_entropy(logits.flatten(0, 1), batch['y'].flatten())
        g_loss = gd.gate_loss(glogits, gd.gate_target(batch['x'], p['gate_k'])) if w > 0 else torch.zeros(())
        report = counter.record(gu.guarded_step(opt, model.parameters(), task_loss + w * g_loss, clip_norm=p['clip']))
        loss = report['loss']; hold = float((gates.detach() <= .5).float().mean()); last_hold = hold
        if loss is not None and prev_loss is not None and loss > prev_loss * p['instability']['loss_jump_ratio'] + p['instability']['loss_jump_absolute']:
            jumps.append(dict(step=step, before=prev_loss, after=loss, hold_before=None, hold_after=hold))
        prev_loss = loss if loss is not None else prev_loss
        row = dict(step=step, applied=report['applied'], loss=loss, grad_norm=report['grad_norm_before_clip'], hold_rate=hold)
        if step % p['validation_every'] == 0 or step == p['steps']:
            stats, _, _ = gs.predict(model, vx, vy, p)
            row['validation'] = gs.json_safe(stats)
            if best is None or stats['final_accuracy'] > best['final_accuracy']:
                best = dict(step=step, **stats)
        sw.append_line(directory / 'updates.jsonl', gs.json_safe(row))
    s = counter.summary()
    result = dict(condition=c, ident=ident(c), steps=p['steps'], guard=s, protocol_hash=manifest['protocol_hash'],
                  max_loss_jump=max((j['after'] / max(j['before'], 1e-9) for j in jumps), default=0.), loss_jumps=len(jumps), jump_steps=[j['step'] for j in jumps],
                  final_train_loss=gs.finite_or_none(prev_loss), best_validation_accuracy=best['final_accuracy'] if best else None,
                  best_validation_step=best['step'] if best else None, hold_rate_last=last_hold,
                  unstable=bool(jumps) or s['skipped'] > 0 or s['max_grad_norm_before_clip'] > p['instability']['grad_norm'],
                  seconds_contended=time.perf_counter() - t0)
    sw.write_json(directory / 'probe.json', gs.json_safe(result))
    return result


def analyze(out):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    rows = {}
    for c in conditions(p):
        path = out / ident(c) / 'probe.json'
        if path.exists():
            rows[ident(c)] = json.loads(path.read_text(encoding='utf8'))
    by_arm = {}
    for r in rows.values():
        by_arm.setdefault(r['condition']['arm'], []).append(r)
    reco = {}
    for arm, rs in by_arm.items():
        stable = [r for r in rs if not r['unstable']]
        best = max(stable, key=lambda r: (r['best_validation_accuracy'] or 0.)) if stable else None
        reco[arm] = dict(stable_lrs=[r['condition']['lr'] for r in stable], unstable_lrs=[r['condition']['lr'] for r in rs if r['unstable']],
                         best_stable_lr=best['condition']['lr'] if best else None,
                         best_stable_validation=best['best_validation_accuracy'] if best else None)
    result = dict(status='DEVELOPMENT_PROBE', protocol_hash=manifest['protocol_hash'], conditions=rows, recommendation=reco,
                  caveat='development probe: validation only, one seed, no test or neutral endpoints; any protocol change is a separate explicit decision')
    sw.write_json(out / 'analysis.json', gs.json_safe(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--run-all', action='store_true'); group.add_argument('--analyze', action='store_true')
    parser.add_argument('--out', default='results/U2c-lr')
    args = parser.parse_args()
    if args.freeze:
        result = dict(protocol_hash=freeze(args.out)['protocol_hash'])
    elif args.run_all:
        m = verify_frozen(args.out); result = {}
        for c in conditions(m['protocol']):
            r = run_one(args.out, c)
            result[ident(c)] = dict(unstable=r['unstable'], jumps=r['loss_jumps'], skipped=r['guard']['skipped'],
                                    max_norm=r['guard']['max_grad_norm_before_clip'], val=r['best_validation_accuracy'], hold=r['hold_rate_last'])
            print(ident(c), json.dumps(result[ident(c)]), f"{r['seconds_contended']:.0f}s", flush=True)
    else:
        result = analyze(args.out)['recommendation']
    print(json.dumps(result, indent=1, default=str))


if __name__ == '__main__':
    main()
