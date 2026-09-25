"""U5-scan: run the exact-preservation gate twice, changing only what the gate is allowed to read.

The question comes from putting our own measurement beside a line that already solved the same shape of problem.
Ours: 2068 bytes/s at 30M, because the decoder loop is sequential over 512 timesteps. Theirs (Mamba, S4): the
same limit, defeated by a parallel scan - which needs `s_t = a_t*s_{t-1} + b_t` with both coefficients free of s.
Our gate reads `[e_t, s_{t-1}]`, so it is not. Mamba's selectivity reads the input alone, so it is.

Hence the cycle: does the campaign's one replicated positive survive the gate losing its view of the state?

  gate_state  the sealed U2c hard_st gate, reproduced here bit for bit
  gate_input  the same thing with the head reading e_t only

BOTH ARMS RUN IN THIS CYCLE rather than comparing against U2c's stored numbers. Same machine, same seeds, same
protocol, one changed term - the method rule this campaign learned in U4-sep2 and then failed to apply in
U4-inject, where the arms turned out to differ in two terms and the headline had to be bounded afterwards.

What a positive would and would not license, stated before any data: an input-only gate is NECESSARY for a
parallel scan and NOT SUFFICIENT, because the candidate still carries the Core's additive feedback from
s_{t-1}. A surviving result makes the second and larger change worth testing; it does not make the model
scannable.

`horizon/gated_sweep.py` and `core/model.py` are hashed inside sealed manifests. Everything reusable is imported
from them; only the model factory and the arm names are new, and `tests/test_scanable.py` proves the variant is
bit-identical to the sealed gate in the arm that stands for it.

Usage: python -B -m horizon.scan_sweep --freeze --out results/U5scan
       python -B -m horizon.scan_sweep --run-all --out results/U5scan
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from horizon import gated as gd
from horizon import gated_sweep as gs
from horizon import guards as gu
from horizon import scanable as sc
from horizon import sweep as sw
from horizon import task_families as tf

ARMS = ('gate_state', 'gate_input')
SOURCES = gs.SOURCES + ('horizon/gated.py', 'horizon/scanable.py', 'horizon/scan_sweep.py')


def protocol():
    """U2c's protocol with a different arm list. Nothing else moves, so the two cycles stay comparable."""
    p = dict(gs.protocol())
    p.update(version=1, arms=list(ARMS), gate_loss_weight=0.,
             purpose='U5-scan: is the gate dependence on the committed state load-bearing, or can the gate read '
                     'the input alone? An input-only gate is the cheaper of the two conditions a parallel scan '
                     'needs; the candidate feedback is the other and is not touched here.',
             preregistration=dict(
                 prediction='exact preservation survives the gate losing its view of the state, because the '
                            'campaign measured the VALUE of exact preservation, not of state-conditioned '
                            'gating - U2c never varied what the head reads',
                 falsification='if gate_input repair falls more than 15 points below gate_state on either task '
                               'in 2 of 3 seeds, the state dependence is load-bearing and the scan route closes',
                 thresholds=dict(survives_pp=15., seeds_required=2, tasks_required=2),
                 licenses='a positive makes removing the candidate feedback worth testing; it does NOT make the '
                          'model scannable, since that second state dependence remains'))
    return p


def mode_of(arm):
    return 'hard_st'                      # both arms are the replicated positive; only the gate input differs


def gate_input_of(arm):
    return 'input' if arm == 'gate_input' else 'state'


def conditions(p):
    return [dict(arm=arm, task=task, seed=seed, width=p['width'], heads=p['heads'], window=p['window'],
                 lr=p['lr'], stage='u5scan')
            for arm in p['arms'] for task in p['tasks'] for seed in p['primary_seeds']]


def ident(c):
    return 'u5-%s-%s-w%d-s%d' % (c['arm'], c['task'], c['width'], c['seed'])


def make_model(c, p):
    core = sw.make_model(c, p)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'] + 7)          # the same stream U2c used, so the cores start identically
        return sc.ScanGatedCore(core, mode=mode_of(c['arm']), k=p['gate_k'], gate_input=gate_input_of(c['arm']))


def source_hashes():
    return {name: sw.sha(gs.ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = dict(p or protocol())
    manifest = dict(protocol=p, protocol_hash=sw.digest(p), sources=source_hashes(), frozen_utc=time.time())
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
    torch.set_num_threads(manifest['protocol'].get('threads', 1))
    return manifest


def run_one(out, c):
    """The U2c training loop, with this cycle's model factory. Structure kept identical on purpose."""
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    if ident(c) not in {ident(x) for x in conditions(p)}:
        raise ValueError('unknown condition ' + ident(c))
    directory = out / ident(c); directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        return json.loads((directory / 'train.json').read_text(encoding='utf8'))
    if any(directory.iterdir()):
        raise ValueError('incomplete condition directory requires audit; no automatic reuse: ' + str(directory))
    (directory / 'active.lock').write_text(json.dumps(dict(pid=os.getpid(), time=time.time())), encoding='utf8')
    try:
        wall = time.perf_counter()
        model = make_model(c, p)
        opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=p['weight_decay'])
        torch.save(sw.cpu_state(model), directory / 'initial.pt')
        vx, vy, _ = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'validation', p['validation_n'], p['train_length'])
        counter = gu.GuardCounter(); best, selected, train_seconds = None, None, 0.
        for step in range(1, p['steps'] + 1):
            start = time.perf_counter()
            batch = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], p['train_length'])
            model.train()
            logits, gates, glogits = model(batch['x'])
            loss = F.cross_entropy(logits.flatten(0, 1), batch['y'].flatten())   # no gate supervision in either arm
            report = counter.record(gu.guarded_step(opt, model.parameters(), loss, clip_norm=p['clip']))
            train_seconds += time.perf_counter() - start
            row = dict(step=step, applied=report['applied'], loss=report['loss'],
                       task_loss=float(loss.detach()) if math.isfinite(float(loss.detach())) else None,
                       grad_norm=report['grad_norm_before_clip'],
                       hold_rate=float((gates.detach() <= .5).float().mean()),
                       input_hash=batch['input_hash'], source_seed=batch['source_seed'])
            if step % p['validation_every'] == 0 or step == p['steps']:
                stats, _, _ = gs.predict(model, vx, vy, p); row['validation'] = gs.json_safe(stats)
                if best is None or stats['final_loss'] < best:
                    best = stats['final_loss']; selected = dict(step=step, model=sw.cpu_state(model), validation=stats)
            sw.append_line(directory / 'updates.jsonl', gs.json_safe(row))
        torch.save(selected['model'], directory / 'selected.pt'); torch.save(sw.cpu_state(model), directory / 'final.pt')
        result = dict(condition=c, ident=ident(c), steps=p['steps'], selected_step=selected['step'],
                      selected_validation=gs.json_safe(selected['validation']), guard=counter.summary(),
                      train_seconds=train_seconds, wall_seconds=time.perf_counter() - wall,
                      protocol_hash=manifest['protocol_hash'], selected_sha256=sw.sha(directory / 'selected.pt'),
                      initial_sha256=sw.sha(directory / 'initial.pt'), updates_sha256=sw.sha(directory / 'updates.jsonl'))
        sw.write_json(directory / 'train.json', gs.json_safe(result))
        sw.write_json(directory / 'complete.json', dict(ident=ident(c), utc=time.time(), train_sha256=sw.sha(directory / 'train.json')))
        return result
    finally:
        (directory / 'active.lock').unlink(missing_ok=True)


def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--run-all', action='store_true')
    parser.add_argument('--out', default='results/U5scan')
    parser.add_argument('--only', default=None)
    args = parser.parse_args()
    if args.freeze:
        m = freeze(args.out)
        print(json.dumps(dict(out=args.out, protocol_hash=m['protocol_hash'], arms=m['protocol']['arms'],
                              conditions=len(conditions(m['protocol']))))); return
    m = verify_frozen(args.out)
    wanted = [c for c in conditions(m['protocol']) if args.only in (None, ident(c))]
    if not wanted:
        raise SystemExit('no condition named %r' % args.only)
    for c in wanted:
        r = run_one(args.out, c)
        print(ident(c), 'val %.4f' % r['selected_validation']['final_accuracy'], '| %.0fs' % r['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
